"""
Speech-to-text engine for the speech tab.

Pipeline:
    audio (microphone or file) -> whisper (Vietnamese) -> completed sentences

Design notes:
    * faster-whisper transcribes fixed-size audio windows, but raw windows cut
      mid-sentence and are unreliable for meaning-level translation.
      We instead only emit sentences on a *final* trigger: silence, an end
      punctuation mark, or the window ending. Sentences are then translated as
      whole units, which is what the requirement asks for.
    * Audio is always decoded in-process into a float32 array. Handing a path to
      faster-whisper breaks on this stack: its decode_audio() calls
      av.open(metadata_errors=...), which the installed PyAV does not accept.
    * Two model presets, because quality and speed trade off hard in Vietnamese:
      large-v3-turbo is far more accurate but ~3.6x realtime, small is ~16x
      realtime with noticeably worse proper nouns.

Requires: pip install sounddevice faster-whisper numpy av
"""

from __future__ import annotations

import logging
import os
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

# faster-whisper downloads through huggingface_hub, which reads these once at
# import time. backend.py sets them too, but this module can be imported on its
# own, and pointing them at ~/.config/dich/models is what keeps a fresh install
# from re-downloading a model it already has.
os.environ.setdefault(
    "HF_HOME", str(Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich")))
)
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich")) / "models"),
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    str(Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich")) / "models"),
)

try:  # GPU transcription is optional: absent mlx-audio falls back to the CPU
    from . import mlx_stt
except ImportError:  # pragma: no cover
    mlx_stt = None

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Window length is the quality/latency dial, and its right value depends on the
# backend, because every transcribe() call carries fixed overhead:
#     2.0s window, small on CPU  -> ~2x realtime   (only speed that keeps up)
#     2.0s window, medium on CPU -> 0.49x realtime (falls behind)
#    10.0s window, medium on CPU -> 2.2x realtime  (lags the speaker by ~10s)
#     5.0s window, turbo on GPU  -> ~160 ms        (keeps up with room to spare)
# On the GPU the same 5s window costs little and carries more context than 2s,
# which measurably improves Vietnamese accuracy; on the CPU a 5s window would
# put the transcriber behind a live speaker, so the short window stays there.
# These are the defaults only — `window_seconds` in the config overrides them.
WINDOW_SECONDS_CPU = 2.0
WINDOW_SECONDS_GPU = 5.0
# Kept as the CPU figure so older callers and configs keep the meaning they had.
WINDOW_SECONDS = WINDOW_SECONDS_CPU
MIN_SENTENCE_CHARS = 8  # drop tiny fragments ("uh", "vâng" alone)
# Segments to accumulate before flushing. Whisper returns several segments per
# window; without a cap, continuous speech parks everything in _pending.
MAX_SEGMENTS_PER_EMIT = 4
# How much audio to hold before the oldest is dropped. A transcriber that
# cannot keep up would otherwise grow this without limit.
BUFFER_MAX_SECONDS = 600.0
# The tail share of a window that must be quieter than the window average for
# the utterance to count as finished.
TRAILING_SHARE = 0.25
TRAILING_RATIO = 0.2
# Peak amplitude below which a window is treated as silence. The decibel
# threshold in _trailing_silence uses an RMS figure; this is the loudest single
# sample, which catches the quiet-but-warm mic noise a real device produces.
SILENCE_FLOOR = 0.003
# Whisper reports how likely a segment is to be non-speech. Above this, the
# segment is a silence artefact rather than speech. (The CPU path never needed
# it because faster-whisper's vad_filter removed those windows; the GPU path
# has no such filter, so the file path must drop them itself.)
NO_SPEECH_THRESHOLD = 0.6
# Batch/file transcription favours accuracy; live mic keeps up with a speaker.
FILE_MODEL = "large-v3-turbo"
LIVE_MODEL = "small"
DEFAULT_MODEL = LIVE_MODEL

# (model id, UI label carrying its measured realtime cost)
SPEECH_MODELS = [
    ("tiny", "tiny — very rough, 16× realtime"),
    ("base", "base — rough, 8× realtime"),
    ("small", "small — keeps up on CPU, ~2× realtime"),
    ("medium", "medium — better names, 0.5× at 2s / 2.2× at 10s"),
    ("large-v3-turbo", "large-v3-turbo — best, 31× on GPU (GPU default)"),
    ("qwen3-0.6b", "Qwen3-ASR 0.6B — fastest, GPU only"),
    ("qwen3-1.7b", "Qwen3-ASR 1.7B — strong, GPU only"),
]
SPEECH_MODEL_IDS = [name for name, _label in SPEECH_MODELS]


def speech_model_label(model: str) -> str:
    """UI label for a model id, carrying its measured realtime cost."""
    for name, label in SPEECH_MODELS:
        if name == model:
            return label
    return model


_END_PUNCT = ".?!。！？…"
# vietnamese final particles / connectors that usually close a sentence
_END_WORDS = {
    "rồi",
    "xong",
    "nhé",
    "nha",
    "nha bạn",
    "được",
    "cảm ơn",
    "vâng",
    "dạ",
    "ừ",
    "ừa",
    "thôi",
    "nào",
    "sao",
    "không",
    "chưa",
}


class SpeechEngine:
    """Records from the default microphone and yields completed Vietnamese sentences."""

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL,
        device_index: Optional[int] = None,
        window_seconds: float = WINDOW_SECONDS,
        buffer_max_seconds: float = BUFFER_MAX_SECONDS,
    ):
        self.model_size = model_size
        self.device_index = device_index
        self.window_seconds = max(0.5, float(window_seconds))
        # The cap stops a transcriber that cannot keep up from growing the
        # buffer without limit, while leaving enough audio to catch up after a
        # pause. 600s of mono float32 is roughly 38 MB; the ceiling here allows
        # the several-hundred-MB case where a backend pass re-reads the audio.
        self.buffer_max_seconds = max(10.0, float(buffer_max_seconds))
        # Numpy blocks, not a list of Python floats: the audio callback fires
        # ~30x a second and copying through .tolist() then back through
        # np.asarray() costs more than the transcription for a 3s window.
        self._buffer: "deque" = deque()
        self._buffered = 0
        self._dropped = 0
        self._buffer_lock = threading.Lock()
        self._pending = ""
        self._pending_segments = 0
        self._last_samples = None
        # Running peak of the captured audio, for the input-level meter. It
        # decays on every read so the meter follows speech instead of latching
        # onto the loudest moment of the session.
        self._peak = 0.0
        # Archived raw microphone input (16 kHz mono float32) for later
        # re-transcription or refinement. Kept separately from the transcribe
        # buffer so saving a recording does not disturb the live pipeline.
        self._archive_enabled = False
        self._archive_dir: Optional[Path] = None
        self._archive_blocks: list = []
        self._archive_lock = threading.Lock()
        self._last_archive_path: Optional[Path] = None
        self._final: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stream = None

    # -- lifecycle --------------------------------------------------------
    def start(self, archive_dir: Optional[str] = None):
        import sounddevice as sd

        self._model = _whisper_model(self.model_size)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device_index,
            callback=self._on_audio,
            blocksize=int(SAMPLE_RATE * 0.5),
        )
        self._stop.clear()
        self._stream.start()
        self._threads = [
            threading.Thread(target=self._transcribe_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

        if archive_dir:
            self._archive_dir = Path(archive_dir)
            self._archive_dir.mkdir(parents=True, exist_ok=True)
            self._archive_blocks = []
            self._archive_enabled = True

    def _on_audio(self, indata, frames, time_info, status):  # noqa: ARG001
        """Append one input block; drop the oldest audio once past the ceiling."""
        with self._buffer_lock:
            mono = np.ascontiguousarray(indata[:, 0])
            self._buffer.append((mono, len(mono)))
            self._buffered += len(mono)
            if mono.size:
                self._peak = max(self._peak, float(np.abs(mono).max()))
            if self._archive_enabled:
                with self._archive_lock:
                    self._archive_blocks.append(mono.copy())
            # Drop the oldest audio once past the cap. A transcriber that
            # cannot keep up would otherwise grow this without limit.
            limit = int(self.buffer_max_seconds * SAMPLE_RATE)
            while self._buffered > limit and len(self._buffer) > 1:
                dropped = self._buffer.popleft()
                self._buffered -= dropped[1]
                self._dropped += dropped[1]

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        # Flush whatever is left as a final sentence.
        self._flush_pending()
        return self._finalize_archive()

    def close(self):
        self.stop()

    @property
    def archive_dir(self) -> Optional[Path]:
        """Directory recordings are saved into, or None when archiving is off."""
        return self._archive_dir

    @property
    def last_archive_path(self) -> Optional[Path]:
        """Path of the most recently written archive WAV, if any."""
        return self._last_archive_path

    def _finalize_archive(self) -> Optional[Path]:
        """
        Write the archived microphone input as a 16-bit WAV file.

        Returns the path written, or None when archiving was not enabled or
        nothing was captured.
        """
        import wave

        if not self._archive_enabled:
            return None
        self._archive_enabled = False
        with self._archive_lock:
            blocks = list(self._archive_blocks)
            self._archive_blocks = []
        if not blocks or self._archive_dir is None:
            return None
        audio = np.concatenate(blocks)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        path = self._archive_dir / f"recording-{stamp}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(
                (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            )
        self._last_archive_path = path
        logger.info(f"Archived recording to {path}")
        return path

    # -- audio capture ----------------------------------------------------
    def buffer_usage(self) -> dict:
        """
        Memory held by the capture buffer, plus where it stands against the
        window.

        `behind_by` is the number the UI should show: it is how far the
        transcriber trails a live speaker, and a countdown that reads 0.0 while
        nothing is arriving is worse than saying so.
        """
        with self._buffer_lock:
            samples = self._buffered
            blocks = len(self._buffer)
            dropped = self._dropped
            peak = self._peak
        # Decay the meter on each read. The UI polls every 500 ms, so this
        # settles within a couple of seconds of the speaker going quiet.
        self._peak = peak * 0.6
        return {
            "samples": samples,
            "seconds": samples / SAMPLE_RATE,
            "bytes": samples * 4,  # float32 mono
            "blocks": blocks,
            "dropped_seconds": dropped / SAMPLE_RATE,
            "window_seconds": self.window_seconds,
            "buffer_max_seconds": self.buffer_max_seconds,
            "peak": peak,
            # How far the transcriber trails the speaker. Positive means it is
            # behind; a countdown that reads 0.0 while nothing is arriving is
            # worse than saying so.
            "behind_by": max(0.0, samples / SAMPLE_RATE - self.window_seconds),
        }

    def _take(self, seconds: float):
        """
        Pop up to `seconds` of audio as one contiguous float32 array.

        Blocks are drained only when a whole window is available, so a partial
        window stays buffered until it can be transcribed in one piece.
        """
        import numpy as np

        want = int(seconds * SAMPLE_RATE)
        with self._buffer_lock:
            if self._buffered < want:
                return None
            parts, remaining = [], want
            while remaining > 0 and self._buffer:
                block, length = self._buffer[0]
                if length <= remaining:
                    parts.append(block)
                    remaining -= length
                    self._buffer.popleft()
                    self._buffered -= length
                else:
                    parts.append(block[:remaining])
                    self._buffer[0] = (block[remaining:], length - remaining)
                    remaining = 0
            if not parts:
                return None
            return parts[0] if len(parts) == 1 else np.concatenate(parts)

    # -- transcription ----------------------------------------------------
    def _transcribe_loop(self):
        while not self._stop.is_set():
            audio = self._take(self.window_seconds)
            if audio is None:
                time.sleep(0.05)
                continue
            self._process_window(audio)

    def _process_window(self, audio: list):
        """
        Transcribe one window of audio into the pending sentence buffer.

        The Whisper settings matter more than they look:
          * vad_filter - without it, every window of near-silence is transcribed
            anyway, and Whisper hallucinates on silence. All-zero audio came
            back as a full sentence of YouTube outro text.
          * condition_on_previous_text - when true, a hallucination feeds itself
            and the same phrase repeats for the rest of the session.
          * initial_prompt - a Vietnamese hint is echoed back verbatim by the
            decoder on silence, so it does more harm than good here.
        """
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        if samples.size == 0:
            return
        if float(np.abs(samples).max()) < SILENCE_FLOOR:
            return  # effectively quiet; do not let the model invent text
        try:
            segments, _info = self._model.transcribe(
                samples,
                language="vi",
                beam_size=5,
                vad_filter=True,  # drop non-speech
                condition_on_previous_text=False,  # no hallucination feedback
            )
        except Exception as e:  # noqa: BLE001 - a failed window must not kill capture
            logger.warning(f"Transcribe window failed: {type(e).__name__}: {e}")
            return

        self._last_samples = samples
        self._ingest_segments(segments, len(samples) / SAMPLE_RATE)

    def _ingest_segments(self, segments, window_seconds: float) -> None:
        """
        Fold transcribed segments into sentences and queue the finished ones.

        Split out from _process_window so the sentence-assembly rules can be
        exercised without a microphone: they are the part with the subtle
        failure modes.
        """
        for seg in segments:
            text = _strip_hallucinations(seg.text.strip())
            # A repeated phrase is Whisper's other silence artefact, and the
            # marker list cannot catch it because the phrase itself is real.
            if not text or _is_repetition(text):
                continue
            self._pending = f"{self._pending} {text}".strip()
            self._pending_segments += 1

            # Flush on any of three triggers. The first alone is not enough: a
            # sentence-final marker frequently never appears in speech, and
            # comparing a segment's length against the whole window cannot fire
            # because a segment cannot span the window. Without the segment
            # count, continuous speech accumulates in _pending and nothing is
            # ever emitted.
            near_window_end = seg.end >= window_seconds - 1.5
            if (
                self._is_complete(self._pending)
                or self._pending_segments >= MAX_SEGMENTS_PER_EMIT
                or near_window_end
            ):
                self._emit(self._pending)
                self._pending = ""

        # Trailing silence closes the utterance. Judged against the window's own
        # level rather than a fixed number: an absolute 0.005 sits above ordinary
        # room noise, so it never fired in a quiet room.
        if self._pending and self._trailing_silence(self._last_samples):
            self._emit(self._pending)
            self._pending = ""

    def _is_complete(self, text: str) -> bool:
        """Heuristic: does this chunk look like a finished sentence?"""
        stripped = text.rstrip()
        # endswith() with a plain string checks the whole string as one suffix,
        # so ".?!。！？…" matched only text ending in that entire sequence —
        # never. The tuple makes each mark a candidate.
        if stripped.endswith(tuple(_END_PUNCT)):
            return True
        last = stripped.split()[-1].lower().strip(",.!?…") if stripped.split() else ""
        return last in _END_WORDS

    def _trailing_silence(self, samples) -> bool:
        """
        True if the tail of the window is much quieter than the window as a
        whole.

        Relative, not absolute: an absolute RMS threshold sits above ordinary
        room noise, so in a quiet room it never fired and an utterance was left
        hanging in `_pending` even though the speaker had finished.
        """
        import numpy as np

        if samples is None or len(samples) == 0:
            return False
        tail = samples[int(len(samples) * (1.0 - TRAILING_SHARE)) :]
        if len(tail) == 0:
            return False
        whole = float(np.sqrt(np.mean(samples**2)))
        tail_rms = float(np.sqrt(np.mean(tail**2)))
        if whole <= 0.0:
            return False
        return tail_rms < whole * TRAILING_RATIO

    def _emit(self, text: str):
        cleaned = " ".join(text.split())
        if len(cleaned) < MIN_SENTENCE_CHARS:
            return
        self._final.put(cleaned)
        self._pending_segments = 0

    def _flush_pending(self):
        if self._pending.strip():
            self._emit(self._pending)
            self._pending = ""

    def latest_partial(self) -> str:
        """
        The utterance currently being assembled.

        This is the same `_pending` buffer the sentence rules build on, so a
        partial transcript is simply a view of it: it fills as windows are
        transcribed and empties when a sentence is emitted. Nothing extra to
        keep in sync, and nothing to leak into the translation path.
        """
        return self._pending

    # -- consumption ------------------------------------------------------
    def final_sentences(self) -> Iterator[str]:
        while True:
            try:
                yield self._final.get(timeout=0.25)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue

    def drain_final_sentences(self) -> list[str]:
        out = []
        while True:
            try:
                out.append(self._final.get_nowait())
            except queue.Empty:
                return out


_HALLUCINATION_MARKERS = (
    "subscribe",
    "đăng ký kênh",
    "không bỏ lỡ",
    "video hấp dẫn",
    "hẹn gặp lại",
    "video này",
    "video sau",
    "cảm ơn các bạn",
    "cảm ơn mọi người",
    "cảm ơn quý vị",
    "quay lại trong video",
    "nhấn like",
    "hãy like",
    "xin chào và chào mừng",
    "chào mừng bạn đến với",
    "giữ nguyên dấu",
    "transcribe the following",
    "thank you for watching",
    "please subscribe",
    "thanks for watching",
)


def _is_hallucination(text: str) -> bool:
    """True when the whole string is a decoding artefact."""
    lowered = text.lower()
    return any(marker in lowered for marker in _HALLUCINATION_MARKERS)


def _is_repetition(text: str) -> bool:
    """
    True when the text is the same phrase repeated, which is Whisper's other
    silence artefact and the one the marker list cannot catch because the
    phrase itself is not boilerplate.

    Seen live on speaker echo: "Công phí dưỡng, công phí dưỡng, công phí
    dưỡng..." -- a real phrase, repeated into nonsense. Only flags when a span
    repeats, since a word naturally recurring in speech is normal.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < 6:
        return False
    for size in (2, 3, 4, 5):
        if len(words) < size * 2:
            continue
        unit = tuple(words[:size])
        if len(unit) != len(set(unit)):
            continue  # "của của của" is a genuine repeat of one word
        run = 1
        for start in range(size, len(words) - size + 1, size):
            if tuple(words[start : start + size]) == unit:
                run += 1
            else:
                break
        if run >= 3:
            return True
    return False


def _strip_hallucinations(text: str) -> str:
    """
    Remove hallucinated boilerplate from *within* a sentence.

    Whisper's silence artefacts do not always arrive as a clean standalone
    segment: they get spliced into real speech, so "à không con đi ở nhà Cảm
    ơn các bạn và hẹn gặp lại video này cho thèo à" is one segment that must be
    trimmed rather than dropped. Sentences are split on punctuation and any
    part containing a marker is discarded.
    """
    import re

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    kept = [p for p in parts if p.strip() and not _is_hallucination(p)]
    return " ".join(kept).strip()


# ---------------------------------------------------------------------------
# Module-level singleton used by the UI
# ---------------------------------------------------------------------------
_engine: Optional[SpeechEngine] = None


def list_input_devices() -> List[tuple]:
    """
    Enumerate microphones as (device_index, display_name) pairs.

    The index is what PortAudio/sounddevice needs; the name is what the UI
    shows. Returns [] when no input device exists, so the caller can say so
    rather than crashing on an empty list.
    """
    try:
        import sounddevice as sd
    except ImportError:
        return []

    try:
        devices = sd.query_devices()
    except Exception as e:  # noqa: BLE001 - audio subsystem can be unavailable
        logger.warning(f"Could not query audio devices: {type(e).__name__}: {e}")
        return []

    result: List[tuple] = []
    for index, info in enumerate(devices):
        if info.get("max_input_channels", 0) > 0:
            name = info.get("name", f"device {index}")
            rate = info.get("default_samplerate", 0)
            result.append((index, f"{name} ({rate:.0f} Hz)"))
    return result


def default_input_device() -> Optional[int]:
    """Index of the system default microphone, or None if there isn't one."""
    try:
        import sounddevice as sd

        info = sd.query_devices(kind="input")
        return int(info["index"])
    except Exception:  # noqa: BLE001 - no audio subsystem or no device
        return None


def _validate_device(device_index: Optional[int]) -> Optional[int]:
    """
    Fall back to the system default when the configured index is stale.

    Happens after a mic is unplugged or device order changes, which would
    otherwise fail at stream start with a PortAudio error the user cannot act on.
    """
    if device_index is None:
        return None
    valid = [index for index, _name in list_input_devices()]
    if device_index in valid:
        return device_index
    logger.warning(
        f"Configured mic {device_index} is no longer present; "
        f"using the system default instead"
    )
    return None


def start_mic_stream(
    model_size: Optional[str] = None,
    device_index: Optional[int] = None,
    window_seconds: Optional[float] = None,
    buffer_max_seconds: float = BUFFER_MAX_SECONDS,
    archive_dir: Optional[str] = None,
) -> SpeechEngine:
    """
    Open the microphone and begin transcribing.

    Args:
        model_size: Whisper model. None uses the backend-appropriate default
            (the accurate model on the GPU, `small` on the CPU).
        device_index: Microphone to open; None uses the system default.
        window_seconds: Audio per transcription window. Shorter means the first
            words appear sooner, at the cost of realtime factor. None uses the
            backend-appropriate default.
        buffer_max_seconds: Audio to hold before the oldest is dropped.
        archive_dir: When set, raw microphone input is saved as WAV files here
            for later re-transcription or refinement.
    """
    global _engine
    if _engine is None:
        _engine = SpeechEngine(
            model_size=model_size or default_live_model(),
            device_index=_validate_device(device_index),
            window_seconds=(
                default_window_seconds() if window_seconds is None else window_seconds
            ),
            buffer_max_seconds=buffer_max_seconds,
        )
    if getattr(_engine, "_stream", None) is None:
        if archive_dir is not None:
            _engine.start(archive_dir=archive_dir)
        else:
            _engine.start()
    return _engine


def stop_mic_stream() -> Optional[Path]:
    """
    Stop the microphone stream.

    Returns the path of the archived recording, if archiving was enabled.
    """
    global _engine
    if _engine is not None:
        path = _engine.stop()
        _engine = None
        return path
    return None


# ---------------------------------------------------------------------------
# Audio file input
# ---------------------------------------------------------------------------


def _whisper_model(name: str, backend: Optional[str] = None):
    """
    Build a Whisper model on the configured backend.

    GPU first: the same weights are ~10x faster on MLX (see mlx_stt), which is
    what lets the accurate model also be the live one. The CPU path stays as
    the fallback, and is what runs when mlx-audio is absent.
    """
    if (backend or _configured_backend()) == "mlx" and mlx_stt is not None:
        if mlx_stt.available():
            return mlx_stt.make_model(name)
        logger.warning("stt_backend=mlx but no GPU is reachable; using the CPU")
    if mlx_stt is not None and mlx_stt.is_gpu_only(name):
        # A GPU-only preset (Qwen3) has no faster-whisper equivalent; silently
        # handing "qwen3-0.6b" to WhisperModel would fail at load time.
        logger.warning(f"{name} needs the GPU backend; falling back to {LIVE_MODEL}")
        name = LIVE_MODEL
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "Speech needs faster-whisper. Install: pip install faster-whisper"
        ) from e
    return WhisperModel(name, device="cpu", compute_type="int8")


def stt_backend(config_value: str = "auto", gpu_available: bool = False) -> str:
    """
    Which STT backend a config value selects: "mlx" or "ctranslate2".

    "auto" prefers the GPU, because on this machine it is strictly faster and
    the accuracy/latency tradeoff the CPU forced (small keeps up, turbo does
    not) does not exist there. An explicit request is honoured even when it
    cannot run, so the fallback warning is visible rather than silent.
    """
    choice = (config_value or "auto").strip().lower()
    if choice in ("mlx", "gpu"):
        return "mlx"
    if choice in ("ctranslate2", "cpu", "faster-whisper"):
        return "ctranslate2"
    return "mlx" if gpu_available else "ctranslate2"


def _gpu_available() -> bool:
    try:
        return mlx_stt is not None and mlx_stt.available()
    except Exception:  # noqa: BLE001 - a broken probe must not break startup
        return False


def _configured_backend() -> str:
    choice = "auto"
    try:
        from . import backend as _backend

        choice = str(_backend.CONFIG.get("stt_backend", "auto"))
    except Exception:  # noqa: BLE001 - config is unavailable in some entry points
        pass
    return stt_backend(choice, gpu_available=_gpu_available())


def default_window_seconds() -> float:
    """
    Window length to use when the config does not name one.

    The GPU can afford the longer window (see the constants at the top): more
    context per decode for a latency the model still absorbs. The CPU cannot, so
    it keeps the short window.
    """
    try:
        return (
            WINDOW_SECONDS_GPU if _configured_backend() == "mlx" else WINDOW_SECONDS_CPU
        )
    except Exception:  # noqa: BLE001 - a broken probe must not break the UI
        return WINDOW_SECONDS_CPU


def default_live_model() -> str:
    """
    Live model to use when the config does not name one.

    `small` was chosen for the CPU, where it is the only model that keeps up.
    On the GPU the accurate model is also the fast one — turbo decodes a 5s
    window in ~160 ms — so the tradeoff that forced `small` no longer exists,
    and the accurate model is what makes unfamiliar voices and proper nouns
    come through. An explicit `mic_model` in the config still wins.
    """
    try:
        if _configured_backend() == "mlx":
            return "large-v3-turbo"
    except Exception:  # noqa: BLE001 - a broken probe must not break the UI
        pass
    return LIVE_MODEL


def load_audio(
    path: str,
    sample_rate: int = SAMPLE_RATE,
    start: float = 0.0,
    duration: Optional[float] = None,
    on_progress: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """
    Decode any audio file to a mono float32 array at `sample_rate`.

    Decoded here rather than passed as a path because faster-whisper's
    decode_audio() calls av.open(metadata_errors=...), which the installed PyAV
    rejects. Decoding at load time also lets a chunk of a long recording be
    read without holding the whole thing in memory.

    `on_progress` receives decoded percent (0-100) as frames are read, so a
    minute-long file is not a blank stare.
    """
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: List[bytes] = []
    collected = 0.0
    position = 0.0
    reported = -1

    # PyAV renamed/dropped `metadata_errors` across versions. faster-whisper
    # passes it unconditionally and explodes on this stack, so the call is made
    # defensively rather than committing to either form.
    try:
        container = av.open(path, metadata_errors="ignore")
    except TypeError:
        container = av.open(path)

    with container:
        # container.duration is in av.time_base units and can be unset (VBR
        # mp3 and friends); fall back to the stream's own duration.
        total = None
        if getattr(container, "duration", None):
            total = float(container.duration) / av.time_base
        elif container.streams.audio[0].duration:
            stream = container.streams.audio[0]
            total = float(stream.duration) * float(stream.time_base)

        for frame in container.decode(container.streams.audio[0]):
            frame_seconds = float(frame.samples) / max(frame.sample_rate, 1)
            if position + frame_seconds <= start:
                position += frame_seconds  # fast-forward past `start`
                continue
            for packet in resampler.resample(frame):
                pcm = packet.to_ndarray().reshape(-1)
                chunks.append(pcm.tobytes())
                collected += len(pcm) / sample_rate
            position += frame_seconds
            if on_progress is not None and total:
                percent = min(100, int(position / total * 100))
                if percent != reported:
                    reported = percent
                    on_progress(percent)
            if duration is not None and collected >= duration:
                break

    if on_progress is not None and reported != 100:
        on_progress(100)

    audio = np.frombuffer(b"".join(chunks), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


class TranscriptionResult:
    """Sentences transcribed from one audio source."""

    def __init__(
        self,
        language: str = "",
        language_probability: float = 0.0,
        model: str = "",
        duration: float = 0.0,
    ):
        self.language = language
        self.language_probability = language_probability
        self.model = model
        self.duration = duration
        self.sentences: List[str] = []
        self.timings: List[tuple] = []

    @property
    def text(self) -> str:
        return "\n".join(self.sentences)

    def __repr__(self) -> str:
        return (
            f"TranscriptionResult(lang={self.language}, "
            f"sentences={len(self.sentences)}, duration={self.duration}s)"
        )


def transcribe_file(
    path: str,
    model_name: str = FILE_MODEL,
    max_seconds: Optional[float] = None,
    on_progress: Optional[Callable[[str, Optional[int]], None]] = None,
) -> TranscriptionResult:
    """
    Transcribe an audio file to Vietnamese.

    Defaults to the accurate-but-slow model: a file is not racing a live
    speaker, so quality wins. Pass LIVE_MODEL when a long recording has to
    finish quickly.

    `on_progress(phase, percent)` reports every stage that can block for a long
    time: "decode" (percent of the file read), "model" (percent None — first
    use downloads gigabytes) and "transcribe" (percent of the audio consumed).
    """
    report = on_progress or (lambda phase, percent: None)

    report("decode", 0)
    audio = load_audio(
        path,
        duration=max_seconds,
        on_progress=lambda percent: report("decode", percent),
    )
    report("decode", 100)
    logger.info(f"Decoded {path}: {len(audio) / SAMPLE_RATE:.1f}s at {SAMPLE_RATE} Hz")

    duration = round(len(audio) / SAMPLE_RATE, 1)
    if audio.size == 0 or float(np.abs(audio).max()) < SILENCE_FLOOR:
        # Never hand a silent buffer to a model on any backend: Whisper invents
        # text on silence, and the GPU path has no vad_filter to hide it. The
        # CPU path's vad_filter was the only thing doing this before, which is
        # why moving to the GPU made silent files start producing sentences.
        logger.info(f"Effectively silent: {path} — not transcribing")
        return TranscriptionResult(model=model_name, duration=duration)

    report("model", None)
    model = _whisper_model(model_name)
    report("transcribe", 0)

    segments, info = model.transcribe(
        audio,
        language="vi",
        beam_size=5,
        vad_filter=True,  # skip silence instead of noise
        condition_on_previous_text=True,  # carry context across sentences
    )

    result = TranscriptionResult(
        language=info.language,
        language_probability=round(float(info.language_probability), 3),
        model=model_name,
        duration=duration,
    )
    total = float(getattr(info, "duration", 0.0) or 0.0) or result.duration
    for seg in segments:
        text = seg.text.strip()
        # A segment Whisper itself flags as non-speech is a silence artefact,
        # not speech that happens to be quiet.
        no_speech = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
        if text and no_speech < NO_SPEECH_THRESHOLD and not _is_hallucination(text):
            result.sentences.append(text)
            result.timings.append((round(seg.start, 2), round(seg.end, 2)))
        if total > 0:
            report("transcribe", min(99, int(seg.end / total * 100)))
    report("transcribe", 100)
    return result


def drain_final_sentences() -> list[str]:
    if _engine is None:
        return []
    return _engine.drain_final_sentences()


def latest_partial() -> str:
    """The sentence still being assembled, or "" when nothing is mid-flight."""
    if _engine is None:
        return ""
    return _engine.latest_partial()


# The idle report must carry every key the live engine reports. The UI reads
# window_seconds unconditionally, and a shorter dict here made the memory
# readout crash — inside open_audio(), before transcription was dispatched.
_IDLE_USAGE = {
    "samples": 0,
    "seconds": 0.0,
    "bytes": 0,
    "blocks": 0,
    "dropped_seconds": 0.0,
    "window_seconds": 0.0,
    "buffer_max_seconds": 0.0,
    "behind_by": 0.0,
    "peak": 0.0,
}


def buffer_usage() -> dict:
    """Buffer memory held by the active capture, or zeros when idle."""
    if _engine is None:
        return dict(_IDLE_USAGE)
    return _engine.buffer_usage()


def final_sentences() -> Iterator[str]:
    if _engine is None:
        return
    yield from _engine.final_sentences()
