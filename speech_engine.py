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
from typing import Iterator, List, Optional

import numpy as np

# faster-whisper downloads through huggingface_hub, which reads these once at
# import time. backend.py sets them too, but this module can be imported on its
# own, and pointing them at ~/.config/dich/models is what keeps a fresh install
# from re-downloading a model it already has.
os.environ.setdefault("HF_HOME", str(Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich"))))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich")) / "models"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(os.environ.get("DICH_HOME", Path.home() / ".config" / "dich")) / "models"))

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# A short window is what makes the first words appear quickly. The cost is the
# realtime factor, because every transcribe() call carries fixed overhead.
# Measured warm for Vietnamese on this M5:
#     2.0s window, medium -> 0.49x realtime  (text appears fast, falls behind)
#     2.0s window, small  -> ~2x realtime    (keeps up)
#    10.0s window, medium -> 2.2x realtime   (lags the speaker by ~10s)
# Falling behind is tolerable while the buffer still holds the audio -- it
# catches up whenever the speaker pauses. Keeping up is why `small` is the
# live default rather than `medium`.
WINDOW_SECONDS = 2.0
SILENCE_SECONDS = 0.8         # trailing silence that ends an utterance
MIN_SENTENCE_CHARS = 8        # drop tiny fragments ("uh", "vâng" alone)
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
# Batch/file transcription favours accuracy; live mic keeps up with a speaker.
# Realtime depends on the window as much as the model, because every
# transcribe() call carries fixed overhead. Measured warm for Vietnamese here:
#     2.0s window, medium -> 0.49x realtime  (text appears fast, falls behind)
#     2.0s window, small  -> ~2x realtime    (keeps up)
#    10.0s window, medium -> 2.2x realtime   (lags the speaker by ~10s)
# Falling behind is tolerable while the buffer holds the audio -- it catches up
# whenever the speaker pauses -- but keeping up is the default, so `small`.
FILE_MODEL = "large-v3-turbo"
LIVE_MODEL = "small"
DEFAULT_MODEL = LIVE_MODEL

# (model id, UI label carrying its measured realtime cost)
SPEECH_MODELS = [
    ("tiny", "tiny — very rough, 16× realtime"),
    ("base", "base — rough, 8× realtime"),
    ("small", "small — keeps up, ~2× realtime (default)"),
    ("medium", "medium — better names, 0.5× at 2s / 2.2× at 10s"),
    ("large-v3-turbo", "large-v3-turbo — best, slowest"),
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
    "rồi", "xong", "nhé", "nha", "nha bạn", "được", "cảm ơn",
    "vâng", "dạ", "ừ", "ừa", "thôi", "nào", "sao", "không", "chưa",
}


class SpeechEngine:
    """Records from the default microphone and yields completed Vietnamese sentences."""

    def __init__(self, model_size: str = DEFAULT_MODEL, device_index: Optional[int] = None,
                 window_seconds: float = WINDOW_SECONDS,
                 buffer_max_seconds: float = BUFFER_MAX_SECONDS):
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
        self._final: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stream = None

    # -- lifecycle --------------------------------------------------------
    def start(self):
        import sounddevice as sd

        self._model = _whisper_model(self.model_size)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=self.device_index, callback=self._on_audio,
            blocksize=int(SAMPLE_RATE * 0.5),
        )
        self._stop.clear()
        self._stream.start()
        self._threads = [
            threading.Thread(target=self._transcribe_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

    def _on_audio(self, indata, frames, time_info, status):  # noqa: ARG001
        """Append one input block; drop the oldest audio once past the ceiling."""
        with self._buffer_lock:
            mono = np.ascontiguousarray(indata[:, 0])
            self._buffer.append((mono, len(mono)))
            self._buffered += len(mono)
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

    def close(self):
        self.stop()

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
        return {
            "samples": samples,
            "seconds": samples / SAMPLE_RATE,
            "bytes": samples * 4,          # float32 mono
            "blocks": blocks,
            "dropped_seconds": dropped / SAMPLE_RATE,
            "window_seconds": self.window_seconds,
            "buffer_max_seconds": self.buffer_max_seconds,
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
                samples, language="vi", beam_size=5,
                vad_filter=True,                     # drop non-speech
                condition_on_previous_text=False,    # no hallucination feedback
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
            if (self._is_complete(self._pending)
                    or self._pending_segments >= MAX_SEGMENTS_PER_EMIT
                    or near_window_end):
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
        if stripped.endswith(_END_PUNCT):
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

        if len(samples) == 0:
            return False
        tail = samples[int(len(samples) * (1.0 - TRAILING_SHARE)):]
        if len(tail) == 0:
            return False
        whole = float(np.sqrt(np.mean(samples ** 2)))
        tail_rms = float(np.sqrt(np.mean(tail ** 2)))
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
            if tuple(words[start:start + size]) == unit:
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


def _prompt_hint() -> str:
    return ""


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
    model_size: str = DEFAULT_MODEL,
    device_index: Optional[int] = None,
    window_seconds: float = WINDOW_SECONDS,
    buffer_max_seconds: float = BUFFER_MAX_SECONDS,
) -> SpeechEngine:
    """
    Open the microphone and begin transcribing.

    Args:
        model_size: Whisper model. Use LIVE_MODEL to keep up with a speaker.
        device_index: Microphone to open; None uses the system default.
        window_seconds: Audio per transcription window. Shorter means the first
            words appear sooner, at the cost of realtime factor.
        buffer_max_seconds: Audio to hold before the oldest is dropped.
    """
    global _engine
    if _engine is None:
        _engine = SpeechEngine(
            model_size=model_size,
            device_index=_validate_device(device_index),
            window_seconds=window_seconds,
            buffer_max_seconds=buffer_max_seconds,
        )
        _engine.start()
    return _engine


def stop_mic_stream():
    global _engine
    if _engine is not None:
        _engine.stop()
        _engine = None


# ---------------------------------------------------------------------------
# Audio file input
# ---------------------------------------------------------------------------

def _whisper_model(name: str):
    """Build a WhisperModel, with an install hint instead of a raw ImportError."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ImportError(
            "Speech needs faster-whisper. Install: pip install faster-whisper"
        ) from e
    return WhisperModel(name, device="cpu", compute_type="int8")


def load_audio(
    path: str,
    sample_rate: int = SAMPLE_RATE,
    start: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    """
    Decode any audio file to a mono float32 array at `sample_rate`.

    Decoded here rather than passed as a path because faster-whisper's
    decode_audio() calls av.open(metadata_errors=...), which the installed PyAV
    rejects. Decoding at load time also lets a chunk of a long recording be
    read without holding the whole thing in memory.
    """
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: List[bytes] = []
    collected = 0.0
    position = 0.0

    # PyAV renamed/dropped `metadata_errors` across versions. faster-whisper
    # passes it unconditionally and explodes on this stack, so the call is made
    # defensively rather than committing to either form.
    try:
        container = av.open(path, metadata_errors="ignore")
    except TypeError:
        container = av.open(path)

    with container:
        for frame in container.decode(container.streams.audio[0]):
            frame_seconds = float(frame.samples) / max(frame.sample_rate, 1)
            if position + frame_seconds <= start:
                position += frame_seconds          # fast-forward past `start`
                continue
            for packet in resampler.resample(frame):
                pcm = packet.to_ndarray().reshape(-1)
                chunks.append(pcm.tobytes())
                collected += len(pcm) / sample_rate
            if duration is not None and collected >= duration:
                break

    audio = np.frombuffer(b"".join(chunks), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


class TranscriptionResult:
    """Sentences transcribed from one audio source."""

    def __init__(self, language: str = "", language_probability: float = 0.0,
                 model: str = "", duration: float = 0.0):
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
) -> TranscriptionResult:
    """
    Transcribe an audio file to Vietnamese.

    Defaults to the accurate-but-slow model: a file is not racing a live
    speaker, so quality wins. Pass LIVE_MODEL when a long recording has to
    finish quickly.
    """
    audio = load_audio(path, duration=max_seconds)
    logger.info(
        f"Decoded {path}: {len(audio) / SAMPLE_RATE:.1f}s at {SAMPLE_RATE} Hz"
    )

    model = _whisper_model(model_name)
    segments, info = model.transcribe(
        audio, language="vi", beam_size=5,
        vad_filter=True,                 # skip silence instead of noise
        condition_on_previous_text=True,  # carry context across sentences
    )

    result = TranscriptionResult(
        language=info.language,
        language_probability=round(float(info.language_probability), 3),
        model=model_name,
        duration=round(len(audio) / SAMPLE_RATE, 1),
    )
    for seg in segments:
        text = seg.text.strip()
        if text and not _is_hallucination(text):
            result.sentences.append(text)
            result.timings.append((round(seg.start, 2), round(seg.end, 2)))
    return result


def drain_final_sentences() -> list[str]:
    if _engine is None:
        return []
    return _engine.drain_final_sentences()


def buffer_usage() -> dict:
    """Buffer memory held by the active capture, or zeros when idle."""
    if _engine is None:
        return {"samples": 0, "seconds": 0.0, "bytes": 0, "blocks": 0}
    return _engine.buffer_usage()


def final_sentences() -> Iterator[str]:
    if _engine is None:
        return
    yield from _engine.final_sentences()
