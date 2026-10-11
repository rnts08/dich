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
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# A longer window matters more than it looks: each transcribe() call carries
# fixed overhead, so short windows multiply it. Measured on this M5 with the
# medium model on Vietnamese speech:
#     3s window  -> 0.84x realtime   (falls behind the speaker)
#     6s window  -> 1.46x realtime
#    10s window  -> 2.00x realtime   (keeps up, and yields *more* segments
#                                     because Whisper sees whole clauses)
# The cost is latency to the first word, which is the right trade here.
WINDOW_SECONDS = 10.0
SILENCE_SECONDS = 0.8         # trailing silence that ends an utterance
MIN_SENTENCE_CHARS = 8        # drop tiny fragments ("uh", "vâng" alone)
# Peak amplitude below which a window is treated as silence. The decibel
# threshold in _trailing_silence uses an RMS figure; this is the loudest single
# sample, which catches the quiet-but-warm mic noise a real device produces.
SILENCE_FLOOR = 0.003
# Batch/file transcription favours accuracy; live mic keeps up with a speaker.
# Measured on this M5 for Vietnamese: small 6.4x realtime but mangles proper
# nouns badly ("Tũ" for "Tú"), medium 2.4x realtime and reads them correctly.
# Medium is the default because a wrong transcript is worse than a lag: it
# trails the speaker by roughly 7 seconds, which is still semi-realtime.
FILE_MODEL = "large-v3-turbo"
LIVE_MODEL = "medium"
DEFAULT_MODEL = LIVE_MODEL

_END_PUNCT = ".?!。！？…"
# vietnamese final particles / connectors that usually close a sentence
_END_WORDS = {
    "rồi", "xong", "nhé", "nha", "nha bạn", "được", "cảm ơn",
    "vâng", "dạ", "ừ", "ừa", "thôi", "nào", "sao", "không", "chưa",
}


class SpeechEngine:
    """Records from the default microphone and yields completed Vietnamese sentences."""

    def __init__(self, model_size: str = DEFAULT_MODEL, device_index: Optional[int] = None):
        self.model_size = model_size
        self.device_index = device_index
        # Numpy blocks, not a list of Python floats: the audio callback fires
        # ~30x a second and copying through .tolist() then back through
        # np.asarray() costs more than the transcription for a 3s window.
        self._buffer: "deque" = deque()
        self._buffered = 0
        self._buffer_lock = threading.Lock()
        self._pending = ""
        self._final: "queue.Queue[str]" = queue.Queue()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._stream = None

    # -- lifecycle --------------------------------------------------------
    def start(self):
        import sounddevice as sd

        self._model = _whisper_model(self.model_size)

        def on_audio(indata, frames, time_info, status):  # noqa: ARG001
            with self._buffer_lock:
                mono = np.ascontiguousarray(indata[:, 0])
                self._buffer.append((mono, len(mono)))
                self._buffered += len(mono)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            device=self.device_index, callback=on_audio,
            blocksize=int(SAMPLE_RATE * 0.5),
        )
        self._stop.clear()
        self._stream.start()
        self._threads = [
            threading.Thread(target=self._transcribe_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

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
            audio = self._take(WINDOW_SECONDS)
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

        for seg in segments:
            text = _strip_hallucinations(seg.text.strip())
            if not text:
                continue
            self._pending = f"{self._pending} {text}".strip()
            if self._is_complete(self._pending) or (seg.end - seg.start) >= WINDOW_SECONDS:
                self._emit(self._pending)
                self._pending = ""

        # Trailing silence closes the utterance.
        if self._pending and self._trailing_silence(samples):
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
        """True if the tail of the window is quiet."""
        import numpy as np

        tail = samples[int(len(samples) * 0.75):]
        if len(tail) == 0:
            return False
        return bool(np.sqrt(np.mean(tail ** 2)) < 0.005)

    def _emit(self, text: str):
        cleaned = " ".join(text.split())
        if len(cleaned) < MIN_SENTENCE_CHARS:
            return
        self._final.put(cleaned)

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
) -> SpeechEngine:
    """
    Open the microphone and begin transcribing.

    Args:
        model_size: Whisper model. Use LIVE_MODEL to keep up with a speaker.
        device_index: Microphone to open; None uses the system default.
    """
    global _engine
    if _engine is None:
        _engine = SpeechEngine(
            model_size=model_size,
            device_index=_validate_device(device_index),
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


def final_sentences() -> Iterator[str]:
    if _engine is None:
        return
    yield from _engine.final_sentences()
