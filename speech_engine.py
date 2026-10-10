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
WINDOW_SECONDS = 3.0          # audio sent to the transcriber at a time
SILENCE_SECONDS = 0.8         # trailing silence that ends an utterance
MIN_SENTENCE_CHARS = 8        # drop tiny fragments ("uh", "vâng" alone)
# Batch/file transcription favours accuracy; live mic favours keeping up.
FILE_MODEL = "large-v3-turbo"
LIVE_MODEL = "small"
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
        self._buffer: "deque[float]" = deque()
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
                self._buffer.extend(indata[:, 0].tolist())

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
    def _take(self, seconds: float) -> list:
        """Pop up to `seconds` of audio from the capture buffer."""
        want = int(seconds * SAMPLE_RATE)
        with self._buffer_lock:
            if len(self._buffer) < want:
                return []
            return [self._buffer.popleft() for _ in range(want)]

    # -- transcription ----------------------------------------------------
    def _transcribe_loop(self):
        while not self._stop.is_set():
            audio = self._take(WINDOW_SECONDS)
            if not audio:
                time.sleep(0.05)
                continue
            self._process_window(audio)

    def _process_window(self, audio: list):
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        segments, _info = self._model.transcribe(
            samples, language="vi", beam_size=5,
            without_timestamps=False, condition_on_previous_text=True,
            initial_prompt=_prompt_hint(),
        )
        for seg in segments:
            text = seg.text.strip()
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


def _prompt_hint() -> str:
    """Bias the decoder toward Vietnamese vocabulary."""
    return (
        "Đây là đoạn hội thoại tiếng Việt. Chép lại chính xác bằng tiếng Việt: "
        "ư, ơ, ă, â, ê, ô, ồ, đ, sắc, huyền, hỏi, ngã, nặng. "
        "Giữ nguyên dấu thanh và dấu câu."
    )


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
        if text:
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
