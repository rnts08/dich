"""
Speech-to-text engine for the speech tab.

Pipeline:
    microphone -> whisper (Vietnamese) -> completed sentences

Design notes:
    * faster-whisper (CTranslate2) transcribes fixed-size audio windows, but raw
      windows cut mid-sentence and are unreliable for meaning-level translation.
      We instead only emit sentences on a *final* trigger: silence, an end
      punctuation mark, or the window ending. Sentences are then translated as
      whole units, which is what the requirement asks for.
    * If the optional deps are missing, importing this module raises ImportError
      and the UI tells the user what to install.

Requires: pip install sounddevice faster-whisper numpy
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque
from typing import Iterator, Optional

SAMPLE_RATE = 16000
WINDOW_SECONDS = 3.0          # audio sent to the transcriber at a time
SILENCE_SECONDS = 0.8         # trailing silence that ends an utterance
MIN_SENTENCE_CHARS = 8        # drop tiny fragments ("uh", "vâng" alone)
MODEL_SIZE = "small"          # good Vietnamese accuracy; use "base"/"medium" to trade

_END_PUNCT = ".?!。！？…"
# vietnamese final particles / connectors that usually close a sentence
_END_WORDS = {
    "rồi", "xong", "nhé", "nha", "nha bạn", "được", "cảm ơn",
    "vâng", "dạ", "ừ", "ừa", "thôi", "nào", "sao", "không", "chưa",
}


class SpeechEngine:
    """Records from the default microphone and yields completed Vietnamese sentences."""

    def __init__(self, model_size: str = MODEL_SIZE, device_index: Optional[int] = None):
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
        import numpy as np  # noqa: F401 - presence check
        import sounddevice as sd

        from faster_whisper import WhisperModel

        self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")

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


def start_mic_stream(model_size: str = MODEL_SIZE) -> SpeechEngine:
    global _engine
    if _engine is None:
        _engine = SpeechEngine(model_size=model_size)
        _engine.start()
    return _engine


def stop_mic_stream():
    global _engine
    if _engine is not None:
        _engine.stop()
        _engine = None


def drain_final_sentences() -> list[str]:
    if _engine is None:
        return []
    return _engine.drain_final_sentences()


def final_sentences() -> Iterator[str]:
    if _engine is None:
        return
    yield from _engine.final_sentences()
