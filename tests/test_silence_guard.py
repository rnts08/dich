"""
Silence must never reach a model on the file path.

Regression cover for the GPU move: the CPU path was saved from silence
hallucination by faster-whisper's `vad_filter`, which the GPU adapter drops
(it is not a mlx-audio argument). Feeding the same silent file to Qwen3 on the
GPU produced a sentence — "ở đâu?" — out of nothing.
"""

import wave
from types import SimpleNamespace

import numpy as np

from dich import speech_engine


def _write_wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((samples * 32767).astype(np.int16).tobytes())


def test_a_silent_file_is_never_sent_to_the_model(tmp_path, monkeypatch):
    wav = tmp_path / "silent.wav"
    _write_wav(wav, np.zeros(16000 * 5, dtype=np.float32))

    def must_not_run(name):
        raise AssertionError("a silent file must not be handed to a model")

    monkeypatch.setattr(speech_engine, "_whisper_model", must_not_run)

    result = speech_engine.transcribe_file(str(wav))

    assert result.sentences == []
    assert result.duration == 5.0


def test_quiet_but_present_audio_still_reaches_the_model(tmp_path, monkeypatch):
    """The guard is a floor, not a gate: real speech at low level must pass."""
    wav = tmp_path / "quiet.wav"
    tone = np.sin(np.linspace(0, 2 * np.pi * 200, 16000 * 2)) * 0.05
    _write_wav(wav, tone.astype(np.float32))

    called = []

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            called.append(True)
            info = SimpleNamespace(
                duration=2.0, language="vi", language_probability=0.9
            )
            return iter([SimpleNamespace(text=" Xin chào.", start=0.0, end=2.0)]), info

    monkeypatch.setattr(speech_engine, "_whisper_model", lambda name: FakeModel())

    result = speech_engine.transcribe_file(str(wav))

    assert called, "audible audio must be transcribed"
    assert result.sentences == ["Xin chào."]


def test_segments_the_model_flags_as_silence_are_dropped(tmp_path, monkeypatch):
    wav = tmp_path / "speech-and-gap.wav"
    _write_wav(wav, (np.random.randn(16000 * 6) * 0.1).astype(np.float32))

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            info = SimpleNamespace(
                duration=6.0, language="vi", language_probability=0.9
            )
            segments = iter(
                [
                    SimpleNamespace(
                        text=" Câu thật.", start=0.0, end=2.0, no_speech_prob=0.01
                    ),
                    SimpleNamespace(
                        text=" ở đâu?", start=3.0, end=5.0, no_speech_prob=0.95
                    ),
                ]
            )
            return segments, info

    monkeypatch.setattr(speech_engine, "_whisper_model", lambda name: FakeModel())

    result = speech_engine.transcribe_file(str(wav))

    assert result.sentences == ["Câu thật."]


def test_the_cpu_path_has_no_no_speech_attribute_and_still_works(tmp_path, monkeypatch):
    """faster-whisper segments carry no no_speech_prob; the filter must allow them."""
    wav = tmp_path / "cpu.wav"
    _write_wav(wav, (np.random.randn(16000 * 2) * 0.1).astype(np.float32))

    class CpuSegment:
        def __init__(self, text):
            self.text = text
            self.start = 0.0
            self.end = 2.0

    class FakeModel:
        def transcribe(self, audio, **kwargs):
            info = SimpleNamespace(
                duration=2.0, language="vi", language_probability=0.9
            )
            return iter([CpuSegment(" Xin chào.")]), info

    monkeypatch.setattr(speech_engine, "_whisper_model", lambda name: FakeModel())

    assert speech_engine.transcribe_file(str(wav)).sentences == ["Xin chào."]
