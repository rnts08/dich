"""
Progress reporting on the file-transcription path.

A file job can run for minutes — a first-use model download alone is gigabytes —
so transcribe_file() and load_audio() report (phase, percent) to a callback
instead of hiding the whole thing behind one blocking call. These tests use a
generated WAV and a fake model: no download, no `model` marker.
"""

import wave
from types import SimpleNamespace

import numpy as np

from dich import speech_engine


def _write_wav(path, seconds=1.0, rate=16000):
    pcm = (np.sin(np.linspace(0, 200 * seconds, int(rate * seconds))) * 3000).astype(
        np.int16
    )
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def test_load_audio_reports_decode_progress(tmp_path):
    wav = tmp_path / "a.wav"
    _write_wav(wav, seconds=1.0)

    seen = []
    audio = speech_engine.load_audio(str(wav), on_progress=seen.append)

    assert len(audio) > 0
    assert seen, "decode must report at least one tick"
    assert seen[-1] == 100
    assert all(0 <= pct <= 100 for pct in seen)


def test_load_audio_progress_is_optional(tmp_path):
    wav = tmp_path / "b.wav"
    _write_wav(wav, seconds=0.5)

    audio = speech_engine.load_audio(str(wav))  # no callback: must still work

    assert len(audio) > 0


def test_transcribe_file_reports_every_phase(tmp_path, monkeypatch):
    wav = tmp_path / "c.wav"
    _write_wav(wav, seconds=1.0)

    class FakeModel:
        def transcribe(self, audio, **kw):
            info = SimpleNamespace(
                duration=1.0, language="vi", language_probability=0.9
            )
            segments = iter([SimpleNamespace(text=" Xin chào.", start=0.0, end=1.0)])
            return segments, info

    monkeypatch.setattr(speech_engine, "_whisper_model", lambda name: FakeModel())

    seen = []
    result = speech_engine.transcribe_file(
        str(wav), on_progress=lambda phase, pct: seen.append((phase, pct))
    )

    phases = [p for p, _ in seen]
    assert phases[0] == "decode"
    assert "model" in phases
    assert "transcribe" in phases
    assert ("model", None) in seen  # indeterminate while the model loads
    assert ("transcribe", 100) in seen
    assert result.sentences == ["Xin chào."]


def test_transcribe_file_progress_is_optional(tmp_path, monkeypatch):
    wav = tmp_path / "d.wav"
    _write_wav(wav, seconds=0.5)

    class FakeModel:
        def transcribe(self, audio, **kw):
            info = SimpleNamespace(
                duration=0.5, language="vi", language_probability=0.9
            )
            return iter([]), info

    monkeypatch.setattr(speech_engine, "_whisper_model", lambda name: FakeModel())

    result = speech_engine.transcribe_file(str(wav))  # no callback

    assert result.sentences == []


def test_decode_phase_gets_a_terminating_100(tmp_path, monkeypatch):
    """A file that yields nothing still ends its decode phase cleanly."""
    wav = tmp_path / "e.wav"
    _write_wav(wav, seconds=0.5)

    class FakeModel:
        def transcribe(self, audio, **kw):
            info = SimpleNamespace(
                duration=0.5, language="vi", language_probability=0.9
            )
            return iter([]), info

    monkeypatch.setattr(speech_engine, "_whisper_model", lambda name: FakeModel())

    seen = []
    speech_engine.transcribe_file(
        str(wav), on_progress=lambda p, pc: seen.append((p, pc))
    )

    decode = [pc for p, pc in seen if p == "decode"]
    assert decode[0] == 0 and decode[-1] == 100
