"""
Backend selection for speech-to-text: GPU (mlx) vs CPU (faster-whisper).

The rule under test is the one the backlog cares about — the accurate model
should stop being the slow one. On the GPU large-v3-turbo is 31x realtime, so
"auto" prefers it; the CPU path stays for machines without mlx-audio.
"""

import sys
from types import SimpleNamespace

from dich import speech_engine


def test_auto_prefers_the_gpu_when_one_is_available():
    assert speech_engine.stt_backend("auto", gpu_available=True) == "mlx"
    assert speech_engine.stt_backend("auto", gpu_available=False) == "ctranslate2"


def test_explicit_choices_win():
    assert speech_engine.stt_backend("gpu", gpu_available=False) == "mlx"
    assert speech_engine.stt_backend("cpu", gpu_available=True) == "ctranslate2"
    assert speech_engine.stt_backend("mlx", gpu_available=False) == "mlx"


def test_configured_backend_reads_the_config(monkeypatch):
    from dich import backend

    monkeypatch.setitem(backend.CONFIG, "stt_backend", "ctranslate2")
    monkeypatch.setattr(speech_engine, "_gpu_available", lambda: True)
    assert speech_engine._configured_backend() == "ctranslate2"


def test_whisper_model_routes_to_mlx(monkeypatch):
    fake = SimpleNamespace(
        available=lambda: True, MLXWhisper=lambda name: ("mlx", name)
    )
    monkeypatch.setattr(speech_engine, "mlx_stt", fake)

    assert speech_engine._whisper_model("large-v3-turbo", backend="mlx") == (
        "mlx",
        "large-v3-turbo",
    )


def test_whisper_model_falls_back_to_cpu_when_no_gpu(monkeypatch):
    class FakeWhisper:
        def __init__(self, name, device, compute_type):
            self.name = name
            self.device = device
            self.compute_type = compute_type

    monkeypatch.setattr(
        speech_engine,
        "mlx_stt",
        SimpleNamespace(available=lambda: False, MLXWhisper=lambda name: None),
    )
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper)
    )

    model = speech_engine._whisper_model("small", backend="mlx")

    assert isinstance(model, FakeWhisper)
    assert model.device == "cpu"
    assert model.name == "small"


def test_whisper_model_uses_cpu_backend_when_asked(monkeypatch):
    class FakeWhisper:
        def __init__(self, name, device, compute_type):
            self.name = name

    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper)
    )

    model = speech_engine._whisper_model("medium", backend="ctranslate2")

    assert model.name == "medium"
