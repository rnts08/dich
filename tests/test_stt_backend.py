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
        available=lambda: True,
        make_model=lambda name: ("mlx", name),
        is_gpu_only=lambda name: False,
    )
    monkeypatch.setattr(speech_engine, "mlx_stt", fake)

    assert speech_engine._whisper_model("large-v3-turbo", backend="mlx") == (
        "mlx",
        "large-v3-turbo",
    )


def test_gpu_only_preset_falls_back_to_a_cpu_model(monkeypatch):
    """Qwen3 has no faster-whisper equivalent; WhisperModel would raise."""

    class FakeWhisper:
        def __init__(self, name, device, compute_type):
            self.name = name

    monkeypatch.setattr(
        speech_engine,
        "mlx_stt",
        SimpleNamespace(
            available=lambda: False,
            make_model=lambda name: None,
            is_gpu_only=lambda name: name.startswith("qwen3"),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisper)
    )

    model = speech_engine._whisper_model("qwen3-0.6b", backend="ctranslate2")

    assert model.name == speech_engine.LIVE_MODEL


def test_speech_models_list_includes_the_gpu_presets():
    ids = [name for name, _label in speech_engine.SPEECH_MODELS]
    assert "large-v3-turbo" in ids
    assert "qwen3-0.6b" in ids
    assert speech_engine.speech_model_label("qwen3-0.6b") != "qwen3-0.6b"


def test_default_window_is_longer_on_the_gpu(monkeypatch):
    """More context per decode where the decode is nearly free."""
    monkeypatch.setattr(speech_engine, "_configured_backend", lambda: "mlx")
    assert speech_engine.default_window_seconds() == speech_engine.WINDOW_SECONDS_GPU

    monkeypatch.setattr(speech_engine, "_configured_backend", lambda: "ctranslate2")
    assert speech_engine.default_window_seconds() == speech_engine.WINDOW_SECONDS_CPU

    # The whole point: the GPU default must actually be the longer one.
    assert speech_engine.WINDOW_SECONDS_GPU > speech_engine.WINDOW_SECONDS_CPU


def test_default_window_survives_a_broken_backend_probe(monkeypatch):
    def boom():
        raise RuntimeError("no mlx")

    monkeypatch.setattr(speech_engine, "_configured_backend", boom)

    assert speech_engine.default_window_seconds() == speech_engine.WINDOW_SECONDS_CPU


def test_start_mic_stream_uses_the_backend_default_when_none_is_given(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            captured["started"] = True

    monkeypatch.setattr(speech_engine, "SpeechEngine", FakeEngine)
    monkeypatch.setattr(speech_engine, "_validate_device", lambda index: None)
    monkeypatch.setattr(speech_engine, "default_window_seconds", lambda: 5.0)
    monkeypatch.setattr(speech_engine, "_engine", None)

    speech_engine.start_mic_stream(window_seconds=None)

    assert captured["window_seconds"] == 5.0
    assert captured["started"] is True


def test_start_mic_stream_still_honours_an_explicit_window(monkeypatch):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def start(self):
            pass

    monkeypatch.setattr(speech_engine, "SpeechEngine", FakeEngine)
    monkeypatch.setattr(speech_engine, "_validate_device", lambda index: None)
    monkeypatch.setattr(speech_engine, "default_window_seconds", lambda: 5.0)
    monkeypatch.setattr(speech_engine, "_engine", None)

    speech_engine.start_mic_stream(window_seconds=2.5)

    assert captured["window_seconds"] == 2.5


def test_whisper_model_falls_back_to_cpu_when_no_gpu(monkeypatch):
    class FakeWhisper:
        def __init__(self, name, device, compute_type):
            self.name = name
            self.device = device
            self.compute_type = compute_type

    monkeypatch.setattr(
        speech_engine,
        "mlx_stt",
        SimpleNamespace(
            available=lambda: False,
            make_model=lambda name: None,
            is_gpu_only=lambda name: False,
        ),
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
