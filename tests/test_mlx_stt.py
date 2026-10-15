"""
The GPU (mlx-audio) STT backend, without downloading or loading a model.

The repo mapping is the bug this guards: `mlx-community/whisper-large-v3-turbo`
carries weights only, and mlx-audio then fails at get_tokenizer() with
"Processor not found". The `-asr-fp16` repos carry the processor and tokenizer.
"""

import sys
from types import SimpleNamespace

import numpy as np

from dich import mlx_stt


def test_repo_for_maps_model_ids_to_the_full_processor_repos():
    assert (
        mlx_stt.repo_for("large-v3-turbo")
        == "mlx-community/whisper-large-v3-turbo-asr-fp16"
    )
    assert mlx_stt.repo_for("small") == "mlx-community/whisper-small-asr-fp16"


def test_repo_for_passes_a_repo_id_through():
    assert mlx_stt.repo_for("someone/custom-whisper") == "someone/custom-whisper"


def _fake_output():
    return SimpleNamespace(
        text=" Đây là một câu. Câu thứ hai.",
        language="vi",
        segments=[
            {
                "text": " Đây là một câu.",
                "start": 0.0,
                "end": 2.5,
                "no_speech_prob": 0.01,
            },
            {"text": " Câu thứ hai.", "start": 2.5, "end": 5.0},
        ],
    )


def _install_fake_model(monkeypatch, calls):
    class FakeModel:
        def generate(self, audio, **kwargs):
            calls.append(kwargs)
            return _fake_output()

    monkeypatch.setattr(mlx_stt, "_load_repo", lambda repo: FakeModel())
    return FakeModel()


def test_transcribe_converts_segments_to_objects_the_engine_reads(monkeypatch):
    calls = []
    _install_fake_model(monkeypatch, calls)
    model = mlx_stt.MLXWhisper("large-v3-turbo")

    segments, info = model.transcribe(np.zeros(16000, dtype=np.float32), language="vi")

    assert [s.text for s in segments] == ["Đây là một câu.", "Câu thứ hai."]
    assert [round(s.end, 1) for s in segments] == [2.5, 5.0]
    assert info.language == "vi"


def test_transcribe_drops_faster_whisper_only_kwargs(monkeypatch):
    """mlx-audio raises on unknown keywords; the engine passes several."""
    calls = []
    _install_fake_model(monkeypatch, calls)
    model = mlx_stt.MLXWhisper("small")

    model.transcribe(
        np.zeros(16000, dtype=np.float32),
        language="vi",
        beam_size=5,
        vad_filter=True,
        condition_on_previous_text=True,
    )

    assert set(calls[0]) == {"language", "condition_on_previous_text"}
    assert calls[0]["condition_on_previous_text"] is True


def test_model_loads_once(monkeypatch):
    loads = []

    class FakeModel:
        def generate(self, audio, **kwargs):
            return _fake_output()

    def fake_load(repo):
        loads.append(repo)
        return FakeModel()

    monkeypatch.setattr(mlx_stt, "_load_repo", fake_load)
    model = mlx_stt.MLXWhisper("medium")

    for _ in range(3):
        model.transcribe(np.zeros(16000, dtype=np.float32))

    assert loads == ["mlx-community/whisper-medium-asr-fp16"]


def test_available_is_false_without_mlx(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    assert mlx_stt.available() is False


def test_available_true_when_metal_reports_available(monkeypatch):
    fake_core = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=fake_core))
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)
    monkeypatch.setitem(
        sys.modules, "mlx_audio.stt.utils", SimpleNamespace(load=lambda r: None)
    )

    assert mlx_stt.available() is True
