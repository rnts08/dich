"""
The Settings speech-model picker fits the machine: GPU-only models are only
offered when a Metal GPU is reachable, the default is the GPU model there,
and a saved model this machine cannot run falls back cleanly.
"""

from types import SimpleNamespace

from dich import speech_engine as se


def test_choices_exclude_gpu_only_models_without_a_gpu(monkeypatch):
    monkeypatch.setattr(se, "_gpu_available", lambda: False)
    labels, ids = se.stt_model_choices()
    assert "qwen3-0.6b" not in ids and "qwen3-1.7b" not in ids
    assert "large-v3-turbo" in ids and "small" in ids
    assert len(labels) == len(ids)


def test_choices_offer_everything_with_a_gpu(monkeypatch):
    monkeypatch.setattr(se, "_gpu_available", lambda: True)
    _labels, ids = se.stt_model_choices()
    for model in se.SPEECH_MODEL_IDS:
        assert model in ids


def test_default_live_model_is_the_gpu_model(monkeypatch):
    # default_live_model picks the GPU default when the backend can use it.
    monkeypatch.setattr(se, "stt_backend", lambda *a, **k: "mlx")
    assert se.default_live_model() == "large-v3-turbo"


def test_saved_gpu_only_model_falls_back_to_the_machine_default(monkeypatch):
    """A config pointing at Qwen3 on a CPU-only machine must not stick."""
    monkeypatch.setattr(se, "_gpu_available", lambda: False)
    labels, ids = se.stt_model_choices()
    shown = {}
    var = SimpleNamespace(get=lambda: "", set=lambda v: shown.update({"label": v}))
    from types import MethodType

    from dich import ui

    app = SimpleNamespace(
        mic_model_var=var,
        mic_model_labels=labels,
        mic_model_ids=ids,
    )
    app._set_mic_model_label = MethodType(ui.DichApp._set_mic_model_label, app)

    app._set_mic_model_label("qwen3-1.7b")

    assert shown["label"] == se.speech_model_label(se.default_live_model())
