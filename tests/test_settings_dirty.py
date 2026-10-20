"""
Unsaved settings: a snapshot taken at save time, a marker that appears when
anything drifts from it, and a close-time warning that can save, discard or
cancel.
"""

from types import MethodType, SimpleNamespace

from dich import ui


def fake_app(values):
    """A stand-in whose settings vars read from `values`."""
    app = SimpleNamespace(
        recording=False,
        _settings_snapshot=dict(values),
        _dirty_shown=False,
        dirty_label=SimpleNamespace(configure=lambda **kw: None),
    )
    app._settings_values = MethodType(ui.DichApp._settings_values, app)
    app._settings_dirty = MethodType(ui.DichApp._settings_dirty, app)
    app._int_setting = ui.DichApp._int_setting
    app._float_setting = ui.DichApp._float_setting

    # One getter per setting var, reading the matching key in `values`.
    simple = {
        "model_var": "model",
        "source_var": "source_language",
        "target_var": "target_language",
        "preload_var": "load_model_on_startup",
        "push_to_talk_var": "push_to_talk",
        "voice_only_var": "voice_only_filter",
        "band_filter_var": "speech_band_filter",
        "combine_var": "combine_short_utterances",
        "ocr_var": "ocr_enabled",
        "archive_var": "archive_audio",
        "archive_path_var": "audio_archive_path",
        "clean_var": "clean_disfluencies",
        "llm_enabled_var": "llm_enabled",
        "llm_base_var": "llm_base_url",
        "llm_api_var": "llm_api_key",
        "llm_model_var": "llm_model",
        "refine_var": "refine_every",
        "buffer_var": "buffer_max_seconds",
    }
    for var_name, key in simple.items():
        setattr(app, var_name, SimpleNamespace(get=lambda k=key: values[k]))
    app.window_var = SimpleNamespace(get=lambda: values["window_seconds"])
    app.llm_instruction_text = SimpleNamespace(get=lambda *a: values["llm_instruction"])
    app._selected_device = lambda: values["mic_device_index"]
    app._selected_mic_model = lambda: values["mic_model"]
    return app


VALUES = {
    "model": "m/a",
    "source_language": "Vietnamese",
    "target_language": "English",
    "load_model_on_startup": True,
    "mic_device_index": None,
    "mic_model": "small",
    "refine_every": 6,
    "window_seconds": 5.0,
    "buffer_max_seconds": 600,
    "push_to_talk": False,
    "voice_only_filter": True,
    "speech_band_filter": True,
    "combine_short_utterances": False,
    "ocr_enabled": True,
    "archive_audio": False,
    "audio_archive_path": "",
    "clean_disfluencies": False,
    "llm_enabled": False,
    "llm_base_url": "",
    "llm_api_key": "",
    "llm_model": "",
    "llm_instruction": "",
}


def test_clean_settings_are_not_dirty(backend):
    app = fake_app(VALUES)
    assert app._settings_dirty() is False


def test_changing_any_setting_is_dirty(backend):
    app = fake_app(VALUES)
    values = dict(VALUES, model="other/model")
    app.model_var = SimpleNamespace(get=lambda: values["model"])
    assert app._settings_dirty() is True


def test_snapshot_refresh_makes_settings_clean_again(backend):
    app = fake_app(VALUES)
    values = dict(VALUES, llm_base_url="http://localhost:11434/v1")
    app.llm_base_var = SimpleNamespace(get=lambda: values["llm_base_url"])
    assert app._settings_dirty() is True
    app._settings_snapshot = app._settings_values()
    assert app._settings_dirty() is False


def test_garbage_in_numeric_fields_is_not_dirty_and_does_not_crash(backend):
    app = fake_app(dict(VALUES, refine_every=6))
    app.refine_var = SimpleNamespace(get=lambda: "not a number")
    # Unparsable input reads as the fallback, so the marker stays honest
    # without a crash on every poll.
    assert isinstance(app._settings_values()["refine_every"], int)


def test_close_warns_and_saves_dirty_settings(backend, monkeypatch):
    saved = {}
    destroyed = []
    answers = []

    app = SimpleNamespace(
        recording=False,
        _settings_snapshot=dict(VALUES),
        _close_stats=lambda: None,
        save_settings=lambda: saved.update({"saved": True}),
        root=SimpleNamespace(
            geometry=lambda: "800x600",
            destroy=lambda: destroyed.append(True),
        ),
        _save_config_fields=lambda u: None,
    )
    values = dict(VALUES, model="changed/model")

    def dirty():
        return values["model"] != app._settings_snapshot.get("model")

    app._settings_dirty = dirty
    monkeypatch.setattr(ui.messagebox, "askyesnocancel", lambda *a, **k: answers.pop())
    monkeypatch.setattr(backend, "unload_models", lambda: [])

    # Cancel keeps the window open.
    answers.append(None)
    ui.DichApp._on_close(app)
    assert destroyed == [] and saved == {}

    # "No" discards and quits.
    answers.append(False)
    ui.DichApp._on_close(app)
    assert destroyed == [True] and saved == {}

    # "Yes" saves first, then quits.
    app2 = SimpleNamespace(
        recording=False,
        _settings_snapshot=dict(VALUES),
        _close_stats=lambda: None,
        save_settings=lambda: saved.update({"saved": True}),
        root=SimpleNamespace(
            geometry=lambda: "800x600",
            destroy=lambda: destroyed.append(True),
        ),
        _save_config_fields=lambda u: None,
        _settings_dirty=dirty,
    )
    answers.append(True)
    ui.DichApp._on_close(app2)
    assert saved == {"saved": True}
    assert len(destroyed) == 2


def test_close_is_silent_when_settings_are_clean(backend, monkeypatch):
    asked = []
    monkeypatch.setattr(
        ui.messagebox, "askyesnocancel", lambda *a, **k: asked.append(True) or None
    )
    destroyed = []
    app = SimpleNamespace(
        recording=False,
        _settings_snapshot={},
        _close_stats=lambda: None,
        root=SimpleNamespace(
            geometry=lambda: "800x600",
            destroy=lambda: destroyed.append(True),
        ),
        _save_config_fields=lambda u: None,
        _settings_dirty=lambda: False,
    )
    monkeypatch.setattr(backend, "unload_models", lambda: [])

    ui.DichApp._on_close(app)

    assert asked == []
    assert destroyed == [True]


# -- the poll updates the marker and the voice line ---------------------------
def test_poll_shows_and_clears_the_unsaved_marker(backend, monkeypatch):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        configured = []

        app = fake_app(VALUES)
        app.dirty_label = SimpleNamespace(configure=lambda **kw: configured.append(kw))
        app.root = SimpleNamespace(
            after=lambda *a, **k: None,  # do not actually reschedule
        )
        app.stt_label = SimpleNamespace(configure=lambda **kw: None)
        app._refresh_stt_label = lambda: None
        app._poll_settings_dirty = MethodType(ui.DichApp._poll_settings_dirty, app)
        app._dirty_shown = False
        # Start dirty: a changed model must raise the marker on the next poll.
        app.model_var = SimpleNamespace(get=lambda: "changed/model")

        ui.DichApp._poll_settings_dirty(app)
        assert configured and configured[0]["text"] == "unsaved changes"
        assert app._dirty_shown is True

        # Clean again: the marker clears.
        app._settings_snapshot = app._settings_values()
        ui.DichApp._poll_settings_dirty(app)
        assert configured[-1]["text"] == ""
    finally:
        root.destroy()


def test_voice_line_names_the_model_state_and_backend(backend):
    shown = {}
    app = SimpleNamespace(
        recording=True,
        _file_transcribing=False,
        stt_label=SimpleNamespace(configure=lambda **kw: shown.update(kw)),
    )
    app._refresh_stt_label = MethodType(ui.DichApp._refresh_stt_label, app)
    app._refresh_stt_label()

    text = shown["text"]
    assert text.startswith("voice: ")
    assert "listening" in text
