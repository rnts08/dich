"""
App lifecycle: remembered geometry, recent files, idle unload, clean quit.

These run without a Tk window — the handlers are pure enough to call on a
stand-in app, which is how the rest of the UI is tested here.
"""

import time
from pathlib import Path
from types import SimpleNamespace

from dich import ui


# -- geometry ---------------------------------------------------------------
def test_saved_geometry_uses_the_config_value(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "window_geometry", "900x700+5+5")
    assert ui.DichApp._saved_geometry(SimpleNamespace()) == "900x700+5+5"


def test_saved_geometry_falls_back_when_unset(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "window_geometry", "")
    assert ui.DichApp._saved_geometry(SimpleNamespace()) == "1280x820"


# -- idle unload ------------------------------------------------------------
def _idle_app(monkeypatch, backend, idle_for, **flags):
    unloaded = []
    hints = []
    scheduled = []
    monkeypatch.setattr(
        backend, "unload_models", lambda: unloaded.append(True) or ["a/model"]
    )
    monkeypatch.setitem(backend.CONFIG, "idle_unload_minutes", flags.get("minutes", 15))
    app = SimpleNamespace(
        recording=flags.get("recording", False),
        _file_transcribing=flags.get("file", False),
        _last_activity=time.time() - idle_for,
        _idle_unloaded=False,
        _idle_tick=lambda: None,
        root=SimpleNamespace(after=lambda ms, fn: scheduled.append(ms)),
        status=SimpleNamespace(set_hint=hints.append),
    )
    return app, unloaded, hints, scheduled


def test_idle_tick_unloads_after_the_configured_idle(monkeypatch, backend):
    app, unloaded, hints, scheduled = _idle_app(monkeypatch, backend, idle_for=16 * 60)

    ui.DichApp._idle_tick(app)

    assert unloaded == [True]
    assert app._idle_unloaded is True
    assert any("idle" in h for h in hints)
    assert scheduled  # reschedules itself


def test_idle_tick_leaves_recently_used_models_alone(monkeypatch, backend):
    app, unloaded, _hints, _scheduled = _idle_app(monkeypatch, backend, idle_for=60)

    ui.DichApp._idle_tick(app)

    assert unloaded == []


def test_idle_tick_will_not_unload_mid_recording(monkeypatch, backend):
    app, unloaded, _hints, _scheduled = _idle_app(
        monkeypatch, backend, idle_for=16 * 60, recording=True
    )

    ui.DichApp._idle_tick(app)

    assert unloaded == []


def test_idle_tick_can_be_disabled(monkeypatch, backend):
    app, unloaded, _hints, _scheduled = _idle_app(
        monkeypatch, backend, idle_for=10 * 60, minutes=0
    )

    ui.DichApp._idle_tick(app)

    assert unloaded == []


def test_note_activity_resets_the_clock_and_the_flag():
    app = SimpleNamespace(_last_activity=0.0, _idle_unloaded=True)
    ui.DichApp.note_activity(app)
    assert app._last_activity > 0
    assert app._idle_unloaded is False


# -- recent files -----------------------------------------------------------
def test_remember_recent_moves_to_front_dedups_and_caps(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "recent_files", [f"/f{i}" for i in range(8)])
    saved = {}
    app = SimpleNamespace(
        _save_config_fields=lambda u: saved.update(u),
        _rebuild_recent_menu=lambda: None,
    )

    ui.DichApp._remember_recent(app, Path("/f3"))

    recent = saved["recent_files"]
    assert recent[0] == "/f3"
    assert recent.count("/f3") == 1
    assert len(recent) == 8  # oldest dropped, not grown


def test_rebuild_recent_menu_disables_when_empty(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "recent_files", [])
    added = []

    class FakeMenu:
        def delete(self, *a):
            pass

        def add_command(self, **kw):
            added.append(kw)

        def add_separator(self):
            pass

    ui.DichApp._rebuild_recent_menu(SimpleNamespace(recent_menu=FakeMenu()))

    assert added and added[0].get("state") == "disabled"


def test_rebuild_recent_menu_lists_files(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "recent_files", ["/a/one.txt", "/a/two.md"])
    added = []

    class FakeMenu:
        def delete(self, *a):
            pass

        def add_command(self, **kw):
            added.append(kw)

        def add_separator(self):
            pass

    ui.DichApp._rebuild_recent_menu(
        SimpleNamespace(
            recent_menu=FakeMenu(),
            _open_recent=lambda p: None,
            _clear_recent=lambda: None,
        )
    )

    labels = [c.get("label") for c in added]
    assert "one.txt" in labels and "two.md" in labels
    assert "Clear List" in labels


# -- clean quit -------------------------------------------------------------
def test_on_close_saves_geometry_unloads_and_destroys(monkeypatch, backend):
    saved = {}
    unloaded = []
    destroyed = []
    monkeypatch.setattr(backend, "unload_models", lambda: unloaded.append(True) or [])
    monkeypatch.setattr(ui.speech_engine, "stop_mic_stream", lambda: None)
    app = SimpleNamespace(
        recording=True,
        root=SimpleNamespace(
            geometry=lambda: "900x700+5+5",
            destroy=lambda: destroyed.append(True),
        ),
        _save_config_fields=lambda u: saved.update(u),
    )

    ui.DichApp._on_close(app)

    assert saved["window_geometry"] == "900x700+5+5"
    assert unloaded == [True]
    assert destroyed == [True]


# -- same-format save and clipboard -----------------------------------------
def test_save_translation_exports_in_the_source_format(monkeypatch, backend, tmp_path):
    exported = {}
    monkeypatch.setattr(
        backend,
        "export_text",
        lambda text, path: exported.update({"text": text, "path": path}) or path,
    )
    monkeypatch.setattr(
        ui.filedialog,
        "asksaveasfilename",
        lambda **kw: str(tmp_path / "out.docx"),
    )
    app = SimpleNamespace(
        current_file=Path("/docs/notes.docx"),
        target_pane=SimpleNamespace(
            markdown_source=lambda: "Hello", get=lambda: "Hello", placeholder="ph"
        ),
        status=SimpleNamespace(set_hint=lambda t: None),
    )

    ui.DichApp.save_translation(app)

    assert exported["path"].endswith("out.docx")  # not forced to .txt
    assert exported["text"] == "Hello"


def test_save_config_fields_merges_and_updates_memory(monkeypatch, backend, tmp_path):
    import json

    path = tmp_path / "config.json"
    path.write_text('{"model": "keep-me"}', encoding="utf-8")
    monkeypatch.setattr(ui, "CONFIG_PATH", path)

    ui.DichApp._save_config_fields(
        SimpleNamespace(_show_error=lambda *a: None), {"recent_files": ["/a"]}
    )

    assert json.loads(path.read_text(encoding="utf-8"))["model"] == "keep-me"
    assert backend.CONFIG["recent_files"] == ["/a"]


def test_translate_clipboard_reports_an_empty_clipboard():
    import tkinter as tk

    hints = []

    def empty_clipboard():
        raise tk.TclError("no clipboard")

    app = SimpleNamespace(
        root=SimpleNamespace(clipboard_get=empty_clipboard),
        status=SimpleNamespace(set_hint=hints.append),
    )

    ui.DichApp.translate_clipboard(app)

    assert any("empty" in h for h in hints)
