"""
Re-scan: a second transcription pass over an archived recording, compared
against the live transcript before anything changes.

The engine side is `transcribe_file` with a model override; these tests pin
the UI flow — the menu choices, the guard against a running file job, the
comparison window, and Keep adopting the new transcript through the same
path Open Audio uses.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from dich import speech_engine as se
from dich import ui


class RecordingWorker:
    """Records worker.run calls instead of running them."""

    def __init__(self):
        self.calls = []

    def run(self, fn, *args, **kwargs):
        self.calls.append((fn, args, kwargs))

    def post(self, fn, *args, **kwargs):  # unused here, kept for symmetry
        self.calls.append((fn, args, kwargs))


def fake_app(**overrides):
    rec = []
    app = SimpleNamespace(
        _file_transcribing=False,
        recording=False,
        stop_recording=lambda: rec.append("stopped"),
        mic_status=SimpleNamespace(configure=lambda **kw: rec.append(("mic", kw))),
        status=SimpleNamespace(
            set_busy=lambda *a, **kw: rec.append(("busy", a, kw)),
            set_hint=lambda *a, **kw: rec.append(("hint", a)),
            set_stats=lambda *a, **kw: rec.append(("stats", a)),
        ),
        _start_memory_poll=lambda: rec.append("poll-start"),
        _stop_memory_poll=lambda: rec.append("poll-stop"),
        worker=RecordingWorker(),
        transcript_pane=SimpleNamespace(get=lambda: "current text", set=rec.append),
        root=SimpleNamespace(after=lambda ms, fn: rec.append(("after", ms, fn))),
        _rec=rec,
    )
    # Methods the flows call through `self` must exist on the stand-in; bind
    # the real implementations so the behaviour under test is the real one.
    import types

    for name in (
        "_on_rescanned",
        "_show_rescan_comparison",
        "_keep_rescan",
        "_new_translator",
        "_adopt_transcript",
        "on_file_transcription_error",
    ):
        setattr(app, name, types.MethodType(getattr(ui.DichApp, name), app))
    for key, value in overrides.items():
        setattr(app, key, value)
    return app


# -- menu choices -------------------------------------------------------------
def test_rescan_choices_are_newest_first_and_capped(backend, monkeypatch, tmp_path):
    import os

    for i in range(8):
        p = tmp_path / f"recording-{i}.wav"
        p.write_bytes(b"x")
        os.utime(p, (1000 + i, 1000 + i))
    monkeypatch.setattr(backend, "audio_archive_dir", lambda: tmp_path)

    choices = ui.DichApp._rescan_choices(fake_app())
    assert [p.name for p in choices] == [
        f"recording-{i}.wav" for i in (7, 6, 5, 4, 3, 2)
    ]


def test_rescan_choices_empty_when_no_archive(backend, monkeypatch, tmp_path):
    monkeypatch.setattr(backend, "audio_archive_dir", lambda: tmp_path)
    assert ui.DichApp._rescan_choices(fake_app()) == []


# -- starting a scan ----------------------------------------------------------
def test_start_rescan_refuses_while_a_file_job_is_running(backend):
    app = fake_app(_file_transcribing=True)
    ui.DichApp.start_rescan(app, Path("a.wav"), "small")
    assert ("hint", ("a file is already being transcribed",)) in app._rec
    assert app.worker.calls == []  # nothing was scheduled


def test_start_rescan_stops_recording_and_schedules_transcription(backend):
    app = fake_app(recording=True)
    ui.DichApp.start_rescan(app, Path("a.wav"), "small")

    assert "stopped" in app._rec
    assert app._file_transcribing is True
    fn, args, kwargs = app.worker.calls[0]
    assert fn is se.transcribe_file
    assert args == ("a.wav", "small")
    assert kwargs["on_success"].__func__ is ui.DichApp._on_rescanned
    assert kwargs["on_error"].__func__ is ui.DichApp.on_file_transcription_error


# -- comparison and adoption --------------------------------------------------
@pytest.fixture()
def display_root():
    tkinter = pytest.importorskip("tkinter")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def _find_button(widget, prefix):
    for child in widget.winfo_children():
        if isinstance(child, __import__("tkinter").ttk.Button) and child.cget(
            "text"
        ).startswith(prefix):
            return child
        found = _find_button(child, prefix)
        if found is not None:
            return found
    return None


def test_rescanned_opens_comparison_and_keep_adopts(backend, display_root):
    import tkinter as tk

    sets = []
    # A real Tk root is needed to parent the comparison window; wrap `after`
    # so scheduling is still observable.
    real_after = display_root.after
    afters = []
    display_root.after = lambda ms, fn=None: (
        afters.append(fn),
        real_after(ms, fn),
    )[1]

    app = fake_app(
        root=display_root,
        transcript_pane=SimpleNamespace(get=lambda: "current text", set=sets.append),
    )
    result = SimpleNamespace(
        sentences=["Một câu mới.", "Hai câu mới."], model="small", duration=9.0
    )

    ui.DichApp._on_rescanned(app, result)

    assert app._file_transcribing is False
    assert "poll-stop" in app._rec
    window = app._rescan_window
    assert isinstance(window, tk.Toplevel)

    keep = _find_button(window, "Keep")
    assert keep is not None
    keep.invoke()  # the button's own command must drive adoption

    # Adoption went through the same path as Open Audio: a fresh translator,
    # the pane replaced, and the translation stream scheduled.
    assert isinstance(app.translator, backend.ContextualTranslator)
    assert sets == ["Một câu mới.\nHai câu mới."]
    assert afters and all(callable(f) for f in afters), (
        "the translation stream was never scheduled"
    )
    assert not window.winfo_exists()  # the comparison window closed


def test_rescanned_empty_result_disables_keep(backend, display_root):
    app = fake_app(root=display_root)
    result = SimpleNamespace(sentences=[], model="small", duration=5.0)

    ui.DichApp._on_rescanned(app, result)

    keep = _find_button(app._rescan_window, "Keep")
    assert keep is not None
    assert "disabled" in keep.state()
    assert app.worker.calls == []  # nothing adopted


def test_rescanned_discard_leaves_everything_alone(backend, display_root):
    app = fake_app(root=display_root)
    result = SimpleNamespace(sentences=["Một câu."], model="small", duration=5.0)

    ui.DichApp._on_rescanned(app, result)
    discard = _find_button(app._rescan_window, "Discard")
    discard.invoke()

    assert app.worker.calls == []
    assert not hasattr(app, "translator") or not app.worker.calls
