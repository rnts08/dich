"""The Speech tab's cleanup and rewrite wiring, without a Tk window.

The rules and the request shape are tested elsewhere; here the handlers are
driven with stand-in widgets to pin the behaviour a user sees: rewrites are
refused until the feature is configured, the right text goes to each target, a
no-op rewrite is not offered, and applying a transcript rewrite re-translates.
"""

from types import SimpleNamespace

from dich import ui


def _app(events, worker=None):
    app = SimpleNamespace(
        transcript_pane=SimpleNamespace(
            get=lambda: "ừm xin chào",
            set=lambda t: events.append(("transcript", t)),
        ),
        speech_translation_pane=SimpleNamespace(
            markdown_source=lambda: "Hello there",
            get=lambda: "Hello there",
            set=lambda t: events.append(("translation", t)),
        ),
        status=SimpleNamespace(
            set_busy=lambda *a, **k: events.append(("busy", a)),
            set_stats=lambda *a, **k: events.append(("stats", a)),
            set_hint=lambda t: events.append(("hint", t)),
        ),
        worker=worker or SimpleNamespace(run=lambda *a, **k: None),
        root=SimpleNamespace(),
    )
    # Methods the handlers reach through self; a plain function is not bound off
    # a SimpleNamespace, so bind them to the stand-in instance explicitly.
    app._speech_text = lambda kind: ui.DichApp._speech_text(app, kind)
    app._on_speech_error = lambda exc, tb: None
    app._on_rewrite_error = lambda exc, tb: None
    app._on_replaced = lambda: ui.DichApp._on_replaced(app)
    return app


def test_rewrite_is_refused_until_configured(monkeypatch):
    monkeypatch.setattr(ui.backend, "llm_available", lambda: False)
    shown = {}
    monkeypatch.setattr(
        ui.messagebox, "showinfo", lambda title, msg: shown.update(title=title)
    )
    ui.DichApp.rewrite_api(_app([]), "translation")
    assert "not set up" in shown["title"].lower()


def test_rewrite_sends_the_transcript_for_the_transcript_target(monkeypatch):
    monkeypatch.setattr(ui.backend, "llm_available", lambda: True)
    captured = {}
    worker = SimpleNamespace(
        run=lambda fn, *a, **k: captured.update(fn=fn, args=a, kwargs=k)
    )
    ui.DichApp.rewrite_api(_app([], worker=worker), "transcript")
    assert captured["fn"] is ui.backend.rewrite_with_llm
    assert captured["args"] == ("ừm xin chào", "transcript")


def test_rewrite_sends_the_english_for_the_translation_target(monkeypatch):
    monkeypatch.setattr(ui.backend, "llm_available", lambda: True)
    captured = {}
    worker = SimpleNamespace(run=lambda fn, *a, **k: captured.update(args=a))
    ui.DichApp.rewrite_api(_app([], worker=worker), "translation")
    assert captured["args"] == ("Hello there", "translation")


def test_rewrite_of_empty_text_is_refused(monkeypatch):
    monkeypatch.setattr(ui.backend, "llm_available", lambda: True)
    shown = {}
    monkeypatch.setattr(
        ui.messagebox, "showinfo", lambda title, msg: shown.update(title=title)
    )
    app = _app([])
    app.transcript_pane.get = lambda: "   "
    ui.DichApp.rewrite_api(app, "transcript")
    assert "nothing to rewrite" in shown["title"].lower()


def test_a_rewrite_with_no_change_is_not_offered():
    events = []
    app = _app(events)
    called = {}
    app._show_rewrite_result = lambda kind, text: called.update(kind=kind)
    ui.DichApp._on_rewritten(app, "translation", "same", "same")
    assert called == {}
    assert any(e[0] == "stats" and "no change" in e[1][0] for e in events)


def test_a_changed_rewrite_opens_the_result():
    event_calls = []
    app = _app(event_calls)
    called = {}
    app._show_rewrite_result = lambda kind, text: called.update(kind=kind, text=text)
    ui.DichApp._on_rewritten(app, "translation", "old", "new")
    assert called == {"kind": "translation", "text": "new"}


def test_applying_a_transcript_rewrite_re_translates():
    events = []
    calls = {}
    worker = SimpleNamespace(
        run=lambda fn, *a, **k: calls.update(fn=fn, args=a, kwargs=k)
    )
    app = _app(events, worker=worker)
    app.translator = SimpleNamespace(replace=lambda lines: None)

    ui.DichApp._apply_transcript_rewrite(app, "dòng một\n\ndòng hai\n")

    assert ("transcript", "dòng một\n\ndòng hai") in events
    assert calls["fn"] is app.translator.replace
    assert calls["args"] == (["dòng một", "dòng hai"],)


def test_on_replaced_shows_the_new_translation():
    events = []
    app = _app(events)
    app.translator = SimpleNamespace(
        render_refined=lambda: "EN",
        stats=lambda: {"refined": 2, "utterances": 2},
    )
    ui.DichApp._on_replaced(app)
    assert ("translation", "EN") in events
    assert any(e[0] == "stats" and "2 refined" in e[1][0] for e in events)


def test_on_cleaned_reports_and_refreshes_both_panes():
    events = []
    app = _app(events)
    app.translator = SimpleNamespace(
        transcript=["tôi nghĩ"], render_refined=lambda: "I"
    )
    ui.DichApp._on_cleaned(app, {"changed": True, "removed": 2})
    assert ("transcript", "tôi nghĩ") in events
    assert ("translation", "I") in events
    assert any(e[0] == "stats" and "removed 2 fillers" in e[1][0] for e in events)


def test_cleanup_speech_needs_a_transcript(monkeypatch):
    shown = {}
    monkeypatch.setattr(
        ui.messagebox, "showinfo", lambda title, msg: shown.update(title=title)
    )
    app = _app([])
    app.translator = SimpleNamespace(transcript=[])
    ui.DichApp.cleanup_speech(app)
    assert "nothing to clean" in shown["title"].lower()
