"""
The speech tab's file-transcription lifecycle, without a Tk window.

Regression cover: `_on_speech_error` was wired to transcribe_file's on_error
and never cleared anything — a decode/download failure left the pulsing busy
bar, the "transcribing…" label and the 500 ms memory poll running forever, and
said "translation failed" for what was a transcription failure.
"""

from types import SimpleNamespace

from dich import ui


def _file_error_app(events):
    """Minimal app stand-in for on_file_transcription_error."""
    app = SimpleNamespace(
        _memory_polling=True,
        _file_transcribing=True,
        status=SimpleNamespace(
            set_stats=lambda *a, **k: events.append(("stats", a)),
            set_hint=lambda t: events.append(("hint", t)),
            set_memory=lambda t: events.append(("memory", t)),
            reset=lambda: events.append(("reset",)),
        ),
        mic_status=SimpleNamespace(configure=lambda **kw: events.append(("mic", kw))),
        speech_translation_pane=SimpleNamespace(
            append=lambda t: events.append(("pane", t))
        ),
        root=SimpleNamespace(config=lambda **kw: events.append(("cursor", kw))),
    )
    # methods the handler reaches through self — a lambda binds what a
    # SimpleNamespace instance attribute cannot (it stores functions plain)
    app._stop_memory_poll = lambda: ui.DichApp._stop_memory_poll(app)
    return app


def test_file_error_clears_busy_state_and_stops_poll():
    """A failed file job must leave no spinning poll and no stale label."""
    events = []
    app = _file_error_app(events)

    ui.DichApp.on_file_transcription_error(app, RuntimeError("decode failed"), "tb")

    assert app._memory_polling is False
    assert app._file_transcribing is False
    # the memory poll chain is re-armed only by _poll_memory; stopping means
    # no further after() was requested by this call.
    assert not any(e[0] == "after" for e in events)
    # mic status is visibly failed, not "transcribing…"
    mic = [e[1] for e in events if e[0] == "mic"]
    assert mic and all("text" in kw for kw in mic)
    assert any("failed" in kw.get("text", "") for kw in mic)


def test_empty_result_reports_no_speech():
    """A file that yields no sentences is named as such, not reported as success."""
    events = []
    result = SimpleNamespace(sentences=[], duration=12.0)
    app = SimpleNamespace(
        mic_status=SimpleNamespace(configure=lambda **kw: events.append(("mic", kw))),
        status=SimpleNamespace(
            set_stats=lambda *a, **k: events.append(("stats", a)),
            set_hint=lambda t: events.append(("hint", t)),
        ),
        transcript_pane=SimpleNamespace(set=lambda t: events.append(("pane", t))),
        root=SimpleNamespace(after=lambda delay, fn: events.append(("after", delay))),
    )

    ui.DichApp._on_audio_transcribed(app, result)

    mic_text = " ".join(e[1].get("text", "") for e in events if e[0] == "mic")
    assert "no speech" in mic_text
    hint = next(e[1] for e in events if e[0] == "hint")
    assert "silence is deliberately not transcribed" in hint.lower()
    assert any(e[0] == "stats" and e[1][0] == "0 sentences" for e in events)
    # finalize is still scheduled so the session is torn down properly
    assert ("after", 0) in events


def test_non_empty_result_still_reports_the_count():
    events = []
    result = SimpleNamespace(sentences=["Xin chào.", "Tạm biệt."], duration=3.0)
    app = SimpleNamespace(
        mic_status=SimpleNamespace(configure=lambda **kw: events.append(("mic", kw))),
        status=SimpleNamespace(
            set_stats=lambda *a, **k: events.append(("stats", a)),
            set_hint=lambda t: events.append(("hint", t)),
        ),
        transcript_pane=SimpleNamespace(set=lambda t: events.append(("pane", t))),
        root=SimpleNamespace(after=lambda delay, fn: events.append(("after", delay))),
    )

    ui.DichApp._on_audio_transcribed(app, result)

    mic_text = next(e[1]["text"] for e in events if e[0] == "mic")
    assert mic_text == "2 sentences, 3s"
    assert not any(e[0] == "hint" for e in events)


def test_finalized_with_no_utterances_says_no_speech():
    """The trailing "done · 0 refined" must not claim success when nothing was said."""
    events = []
    app = SimpleNamespace(
        _memory_polling=True,
        status=SimpleNamespace(
            set_stats=lambda *a, **k: events.append(("stats", a)),
            set_memory=lambda t: None,
        ),
        speech_translation_pane=SimpleNamespace(set=lambda t: None),
        translator=SimpleNamespace(
            render_refined=lambda: "",
            stats=lambda: {"refined": 0, "utterances": 0, "provisional": 0},
        ),
    )
    app._stop_memory_poll = lambda: ui.DichApp._stop_memory_poll(app)

    ui.DichApp._on_finalized(app, ok=False)

    assert any(e[0] == "stats" and e[1][0] == "no speech detected" for e in events)
    assert not any(e[1][0].startswith("done") for e in events if e[0] == "stats")


def test_file_progress_updates_status_and_label():
    """Percent phases drive the real progress bar; None is indeterminate."""
    events = []
    app = SimpleNamespace(
        status=SimpleNamespace(
            set_busy=lambda *a, **k: events.append(("busy", a, k)),
            update=lambda pct, stats: events.append(("progress", pct, stats)),
        ),
        mic_status=SimpleNamespace(configure=lambda **kw: events.append(("mic", kw))),
    )

    ui.DichApp._on_file_progress(app, "transcribe", 42)

    assert any(e[0] == "progress" and e[1] == 42 for e in events)
    assert any(e[0] == "mic" and "transcribe" in e[1].get("text", "") for e in events)

    # a new phase with no percentage (model load) must switch to the pulsing bar
    ui.DichApp._on_file_progress(app, "model", None)

    assert any(e[0] == "busy" and "model" in e[1][0] for e in events)
    assert any(e[0] == "mic" and "model" in e[1].get("text", "") for e in events)


def test_open_audio_forwards_progress_through_the_worker(monkeypatch):
    """
    The engine reports from a worker thread, so progress must be marshalled
    with Worker.post — touching Tk from that thread raises.
    """
    posted = []
    captured = {}

    def fake_transcribe_file(path, model_name, on_progress=None):
        captured["on_progress"] = on_progress
        return SimpleNamespace(sentences=[], duration=1.0)

    app = SimpleNamespace(
        recording=False,
        audio_path=None,
        mic_status=SimpleNamespace(configure=lambda **kw: None),
        status=SimpleNamespace(set_busy=lambda *a, **k: None, set_hint=lambda t: None),
        transcript_pane=SimpleNamespace(set=lambda t: None),
        speech_translation_pane=SimpleNamespace(set=lambda t: None),
        worker=SimpleNamespace(
            run=lambda fn, *a, **kw: captured.update(fn=fn, kwargs=kw),
            post=lambda fn, *a: posted.append((fn, a)),
        ),
        root=SimpleNamespace(after=lambda d, fn: None),
        _new_translator=lambda: None,
        _start_memory_poll=lambda: None,
    )
    app._on_audio_transcribed = lambda result: None
    app.on_file_transcription_error = lambda exc, tb: None
    app._on_file_progress = lambda phase, percent: None
    monkeypatch.setattr(
        ui,
        "speech_engine",
        SimpleNamespace(
            transcribe_file=fake_transcribe_file, FILE_MODEL="large-v3-turbo"
        ),
    )
    monkeypatch.setattr(ui.filedialog, "askopenfilename", lambda **kw: "/tmp/x.wav")

    ui.DichApp.open_audio(app)

    assert captured["fn"] is fake_transcribe_file
    progress = captured["kwargs"]["on_progress"]
    progress("transcribe", 10)
    assert posted and posted[0][0] == app._on_file_progress
    assert posted[0][1] == ("transcribe", 10)


def test_file_error_names_the_failure_as_transcription():
    """Not "translation failed": the job that failed was the file transcription."""
    events = []
    app = _file_error_app(events)

    ui.DichApp.on_file_transcription_error(app, ValueError("not audio"), "tb")

    stats = [e[1][0] for e in events if e[0] == "stats"]
    assert any("transcription failed" in s for s in stats)
    assert any(e[0] == "hint" and e[1] == "not audio" for e in events)
    pane = [e[1] for e in events if e[0] == "pane"]
    assert pane and "transcription failed" in pane[0]
