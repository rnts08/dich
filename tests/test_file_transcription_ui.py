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
