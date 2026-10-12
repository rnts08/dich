"""
The buffer-memory readout and its idle schema.

Regression cover: the idle dict lacked `window_seconds`, so _poll_memory's
first tick — run synchronously inside open_audio() — raised KeyError and
aborted file transcription before worker.run(transcribe_file…) was reached.
No transcription, no buffer line, status stuck at "transcribing…".
"""

from types import SimpleNamespace

from dich import speech_engine, ui


def test_idle_usage_matches_live_schema(monkeypatch):
    """The idle report must carry every key a live engine reports."""
    engine = speech_engine.SpeechEngine()
    live_keys = set(engine.buffer_usage())
    assert live_keys, "the live engine must report something"

    monkeypatch.setattr(speech_engine, "_engine", None)
    idle = speech_engine.buffer_usage()
    assert set(idle) == live_keys


def test_poll_memory_first_tick_survives_idle_engine():
    """
    open_audio() calls _start_memory_poll() before dispatching the file job,
    so the first tick runs synchronously. With no engine ever started (the
    common case: a fresh app going straight to File -> Open) it must render
    the idle buffer line and re-arm — not raise.
    """
    events = []
    app = SimpleNamespace(
        _memory_polling=True,
        recording=False,
        status=SimpleNamespace(set_memory=lambda text: events.append(("memory", text))),
        mic_status=SimpleNamespace(configure=lambda **kw: events.append(("mic", kw))),
        root=SimpleNamespace(after=lambda delay, fn: events.append(("after", delay))),
    )
    app._poll_memory = ui.DichApp._poll_memory  # the method re-arms via self.

    ui.DichApp._poll_memory(app)  # must not raise

    assert any(e[0] == "memory" and e[1].startswith("buffer") for e in events)
    assert ("after", 500) in events  # the poll chain is still alive


def test_poll_memory_contains_render_errors(monkeypatch):
    """A broken reading is logged; it never propagates to the caller."""

    def boom():
        raise RuntimeError("sensor offline")

    monkeypatch.setattr(ui, "speech_engine", SimpleNamespace(buffer_usage=boom))
    events = []
    app = SimpleNamespace(
        _memory_polling=True,
        recording=True,
        status=SimpleNamespace(set_memory=lambda text: None),
        mic_status=SimpleNamespace(configure=lambda **kw: None),
        root=SimpleNamespace(after=lambda delay, fn: events.append(("after", delay))),
    )
    app._poll_memory = ui.DichApp._poll_memory  # the method re-arms via self.

    ui.DichApp._poll_memory(app)  # must not raise

    assert ("after", 500) in events
