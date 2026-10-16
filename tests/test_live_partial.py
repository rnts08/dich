"""
Partial transcripts while the speaker is still talking.

The sentence rules already accumulate text in `_pending`; the live view is that
same buffer, so a partial cannot drift from what will be emitted, and nothing
extra has to be cleared when the sentence lands. These tests pin that down plus
the UI hop that renders it.
"""

from types import SimpleNamespace

import numpy as np

from dich import speech_engine, ui


class FakeModel:
    def __init__(self, segments):
        self._segments = segments

    def transcribe(self, audio, **kwargs):
        info = SimpleNamespace(duration=2.0, language="vi", language_probability=0.9)
        return iter(self._segments), info


def _segment(text, end=0.0):
    return SimpleNamespace(text=text, start=0.0, end=end)


def _engine(segments):
    engine = speech_engine.SpeechEngine(model_size="small")
    engine._model = FakeModel(segments)
    return engine


def _window(seconds=4.0):
    # A voice-band harmonic stack with a constant envelope: the voice gate
    # accepts it as speech, and the flat level means the trailing-silence rule
    # cannot fire on its own.
    n = int(speech_engine.SAMPLE_RATE * seconds)
    t = np.arange(n) / speech_engine.SAMPLE_RATE
    tone = (
        0.3 * np.sin(2 * np.pi * 400 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1600 * t)
    )
    return tone.astype(np.float32)


def test_partial_is_the_sentence_still_being_assembled():
    engine = _engine([_segment(" xin chào")])

    engine._process_window(_window())

    # "chào" is not a sentence end, so it stays pending — and visible.
    assert engine.latest_partial() == "xin chào"
    assert engine.drain_final_sentences() == []


def test_partial_clears_the_moment_the_sentence_is_emitted():
    engine = _engine([_segment(" xin chào.")])

    engine._process_window(_window())

    assert engine.latest_partial() == ""
    assert engine.drain_final_sentences() == ["xin chào."]


def test_module_level_partial_is_empty_without_a_session(monkeypatch):
    monkeypatch.setattr(speech_engine, "_engine", None)

    assert speech_engine.latest_partial() == ""


def test_module_level_partial_reads_the_live_engine(monkeypatch):
    engine = _engine([_segment(" tạm biệt bạn")])
    engine._process_window(_window())
    monkeypatch.setattr(speech_engine, "_engine", engine)

    assert speech_engine.latest_partial() == "tạm biệt bạn"


def test_show_partial_labels_and_clears_the_hint():
    seen = []
    app = SimpleNamespace(status=SimpleNamespace(set_hint=seen.append))

    ui.DichApp._show_partial(app, "xin chào")
    ui.DichApp._show_partial(app, "")

    assert seen == ["hearing: xin chào", ""]


def test_render_speech_pushes_the_partial_through_the_worker(monkeypatch):
    """The loop runs on its own thread, so the hint must hop via Worker.post."""
    posted = []
    shown = []

    app = SimpleNamespace(
        recording=True,
        _last_partial="",
        worker=SimpleNamespace(
            post=lambda fn, *a: posted.append((fn, a)),
            run=lambda *a, **k: None,
        ),
        translator=SimpleNamespace(),
        transcript_pane=SimpleNamespace(append=lambda *a: None),
    )
    app._show_partial = shown.append  # the bound handler the loop must use

    def drain():
        app.recording = False  # one pass is enough to observe the hop
        return []

    monkeypatch.setattr(speech_engine, "latest_partial", lambda: "xin chào")
    monkeypatch.setattr(speech_engine, "drain_final_sentences", drain)

    ui.DichApp._render_speech(app)

    assert posted == [(app._show_partial, ("xin chào",))]
    assert shown == []  # post() did not run it inline; the worker will


def test_render_speech_does_not_repaint_an_unchanged_partial(monkeypatch):
    """Whisper re-reports the same pending text each window; don't repaint it."""
    posted = []

    app = SimpleNamespace(
        recording=True,
        _last_partial="xin chào",  # already showing
        worker=SimpleNamespace(
            post=lambda fn, *a: posted.append((fn, a)),
            run=lambda *a, **k: None,
        ),
        translator=SimpleNamespace(),
        transcript_pane=SimpleNamespace(append=lambda *a: None),
    )

    def drain():
        app.recording = False
        return []

    monkeypatch.setattr(speech_engine, "latest_partial", lambda: "xin chào")
    monkeypatch.setattr(speech_engine, "drain_final_sentences", drain)

    ui.DichApp._render_speech(app)

    assert posted == []
