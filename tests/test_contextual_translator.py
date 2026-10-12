"""ContextualTranslator: refinement cadence, rendering, stats — model faked.

The refine-every-N mechanism is the reason speech output gets *better* as
context accumulates, so its accounting is worth pinning down without a model.
"""
import pytest

from dich import backend


@pytest.fixture()
def translator(backend, monkeypatch):
    monkeypatch.setattr(
        backend, "translate_utterance",
        lambda sentence, history, model=None: f"EN<{sentence}>",
    )
    monkeypatch.setattr(
        backend, "translate_block",
        lambda sentences, model=None, history=None:
            "\n".join(f"BLOCK {s}" for s in sentences),
    )
    return backend.ContextualTranslator(refine_every=3)


def test_add_accumulates_provisionally(backend, monkeypatch):
    t = backend.ContextualTranslator(refine_every=100)
    monkeypatch.setattr(
        backend, "translate_utterance",
        lambda sentence, history, model=None: f"EN<{sentence}>")
    assert t.add("câu một") == "EN<câu một>"
    assert t.add("câu hai") == "EN<câu hai>"
    assert t.stats()["provisional"] == 2
    assert t.stats()["refined"] == 0


def test_empty_add_is_ignored(translator):
    assert translator.add("   ") == ""
    assert translator.transcript == []


def test_refine_fires_at_the_cadence(translator):
    for i in range(3):
        translator.add(f"câu {i}")
    stats = translator.stats()
    assert stats["utterances"] == 3
    assert stats["refined"] == 3          # all three, re-read as one passage
    assert stats["provisional"] == 0
    assert stats["refined_blocks"] == 1
    assert translator.render_refined() == "BLOCK câu 0\nBLOCK câu 1\nBLOCK câu 2"


def test_lines_after_a_refinement_stay_provisional(translator):
    for i in range(3):
        translator.add(f"câu {i}")
    translator.add("câu 3")
    assert translator.stats()["refined"] == 3
    assert translator.stats()["provisional"] == 1
    assert translator.block_start == 3
    rendered = translator.render_refined()
    assert "BLOCK câu 2" in rendered
    assert "EN<câu 3>" in rendered        # the provisional line still shows


def test_finalize_refines_the_trailing_block(backend, monkeypatch):
    monkeypatch.setattr(
        backend, "translate_utterance",
        lambda sentence, history, model=None: f"EN<{sentence}>")
    monkeypatch.setattr(
        backend, "translate_block",
        lambda sentences, model=None, history=None:
            "\n".join(f"BLOCK {s}" for s in sentences))
    t = backend.ContextualTranslator(refine_every=6)
    t.add("câu một")
    t.add("câu hai")
    assert t.finalize() is True           # two of six: still pending
    assert t.stats()["provisional"] == 0
    assert t.finalize() is False          # nothing left to refine


def test_refine_with_nothing_pending_is_a_noop(translator):
    assert translator.refine() is False


def test_failed_refine_keeps_earlier_results(translator, monkeypatch):
    def boom(sentences, model=None, history=None):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(backend, "translate_block", boom)
    translator.add("câu một")
    assert translator.refine() is False
    assert translator.stats()["last_error"] == "RuntimeError: model unavailable"
    assert translator.stats()["provisional"] == 1   # line survives as provisional


def test_failing_listener_cannot_break_the_flow(translator):
    def bad_listener(*_args):
        raise ValueError("listener exploded")
    t = backend.ContextualTranslator(refine_every=2, on_update=bad_listener)
    assert t.add("câu một") == "EN<câu một>"        # must not raise
    assert t.transcript == ["câu một"]
