"""Transcript cleanup on the ContextualTranslator, with the model faked.

The rules themselves are covered in test_cleanup; this file pins the wiring —
when cleanup runs (never by default, always on demand), that a cleaned
transcript is re-translated, and that a cleanup with nothing to do leaves the
existing — possibly expensive — translations alone.
"""

from dich import backend


def _fake_model(monkeypatch):
    monkeypatch.setattr(
        backend,
        "translate_utterance",
        lambda sentence, history, model=None: f"EN<{sentence}>",
    )
    monkeypatch.setattr(
        backend,
        "translate_block",
        lambda sentences, model=None, history=None: "\n".join(
            f"B<{s}>" for s in sentences
        ),
    )


def test_normalize_only_strips_by_default(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator()
    assert t.normalize("  ừm tôi tôi nghĩ  ") == "ừm tôi tôi nghĩ"


def test_normalize_cleans_when_enabled(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator(clean_disfluencies=True)
    assert t.normalize("ừm tôi tôi nghĩ") == "tôi nghĩ"


def test_add_cleans_when_enabled(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator(refine_every=100, clean_disfluencies=True)
    t.add("ừm tôi tôi nghĩ")
    assert t.transcript == ["tôi nghĩ"]


def test_cleanup_rewrites_the_transcript_and_re_translates(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator(refine_every=100)
    t.add("ừm tôi tôi nghĩ")
    assert t.transcript == ["ừm tôi tôi nghĩ"]

    stats = t.cleanup()

    assert stats["changed"] is True
    assert stats["removed"] == 2
    assert t.transcript == ["tôi nghĩ"]
    assert "B<tôi nghĩ>" in t.render_refined()  # re-translated, not left stale


def test_cleanup_with_nothing_to_do_does_not_re_translate(backend, monkeypatch):
    calls = []

    def fake(sentence, history, model=None):
        calls.append(sentence)
        return f"EN<{sentence}>"

    monkeypatch.setattr(backend, "translate_utterance", fake)
    monkeypatch.setattr(
        backend,
        "translate_block",
        lambda sentences, model=None, history=None: "placeholder",
    )
    t = backend.ContextualTranslator(refine_every=100)
    t.add("Xin chào các bạn.")
    calls.clear()

    stats = t.cleanup()

    assert stats["changed"] is False
    assert calls == []  # no second model call over unchanged text
    assert t.transcript == ["Xin chào các bạn."]


def test_cleanup_on_an_empty_translator_is_a_noop(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator()
    assert t.cleanup() == {"removed": 0, "changed": False}


def test_cleanup_drops_lines_that_were_only_filler(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator(refine_every=100)
    t.add("ừm")
    # "ừm" is stripped on arrival only when cleanup is enabled; here it is
    # stored, then the on-demand cleanup removes it entirely.
    assert t.transcript == ["ừm"]

    t.cleanup()

    assert t.transcript == []


def test_replace_resets_and_translates(backend, monkeypatch):
    _fake_model(monkeypatch)
    t = backend.ContextualTranslator(refine_every=100)
    t.add("câu một")
    t.replace(["dòng một", "dòng hai"])
    assert t.transcript == ["dòng một", "dòng hai"]
    assert t.stats()["utterances"] == 2
