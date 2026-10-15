"""
Glossary: pin proper nouns to a chosen translation.

The directive must be empty when nothing is configured, so ordinary output is
byte-identical to the un-glossaried app.
"""


def test_directive_is_empty_without_a_glossary(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "glossary", {})
    assert backend.glossary_directive() == ""


def test_directive_lists_pairs(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "glossary", {"Hà Nội": "Hanoi"})
    directive = backend.glossary_directive()
    assert "Hà Nội -> Hanoi" in directive


def test_directive_ignores_blank_entries(monkeypatch, backend):
    monkeypatch.setitem(
        backend.CONFIG, "glossary", {"": "x", "keep": "y", "drop": "  "}
    )
    directive = backend.glossary_directive()
    assert "keep -> y" in directive
    assert "drop" not in directive


def test_directive_survives_a_non_dict_glossary(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "glossary", ["not", "a", "dict"])
    assert backend.glossary_directive() == ""


def test_chat_prompt_carries_the_glossary(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "glossary", {"Hà Nội": "Hanoi"})
    engine = backend.LocalEngine("test/model")
    messages = engine.build_messages("Tôi ở Hà Nội")
    assert "Hà Nội -> Hanoi" in messages[0]["content"]


def test_translategemma_prompt_carries_the_glossary(monkeypatch, backend):
    monkeypatch.setitem(backend.CONFIG, "glossary", {"Hà Nội": "Hanoi"})
    engine = backend.LocalEngine("mlx-community/translategemma-4b-it-4bit")
    messages = engine.build_messages("Tôi ở Hà Nội")
    assert "Hà Nội -> Hanoi" in messages[0]["content"][0]["text"]
