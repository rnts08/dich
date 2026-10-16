"""
Saving a speech session: the transcript and its translation together.

The speech tab used to have no save at all — the panes could be read but not
kept. The export always carries *both*, so a later reader has the Vietnamese
transcript next to the English rather than the translation alone.
"""

from dich import backend


def test_plain_text_export_has_both_sections():
    doc = backend.format_speech_export(
        "Xin chào các bạn.", "Hello everyone.", source="Vietnamese", target="English"
    )

    assert "Vietnamese (transcription)" in doc
    assert "English (translation)" in doc
    assert "Xin chào các bạn." in doc
    assert "Hello everyone." in doc
    # the transcript comes first, so the source is always readable beside it
    assert doc.index("Xin chào") < doc.index("Hello everyone.")


def test_markdown_export_aligns_paragraphs_side_by_side():
    doc = backend.format_speech_export("A1\n\nA2", "B1\n\nB2", fmt="md")

    assert "| Vietnamese | English |" in doc
    assert "| A1 | B1 |" in doc
    assert "| A2 | B2 |" in doc


def test_markdown_export_pads_the_shorter_side():
    doc = backend.format_speech_export("A1\n\nA2", "B1", fmt="md")

    assert "| A2 |  |" in doc


def test_markdown_export_escapes_table_pipes():
    doc = backend.format_speech_export("a | b", "c", fmt="md")

    assert "a \\| b" in doc


def test_export_tolerates_empty_sides():
    doc = backend.format_speech_export("", "")
    assert "Vietnamese (transcription)" in doc

    md = backend.format_speech_export("", "", fmt="md")
    assert "| Vietnamese | English |" in md
