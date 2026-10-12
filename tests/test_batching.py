"""Batch splitting must respect the character limit, not just the batch count."""


def test_single_paragraph_over_char_limit_is_split(backend):
    # The fast path used to return the whole text whenever the paragraph count
    # was under the batch size, ignoring char_limit — a 3-paragraph 5 000-char
    # document went out as one prompt and got truncated.
    text = "Câu một. " * 200  # one paragraph, ~1 800 chars
    chunks = backend._split_batches(text, size=8, char_limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)
    assert "".join(chunks) == text  # splitting loses no characters


def test_multi_paragraph_with_oversized_paragraph_is_split(backend):
    text = "Ngắn.\n\n" + "Câu dài. " * 100 + "\n\nCuối."
    chunks = backend._split_batches(text, size=8, char_limit=500)
    assert len(chunks) > 1
    assert "".join(chunks) == text
    # The limit is soft w.r.t. paragraph separators: "\n\n" is joined into the
    # chunk but never counted, so allow a few bytes of separator slack.
    assert all(len(c) <= 500 + 10 for c in chunks)


def test_short_text_is_unchanged(backend):
    text = "Một đoạn ngắn."
    assert backend._split_batches(text, size=8, char_limit=500) == [text]
