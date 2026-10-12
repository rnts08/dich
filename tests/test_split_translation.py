"""Block translations map back to per-utterance lines without losing text."""


def test_exact_count(backend):
    assert backend._split_translation("a\nb\nc", 3) == ["a", "b", "c"]


def test_blank_lines_are_ignored(backend):
    assert backend._split_translation("a\n\n  b  \n\n", 2) == ["a", "b"]


def test_surplus_folds_into_the_last_line(backend):
    # The model sometimes merges utterances; nothing may be silently dropped.
    assert backend._split_translation("a\nb\nc", 2) == ["a", "b c"]


def test_deficit_pads_with_empty(backend):
    # A short result leaves later utterances provisional, not hallucinated.
    assert backend._split_translation("a", 3) == ["a", "", ""]
