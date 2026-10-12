"""Stop-token handling — the <end_of_turn> runaway, in miniature.

Gemma-family models end a turn with <end_of_turn>, which is not in mlx-lm's
eos set. Unhandled, a short sentence ran to max_tokens: ~10s and thousands of
junk characters per utterance.
"""


class FakeTokenizer:
    def __init__(self, mapping=None, fail=False):
        self.mapping = mapping or {}
        self.eos_token_ids = {1}
        self.fail = fail

    def convert_tokens_to_ids(self, token):
        if self.fail:
            raise ValueError("no such helper")
        return self.mapping.get(token)


def _engine(backend, tokenizer):
    engine = backend.LocalEngine("fake/model")
    engine._tokenizer = tokenizer        # __init__ loads nothing
    return engine


def test_end_of_turn_is_added_to_the_stop_set(backend):
    engine = _engine(backend, FakeTokenizer({"<end_of_turn>": 109}))
    engine._align_stop_tokens()
    assert engine._tokenizer.eos_token_ids == {1, 109}


def test_aligning_twice_does_not_duplicate(backend):
    engine = _engine(backend, FakeTokenizer({"<end_of_turn>": 109}))
    engine._align_stop_tokens()
    engine._align_stop_tokens()
    assert engine._tokenizer.eos_token_ids == {1, 109}


def test_unknown_token_leaves_the_stop_set_alone(backend):
    engine = _engine(backend, FakeTokenizer())
    engine._align_stop_tokens()
    assert engine._tokenizer.eos_token_ids == {1}


def test_tokenizer_without_the_helper_is_survivable(backend):
    engine = _engine(backend, FakeTokenizer(fail=True))
    engine._align_stop_tokens()           # must not raise
    assert engine._tokenizer.eos_token_ids == {1}


def test_trim_cuts_at_end_of_turn(backend):
    out = backend._trim_stop_tokens("Được.<end_of_turn>" + " junk" * 2000)
    assert out == "Được."
    assert len(out) < 60                  # the 6000-character runaway


def test_trim_cuts_at_eos_variants_and_strips(backend):
    assert backend._trim_stop_tokens("A<eos>B") == "A"
    assert backend._trim_stop_tokens("A<end_of_text>B") == "A"
    assert backend._trim_stop_tokens("  nothing to cut  ") == "nothing to cut"
