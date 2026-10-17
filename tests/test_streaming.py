"""
Streaming translated output: deltas arrive while the model is generating, a
streamed run assembles exactly the text a batch run produces, and a fallback
retry tells the client to start over.

The engine is faked at `get_engine`, so no weights are needed; the real
`run_messages` path is covered by the model-marked golden tests, which now
exercise `stream_generate` end to end.
"""

from types import SimpleNamespace

import pytest

from dich import backend, ui


class StreamingFakeEngine:
    """Stands in for LocalEngine: emits word-by-word deltas like mlx-lm."""

    def __init__(self, outputs):
        # What each call to run() returns, in order.
        self.outputs = list(outputs)
        self.chunks_seen = []

    def run(self, chunk, max_tokens, on_token=None):
        self.chunks_seen.append(chunk)
        text = self.outputs.pop(0)
        words = text.split(" ")
        if on_token is not None:
            for word in words[:-1]:
                on_token(word + " ")
            on_token(words[-1])
        return text, len(words)

    def count_tokens(self, text: str) -> int:
        return len(text.split())


# -- streaming vs batch -------------------------------------------------------
def test_streamed_text_equals_batch_text(backend_cache, monkeypatch):
    engine = StreamingFakeEngine(["Xin chào thế giới."])
    monkeypatch.setattr(backend, "get_engine", lambda name: engine)

    batch = backend.translate_text("Xin chào thế giới.", use_cache=False)

    engine = StreamingFakeEngine(["Xin chào thế giới."])
    monkeypatch.setattr(backend, "get_engine", lambda name: engine)
    deltas = []
    streamed = backend.translate_text(
        "Xin chào thế giới.", use_cache=False, on_token=deltas.append
    )

    assert streamed.text == batch.text
    # The run announces itself with a reset, then streams the whole text.
    assert deltas[0] is None
    assert "".join(d for d in deltas if d is not None) == batch.text


def test_multi_chunk_stream_matches_the_joined_result(backend_cache, monkeypatch):
    # The splitter itself is covered by test_batching; this pins the stream
    # protocol across chunk boundaries, so the splitter is fixed to two chunks.
    monkeypatch.setattr(
        backend, "_split_batches", lambda text, size, char_limit: [text[:11], text[13:]]
    )
    engine = StreamingFakeEngine(["Một hai ba.", "Bốn năm sáu."])
    monkeypatch.setattr(backend, "get_engine", lambda name: engine)
    deltas = []
    result = backend.translate_text(
        "Một hai ba.\n\nBốn năm sáu.", use_cache=False, on_token=deltas.append
    )

    # The batch path joins chunks with "\n"; the stream must assemble the
    # same text, separator included.
    assert result.batches == 2
    assert "".join(d for d in deltas if d is not None) == result.text
    assert result.text == "Một hai ba.\nBốn năm sáu.\n"


def test_streaming_without_a_callback_changes_nothing(backend_cache, monkeypatch):
    engine = StreamingFakeEngine(["Một câu bình thường."])
    monkeypatch.setattr(backend, "get_engine", lambda name: engine)
    result = backend.translate_text("Một câu bình thường.", use_cache=False)
    assert result.text == "Một câu bình thường."


# -- retries and cache --------------------------------------------------------
def test_fallback_retry_resets_the_stream(backend_cache, monkeypatch):
    class ExplodingEngine:
        def run(self, chunk, max_tokens, on_token=None):
            raise RuntimeError("primary down")

    good = StreamingFakeEngine(["Câu đúng."])
    engines = {"primary/model": ExplodingEngine(), "fallback/model": good}
    monkeypatch.setattr(backend, "get_engine", lambda name: engines[name])
    monkeypatch.setitem(backend.CONFIG, "fallback_models", ["fallback/model"])

    deltas = []
    result = backend.translate_text(
        "một câu.",
        model="primary/model",
        use_cache=False,
        on_token=deltas.append,
    )

    assert result.fell_back
    assert deltas[0] is None  # the first attempt announces itself
    assert deltas[1:].count(None) == 1  # the retry resets the client
    assert "".join(d for d in deltas if d is not None) == "Câu đúng."


def test_cache_hit_never_streams(backend_cache):
    backend.cache_put("đã dịch.", "Already translated.")
    deltas = []
    result = backend.translate_text("đã dịch.", on_token=deltas.append)
    assert result.cached
    assert deltas == []


def test_on_token_must_be_callable(backend_cache):
    with pytest.raises(TypeError):
        backend.translate_text("một câu.", on_token="yes")


# -- UI plumbing --------------------------------------------------------------
def test_on_token_routes_reset_and_deltas(backend):
    """None clears the pane; every delta appends — both via the target pane."""
    appended, resets = [], []
    app = SimpleNamespace(
        target_pane=SimpleNamespace(
            stream_append=appended.append,
            stream_reset=lambda: resets.append(True),
        )
    )
    ui.DichApp._on_token(app, "Xin ")
    ui.DichApp._on_token(app, "chào.")
    ui.DichApp._on_token(app, None)
    assert appended == ["Xin ", "chào."]
    assert resets == [True]


def test_text_pane_stream_accumulates_then_the_final_render_wins(requires_display):
    from dich.ui import TextPane

    pane = TextPane(requires_display, "Translation", editable=False, markdown=True)
    pane.set("")  # translate_current clears the pane before the run

    pane.stream_append("Hello ")
    pane.stream_append("**world**\n")
    assert pane.get() == "Hello **world**"
    assert pane.markdown_source() == "Hello **world**\n"

    pane.stream_reset()
    assert pane.get() == ""
    assert pane.markdown_source() == ""

    pane.stream_append("Attempt two")
    pane.set("Final")  # the finished translation re-renders over the stream
    assert pane.get() == "Final"
    assert pane.markdown_source() == "Final"
