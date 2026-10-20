"""
Statistics: the LLM speeds captured from stream_generate land in
TranslationResult and the stats view, and the lifetime counters persist and
accumulate in config.json.
"""

import json
import time
from types import SimpleNamespace

import pytest

from dich import backend


@pytest.fixture(autouse=True)
def clean_stats(backend):
    """Zero the counters in memory and on disk around each test."""
    backend.CONFIG["stats"] = {}
    if backend.CONFIG_PATH.exists():
        data = json.loads(backend.CONFIG_PATH.read_text())
        data["stats"] = {}
        backend.CONFIG_PATH.write_text(json.dumps(data))
    yield
    backend.CONFIG["stats"] = {}


# -- LLM generation speeds ----------------------------------------------------
class _FakeTokenizer:
    def encode(self, text):
        return text.split()


def _fake_stream(responses):
    def stream_generate(model, tokenizer, prompt=None, max_tokens=256, **kwargs):
        yield from responses

    return stream_generate


@pytest.fixture()
def streaming_engine(monkeypatch):

    engine = backend.LocalEngine("test/model")
    monkeypatch.setattr(engine, "load", lambda: (object(), _FakeTokenizer()))
    monkeypatch.setattr(engine, "render_prompt", lambda messages: "PROMPT")
    monkeypatch.setattr(engine, "count_tokens", lambda text: len(text.split()))
    return engine


def test_run_messages_captures_generation_speeds(streaming_engine, monkeypatch):
    import mlx_lm

    responses = [
        SimpleNamespace(
            text="Xin ",
            prompt_tps=0.0,
            generation_tps=1.0,
            prompt_tokens=10,
            generation_tokens=1,
        ),
        SimpleNamespace(
            text="chào.",
            prompt_tps=812.5,
            generation_tps=42.0,
            prompt_tokens=10,
            generation_tokens=2,
        ),
    ]
    monkeypatch.setattr(mlx_lm, "stream_generate", _fake_stream(responses))

    text, _tokens = streaming_engine.run_messages([], max_tokens=64)

    assert text == "Xin chào."
    assert backend.LLM_STATS["prompt_tps"] == 812.5
    assert backend.LLM_STATS["generation_tps"] == 42.0
    assert backend.LLM_STATS["generation_tokens"] == 2
    assert backend.LLM_STATS["model"] == "test/model"


def test_translate_text_attaches_speeds_and_counts_totals(backend_cache, monkeypatch):
    # A fake engine at get_engine level: translate_text never touches weights.
    class FakeEngine:
        def run(self, chunk, max_tokens, on_token=None):
            backend.LLM_STATS.clear()
            backend.LLM_STATS.update(
                {
                    "model": "test/model",
                    "prompt_tps": 800.0,
                    "generation_tps": 40.0,
                    "peak_memory_gb": 2.2,
                    "at": time.time(),
                }
            )
            return "Một bản dịch.", 3

    monkeypatch.setattr(backend, "get_engine", lambda name: FakeEngine())

    result = backend.translate_text("xin chào", use_cache=False)

    assert result.generation_tps == 40.0
    assert result.prompt_tps == 800.0
    assert result.peak_memory_gb == 2.2
    stats = backend.get_stats()
    assert stats["translations"] == 1
    assert stats["words_translated"] >= 2


def test_stale_llm_stats_are_not_attached(backend_cache, monkeypatch):
    """Speeds from a previous run must not leak into this result."""

    class FakeEngine:
        def run(self, chunk, max_tokens, on_token=None):
            return "bản dịch", 2

    monkeypatch.setattr(backend, "get_engine", lambda name: FakeEngine())
    backend.LLM_STATS.clear()
    backend.LLM_STATS.update(
        {"model": "old", "generation_tps": 99.0, "at": time.time() - 3600}
    )

    result = backend.translate_text("xin chào", use_cache=False)

    assert result.generation_tps == 0.0


def test_cache_hit_counts_as_a_cache_hit(backend_cache):
    backend.cache_put("đã dịch.", "Already translated.")
    result = backend.translate_text("đã dịch.")
    assert result.cached
    assert backend.get_stats().get("cache_hits") == 1


# -- lifetime counters --------------------------------------------------------
def test_bump_stats_accumulates_and_persists(backend):
    backend.bump_stats(words_translated=3)
    backend.bump_stats(words_translated=2, sentences_transcribed=1)

    stats = backend.get_stats()
    assert stats["words_translated"] == 5
    assert stats["sentences_transcribed"] == 1

    with open(backend.CONFIG_PATH, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["stats"]["words_translated"] == 5


def test_bump_stats_merges_with_other_config_writes(backend):
    # Something else writes the file between two bumps; the counters survive.
    data = json.loads(backend.CONFIG_PATH.read_text())
    data["unrelated"] = True
    backend.CONFIG_PATH.write_text(json.dumps(data))

    backend.bump_stats(words_translated=1)
    backend.bump_stats(words_translated=1)

    with open(backend.CONFIG_PATH, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk["unrelated"] is True
    assert on_disk["stats"]["words_translated"] == 2
