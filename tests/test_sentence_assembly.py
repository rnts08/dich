"""Sentence assembly: the three flush triggers, trailing silence, and the drops.

_ingest_segments is where the subtle failure modes live — the shipped bugs
(the sentence flush that never fired) were all here. Feeding it fake segments
exercises the rules without a microphone or a model.
"""

import numpy as np

from dich import speech_engine


class Seg:
    """Duck-typed stand-in for a faster-whisper segment."""

    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end


def _engine():
    return speech_engine.SpeechEngine()


def test_flush_on_end_punctuation():
    eng = _engine()
    eng._ingest_segments([Seg("Xin chào các bạn.", 0.0, 0.4)], 2.0)
    assert eng.drain_final_sentences() == ["Xin chào các bạn."]


def test_flush_on_every_end_mark():
    # The punctuation check passed the marker string as ONE suffix to
    # endswith(), so it matched only text ending in the entire sequence —
    # never. Each mark must trigger on its own.
    for mark in (".", "?", "!", "…"):
        eng = _engine()
        eng._ingest_segments([Seg(f"Câu nói{mark}", 0.0, 0.4)], 2.0)
        assert eng.drain_final_sentences() == [f"Câu nói{mark}"], mark


def test_flush_on_vietnamese_end_word():
    eng = _engine()
    eng._ingest_segments([Seg("Tôi đi chợ rồi", 0.0, 0.4)], 2.0)
    assert eng.drain_final_sentences() == ["Tôi đi chợ rồi"]


def test_flush_on_segment_count_during_continuous_speech():
    # No punctuation anywhere: without the segment-count trigger, pending
    # accumulates forever and nothing is ever emitted.
    eng = _engine()
    segs = [
        Seg("một hai ba", i * 0.3, i * 0.3 + 0.2)
        for i in range(speech_engine.MAX_SEGMENTS_PER_EMIT)
    ]
    eng._ingest_segments(segs, 30.0)
    out = eng.drain_final_sentences()
    assert len(out) == 1
    assert out[0] == ("một hai ba " * 4).strip()


def test_flush_near_window_end():
    eng = _engine()
    # 1.9 >= 2.0 - 1.5: the tail of a window is flushed even mid-sentence.
    eng._ingest_segments([Seg("phần này nghe rõ", 0.0, 1.9)], 2.0)
    assert eng.drain_final_sentences() == ["phần này nghe rõ"]


def test_trailing_silence_closes_the_utterance():
    eng = _engine()
    eng._last_samples = np.ones(32000, dtype=np.float32) * 0.5
    eng._last_samples[-8000:] = 0.0001  # quiet tail -> silence flush
    eng._ingest_segments([Seg("các bạn nghe rõ gì", 0.0, 1.0)], 30.0)
    assert eng.drain_final_sentences() == ["các bạn nghe rõ gì"]


def test_no_flush_without_a_trigger():
    eng = _engine()
    eng._last_samples = np.ones(32000, dtype=np.float32) * 0.5  # loud tail
    eng._ingest_segments([Seg("một hai", 0.0, 0.3)], 30.0)
    assert eng.drain_final_sentences() == []
    assert eng._pending == "một hai"  # held for the next window


def test_short_fragments_are_dropped_by_emit():
    # MIN_SENTENCE_CHARS keeps "ừ" and similar grunts out of the transcript.
    eng = _engine()
    eng._ingest_segments([Seg("ừ", 0.0, 0.4)], 2.0)
    assert eng.drain_final_sentences() == []


def test_whitespace_is_normalised():
    eng = _engine()
    eng._ingest_segments([Seg(" Xin   chào \n các bạn. ", 0.0, 0.4)], 2.0)
    assert eng.drain_final_sentences() == ["Xin chào các bạn."]


def test_hallucinated_segment_is_dropped():
    eng = _engine()
    text = "Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn."
    eng._ingest_segments([Seg(text, 0.0, 0.4)], 2.0)
    assert eng.drain_final_sentences() == []
    assert eng._pending == ""


def test_repeated_phrase_is_dropped():
    eng = _engine()
    eng._ingest_segments(
        [
            Seg(
                "Công phí dưỡng, công phí dưỡng, công phí dưỡng, công phí dưỡng",
                0.0,
                0.4,
            )
        ],
        2.0,
    )
    assert eng.drain_final_sentences() == []


def test_real_speech_survives_the_filters():
    eng = _engine()
    eng._ingest_segments(
        [Seg("Bài báo này nói về kinh tế Việt Nam năm nay.", 0.0, 0.4)], 2.0
    )
    assert eng.drain_final_sentences() == [
        "Bài báo này nói về kinh tế Việt Nam năm nay."
    ]
