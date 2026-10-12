"""Drop-oldest accounting in the capture buffer.

The overflow logic lived inside the start() closure, unreachable without a
live microphone. It is now SpeechEngine._on_audio, so the buffer ceiling —
the only unbounded thing in the audio path — is testable directly.
"""
import numpy as np

from dich import speech_engine


def _block(marker: float, samples: int = speech_engine.SAMPLE_RATE // 10):
    """One 0.1 s input block filled with a distinct marker value."""
    return np.full((samples, 1), marker, dtype=np.float32)


def test_overflow_drops_oldest_and_counts():
    eng = speech_engine.SpeechEngine()
    eng.buffer_max_seconds = 1.0            # bypass the 10 s constructor clamp
    limit = int(eng.buffer_max_seconds * speech_engine.SAMPLE_RATE)
    block_samples = speech_engine.SAMPLE_RATE // 10

    for i in range(20):
        eng._on_audio(_block(float(i)), 0, None, None)

    assert eng._dropped > 0
    assert eng._buffered <= limit
    # Everything that came in either sits in the buffer or was counted dropped.
    assert eng._dropped == 20 * block_samples - eng._buffered
    # The oldest audio left first: the buffer head is a recent block, not block 0.
    assert eng._buffer[0][0][0] >= 10.0


def test_single_block_over_ceiling_is_kept():
    # The len > 1 guard keeps at least one block, so the transcriber always
    # has something to read even when one block exceeds the whole ceiling.
    eng = speech_engine.SpeechEngine()
    eng.buffer_max_seconds = 1.0
    big = np.full((speech_engine.SAMPLE_RATE * 2, 1), 1.0, dtype=np.float32)
    eng._on_audio(big, 0, None, None)
    assert eng._dropped == 0
    assert len(eng._buffer) == 1


def test_blocks_under_ceiling_are_all_kept():
    eng = speech_engine.SpeechEngine()
    eng.buffer_max_seconds = 10.0
    for i in range(5):
        eng._on_audio(_block(float(i)), 0, None, None)
    assert eng._dropped == 0
    assert len(eng._buffer) == 5
