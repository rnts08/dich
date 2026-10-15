"""Input-level meter: `buffer_usage()` reports the captured peak and decays it."""

import numpy as np
import pytest

from dich import speech_engine


def test_buffer_usage_reports_peak_and_decays_on_read():
    engine = speech_engine.SpeechEngine()
    with engine._buffer_lock:
        engine._buffered = 16000  # 1 s of 16 kHz mono
        engine._peak = 0.4

    first = engine.buffer_usage()
    assert first["peak"] == pytest.approx(0.4)
    assert first["seconds"] == pytest.approx(1.0)

    # Every read decays the meter so it follows speech rather than latching.
    second = engine.buffer_usage()
    assert second["peak"] == pytest.approx(0.4 * 0.6)


def test_audio_callback_records_the_peak():
    engine = speech_engine.SpeechEngine()
    engine._on_audio(np.array([[0.5], [-0.3]], dtype=np.float32), 2, None, None)
    assert engine._peak == pytest.approx(0.5)

    # A louder block takes the max; a quieter one does not lower it.
    engine._on_audio(np.array([[0.1], [0.05]], dtype=np.float32), 2, None, None)
    assert engine._peak == pytest.approx(0.5)


def test_idle_usage_carries_a_zero_peak():
    usage = speech_engine.buffer_usage()
    assert usage["peak"] == 0.0
