"""The hallucination and repetition filters, as pure functions.

The trade-off is deliberate and asserted here as such: a genuinely spoken
"cảm ơn các bạn" is dropped because in YouTube-trained Whisper it is
overwhelmingly an artefact.
"""

import numpy as np

from dich import speech_engine


def test_youtube_boilerplate_is_caught():
    text = "Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn."
    assert speech_engine._is_hallucination(text)


def test_the_deliberate_tradeoff():
    # Documented limitation, not a bug: real speech matching a marker is dropped.
    assert speech_engine._is_hallucination("Cảm ơn các bạn và hẹn gặp lại.")


def test_marker_spliced_into_real_speech_is_trimmed():
    text = "À không, con đi ở nhà. Cảm ơn các bạn và hẹn gặp lại video này."
    assert speech_engine._strip_hallucinations(text) == "À không, con đi ở nhà."


def test_repetition_of_a_real_phrase_is_flagged():
    assert speech_engine._is_repetition("cảm ơn cảm ơn cảm ơn cảm ơn cảm ơn cảm ơn")


def test_natural_repetition_is_not_flagged():
    assert not speech_engine._is_repetition(
        "Bộ trưởng bộ giáo dục nói về bộ sách giáo khoa mới của bộ."
    )


def test_too_short_to_judge():
    assert not speech_engine._is_repetition("cảm ơn cảm ơn")


def test_empty_trailing_samples_are_not_silence():
    eng = speech_engine.SpeechEngine()
    assert not eng._trailing_silence(np.array([], dtype=np.float32))


def test_no_samples_yet_are_not_silence():
    # _last_samples is None until the first window is transcribed; judging
    # trailing silence on it used to raise TypeError.
    eng = speech_engine.SpeechEngine()
    assert not eng._trailing_silence(None)


def test_uniform_audio_is_not_trailing_silence():
    eng = speech_engine.SpeechEngine()
    assert not eng._trailing_silence(np.ones(32000, dtype=np.float32) * 0.3)
