"""
The voice-only microphone filter.

The bug this covers: on the GPU path there is no faster-whisper ``vad_filter``,
so a noisy window (a fan, a keyboard, room tone) was handed straight to Whisper,
which invented text for it. The gate rejects those windows *before* the model
sees them. These tests use synthetic signals — no microphone, no model.
"""

import numpy as np

from dich import speech_engine
from dich import voice_filter as vf

SR = vf.SAMPLE_RATE


def _seconds(seconds: float) -> np.ndarray:
    return np.arange(int(SR * seconds)) / SR


def _white_noise(seconds=2.0, amp=0.2):
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(SR * seconds)) * amp).astype(np.float32)


def _voice(seconds=2.0, amp=1.0):
    """A voice-band harmonic stack: what a voiced window looks like."""
    t = _seconds(seconds)
    tone = (
        0.3 * np.sin(2 * np.pi * 400 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1600 * t)
    )
    return (amp * tone).astype(np.float32)


# -- the primitives --------------------------------------------------------
def test_voice_band_ratio_separates_noise_from_a_voice():
    assert vf.voice_band_ratio(_white_noise()) < vf.VOICE_BAND_MIN_RATIO
    assert vf.voice_band_ratio(_voice()) > vf.VOICE_BAND_MIN_RATIO


def test_spectral_flatness_is_low_for_a_tone_high_for_noise():
    assert vf.spectral_flatness(_voice()) < 0.2
    assert vf.spectral_flatness(_white_noise()) > 0.5


def test_zero_crossing_rate_bounds():
    assert vf.zero_crossing_rate(np.zeros(SR)) == 0.0
    assert vf.zero_crossing_rate(_white_noise()) > 0.4  # hiss sits high
    assert 0.02 < vf.zero_crossing_rate(_voice()) < 0.2  # a voice sits in between


# -- the gate --------------------------------------------------------------
def test_noise_is_not_voice():
    assert vf.VoiceGate().is_voice(_white_noise()) is False


def test_silence_is_not_voice():
    assert vf.VoiceGate().is_voice(np.zeros(SR * 2, dtype=np.float32)) is False


def test_a_constant_offset_is_not_voice():
    # A DC level is the fixture that used to stand in for audio; it is not speech.
    assert vf.VoiceGate().is_voice(np.full(SR * 2, 0.5, dtype=np.float32)) is False


def test_mains_hum_is_not_voice():
    t = _seconds(2.0)
    hum = (0.3 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    assert vf.VoiceGate().is_voice(hum) is False


def test_a_voice_band_window_is_voice():
    assert vf.VoiceGate().is_voice(_voice()) is True


def test_a_quiet_voice_is_still_accepted():
    gate = vf.VoiceGate()
    assert gate.is_voice(_voice(amp=0.05)) is True


def test_too_short_a_window_is_not_judged_voice():
    assert vf.VoiceGate().is_voice(_voice(seconds=0.01)) is False


def test_the_noise_floor_rises_with_room_noise_but_speech_still_passes():
    gate = vf.VoiceGate()
    noise = _white_noise()
    for _ in range(10):
        gate.is_voice(noise)
    assert gate.noise_floor > vf.NOISE_FLOOR_INIT
    # A voice is well above the learned floor, so it must still get through.
    assert gate.is_voice(_voice()) is True


# -- integration with the engine ------------------------------------------
class _FakeModel:
    def __init__(self, called):
        self._called = called

    def transcribe(self, audio, **kwargs):
        self._called.append(True)
        from types import SimpleNamespace

        info = SimpleNamespace(duration=2.0, language="vi", language_probability=0.9)
        return iter([]), info


def test_engine_does_not_send_noise_to_the_model():
    called = []
    engine = speech_engine.SpeechEngine(model_size="small")
    engine._model = _FakeModel(called)

    engine._process_window(_white_noise(seconds=4.0))

    assert called == []


def test_engine_sends_a_voice_window_to_the_model():
    called = []
    engine = speech_engine.SpeechEngine(model_size="small")
    engine._model = _FakeModel(called)

    engine._process_window(_voice(seconds=4.0))

    assert called == [True]


def test_the_filter_can_be_turned_off():
    called = []
    engine = speech_engine.SpeechEngine(model_size="small", voice_only_filter=False)
    engine._model = _FakeModel(called)

    engine._process_window(_white_noise(seconds=4.0))

    assert called == [True]


def test_two_quiet_windows_close_a_hanging_utterance():
    from types import SimpleNamespace

    class Seg:
        text = " xin chào các bạn"
        start = 0.0
        end = 1.0

    class Model:
        def transcribe(self, audio, **kwargs):
            info = SimpleNamespace(
                duration=2.0, language="vi", language_probability=0.9
            )
            return iter([Seg()]), info

    engine = speech_engine.SpeechEngine(model_size="small")
    engine._model = Model()
    engine._process_window(_voice(seconds=4.0))
    assert engine.latest_partial() == "xin chào các bạn"

    engine._process_window(np.zeros(SR * 4, dtype=np.float32))
    assert engine.latest_partial() == "xin chào các bạn"  # one quiet window: hold

    engine._process_window(np.zeros(SR * 4, dtype=np.float32))
    assert engine.latest_partial() == ""
    assert engine.drain_final_sentences() == ["xin chào các bạn"]
