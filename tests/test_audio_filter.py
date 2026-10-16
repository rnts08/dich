"""
The speech-band front-end.

The bug this covers: the microphone hears the whole room, and everything below
the voice band (mains hum, traffic rumble, a desk knock) both buries quiet speech
and fools the voice gate into seeing a low voice-band share. The engine runs
``band_pass`` ahead of the gate and the model, so only the frequencies a voice
occupies survive. These tests use synthetic signals — no microphone, no model.
"""

import numpy as np

from dich import audio_filter as af
from dich import speech_engine
from dich import voice_filter as vf

SR = af.SAMPLE_RATE


def _seconds(seconds: float) -> np.ndarray:
    return np.arange(int(SR * seconds)) / SR


def _tone(hz: float, seconds=2.0, amp=0.3) -> np.ndarray:
    return (amp * np.sin(2 * np.pi * hz * _seconds(seconds))).astype(np.float32)


def _voice(seconds=2.0, amp=1.0) -> np.ndarray:
    t = _seconds(seconds)
    tone = (
        0.3 * np.sin(2 * np.pi * 400 * t)
        + 0.2 * np.sin(2 * np.pi * 800 * t)
        + 0.1 * np.sin(2 * np.pi * 1600 * t)
    )
    return (amp * tone).astype(np.float32)


def _rms(x) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


# -- the filter ------------------------------------------------------------
def test_mains_hum_is_removed():
    hum = _tone(50.0)
    filtered = af.band_pass(hum)
    # A 50 Hz hum is two octaves below speech and lands near -24 dB.
    assert _rms(filtered) < 0.1 * _rms(hum)


def test_top_end_hiss_is_trimmed():
    hiss = _tone(7900.0)
    filtered = af.band_pass(hiss)
    # The top edge only trims the last kilohertz; it must not gut the sibilants.
    assert _rms(filtered) < 0.7 * _rms(hiss)


def test_a_voice_passes_through():
    voice = _voice()
    filtered = af.band_pass(voice)
    # Almost all of a voice is in band, so the level is essentially unchanged.
    assert 0.8 * _rms(voice) < _rms(filtered) < 1.2 * _rms(voice)


def test_output_keeps_length_and_dtype():
    voice = _voice(seconds=1.5)
    filtered = af.band_pass(voice)
    assert filtered.shape == voice.shape
    assert filtered.dtype == np.float32


def test_a_too_short_window_is_returned_unchanged():
    tiny = np.ones(8, dtype=np.float32)
    np.testing.assert_array_equal(af.band_pass(tiny), tiny)


def test_silence_stays_silent():
    silence = np.zeros(SR * 2, dtype=np.float32)
    np.testing.assert_allclose(af.band_pass(silence), silence, atol=1e-6)


# -- the diagnostic --------------------------------------------------------
def test_out_of_band_ratio_flags_noise_not_speech():
    assert af.out_of_band_ratio(_tone(50.0)) > 0.8
    assert af.out_of_band_ratio(_tone(7900.0)) > 0.5
    assert af.out_of_band_ratio(_voice()) < 0.2


# -- integration with the engine ------------------------------------------
class _FakeModel:
    def __init__(self, called):
        self._called = called

    def transcribe(self, audio, **kwargs):
        self._called.append(True)
        from types import SimpleNamespace

        info = SimpleNamespace(duration=2.0, language="vi", language_probability=0.9)
        return iter([]), info


def _hum_plus_voice(seconds=4.0) -> np.ndarray:
    """A quiet voice buried under a loud 50 Hz hum: the room-noise case."""
    hum = _tone(50.0, seconds, amp=1.0)
    voice = _voice(seconds, amp=0.6)
    return (hum + voice).astype(np.float32)


def test_the_band_filter_rescues_a_voice_buried_in_hum():
    # Without the front-end the gate sees a hum-dominated window and refuses it.
    raw = _hum_plus_voice()
    assert vf.VoiceGate().is_voice(raw) is False

    # With it, the hum is gone and the voice is heard.
    assert vf.VoiceGate().is_voice(af.band_pass(raw)) is True


def test_engine_sends_a_hum_buried_voice_only_with_the_band_filter():
    called = []
    engine = speech_engine.SpeechEngine(model_size="small", speech_band_filter=True)
    engine._model = _FakeModel(called)
    engine._process_window(_hum_plus_voice())
    assert called == [True]

    off = []
    engine_off = speech_engine.SpeechEngine(
        model_size="small", speech_band_filter=False
    )
    engine_off._model = _FakeModel(off)
    engine_off._process_window(_hum_plus_voice())
    assert off == []


def test_engine_still_rejects_noise_with_the_band_filter_on():
    called = []
    engine = speech_engine.SpeechEngine(model_size="small", speech_band_filter=True)
    engine._model = _FakeModel(called)
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(SR * 4) * 0.2).astype(np.float32)
    engine._process_window(noise)
    assert called == []
