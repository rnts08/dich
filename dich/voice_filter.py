"""
Voice-only microphone filter.

The problem this solves: Whisper (on any backend) hallucinates when it is handed
a window that is *not* speech. A fan, a keyboard, a TV in the next room or plain
room tone all get "transcribed" into invented words, and on the GPU path there is
no faster-whisper ``vad_filter`` to hide it. The cheap energy floor in
speech_engine (``SILENCE_FLOOR``) only catches near-silence; a noisy room sails
straight through it.

This is a dependency-free (numpy) voice activity gate that runs *before* a window
is offered to the model, so noise never becomes text. Three independent tests
must agree that a window looks like a voice:

  * **Voice-band energy** — speech concentrates energy between roughly 300 Hz and
    3.4 kHz, while broadband noise (hiss, fans) spreads it across the whole
    0-8 kHz band, so the voice-band share of a noise window is much lower.
  * **Spectral flatness** — the geometric/arithmetic mean ratio of the spectrum.
    A voice is peaky (near 0); white noise is flat (near 1).
  * **Zero-crossing rate** — a coarse shape check. Mains hum and rumble sit far
    below a voice; hiss sits above it.

An adaptive noise floor adds the fourth condition: a window must also be louder
than the room's own background, which is learned from windows that are *not*
speech, so a quiet utterance after loud noise is still heard. The floor is
measured on the voice band rather than the whole signal, so a low rumble two
octaves below speech cannot raise the bar a quiet voice has to clear.

The engine runs a speech-band front-end (``audio_filter``) ahead of this gate, so
by the time a window arrives here most out-of-band noise — hum, rumble, top-end
hiss — is already gone and the voice-band share is a cleaner signal.

The gate is deliberately permissive. A false negative (dropping real speech) is
worse than a false positive (transcribing a noisy window), so the thresholds lean
toward accepting borderline windows; the filters in speech_engine still catch the
hallucinations that slip through.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000

# Speech energy lives here on the telephone band; outside it is mostly noise.
VOICE_BAND_LOW = 300.0
VOICE_BAND_HIGH = 3400.0
# Minimum share of a window's energy that must sit in the voice band. Broadband
# noise lands near (3400-300)/8000 = 0.39, so 0.5 separates the two cleanly while
# still admitting voiced speech, which sits well above it.
VOICE_BAND_MIN_RATIO = 0.5
# Spectral flatness ceiling. Hiss and fans approach 1.0; a voiced window is well
# under 0.5. 0.7 keeps sung/whispered speech in and steady noise out.
MAX_SPECTRAL_FLATNESS = 0.7
# Zero-crossing-rate window. Below the floor is DC/rumble; above the ceiling is
# hiss. A 16 kHz voice sits comfortably inside.
ZCR_MIN = 0.002
ZCR_MAX = 0.4
# A window must be this many times louder than the learned noise floor.
NOISE_FLOOR_MARGIN = 2.0
# Speed of the adaptive noise floor. It drops quickly toward a quieter window
# and creeps up slowly toward a louder one, so a cough or a door slam does not
# raise the bar a voice has to clear on the very next window.
NOISE_FLOOR_ADAPT_DOWN = 0.2
NOISE_FLOOR_ADAPT_UP = 0.02
# Where the learned floor starts, before any non-speech window has been seen.
NOISE_FLOOR_INIT = 0.003
# An absolute voice-band RMS below which a window is never speech, whatever its
# shape. It sits under the quietest usable utterance (a level sweep transcribed
# Vietnamese down to about 0.002 RMS) and only rejects a near-inaudible tone the
# model could otherwise read words into.
VOICE_ABS_MIN = 0.001
# Windows shorter than this cannot carry a stable spectrum.
MIN_WINDOW_SECONDS = 0.05


def voice_band_ratio(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low: float = VOICE_BAND_LOW,
    high: float = VOICE_BAND_HIGH,
) -> float:
    """Share of a window's spectral energy that falls inside the voice band."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = samples.size
    if n < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(n))) ** 2
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
    band = float(spectrum[(freqs >= low) & (freqs <= high)].sum())
    return band / total


def spectral_flatness(samples: np.ndarray) -> float:
    """
    Geometric/arithmetic mean of the power spectrum, in [0, 1].

    A pure tone approaches 0; white noise approaches 1. DC is dropped so a
    constant offset cannot read as a peak.
    """
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = samples.size
    if n < 2:
        return 1.0
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(n)))[1:]
    if spectrum.size == 0:
        return 1.0
    spectrum = spectrum + 1e-12
    geometric = float(np.exp(np.mean(np.log(spectrum))))
    arithmetic = float(np.mean(spectrum))
    if arithmetic <= 0.0:
        return 1.0
    return min(1.0, geometric / arithmetic)


def zero_crossing_rate(samples: np.ndarray) -> float:
    """Fraction of adjacent sample pairs that change sign."""
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    if samples.size < 2:
        return 0.0
    signs = np.sign(samples)
    signs[signs == 0] = 1.0
    return float(np.mean(signs[1:] != signs[:-1]))


class VoiceGate:
    """
    Stateful voice detector that adapts to the background noise of a room.

    One instance per recording session. Call :meth:`is_voice` with each window in
    order; the noise floor is only updated from windows judged *not* to be voice,
    so speech never raises the bar it has to clear.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = int(sample_rate)
        self.noise_floor = NOISE_FLOOR_INIT  # learned from non-speech windows

    def rms(self, samples: np.ndarray) -> float:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    def is_voice(self, samples: np.ndarray) -> bool:
        """True when the window looks like a voice rather than noise or silence."""
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        if samples.size < int(MIN_WINDOW_SECONDS * self.sample_rate):
            return False

        full = self.rms(samples)
        if full <= 0.0:
            self._adapt(0.0, speech=False)
            return False

        band = voice_band_ratio(samples, self.sample_rate)
        flat = spectral_flatness(samples)
        zcr = zero_crossing_rate(samples)
        # The floor is measured on the *voice band* alone (Parseval: full RMS
        # scaled by the square root of the band's share), so a rumble two
        # octaves below speech cannot raise the bar a quiet voice has to clear.
        level = full * float(np.sqrt(max(0.0, band)))

        shaped = (
            band >= VOICE_BAND_MIN_RATIO
            and flat <= MAX_SPECTRAL_FLATNESS
            and ZCR_MIN <= zcr <= ZCR_MAX
        )
        loud = level >= max(self.noise_floor * NOISE_FLOOR_MARGIN, VOICE_ABS_MIN)

        speech = bool(shaped and loud)
        self._adapt(level, speech)
        return speech

    def _adapt(self, level: float, speech: bool) -> None:
        """Learn the noise floor from non-speech windows only."""
        if speech:
            return
        rate = (
            NOISE_FLOOR_ADAPT_DOWN if level < self.noise_floor else NOISE_FLOOR_ADAPT_UP
        )
        self.noise_floor = (1.0 - rate) * self.noise_floor + rate * level
