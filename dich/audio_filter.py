"""
Speech-band front-end for the microphone.

A microphone hears everything in the room: mains hum at 50/60 Hz, the rumble of
traffic or a desk knock, the low whoosh of a laptop fan and hiss above the
speech range. None of it is a voice, but Whisper will still try to read words
into it if it reaches the model. This module keeps only the band a human voice
actually occupies (roughly 80 Hz to 7 kHz) so the noise that sits outside it is
removed *before* the voice gate and the model see it.

The filter is a zero-phase band-pass built from a Butterworth magnitude
response and applied in the frequency domain, so it needs nothing beyond numpy —
the same dependency budget as ``voice_filter``. Zero-phase matters here: a
causal IIR filter would smear the signal forward in time, which distorts the
alignment the sentence assembler relies on.

The high-pass edge is the one that earns its keep. Real microphones pick up far
more energy below 100 Hz than above it, and that low-frequency noise both buries
quiet speech and gives the voice gate a misleadingly low voice-band share. Removing
it raises the speech-to-noise ratio at the source.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
# The band a voice occupies. The low edge drops hum and rumble; the high edge
# trims the top-end hiss and breath noise while leaving the sibilants Whisper
# expects. Cutting lower (a telephone 3.4 kHz) would measurably hurt recognition,
# so it is not done.
SPEECH_LOW_HZ = 80.0
SPEECH_HIGH_HZ = 7000.0
# Butterworth order per edge. 6 is a 36 dB/octave rolloff: steep enough to bury
# hum two octaves down (a 50 Hz tone lands around -24 dB), gentle enough to avoid
# ringing in the passband. The top edge only trims the last kilohertz of hiss and
# breath noise; the sibilants Whisper needs sit below it.
FILTER_ORDER = 6


def band_mask(
    freqs: np.ndarray,
    low: float = SPEECH_LOW_HZ,
    high: float = SPEECH_HIGH_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """Butterworth band-pass magnitude for each frequency, in [0, 1]."""
    f = np.maximum(np.asarray(freqs, dtype=np.float64), 1e-9)
    high_pass = 1.0 / np.sqrt(1.0 + (low / f) ** (2 * order))
    low_pass = 1.0 / np.sqrt(1.0 + (f / high) ** (2 * order))
    return (high_pass * low_pass).astype(np.float64)


def band_pass(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low: float = SPEECH_LOW_HZ,
    high: float = SPEECH_HIGH_HZ,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Return `samples` with everything outside the speech band attenuated.

    Zero-phase: the spectrum is scaled and transformed back, so the waveform
    keeps its shape and only its spectrum changes. Windows shorter than a few
    cycles of the low edge cannot be filtered meaningfully and are returned
    unchanged.
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = x.size
    if n < 16 or high <= 0:
        return x.copy()
    freqs = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
    spectrum = np.fft.rfft(x) * band_mask(freqs, low, high, order)
    return np.fft.irfft(spectrum, n).astype(np.float32)


def out_of_band_ratio(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    low: float = SPEECH_LOW_HZ,
    high: float = SPEECH_HIGH_HZ,
) -> float:
    """
    Share of energy that sits *outside* the speech band, in [0, 1].

    A diagnostic for tests and the UI: a clean voice window is mostly in-band (a
    low value); a rumble or a high hiss is mostly out (a value near 1).
    """
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    n = x.size
    if n < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    total = float(spectrum.sum())
    if total <= 0.0:
        return 0.0
    freqs = np.fft.rfftfreq(n, 1.0 / float(sample_rate))
    inside = float(spectrum[(freqs >= low) & (freqs <= high)].sum())
    return max(0.0, min(1.0, 1.0 - inside / total))
