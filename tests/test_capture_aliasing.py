"""
The capture buffer must hold a copy of each callback's audio.

sounddevice's callback `indata` wraps PortAudio's own buffer memory, which is
reused on the next invocation. `_on_audio` used to store
`np.ascontiguousarray(indata[:, 0])` — a *view* for a mono stream — so every
buffered block silently showed the latest callback's audio: windows became
looped garbage and Whisper hallucinated over them, while archived recordings
(which always copied) stayed clean. That is the aliasing a live mic session
suffered; these tests pin the copy.
"""

import numpy as np

from dich import speech_engine as se


def test_captured_blocks_survive_portaudio_buffer_reuse():
    engine = se.SpeechEngine(window_seconds=1)
    mem = np.zeros((8000, 1), dtype=np.float32)  # one reused callback buffer

    mem[:] = 0.1  # first callback writes its audio into the shared memory
    engine._on_audio(mem, 8000, None, None)
    mem[:] = -0.2  # second callback overwrites the same memory
    engine._on_audio(mem, 8000, None, None)

    got = engine._take(0.5)  # one callback block, before the next overwrites it
    assert got is not None
    assert np.allclose(got, 0.1, atol=1e-6), (
        "the first block was aliased: the window shows the overwritten memory"
    )
    got2 = engine._take(0.5)
    assert np.allclose(got2, -0.2, atol=1e-6)


def test_window_assembly_is_unaffected_for_multi_block_windows():
    """Blocks that arrive in pieces still assemble in order, once copied."""
    engine = se.SpeechEngine(window_seconds=1)
    mem = np.zeros((4000, 1), dtype=np.float32)  # smaller callbacks, 4 per window

    values = [0.1, 0.2, 0.3, 0.4]
    for v in values:
        mem[:] = v
        engine._on_audio(mem, 4000, None, None)

    got = engine._take(1.0)
    assert got is not None and len(got) == 16000
    for i, v in enumerate(values):
        assert np.allclose(got[i * 4000 : (i + 1) * 4000], v, atol=1e-6)


def test_partial_window_stays_buffered():
    engine = se.SpeechEngine(window_seconds=1)
    mem = np.zeros((4000, 1), dtype=np.float32)
    mem[:] = 0.5
    engine._on_audio(mem, 4000, None, None)
    assert engine._take(1.0) is None  # 0.25s is not a whole window yet
