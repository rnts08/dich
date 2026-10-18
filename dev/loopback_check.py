#!/usr/bin/env python3
"""
Test the entire live microphone path digitally, with no person and no speaker.

The live mic is the one input that cannot be exercised from recorded files,
because the engine's capture, gate and windowing only run on a real input
device. A loopback driver makes them runnable anyway: with BlackHole
(``brew install blackhole-2ch``) installed, this script plays a real
recording into the driver's output while the app's *own* live engine — the
same filters, windowing, sentence assembly and transcription the Record
button uses — captures from the driver's input.

One full-duplex stream drives both sides: macOS (AUHAL) refuses two
half-duplex streams on the same device in one process, but a duplex stream
works, and its callback feeds `SpeechEngine._on_audio` — the exact callback
the real microphone InputStream uses, at the same 16 kHz.

If a real recording produces sentences here, the live path works end to end
and any remaining difference is at the physical microphone or its room. Run
``dev/mic_check.py`` for that diagnosis.

    .venv/bin/python3 dev/loopback_check.py test_data/speech/vi_150s.wav
    .venv/bin/python3 dev/loopback_check.py recording.wav --model small --window 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


from dich import backend  # noqa: E402
from dich import speech_engine as se  # noqa: E402


def find_loopback() -> int | None:
    """The device index of a BlackHole driver, resolved fresh every run."""
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if "blackhole" in d["name"].lower():
            return i
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path, help="an audio file to play in")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="loopback device index (default: auto-detect BlackHole)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="live speech model (default: the engine's GPU/CPU default)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help="live window seconds (default: the engine's default)",
    )
    parser.add_argument(
        "--quiet",
        type=float,
        default=6.0,
        help="seconds of tail silence after playback before stopping",
    )
    parser.add_argument(
        "--no-voice-gate",
        action="store_true",
        help="disable the voice-only filter (a Settings toggle). "
        "Real meeting recordings measure below the gate's "
        "voice-band bar; clean speech does not need this.",
    )
    parser.add_argument(
        "--archive",
        default=None,
        help="where to save the captured WAV (default: the app's archive)",
    )
    args = parser.parse_args()

    index = args.device
    if index is None:
        index = find_loopback()
        if index is None:
            print("No BlackHole device found. Install it with:")
            print("    brew install blackhole-2ch")
            print("…then re-run, or pass --device <device index>.")
            return 2
    print(f"Loopback device: {index}")

    audio = se.load_audio(str(args.audio))
    duration = len(audio) / se.SAMPLE_RATE
    if audio.size == 0:
        print(f"Could not decode {args.audio}")
        return 2
    print(f"Playing {args.audio.name} ({duration:.0f}s) through the loopback…")
    print()

    engine = se.SpeechEngine(
        model_size=args.model or se.default_live_model(),
        window_seconds=args.window
        if args.window is not None
        else se.default_window_seconds(),
        voice_only_filter=not args.no_voice_gate,
    )
    archive_dir = args.archive or str(backend.audio_archive_dir())
    engine.prepare(archive_dir=archive_dir)
    # The module-level drain/latest API reads this global; registering the
    # engine here is what makes drain_final_sentences() see our session.
    se._engine = engine
    print(
        f"Live engine: model={engine.model_size} window={engine.window_seconds}s "
        f"backend={se.stt_backend()} voice_gate={engine.voice_only_filter}"
    )

    # The duplex callback is the loopback itself: what goes into outdata comes
    # back on indata, and indata is fed to the engine's own capture callback.
    play_pos = 0

    def duplex(indata, outdata, frames, time_info, status):  # noqa: ANN001
        nonlocal play_pos
        engine._on_audio(indata, frames, time_info, status)
        chunk = audio[play_pos : play_pos + frames]
        outdata[: len(chunk), 0] = chunk
        if len(chunk) < frames:
            outdata[len(chunk) :, 0] = 0.0  # silence after the file ends
        play_pos += frames

    import sounddevice as sd

    sentences: list[str] = []
    first_at: float | None = None
    started = time.monotonic()
    with sd.Stream(
        device=index,
        samplerate=se.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=duplex,
        blocksize=int(se.SAMPLE_RATE * 0.5),
    ):
        deadline = started + duration + args.quiet
        try:
            while time.monotonic() < deadline:
                for sentence in se.drain_final_sentences():
                    now = time.monotonic() - started
                    if first_at is None:
                        first_at = now
                    print(f"\r{'':80}\r", end="")  # clear the partial line
                    print(f"  {now:6.1f}s  {sentence}")
                    sentences.append(sentence)
                if play_pos < len(audio):
                    partial = se.latest_partial()
                    if partial:
                        print(f"\r  …{partial[-48:]}", end="", flush=True)
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\nInterrupted.")
    archived = se.stop_mic_stream()  # flushes the trailing utterance too
    for sentence in se.drain_final_sentences():
        print(f"  final  {sentence}")
        sentences.append(sentence)

    print()
    print(f"Sentences: {len(sentences)}")
    if first_at is not None:
        print(
            f"First sentence arrived {first_at:.1f}s after playback started "
            f"(the recording itself is {duration:.0f}s)"
        )
    if archived:
        print(f"Captured audio archived to {archived}")
        print("Compare it with another model: Speech tab → Re-scan ▾")

    if not sentences:
        print(
            "\nThe live path produced nothing for real speech. That is a bug or a\n"
            "gate miscalibration — check dev/mic_check.py's verdict on the same\n"
            "file and file the numbers with it."
        )
        return 1
    print(
        "\nLive path OK: capture, gate, windowing and transcription all produced text."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
