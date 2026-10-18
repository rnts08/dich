#!/usr/bin/env python3
"""
Diagnose the live microphone path, link by link.

The one untested input in the app is a human speaking into the microphone;
recorded audio fed through speakers has always worked. When the live mic
produces nothing, this script names the failing link instead of leaving a
hunch:

  1. the device captures (almost) nothing          -> level problem
  2. audio arrives but the voice gate rejects it   -> gate/noise problem
  3. windows pass the gate                          -> the failure is further
                                                       down (model/backend)

It records straight from the chosen device, then replays the *app's own*
chain over the captured audio: the speech-band filter, then the VoiceGate,
on the same window size the live engine would use. Optionally (``--transcribe``)
it also runs the model over the windows that pass, so the output can be read
next to what was actually said.

Run it and read the printed script aloud while it records:

    .venv/bin/python3 dev/mic_check.py --list
    .venv/bin/python3 dev/mic_check.py --seconds 12
    .venv/bin/python3 dev/mic_check.py --seconds 12 --transcribe
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dich import speech_engine as se  # noqa: E402
from dich.audio_filter import band_pass  # noqa: E402
from dich.voice_filter import VoiceGate, spectral_flatness, voice_band_ratio  # noqa: E402

# A few short lines that cover the sounds the gate and the model must handle:
# breathy starts, Vietnamese tones, a proper noun, a question.
SCRIPT = [
    "Xin chào, đây là một buổi kiểm tra microphone.",
    "Tôi đang đọc chậm và rõ ràng để kiểm tra âm lượng.",
    "Hà Nội là thủ đô của Việt Nam.",
    "Cuộc họp bắt đầu lúc ba giờ chiều, đúng không?",
]


def dbfs(samples: np.ndarray) -> float:
    """Level in dBFS; -120 for digital silence."""
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    return 20.0 * np.log10(max(peak, 1e-6))


def list_devices() -> None:
    print("Input devices:")
    for index, name in se.list_input_devices():
        marker = ""
        if index == se.default_input_device():
            marker = "   <- system default"
        print(f"  {index:3d}  {name}{marker}")


def record_with_meter(device_index: int | None, seconds: float) -> np.ndarray:
    """Capture while printing a live level readout, like the app's meter."""
    import sounddevice as sd

    chunks: list[np.ndarray] = []
    peak = 0.0

    def callback(indata, _frames, _time_info, _status):  # noqa: ANN001
        nonlocal peak
        mono = np.ascontiguousarray(indata[:, 0])
        chunks.append(mono.copy())
        peak = max(peak, float(np.abs(mono).max()))
        bar = "#" * int(min(1.0, peak) * 40)
        print(
            f"\r  level {dbfs(mono):7.1f} dBFS |{bar:<40}| peak {peak:.4f}",
            end="",
            flush=True,
        )

    with sd.InputStream(
        samplerate=se.SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_index,
        callback=callback,
        blocksize=int(se.SAMPLE_RATE * 0.5),
    ):
        time.sleep(seconds)
    print()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def analyse(audio: np.ndarray, window_seconds: float, gate: VoiceGate) -> list[dict]:
    """Run the app's chain (speech-band filter, then gate) over each window."""
    window = int(window_seconds * se.SAMPLE_RATE)
    rows = []
    for start in range(0, len(audio) - window // 2, window):
        chunk = audio[start : start + window]
        if chunk.size < window // 2:
            break
        filtered = band_pass(chunk, se.SAMPLE_RATE) if chunk.size > 31 else chunk
        rows.append(
            {
                "t0": start / se.SAMPLE_RATE,
                "db": dbfs(chunk),
                "peak": float(np.abs(chunk).max()),
                "band": voice_band_ratio(filtered),
                "flat": spectral_flatness(filtered),
                "voice": gate.is_voice(filtered),
            }
        )
    return rows


def verdict(rows: list[dict], audio: np.ndarray) -> str:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    voiced = sum(1 for r in rows if r["voice"])
    lines = ["", "Verdict:"]
    if peak < se.SILENCE_FLOOR:
        lines.append(
            f"  x LEVEL — the whole capture peaked at {peak:.4f}, under the\n"
            f"    silence floor {se.SILENCE_FLOOR}. The engine would never\n"
            "    transcribe this. Check the input device, input volume and\n"
            "    System Settings -> Privacy & Security -> Microphone."
        )
    elif voiced == 0:
        lines.append(
            f"  x GATE — audio arrives (peak {peak:.4f}, above the floor) but\n"
            "    no window looks like a voice to the gate. Either the wrong\n"
            "    device was captured, or the sound is noise-shaped (fan,\n"
            "    music, distorted input). Compare a known-good recording:"
        )
        lines.append(
            "      .venv/bin/python3 dev/loopback_check.py test_data/speech/vi_150s.wav"
        )
    else:
        lines.append(
            f"  o CHAIN OK — {voiced}/{len(rows)} windows passed the gate at a\n"
            f"    healthy level (peak {peak:.4f}). If the app still shows\n"
            "    nothing live, the failure is in the model path; re-run with\n"
            "    --transcribe to see what the engine would emit."
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="list input devices")
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="input device index (default: the configured mic)",
    )
    parser.add_argument(
        "--seconds", type=float, default=12.0, help="how long to record (default: 12)"
    )
    parser.add_argument(
        "--window",
        type=float,
        default=None,
        help="analysis window seconds (default: the engine's)",
    )
    parser.add_argument(
        "--transcribe",
        action="store_true",
        help="also run the speech model over the passing windows",
    )
    args = parser.parse_args()

    if args.list:
        list_devices()
        return 0

    device = args.device
    if device is None:
        device = se._validate_device(se.default_input_device())
    window_seconds = args.window or se.default_window_seconds()

    name = next((n for i, n in se.list_input_devices() if i == device), "?")
    print(f"Device {device}: {name}")
    print(f"Window {window_seconds:.1f}s — the same size the live engine uses")
    print()
    print("Read this aloud, one line every ~3 seconds:")
    for line in SCRIPT:
        print(f"    {line}")
    print(f"\nRecording for {args.seconds:.0f}s — start speaking now…")

    gate = VoiceGate()
    audio = record_with_meter(device, args.seconds)
    if audio.size == 0:
        print("Nothing captured at all — the device refused to open or gave no data.")
        return 1

    rows = analyse(audio, window_seconds, gate)
    print()
    print(f"{'t(s)':>6}  {'dBFS':>7}  {'peak':>7}  {'band':>5}  {'flat':>5}  gate")
    for r in rows:
        print(
            f"{r['t0']:6.1f}  {r['db']:7.1f}  {r['peak']:7.4f}"
            f"  {r['band']:5.2f}  {r['flat']:5.2f}  {'VOICE' if r['voice'] else 'reject'}"
        )
    print(verdict(rows, audio))

    if args.transcribe:
        voiced_windows = [r for r in rows if r["voice"]]
        if not voiced_windows:
            print("\nNo windows passed the gate, so there is nothing to transcribe.")
            return 0
        model_name = se.default_live_model()
        print(
            f"\nTranscribing the {len(voiced_windows)} passing windows with "
            f"'{model_name}' (first use downloads the model)…"
        )
        model = se._whisper_model(model_name)
        window = int(window_seconds * se.SAMPLE_RATE)
        for r in voiced_windows:
            start = int(r["t0"] * se.SAMPLE_RATE)
            chunk = audio[start : start + window]
            segments, _info = model.transcribe(
                chunk, language="vi", beam_size=5, condition_on_previous_text=False
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            label = text if text else "(nothing)"
            print(f"  {r['t0']:6.1f}s  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
