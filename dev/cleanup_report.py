#!/usr/bin/env python3
"""
Report what the disfluency cleaner removes from real transcripts.

Run this locally over a folder of meeting transcripts (RTF or plain text) to see
the before/after for each file. It exists because the rules in ``dich/cleanup``
were tuned against real meeting-room audio, where the dominant artefact is not a
stutter but a Whisper *loop*; rerun this after changing a rule to check the
change against the real thing rather than a hunch.

    .venv/bin/python3 dev/cleanup_report.py real-world-data

RTF files are converted with macOS ``textutil`` (skipped if it is absent). A
transcript may hold the Vietnamese first and an English interpretation after a
line that is only an em dash; only the Vietnamese part is cleaned. Nothing here
is committed — the recordings stay local.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dich import cleanup as cl  # noqa: E402


def _read(path: Path) -> str | None:
    if path.suffix.lower() == ".rtf":
        try:
            return subprocess.run(
                ["textutil", "-convert", "txt", "-stdout", str(path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
    return path.read_text(encoding="utf-8", errors="replace")


def _vietnamese(text: str) -> str:
    """Keep only the part before a lone em-dash separator, if there is one."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip() == "—":
            return "\n".join(lines[:index])
    return text


def report(path: Path) -> None:
    text = _read(path)
    if text is None:
        print(f"{path.name}: skipped (could not read)")
        return
    raw = _vietnamese(text).strip()
    cleaned = cl.cleanup(raw)
    stats = cl.cleanup_stats(raw, cleaned)
    print(f"\n######## {path.name}")
    print(
        f"  words {stats['words_before']} -> {stats['words_after']}"
        f"  (removed {stats['removed']}, saved {stats['saved_chars']} chars)"
    )
    if not stats["changed"]:
        print("  nothing to clean")
        return
    diff = difflib.unified_diff(
        raw.splitlines(),
        cleaned.splitlines(),
        lineterm="",
        n=0,
    )
    for line in diff:
        if line.startswith(("---", "+++", "@@")):
            continue
        print(f"  {line}")


def main(argv: list[str]) -> int:
    target = Path(argv[1]) if len(argv) > 1 else ROOT / "real-world-data"
    if not target.is_dir():
        print(f"not a directory: {target}", file=sys.stderr)
        return 2
    files = sorted(p for p in target.iterdir() if p.suffix.lower() in {".txt", ".rtf"})
    if not files:
        print(f"no .txt/.rtf transcripts in {target}", file=sys.stderr)
        return 1
    for path in files:
        report(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
