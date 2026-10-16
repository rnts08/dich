#!/usr/bin/env python3
"""
Practice the whole speech path on real recordings: messy Vietnamese in, clean
contextual English out.

This runs a transcript through the *app's own* pipeline — the rule-based cleanup
followed by the contextual translator — and prints the Vietnamese it cleaned, the
English it produced and the human reference for comparison. It is how the cleanup
rules get exercised against real meeting-room speech instead of a hunch: if the
cleaned text does not read as coherent Vietnamese, the translation that follows
cannot be good either.

Run it locally (it needs the model):

    HF_HUB_OFFLINE=1 .venv/bin/python3 dev/pipeline_report.py real-world-data
    HF_HUB_OFFLINE=1 .venv/bin/python3 dev/pipeline_report.py real-world-data \\
        --file sala-v3.rtf --limit 20

RTF files are converted with macOS ``textutil``. Transcripts may hold the
Vietnamese first and an English interpretation after a line that is only an em
dash. Nothing here is committed — the recordings stay local.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dich import backend  # noqa: E402
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


def _split(text: str) -> tuple[str, str]:
    """Vietnamese before a lone em dash, the human reference after it."""
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.strip() == "—":
            return "\n".join(lines[:index]), "\n".join(lines[index + 1 :])
    return text, ""


def _utterances(text: str) -> list[str]:
    return [line.strip() for line in text.split("\n") if line.strip()]


def report(path: Path, limit: int | None, translate: bool) -> None:
    raw = _read(path)
    if raw is None:
        print(f"{path.name}: skipped (could not read)")
        return
    vietnamese, reference = _split(raw)
    source = _utterances(vietnamese)
    if limit:
        source = source[:limit]

    # Clean the whole block, not line by line: the loop collapse works across
    # lines, so a per-line pass would miss exactly the artefact that matters.
    cleaned_block = cl.cleanup("\n".join(source))
    cleaned = _utterances(cleaned_block)
    stats = cl.cleanup_stats("\n".join(source), cleaned_block)

    print(f"\n################ {path.name}")
    print(
        f"  input {len(source)} utterances, {stats['words_before']} words"
        f"  ->  cleaned {len(cleaned)}, {stats['words_after']} words"
        f"  (removed {stats['removed']})"
    )
    print("\n---- cleaned Vietnamese ----")
    for line in cleaned:
        print(f"  {line}")

    if not translate:
        return

    translator = backend.ContextualTranslator(clean_disfluencies=True)
    translator.replace(cleaned)
    print("\n---- app translation ----")
    for line in translator.render_refined().split("\n"):
        print(f"  {line}")
    print(f"  {translator.stats()}")

    if reference.strip():
        print("\n---- human reference ----")
        for line in _utterances(reference):
            print(f"  {line}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=str(ROOT / "real-world-data"),
        help="folder of .txt/.rtf transcripts (default: real-world-data)",
    )
    parser.add_argument("--file", help="only this file name")
    parser.add_argument("--limit", type=int, help="translate only the first N lines")
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="clean only; do not load the model",
    )
    args = parser.parse_args(argv[1:])

    target = Path(args.directory)
    if not target.is_dir():
        print(f"not a directory: {target}", file=sys.stderr)
        return 2
    files = sorted(p for p in target.iterdir() if p.suffix.lower() in {".txt", ".rtf"})
    if args.file:
        files = [p for p in files if p.name == args.file]
    if not files:
        print(f"no matching transcripts in {target}", file=sys.stderr)
        return 1

    for path in files:
        report(path, args.limit, translate=not args.no_translate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
