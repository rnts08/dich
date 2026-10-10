#!/usr/bin/env python3
"""
Bump the patch version and record it in git.

Policy (set by the project owner):
  - 0.1.x  document translation (text -> translated document)
  - 0.2.0  speech-to-text in development; patch bumps continue from there

Usage:
  ./dev/bump_version.py            # bump patch (0.1.1 -> 0.1.2)
  ./dev/bump_version.py minor      # bump minor (0.1.2 -> 0.2.0)
  ./dev/bump_version.py --dry      # show the result without changing anything

The script updates VERSION, commits it, and tags v<version>. Run it after the
real changes are committed, or with --no-commit to update VERSION only.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = ROOT / "VERSION"
SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_version() -> tuple[int, int, int]:
    text = VERSION_FILE.read_text(encoding="utf-8").strip()
    match = SEMVER.match(text)
    if not match:
        sys.exit(f"VERSION file must contain MAJOR.MINOR.PATCH, got {text!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def bump(part: str) -> tuple[int, int, int]:
    major, minor, patch = read_version()
    if part == "major":
        return major + 1, 0, 0
    if part == "minor":
        return major, minor + 1, 0
    return major, minor, patch + 1


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", nargs="?", default="patch",
                        choices=["patch", "minor", "major"])
    parser.add_argument("--dry", action="store_true",
                        help="show the next version and exit")
    parser.add_argument("--no-commit", action="store_true",
                        help="update VERSION but do not commit or tag")
    parser.add_argument("--message", "-m", default="",
                        help="commit message; defaults to the version bump")
    args = parser.parse_args()

    version = bump(args.part)
    new = f"{version[0]}.{version[1]}.{version[2]}"
    old = ".".join(str(p) for p in read_version())

    if args.dry:
        print(new)
        return 0

    if git("status", "--porcelain") and not args.no_commit:
        # Uncommitted work must land in its own commit first, so the tag points
        # at a commit that actually contains the change it describes.
        sys.exit(
            "Working tree is dirty. Commit your changes first so the version "
            "tag points at the commit it describes."
        )

    VERSION_FILE.write_text(f"{new}\n", encoding="utf-8")
    print(f"{old} -> {new}")

    if args.no_commit:
        return 0

    message = args.message or f"chore: release {new}"
    git("add", str(VERSION_FILE))
    git("commit", "-m", message)
    git("tag", "-a", f"v{new}", "-m", f"Release {new}")
    print(f"committed and tagged v{new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
