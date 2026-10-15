#!/usr/bin/env bash
# Build Dịch.app for macOS. Requires PyInstaller (pip install pyinstaller)
# and a Python environment with the runtime dependencies installed
# (mlx, mlx-lm, sounddevice, ...). Run from the repo root.
#
set -euo pipefail

cd "$(dirname "$0")/.."

if ! .venv/bin/python3 -c "import PyInstaller" 2>/dev/null; then
    echo "PyInstaller is not installed in .venv. Install it once:"
    echo "  .venv/bin/pip install pyinstaller"
    exit 1
fi

rm -rf build/dich dist/dich

echo "Building Dịch.app (this takes a few minutes)…"
.venv/bin/pyinstaller packaging/dich.spec \
    --distpath dist/dich \
    --workpath build/dich

echo
echo "Done: dist/dich/Dịch.app"
echo
echo "Notes:"
echo "  * Codesign with an Apple Developer identity before shipping:"
echo "      codesign --deep --force --sign \"<IDENTITY>\" dist/dich/Dịch.app"
echo "  * Notarize via Apple's notary service, then staple:"
echo "      xcrun notarytool submit ... --wait && xcrun stapler staple ..."
echo "  * These steps need an Apple Developer account and cannot be done from CI."
echo "  * First launch asks for microphone permission; grant it in"
echo "    System Settings → Privacy & Security → Microphone."
