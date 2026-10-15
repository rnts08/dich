# -*- mode: python -*-

"""
Build the standalone macOS .app: `pyinstaller packaging/dich.spec`

Prerequisites (installed once, offline after that):
    pip install pyinstaller

Then:
    cd /path/to/mlx-translation-tool
    source .venv/bin/activate        # or whatever venv holds the runtime deps
    pyinstaller packaging/dich.spec

Output lands in dist/dich/Dịch.app. Codesigning and notarization (Apple
Developer account) are manual steps — they cannot be done from this repo
without credentials, and are intentionally not part of any CI.
"""

block_cipher = None

a = Analysis(
    ["dich/__main__.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[
        # stdlib UI
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
        "tkinter.simpledialog",
        # runtime stack
        "mlx",
        "mlx.core",
        "mlx.nn",
        "mlx_lm",
        "mlx.audio",
        "mlx_audio",
        "sounddevice",
        "docx",
        "striprtf",
        "httpx",
        "av",
        "numpy",
        # the app itself
        "dich",
        "dich.backend",
        "dich.mlx_stt",
        "dich.speech_engine",
        "dich.theme",
        "dich.ui",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="dich",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

app = BUNDLE(
    exe,
    name="Dịch.app",
    icon=None,
    bundle_identifier=None,
    info_plist={
        "NSMicrophoneUsageDescription": "Dịch records from the microphone to "
        "transcribe Vietnamese speech into English.",
        "LSUIElement": False,
    },
)
