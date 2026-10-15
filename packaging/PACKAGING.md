Packaging notes for Dịch (Vietnamese → English translator).
=============================================================

Standalone macOS app
--------------------

The project can be bundled into a double-clickable `Dịch.app` with PyInstaller:

    .venv/bin/pip install pyinstaller
    .venv/bin/pyinstaller packaging/dich.spec

The result is `dist/dich/Dịch.app`. See `packaging/build_app.sh` for the exact
invocation and the codesign/notarize steps, which require an Apple Developer
account and are intentionally manual — there is no CI in this project and
none will be added.

First run
---------

1. `python -m dich` (from the virtualenv described in README.md).
2. The very first translation downloads and loads the model (~4 GB), so it
   takes a while; after that it stays resident.
3. Grant microphone permission in System Settings → Privacy & Security →
   Microphone before recording.
4. Models live under `~/.config/dich/models` (set by `HF_HUB_CACHE`) and are
   reused between runs.

Microphone permission is required for the speech tab; without it the app
reports "microphone unavailable" and suggests the System Settings pane.

Model management
----------------

Settings shows every model already on disk with its size and an estimate of
the RAM it needs. Switching to a model that is not cached asks for confirmation
before it downloads several gigabytes. "Unload models" frees RAM at any time,
and models idle for `idle_unload_minutes` (default 15) are unloaded
automatically.

Glossary
--------

Proper nouns can be pinned to a preferred translation in `config.json`:

    "glossary": {
      "Hà Nội": "Hanoi",
      "Hồ Chí Minh": "Ho Chi Minh City"
    }

The directive is added to the translation prompt; when empty, output is
byte-identical to the un-glossaried app.

Window size and position, the last eight opened files, and the model choice
are remembered between runs. Quitting unloads the models and hands their RAM
back.
