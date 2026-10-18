Installing Dịch from source
===========================

This is the reproducible setup from a clean Apple Silicon Mac, ending with
`python -m dich` running. (`PACKAGING.md` covers bundling a double-clickable
`.app`, which is a separate, deferred step.) Every command runs in Terminal.

The fastest path is the bundled script, which does steps 1–4 and 5's launch:

    ./install.sh install     # venv + all dependencies + pre-commit gate
    ./install.sh run         # launch the app
    ./install.sh status      # what is installed, tkinter, hooks, model cache

The steps below are what it does, for when you want them one at a time.

1. Install Python 3.11 or newer
--------------------------------

Either distribution works; pick one.

**Homebrew:**

    brew install python

**python.org installer:** download the macOS arm64 universal2 installer from
<https://www.python.org/downloads/> and run it. It includes Tk, which the UI
needs.

Verify both Python and Tk are present:

    python3 --version
    python3 -c "import tkinter; print('tk ok')"

If the second command fails on Homebrew Python, install Tk too:

    brew install python-tk

2. Get the source
-----------------

    git clone https://github.com/rnts08/dich.git
    cd dich

3. Create a virtualenv and install dependencies
-----------------------------------------------

    python3 -m venv .venv
    .venv/bin/pip install mlx-lm httpx pypdf python-docx striprtf
    # for the Speech tab:
    .venv/bin/pip install sounddevice faster-whisper av numpy
    # optional — runs Whisper on the Metal GPU instead of the CPU:
    .venv/bin/pip install mlx-audio

`mlx-audio` is strongly recommended on Apple Silicon: on the GPU,
`large-v3-turbo` runs at 31× realtime where the CPU path manages 3.6×, and
the GPU is the default when it is installed and reachable.

4. Enable the pre-commit gate (contributing)
--------------------------------------------

The local gate runs ruff and the fast test suite on every commit:

    git config core.hooksPath .githooks
    .venv/bin/pip install -r requirements-dev.txt

Verify the suite passes on your machine:

    .venv/bin/python3 -m pytest -m "not model"   # fast subset
    .venv/bin/python3 -m pytest                  # + model goldens (needs the
                                                 #   translation weights cached)

5. First run
------------

    .venv/bin/python3 -m dich

What happens on first use:

- The very first translation downloads the translation model — roughly 4 GB
  for `translategemma-4b-it-4bit`. The status bar shows a blocking progress
  bar while it loads; after that the model stays resident.
- The first transcription downloads the speech model (the GPU default,
  `large-v3-turbo`, is about 1.6 GB).
- Recording asks macOS for microphone permission. Grant it in
  **System Settings → Privacy & Security → Microphone** — the app reports
  "microphone unavailable" until it is granted.

6. Where everything lives
-------------------------

All app state sits under `~/.config/dich`:

    ~/.config/dich/
    ├── config.json    model, languages, microphone, refine cadence
    └── models/        every downloaded model — delete a subdirectory to
                       free space; it re-downloads when next needed

Useful environment variables:

- `DICH_HOME` relocates the whole directory (useful for a second copy of the
  app); the tests already isolate themselves this way.
- `DICH_LOG_LEVEL=INFO` restores the startup output that is silenced by
  default.

Nothing is sent anywhere: translation runs in-process on MLX, transcription
runs locally, and the optional external-LLM rewrite is off until you turn it
on in Settings.

7. Check the microphone before a real session
---------------------------------------------

Recorded audio fed through speakers is the tested path; a live human speaker
is the one untested input. Before a real session, run the rehearsal script —
it captures your microphone, prints the level and the voice-gate verdict per
window, and says which link in the chain (level, gate, model) is failing:

    .venv/bin/python3 dev/mic_check.py --list        # find your device index
    .venv/bin/python3 dev/mic_check.py --seconds 12  # read the printed script

With BlackHole (`brew install blackhole-2ch`) you can test the entire live
path digitally, without a speaker or a person, by looping a real recording
into the input:

    .venv/bin/python3 dev/loopback_check.py test_data/speech/vi_150s.wav

8. Troubleshooting
------------------

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'mlx'` | You are not in the venv — use `.venv/bin/python3 -m dich`. |
| Tkinter import error | `brew install python-tk`, or use the python.org installer. |
| "microphone unavailable" | Grant microphone permission in System Settings, then restart the app. |
| Recording hears nothing | Run `dev/mic_check.py` — it names the failing link (level, gate or device). |
| Slow transcription | Install `mlx-audio`; without it everything runs on the CPU. |
| Downloads land somewhere odd | `HF_HUB_CACHE` points into `DICH_HOME` by default; unset overrides in your shell profile. |
| A model re-downloads | Its directory under `~/.config/dich/models` was deleted — that is the documented way to free space. |
