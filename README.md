# Dịch

Vietnamese → English translator, running a local MLX model.

## Running

```bash
.venv/bin/python3 ui.py
```

The model loads in-process on the first translation and stays resident, so there
is no separate server process to start.

## Files

| Path | Purpose |
|---|---|
| `ui.py` | Tkinter app: Translate / Speech / Settings tabs |
| `backend.py` | text extraction, prompt strategies, cache, engine, contextual translation |
| `speech_engine.py` | mic + file audio, Whisper Vietnamese → completed sentences |
| `config.json` | model, languages, `fallback_models`, `load_model_on_startup` |
| `VERSION` | single source of truth for the version |
| `dev/bump_version.py` | bumps patch/minor, commits, and tags |
| `test_data/` | sample documents and a Vietnamese speech recording (gitignored) |

## Speech tab

Two ways in:

**Record** — listens to the microphone, transcribes Vietnamese, and translates
  as it goes.
- **Open Audio…** — transcribes an audio file (`mp3`, `wav`, `m4a`, …). This uses
  `large-v3-turbo`, which is far more accurate in Vietnamese than the live model
  but runs at about 3.6× realtime, so a 38-minute recording takes roughly 14
  minutes. The live mic uses `small` (~16× realtime) so it can keep up.

The translation is **not** fixed on first pass. Each utterance is translated with
the preceding ones supplied as conversation turns, so honorifics and pronouns
resolve from the first line. Every six utterances the whole block is re-translated
as one passage and replaces the provisional line-by-line output — so as more
speech is understood, earlier translations get better. `Refine` forces this
immediately. The status bar shows `N refined · M provisional`.

Measured on a 150s Vietnamese sample (22 utterances): the first pass produced
613 characters of fragment-by-fragment English; after the first block refinement
it became 499 characters of coherent prose. The output gets *better*, not just
longer.

## Microphone settings

In Settings → Microphone:

| Setting | Meaning |
|---|---|
| **Input device** | Which microphone. *System default* follows macOS. |
| **Speech model** | `tiny` → `large-v3-turbo`. Only runs while recording. |
| **Refine every** | Utterances between block re-translations. |

Resolution shows the actual sample rate, so a device that cannot do 16 kHz is
visible before it fails. **Rescan** picks up mics plugged in or removed later,
and keeps an unsaved selection.

If a configured device disappears — unplugged, or device order changed — the
engine falls back to the system default and logs why, rather than dying with a
PortAudio error.

Recording opens the microphone off the UI thread, because `sd.InputStream()`
blocks until macOS decides whether to grant microphone permission. In a context
where the permission prompt cannot appear, that open can block indefinitely, and
on the UI thread it would freeze the window instead. The button is disabled while
opening so a double-click cannot open two streams.

## Versioning and commits

`VERSION` holds the current version and `backend.VERSION` reads it, so the UI
header and the file can never disagree.

The version policy:

- **0.1.x** — document translation (open a file, get a translation)
- **0.2.0** — speech-to-text in development
- **0.3.x** — packaging into a stand-alone app
- **0.4.x** — translation features
- **0.5.x** — tests and release

`ROADMAP.md` tracks the remaining work against those milestones.

Bump a patch for every fix, feature, or change; bump a minor when starting a new
milestone (for example 0.1.x → 0.2.0 when speech-to-text begins).

```bash
# make your change, then:
git add -A
git commit -m "fix(pdf): keep line breaks across page boundaries"

# bump, tag, and record the release:
.venv/bin/python3 dev/bump_version.py            # patch
.venv/bin/python3 dev/bump_version.py minor      # minor
.venv/bin/python3 dev/bump_version.py --dry      # preview only
.venv/bin/python3 dev/bump_version.py --no-commit  # update VERSION, stop there
```

`bump_version.py` tags `v<VERSION>` and **refuses to run on a dirty tree**, so a
version tag always points at a commit that actually contains the change it
describes. Commit your work first, then bump.

## Model loading

By default the model loads as soon as the UI starts, so the status dot is orange
"loading…" for a few seconds and then green "ready" — without translating
anything first. Turn off **Settings → Load the model when the app starts** to
load on the first translation instead; the status stays red until then.

The status dot means:

| Colour | Text | Meaning |
|---|---|---|
| orange | `loading…` | model is being loaded |
| green | `ready` | model resident, can translate |
| red | `not loaded` | no model resident yet |

## Commit messages

Conventional Commits: `feat`, `fix`, `refactor`, `perf`, `docs`, `test`,
`chore`. Subject in the imperative mood, ≤50 characters where possible, no
trailing period. Add a body only when the *why* is not obvious from the diff —
especially for bugs whose root cause is subtle, since that is the context a
future debugger needs.

## Dependencies

```bash
pip install mlx-lm pypdf python-docx striprtf markdown
```
