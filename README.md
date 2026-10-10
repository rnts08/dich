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
| `backend.py` | text extraction, prompt strategies, cache, engine, metrics |
| `speech_engine.py` | mic capture + Whisper Vietnamese → completed sentences |
| `config.json` | model, languages, `fallback_models`, `load_model_on_startup` |
| `VERSION` | single source of truth for the version |
| `dev/bump_version.py` | bumps patch/minor, commits, and tags |
| `test_data/` | sample Vietnamese documents in every supported format (gitignored) |

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
