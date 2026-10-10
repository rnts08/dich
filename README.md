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
| `config.json` | model, source/target language, `fallback_models` |
| `VERSION` | single source of truth for the version |
| `dev/bump_version.py` | bumps patch/minor, commits, and tags |
| `test_data/` | sample Vietnamese documents in every supported format (gitignored) |

## Versioning and commits

`VERSION` holds the current version and `backend.VERSION` reads it, so the UI
header and the file can never disagree.

The version policy:

- **0.1.x** — document translation (open a file, get a translation)
- **0.2.0 onwards** — speech-to-text in development; patch bumps continue

Bump a patch for every fix, feature, or change; bump a minor (0.1.x → 0.2.0) when
starting speech-to-text.

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
