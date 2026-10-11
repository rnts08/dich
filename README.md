# Dịch

Vietnamese → English translator, running a local MLX model.

## Running

```bash
.venv/bin/python3 ui.py
```

The model loads in-process on the first translation and stays resident, so there
is no separate server process to start.

## Where things live

Everything the app owns sits under `~/.config/dich`:

```
~/.config/dich/
├── config.json    model, languages, microphone, refine cadence
└── models/        every downloaded model
    ├── models--mlx-community--translategemma-4b-it-4bit/
    ├── models--mlx-community--Qwen3-4B-Instruct-2507-4bit/
    ├── models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/
    └── models--Systran--faster-whisper-{tiny,small,medium}/
```

The model cache is redirected there at import time via `HF_HUB_CACHE`, before
HuggingFace is loaded, because that library reads the variables once at import.
An existing `config.json` sitting beside the code is copied across on first run
so a configured install keeps its settings.

Both were verified to load with the network disabled, so a second run does not
re-download. Set `DICH_HOME` to relocate them, which is also how the tests
relocate them.

To free space, delete a directory under `models/` — it re-downloads the next
time that model is asked for.

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
  minutes. The live mic uses `small` by default, which keeps up with a speaker
  at a 2-second window.

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
| **Transcription window** | Seconds of audio per transcription. |
| **Buffer ceiling** | Seconds of audio held before the oldest is dropped. |

The speech model defaults to `small`, measured for Vietnamese on an M5:


| Model | Realtime | Accuracy |
|---|---|---|
| `tiny` | 16× | very rough |
| `base` | 8× | rough |
| `small` | ~2× at 2s windows | keeps up (default) |
| `medium` | 0.49× at 2s / 2.2× at 10s | better proper nouns |
| `large-v3-turbo` | slowest | best |
`small` is the default because it keeps up at a 2-second window; choose
`medium` or `large-v3-turbo` when a name being heard correctly matters more
than latency.

Realtime depends on the window as much as on the model, since every
`transcribe()` call carries fixed overhead:

| | 2s window | 10s window |
|---|---|---|
| `small` | ~2× | ~6× |
| `medium` | 0.49× | 2.2× |

### Latency and the window

The status bar counts down to the next transcript: `listening… · next transcript
in 1.3s`. With a 2-second window the first words appear **1.2s after speech
starts**; at the old 10-second window they took 10s, which read as the app doing
nothing at all.

A window that produces text fast does not have to keep up -- the buffer holds
the audio and it catches up during a pause. That is what the buffer ceiling is
for: raise it to let a slower model catch up, or to hold a long recording for
re-transcribing. `small` at 2s keeps up (1.3x), so the countdown stays short.
and keeps an unsaved selection.

If a configured device disappears — unplugged, or device order changed — the
engine falls back to the system default and logs why, rather than dying with a
PortAudio error.

### Buffer memory

The status bar shows how much audio is buffered while listening, as
`buffer 2.7 MB · 45.0s`, and the speech tab shows the same figure beside its
own status. Opening a file shows the buffer for that decode too.

It is worth watching because the capture buffer is the only unbounded thing in
the audio path: the device pushes audio in continuously while transcription
drains it one window at a time. A healthy pipeline produces a sawtooth, climbing
to about one window and then dropping when that window is transcribed:

```
t= 2.2s  buffer= 2.0s  125.0 KB
t= 8.3s  buffer= 8.5s  531.2 KB
t=10.4s  buffer= 0.5s   31.2 KB    <- 10s window drained
t=18.6s  buffer= 8.5s  531.2 KB
```

If the model cannot keep up, the sawtooth becomes a ramp and the number grows
without limit — which is what this readout is for. `large-v3-turbo` at 0.3×
realtime is exactly that case, so expect it to climb.

Recording opens the microphone off the UI thread, because `sd.InputStream()`
blocks until macOS decides whether to grant microphone permission. In a context
where the permission prompt cannot appear, that open can block indefinitely, and
on the UI thread it would freeze the window instead. The button is disabled while
opening so a double-click cannot open two streams.

### Whisper hallucination on silence

Whisper's known failure mode is to invent tidy boilerplate when given silence or
noise. Unhandled, feeding pure digital silence through this pipeline produced:

> "Hãy subscribe cho kênh La La School Để không bỏ lỡ những video hấp dẫn."

…repeated for the whole session. Three settings were the cause:

- **`initial_prompt` removed** — the Vietnamese hint was echoed back verbatim
- **`vad_filter=True`** — without it, near-silent windows are transcribed anyway
- **`condition_on_previous_text=False`** — when true, a hallucination feeds
  itself and the same phrase repeats indefinitely

Plus a peak-amplitude floor that skips quiet windows outright, and a marker
filter that strips boilerplate found *inside* real speech, because Whisper
splices it into genuine sentences rather than always emitting it separately.

The cost is that a genuinely spoken "cảm ơn các bạn" is dropped too. In this
domain that phrase is overwhelmingly an artefact, so the trade is deliberate.

## Licence

Non-commercial. © 2026 Tim Bergström. See [LICENCE](LICENCE) for the full terms;
the summary is in Settings → About, with a link to the project home at
<https://github.com/rnts08/dich>.

Third-party models and libraries invoked at runtime — mlx-lm, faster-whisper,
and whatever Hugging Face publishes — keep their own licences, which this
project does not grant. Several of those restrict commercial use too.

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

## Notes

`mlx-audio` 0.5.8 is installed on this machine as a `uv` tool. Its STT cannot
run `mlx-community/whisper-large-v3-turbo` -- it raises `Processor not found.
Make sure the model was loaded with a HuggingFace processor`, including from its
own CLI -- so it is not used. If it can be made to work it would move
transcription onto the GPU, which is the real answer to the latency tradeoff
above rather than trading accuracy for it.
