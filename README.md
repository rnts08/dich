# Dịch

Vietnamese → English translator, running a local MLX model.

## Running

```bash
.venv/bin/python3 -m dich
```

The model loads in-process and stays resident, so there is no separate server
process to start. By default it loads when the app starts; a Settings checkbox
switches that to load-on-first-translation instead.

## Testing

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python3 -m pytest                 # full suite, incl. the model goldens
.venv/bin/python3 -m pytest -m "not model"  # fast subset — what the hook runs
.venv/bin/ruff check .                      # lint
.venv/bin/ruff format .                     # format
```

Tests sandbox their config (`DICH_HOME` points at a temp dir), so the real
`~/.config/dich` is never touched. The `model`-marked golden files pin fixed
translations and skip when the weights are not cached; after an *intended*
output change, regenerate them and review the diff — that review is the gate:

```bash
.venv/bin/python3 dev/regen_goldens.py
```

Quality checks run locally: a pre-commit hook runs ruff (lint + format check)
and the fast test suite on every commit. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

CI is deliberately disabled while the app is being finished; the workflow
comes back from git history when the app ships.

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
The bundled `dich/config.json` seed is copied across on first run so a
configured checkout keeps its settings.

Both were verified to load with the network disabled, so a second run does not
re-download. Set `DICH_HOME` to relocate them — useful for isolating a second
copy of the app.

To free space, delete a directory under `models/` — it re-downloads the next
time that model is asked for.

## Files

| Path | Purpose |
|---|---|
| `dich/__main__.py` | entry point — `python -m dich` starts the app |
| `dich/ui.py` | Tkinter app: Translate / Speech / Settings tabs |
| `dich/backend.py` | text extraction, prompt strategies, cache, engine, contextual translation |
| `dich/speech_engine.py` | mic + file audio, Whisper Vietnamese → completed sentences |
| `dich/voice_filter.py` | voice-only gate that rejects non-speech mic windows |
| `dich/audio_filter.py` | speech-band front-end: drops hum/rumble/hiss before the gate |
| `dich/cleanup.py` | rule-based removal of stutters, repeats, loops and filler sounds |
| `dich/llm.py` | optional OpenAI-compatible rewrite endpoint |
| `dich/theme.py` | palette and shared widget styles |
| `dich/VERSION` | single source of truth for the version |
| `dich/config.json` | first-run seed; the live copy lives at `~/.config/dich` |
| `dev/bump_version.py` | bumps patch/minor, commits, and tags |
| `dev/regen_goldens.py` | regenerates the golden translation files |
| `dev/cleanup_report.py` | before/after of the cleaner over a folder of real transcripts |
| `ruff.toml` | lint + format rules (ruff) |
| `.githooks/pre-commit` | local gate: ruff check, format check, fast test suite |
| `tests/` | pytest suite: unit, Tk, and `model`-marked golden tests |
| `test_data/` | sample documents and a Vietnamese speech recording (gitignored) |

## Translate tab

Open a file (**File → Open Document…**, or `Cmd+O`) or paste text into the left
pane; the translation appears on the right.

| Format | Extracted by |
|---|---|
| PDF | `pypdf` |
| DOCX / DOCM / DOC | `python-docx`, plus macOS `textutil` for legacy `.doc` |
| RTF | `striprtf` |
| Markdown | read as source text |
| TXT | direct |

Supported extensions: `txt md markdown pdf docx docm doc rtf`.

Long documents are split on paragraph boundaries into batches of at most 24
paragraphs and 8000 characters, translated in turn, and re-joined — a context
limit avoids truncation. The status bar shows a determinate progress bar across
batches plus words and tokens/sec.

The translation renders as **Markdown** — headings, bold, italic, code, lists,
quotes — and the model is asked to emit Markdown, so a translated document keeps
its structure rather than arriving as flat text. The raw Markdown is kept
alongside so **Save…** writes the source rather than the decorated copy.

Every finished translation is cached under `/tmp/dich-cache`, keyed on the model,
both languages and the exact text. Re-opening an unchanged document returns
instantly and shows a green `loaded from cache` badge; editing the text changes
the key and re-translates.

**Settings → Debug** holds a Google Translate reference for checking output
against an independent translator, including the source text currently loaded.

A native **menu bar** (File / Edit / View / Help) mirrors the toolbar buttons —
each dispatches to the same handler, so there is one path per action.

## Packaging

The app runs from source with `python -m dich`. A double-clickable macOS app
can be bundled with PyInstaller (no CI; codesigning/notarization need an Apple
Developer account and are manual):

    .venv/bin/pip install pyinstaller
    .venv/bin/pyinstaller packaging/dich.spec

See `packaging/build_app.sh` and `packaging/PACKAGING.md` for the full steps,
first-run notes, microphone permission, and model management.

## Speech tab

Two ways in:

- **Record** — listens to the microphone, transcribes Vietnamese, and translates
  as it goes.
- **Open Audio…** — transcribes an audio file (`mp3`, `wav`, `m4a`, …). This uses
  `large-v3-turbo`, which is far more accurate in Vietnamese than `small` but
  runs at about 3.6× realtime on the CPU, so a 38-minute recording takes roughly
  14 minutes there. The live mic uses `small` on the CPU, and
  `large-v3-turbo` on the GPU, where it keeps up easily (see below).

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

A file job shows its progress in two halves: decoding, then transcribing
(`transcribe 42%`), then translating sentence by sentence (`translating 3/22`), so
a long recording is never a blank bar. **Save…** writes the Vietnamese transcript
and the English translation together — labelled sections in `.txt`, or a
side-by-side table in `.md` — so the source can be checked next to the output.

### Cleaning up disfluencies

Spoken Vietnamese arrives full of the mechanics of talking: hesitation sounds
("ừm", "ờ"), immediate stutters ("tôi tôi"), repeated words and false starts.
Whisper transcribes them faithfully, and they then get translated faithfully.

**Clean up** strips them with plain rules — no model, no network, so it is
instant and works offline. It removes whole-word hesitations, isolated
interjections wedged between pauses, and repetitions at two levels: an immediate
run of words inside a line, and a whole line that repeats the one above it. The
second is the one that matters most on real meeting audio, where Whisper
sometimes gets stuck and emits the same short phrase over and over, one segment
per line — a lecture recording here produced "Tuy nhiên là" twenty times in a
row. Vietnamese reduplication that is a real word ("người người", "ngày ngày",
"từ từ", "chung chung") is always left alone, as is a repeat held apart by a
comma ("cảm ơn, cảm ơn") until it happens a third time. The cleaned transcript is
then re-translated, so both panes reflect it. **Settings → Microphone → “Remove
stutters, repeated words and filler sounds”** does the same automatically as
utterances arrive.

The one thing rules cannot resolve is a false start that only makes sense with
the sentence around it; the external-LLM rewrite below handles that.

`dev/cleanup_report.py` runs the cleaner over a folder of real transcripts and
shows the before/after per file, so a rule change can be checked against real
recordings instead of a hunch:

```bash
.venv/bin/python3 dev/cleanup_report.py real-world-data
```

### Rewriting with an external LLM

For a stronger, meaning-aware rewrite, the app can send the Vietnamese transcript
or the English translation to any **OpenAI-compatible chat endpoint** — OpenAI,
Ollama, LM Studio, llama.cpp's server, vLLM, OpenRouter and so on. This is off by
default and never happens unless it is turned on and a rewrite is invoked.

Set it up in **Settings → External LLM**: enable the checkbox, then fill in a
**base URL** (`https://api.openai.com/v1`, `http://localhost:11434/v1` for
Ollama, `http://localhost:1234/v1` for LM Studio), the **model** name, and an
**API key** (left blank for a local server). **Test connection** proves it
answers. An optional **Instruction** overrides the built-in prompt.

On the Speech tab, **Rewrite ▾** offers the two targets — *transcript
(Vietnamese)* keeps Vietnamese, *translation (English)* keeps English — and the
same actions live in the Edit menu. The result opens in a window to review before
replacing the pane; a transcript rewrite is re-translated. The built-in prompt
asks the model to preserve every detail and not lose context.

> The API key is stored in plain text in `~/.config/dich/config.json`. Leave the
> feature off to keep everything local.

## Microphone settings

In Settings → Microphone:

| Setting | Meaning |
|---|---|
| **Input device** | Which microphone. *System default* follows macOS. |
| **Speech model** | `tiny` → `large-v3-turbo`. Only runs while recording. |
| **Voice-only filter** | Reject windows that are not speech (fans, keyboards, room tone) before the model sees them. |
| **Speech-band filter** | Keep only the frequencies a voice uses (~80 Hz–7 kHz), so hum, rumble and hiss are removed before the gate and the model. |
| **Clean up speech** | Remove stutters, repeated words and filler sounds automatically before translating. |
| **Refine every** | Utterances between block re-translations. |
| **Transcription window** | Seconds of audio per transcription. |
| **Buffer ceiling** | Seconds of audio held before the oldest is dropped. |

The speech model defaults by backend, measured for Vietnamese on an M5:

| Model | Realtime | Accuracy |
|---|---|---|
| `tiny` | 16× | very rough |
| `base` | 8× | rough |
| `small` | ~2× at 2s windows | keeps up (CPU default) |
| `medium` | 0.49× at 2s / 2.2× at 10s | better proper nouns |
| `large-v3-turbo` | 3.6× on CPU, 31× on GPU | best (GPU default) |

On the CPU the default is `small`, the only model that keeps up at a 2-second
window. On the GPU (`mlx-audio`) the default is `large-v3-turbo`: it decodes a
5-second window in ~160 ms there, so the accuracy/latency tradeoff that forced
`small` is gone, and the accurate model is what makes unfamiliar voices and
proper nouns come through. Picking a model in Settings overrides both defaults.

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

A window that produces text fast does not have to keep up — the buffer holds
the audio and it catches up during a pause. That is what the buffer ceiling is
for: raise it to let a slower model catch up, or to hold a long recording for
re-transcribing. `small` at 2s keeps up (1.3x), so the countdown stays short.

The default window follows the backend. On the CPU it stays at 2s, because a
longer window there would put the transcriber behind the speaker. On the GPU
(`mlx-audio`) it is **5s**: the same window costs about 160 ms on
`large-v3-turbo`, and the extra context is measurably more accurate than 2s.
Setting **Transcription window** in Settings overrides either default.

Resolution shows the real sample rate, so a microphone that cannot do 16 kHz is
visible before it fails. **Rescan** picks up mics plugged in or removed later,
and keeps an unsaved selection. If a configured device disappears — unplugged,
or device order changed — the engine falls back to the system default and logs
why, rather than dying with a PortAudio error.

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
without limit — which is what this readout is for. `large-v3-turbo` is far slower
than realtime and is exactly that case, so expect it to climb.

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

A **speech-band front-end** (`dich/audio_filter.py`) runs even earlier: every
microphone window is passed through a zero-phase 80 Hz–7 kHz Butterworth
band-pass, so the mains hum, traffic rumble and top-end hiss that a real
microphone picks up are gone before the gate judges the window and before the
model reads it. The low edge earns its keep — it both un-buries quiet speech and
stops low-frequency noise from pulling the gate's voice-band share down — while
the high edge only trims the last kilohertz of hiss and breath noise.

A **voice-only gate** (`dich/voice_filter.py`) then runs ahead of the model on the
microphone. A window must look like a voice — most of its energy in the 300–3400
Hz band, a low spectral flatness, a plausible zero-crossing rate, and a level
above the room's own learned noise floor — or it is never transcribed. This is
what stops a fan, a keyboard or plain room tone from becoming text, and it is the
only such defence on the GPU path, where faster-whisper's `vad_filter` does not
exist. The noise floor is measured on the voice band rather than the whole
signal, so a low rumble below speech cannot raise the bar a quiet voice has to
clear. A diacritic-free run of English function words is dropped as hallucinated
English as well; written Vietnamese is diacritic-heavy, so real speech is not
mistaken for it.

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

`ROADMAP.md` tracks the longer-horizon milestones; `BACKLOG.md` is the
prioritised list of what to pick up next.

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
pip install mlx-lm httpx pypdf python-docx striprtf
# for speech only:
pip install sounddevice faster-whisper av numpy
pip install mlx-audio   # optional: runs the same Whisper weights on the GPU
```

`tkinter` ships with Python. `PyPDF2` is accepted as a fallback if `pypdf` is
absent.

## Notes

Speech-to-text runs on the **GPU via `mlx-audio`** when it is installed and a
Metal GPU is reachable, and on the CPU via `faster-whisper` otherwise. Select
with `"stt_backend"` in the config: `auto` (default), `mlx`, or `ctranslate2`.

The one trap is the model repo. `mlx-community/whisper-large-v3-turbo` ships
weights only, so mlx-audio cannot build a `WhisperProcessor` and `get_tokenizer`
raises `Processor not found. Make sure the model was loaded with a HuggingFace
processor`. The `-asr-fp16` repos carry `preprocessor_config.json` and the
tokenizer too, so model ids map to `mlx-community/whisper-<id>-asr-fp16`.

Measured here (same Vietnamese audio, warm):

| | CPU (faster-whisper int8) | GPU (mlx-audio fp16) |
|---|---|---|
| large-v3-turbo, 150 s file | 3.1× realtime | **31× realtime** |
| live 2 s window | small: 1009 ms | turbo: 354 ms |
| peak RAM | — | turbo 2.2 GB, Qwen3-ASR 2.7 GB |

That is what removes the accuracy-versus-latency tradeoff: the accurate model is
now also fast enough to run live. Silence is still never sent to the model —
`SILENCE_FLOOR` in `speech_engine` guards every window, because Whisper on any
backend invents text (on this machine: gibberish for the CPU path, short phrases
like "ở đâu?" for the GPU path) when handed a quiet buffer.

---
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/E3J2203KUB)

