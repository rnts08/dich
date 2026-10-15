# Dịch — Backlog

Last reviewed: 2026-10-16 · version **0.2.40** · branch `main`

This is the working list of what is left, ordered by priority rather than by
milestone. `ROADMAP.md` keeps the longer-horizon shape; this file is what to pick
up next.

Priority key — **P0** blocks a release, **P1** needed for a usable stand-alone
app, **P2** real improvement, **P3** nice to have.

---

## Where it stands

Working today, verified against the code rather than assumed:

- **Text translation** — PDF, DOCX, DOCM, DOC, RTF, Markdown, plain text →
  English, batched for long documents, cached so unchanged input is instant.
- **Speech** — mic or audio file → Vietnamese transcript → translation, with a
  partial transcript shown while the speaker is still talking, and translations
  that improve as context accumulates. Runs on the GPU via `mlx-audio`
  (`large-v3-turbo`, 31× realtime), CPU `faster-whisper` as fallback.
- **Config and models** under `~/.config/dich`, loading offline. No server
  process to run.
- **Menu bar, design tokens, rounded controls, markdown rendering, buffer-memory
  readout, licence and About.**
- **Tests** — a 132-case suite (unit + golden-file quality gate) under `tests/`,
  run locally by the pre-commit hook (CI is deliberately off; see below).

Not working, or not present: **no packaging**, **no token-by-token streaming of
the *translation***, and the microphone has never been confirmed working with a
human speaker — only with recorded audio played through speakers.

---

## P0 — blocking

None open. The suite landed in 0.2.26–0.2.35, closing Milestone 0.5 — see
`ROADMAP.md`. Writing the first sentence-assembly test immediately found two
more shipped bugs (the punctuation flush trigger had never fired, and
`_trailing_silence(None)` crashed), which is the argument for the suite in one
sentence. The suite is run locally by the pre-commit hook; CI was removed on
purpose (see the notes at the end).

---

## P1 — needed for a stand-alone app

### Packaging
- [ ] Build a macOS `.app` (pyinstaller or briefcase) so it double-clicks
- [ ] Model weights bundled, or downloaded on first run with visible progress
- [ ] App icon and Finder metadata
- [ ] Codesign + notarize, so it opens without a Gatekeeper complaint
- [ ] `.dmg` optional

### Model management
- [ ] Model picker listing what is **already cached**, not free-text entry — at
  present it is easy to type a name that then downloads 8 GB
- [ ] Show model size and estimated RAM before switching
- [ ] Download progress for a model that is not cached yet
- [ ] Unload the model when idle, to free RAM

### Confirm the microphone end to end
The only untested path with a real person. Recorded audio fed through speakers
works; a human speaking has never been confirmed.

Evidence from recorded material (2026-10-16): a level sweep of real Vietnamese
speech through the live window path still transcribes down to about −40 dB
(peak 0.0076, well under any sane mic gain); only −50 dB (peak 0.0024, below
`SILENCE_FLOOR`) is dropped. Nine 20 s excerpts spread across 38 minutes of a
multi-speaker dictation file all produced text, so varied voices transcribe
from a file. The failure is therefore at the live mic or its environment, not
in the filters — it cannot be reproduced without a person at a microphone.
- [ ] Listen with a Vietnamese speaker and record what actually comes out
- [ ] Check the input device and System Settings → Microphone permission first
- [ ] If accuracy is the problem, act on it; if latency is, tune the window

### Install guide
- [ ] A reproducible, written setup from a clean machine

---

## P2 — real improvements

### Speech quality
- [x] **Move transcription onto the GPU** (0.2.36–0.2.40). The blocker was the
  repo layout, not mlx-audio: `mlx-community/whisper-large-v3-turbo` ships
  weights only, so `get_tokenizer()` raised `Processor not found`. Model ids now
  map to the `-asr-fp16` repos, which carry the full processor. `mlx-audio`
  runs Whisper and Qwen3-ASR; `faster-whisper` remains the CPU fallback, chosen
  by `stt_backend` (`auto` prefers the GPU). Measured on an M5: turbo 31×
  realtime, Qwen3-ASR 0.6B 30×, both far ahead of the CPU.
- [x] Live partial transcript while the user is still speaking (0.2.40) — the
  engine's pending sentence is shown on the status line and updates per window.
- [x] GPU-appropriate live defaults (0.2.40): 5 s window and `large-v3-turbo` on
  the GPU, 2 s and `small` on the CPU. Trade-off worth knowing: a 5 s window
  means the first *sentence* lands later than the old 1.2 s, but the partial
  line fills sooner and accuracy on unfamiliar voices is markedly better.
- [ ] Streaming *translated* output token by token — the engine can already
  produce it; the UI does not use it yet (partials above are the transcript,
  not the translation)
- [ ] Push-to-talk as an alternative to always-on listening
- [ ] Re-scan the retained buffer for an alternative transcript, as originally
  intended
- [ ] Tune silence threshold and minimum sentence length against real speech
  (the floor survived a −40 dB level sweep; the marker list still drops genuine
  "cảm ơn các bạn" — decide whether that trade is right)

### Text translation
- [ ] Preserve PDF layout instead of extracting a flat text stream
- [ ] Preserve headings, tables and lists in DOCX output rather than plain text
- [ ] Side-by-side aligned view: source paragraph ↔ translated paragraph
- [ ] Diff view against the Google Translate reference
- [ ] Export in the same format it was read from (translate a `.docx`, get a
  `.docx`)
- [ ] Batch-translate a whole folder
- [ ] Clipboard translation / translate selected text

### App lifecycle
- [ ] Remember window size and position between runs
- [ ] Resume the last document on launch
- [ ] Recent files list
- [ ] Proper quit handling for in-flight requests

---

## P3 — later

- [ ] Languages beyond Vietnamese → English. The config is already shaped for
  it; the prompt and STT are hard-coded to Vietnamese.
- [ ] Quality scoring against the Google reference, so weak translations are
  flagged
- [ ] OCR for diagrams and images inside PDFs
- [ ] Glossary so proper nouns stay consistent
- [ ] Combine consecutive short utterances into one model call, to cut overhead

---

## Known limitations — deliberate, not backlog

These cost real debugging time. They are constraints, not tasks:

- **mlx-lm cannot load T5.** `VTSNLP/trans_model_vi_en` is a T5 encoder-decoder;
  the server dies with `Model type t5 not supported`. Encoder-decoder models are
  unsupported outright. This is why TranslateGemma is used instead.
- **plamo-2-translate is English↔Japanese only**, so it cannot do Vietnamese.
- **mlx-lm.server cannot carry every prompt format.** Its
  `process_message_content()` flattens a `content` list into a plain string,
  discarding the `source_lang_code`/`target_lang_code` that TranslateGemma
  requires. That is why models load in-process rather than over HTTP.
- **Gemma-family models do not stop on their own.** mlx-lm stops on
  `tokenizer.eos_token_ids`, which excludes `<end_of_turn>`. Left unhandled a
  short sentence takes 10s and emits thousands of junk characters. Handled by
  `_align_stop_tokens()`.
- **Tkinter is not thread-safe.** Anything touching a widget from the model's
  worker thread must go through `Worker.post`.
- **A `content` list cannot be passed to faster-whisper either.** Its
  `decode_audio()` calls `av.open(metadata_errors=...)`, which the installed
  PyAV rejects — audio is decoded in-process instead.
- **Whisper hallucinates on silence on every backend.** The CPU path was only
  saved by faster-whisper's `vad_filter`, which `mlx-audio` does not accept.
  The silence floor in `speech_engine` therefore guards both paths, and the file
  path drops segments the model itself flags as non-speech. Do not remove it to
  "hear more".
- **The mlx-audio repo trap.** Weights-only repos such as
  `mlx-community/whisper-large-v3-turbo` have no processor, so `get_tokenizer()`
  raises `Processor not found`. Model ids map to the `-asr-fp16` repos, which
  carry the tokenizer and preprocessor.
- **The hallucination filter drops genuine speech.** "cảm ơn các bạn" is removed
  deliberately, because in YouTube-trained Whisper it is overwhelmingly an
  artefact. It is also a phrase a real person says — the trade was made for
  silence-hallucination and is worth revisiting against real speech.

---

## Notes for whoever picks this up

- CI is **intentionally disabled** until the app-finishing work lands: the
  workflow file was removed on purpose (restore `.github/workflows/ci.yml`
  from git history when the time comes — do not re-add new CI before then).
  The local gate is `.githooks/pre-commit` (ruff check + format check + the
  fast suite); enable it with `git config core.hooksPath .githooks`.
- `dev/bump_version.py` bumps the patch, commits and tags, and refuses to run on
  a dirty tree. Commit your change first, then bump.
- `DICH_LOG_LEVEL=INFO` restores the startup output that was silenced.
- `DICH_HOME` relocates `~/.config/dich`, which is how the tests should isolate
  themselves.
- Deleting a directory under `~/.config/dich/models` frees space; it
  re-downloads when that model is next asked for.
