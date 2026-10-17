# Dịch — Backlog

Last reviewed: 2026-10-17 · version **0.2.40** · branch `main`

This is the working list of what is left, ordered by priority rather than by
milestone. `ROADMAP.md` keeps the longer-horizon shape; this file is what to pick
up next.

The app is strictly **Vietnamese → English**. Anything about other languages,
speech models per language, or multi-language support has been removed — it will
not be built.

The finished shape is **runnable from source**: an install guide, green CI once
the finishing work lands, and a version tag. Packaging and distribution
(macOS `.app`, codesign, notarize, `.dmg`, bundled weights with download
progress) are in the **deferred queue** at the end — not being worked on until
much later.

Priority key — **P0** blocks a release, **P1** needed for a usable stand-alone
app, **P2** real improvement, **P3** nice to have.

---

## Where it stands

Working today, verified against the code rather than assumed:

- **Text translation** — PDF, DOCX, DOCM, DOC, RTF, Markdown, plain text →
  English, batched for long documents, cached so unchanged input is instant.
  DOCX structure (headings, tables, lists) is preserved block by block on save.
  PDFs use `pdftotext -layout` when pypdf yields nothing, then OCR as a last
  resort — including OCR of images embedded inside text PDFs (`abdf5f5`).
- **Speech** — mic or audio file → Vietnamese transcript → translation, with a
  partial transcript shown while the speaker is still talking, and translations
  that improve as context accumulates. Runs on the GPU via `mlx-audio`
  (`large-v3-turbo`, 31× realtime), CPU `faster-whisper` as fallback.
  Decibel meter, flat progress bar, push-to-talk, mic archive (raw WAV saved for
  re-transcription), idle + VRAM-pressure model unload.
- **Cleanup and rewrite** — a rule-based pass removes stutters, repeated words,
  phrase/line loops and interleaved loops (Whisper getting stuck on real
  meeting audio) and filler sounds from the transcript offline (on demand or
  automatically), and an optional OpenAI-compatible endpoint can rewrite the
  transcript or the translation for meaning. The rewrite is off by default and
  opt-in.
- **Translation fidelity** — side-by-side aligned view (`715b264`), diff vs the
  Google reference (`cd97bdd`), offline quality scoring (`dich/quality.py`),
  same-format export, folder batch, clipboard translation (Cmd+Shift+V), and
  glossary support (`caf5812`).
- **Session** — recent files, resume the last document on launch (`b3e52cb`),
  window geometry remembered, proper quit handling.
- **Config and models** under `~/.config/dich`, loading offline. No server
  process to run. Model picker lists what is cached with size/RAM estimates and
  a download-size warning for uncached models.
- **Menu bar, design tokens, rounded controls, markdown rendering, buffer-memory
  readout, licence and About.**
- **Tests** — a 331-case suite (323 fast + 8 model goldens) under `tests/`,
  run locally by the pre-commit hook (CI is deliberately off; see below).

Not working, or not present: **no token-by-token streaming of the
translation** — `backend.py` translates through a blocking `mlx_generate()`
call, so the UI only ever sees complete results — and the microphone has
never been confirmed working with a human speaker — only with recorded audio
played through speakers.

---

## P0 — blocking

None open.

---

## P1 — needed for a stand-alone app

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
- [ ] `INSTALL.md`: a reproducible, written setup from a clean machine
  (`PACKAGING.md` covers the app bundle; this is about getting `python -m dich`
  running from scratch)

---

## Remaining work — prioritized list

Everything below is partially done or not done at all, in priority order.

1. **Confirm the microphone with a real Vietnamese speaker** (P1) — blocked
   until a person is at the mic with System Settings → Microphone permission
   granted. Cannot be reproduced from recorded audio.

2. **Streaming translated output token by token** (P2) — the engine calls
   blocking `mlx_generate()` (`dich/backend.py`); the UI consumes complete
   results. Switch to `mlx_lm.stream_generate` and append tokens in the
   Translate pane as the batch runs. **In progress.**

3. **Re-scan the retained buffer for an alternative transcript** (P2) — as
   originally intended, give the user a second pass over buffered audio
   (different window or model), compare, keep or discard.

4. **Tune silence threshold and minimum sentence length against real speech**
   (P2) — `SILENCE_FLOOR` survived a −40 dB level sweep, but the values have
   not been tuned against live human speech; the marker list still drops
   genuine "cảm ơn các bạn" — decide whether that trade is right. Blocked
   behind item 1.

5. **Perfect PDF layout** (P3) — `pdftotext -layout` is a fallback; true layout
   fidelity (columns, side-by-side text, reading order) is not. Decide: build
   it or close it as a documented limitation.

---

## Deferred queue — packaging & distribution (not now)

These are intentionally parked until much later. None are being worked on.

- Build a macOS `.app` (pyinstaller or briefcase) so it double-clicks — spec
  exists in `packaging/dich.spec`, untested (PyInstaller not installed here)
- Model weights bundled, or downloaded on first run with visible progress
- App icon and Finder metadata
- Codesign + notarize, so it opens without a Gatekeeper complaint
- `.dmg` optional

---

## Known limitations — deliberate, not backlog

These cost real debugging time. They are constraints, not tasks:

- **mlx-lm cannot load T5.** `VTSNLP/trans_model_vi_en` is a T5 encoder-decoder;
  the server dies with `Model type t5 not supported`. Encoder-decoder models are
  unsupported outright. This is why TranslateGemma is used.
- **plamo-2-translate is English↔Japanese only**, so it cannot do Vietnamese.
  Do not use this model. It is already gone from the config and the model cache
  (verified 2026-10-17).
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
