# Dịch — Backlog

Last reviewed: 2026-10-18 · version **0.2.42** · branch `main`

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
- **Settings** — explicit Save with an unsaved-changes marker and a
  quit-time warning, a machine-fitted speech-model picker (GPU default,
  GPU-only models hidden without a GPU), the active speech model shown
  under the translation model, and a Stats-for-nerds panel (LLM encode/
  decode speeds, lifetime words/sentences/speech-time totals).
- **Tests** — a 349-case suite (341 fast + 8 model goldens) under `tests/`,
  run locally by the pre-commit hook (CI is deliberately off; see below).

Not working, or not present: nothing known broken. The live microphone path
was repaired and verified end to end on 2026-10-18 (see below); what remains
untested is a *human* speaking into it, which the loopback cannot replace for
tuning judgements. PDF layout is a flat text stream.

### The live-mic aliasing bug, found and fixed (2026-10-18)

A BlackHole loopback (`dev/loopback_check.py`) — playing real recordings into
an input device while the app's own live engine captures — showed the live
path transcribing looped garbage and Whisper hallucinating, while the archive
of the very same audio transcribed perfectly. Root cause: `_on_audio` stored
`np.ascontiguousarray(indata[:, 0])`, a *view* of sounddevice's callback
buffer, which PortAudio reuses for the next callback. Every buffered block
silently showed the latest callback's audio. The archive always copied, which
is why saved recordings were clean while live output was not. The fix copies
at capture time (`np.array(...)`, pinned by `tests/test_capture_aliasing.py`).
A human may still speak differently from a loopback, but the software path is
now proven end to end.

---

## P0 — blocking

None open.

---

## P1 — needed for a stand-alone app

### Confirm the microphone end to end
The software path is now verified (2026-10-18): with BlackHole looped back,
real recordings through the app's own live engine produce 36 correct
sentences in 150 s, first at 7.4 s. What a loopback cannot decide is whether
a *human at the real microphone* works — gain, room noise, permission.

- [ ] Listen with a Vietnamese speaker and record what actually comes out
- [ ] Check the input device and System Settings → Microphone permission first
- [ ] If accuracy is the problem, act on it; if latency is, tune the window

### Install guide
- [x] `INSTALL.md`: a reproducible, written setup from a clean machine —
  written 2026-10-18, includes the mic rehearsal and loopback checks

---

## Remaining work — prioritized list

Everything below is partially done or not done at all, in priority order.

1. **Confirm the microphone with a real Vietnamese speaker** (P1) — the
   software path is verified (loopback, see above); this is about gain, room
   noise and permission on the physical microphone.

2. **Streaming translated output token by token** (P2) — **done.** The engine
   generates with `mlx_lm.stream_generate`; the Translate pane appends deltas
   via `Worker.post`, a fallback retry resets the stream, and streamed text
   is pinned identical to the batch text by tests and the model goldens.

3. **Re-scan the retained buffer for an alternative transcript** (P2) —
   **done.** Speech → Re-scan ▾ lists archived recordings, re-transcribes one
   with a chosen model, shows it beside the current transcript, and Keep
   adopts it through the same path Open Audio uses. Verified against a real
   loopback capture: 36 live sentences vs 24 (turbo, whole file) vs 58
   (small), with visibly better reading on the whole-file pass.

4. **Tune silence threshold and the voice gate against real speech** (P2) —
   new evidence from the loopback work (2026-10-18): clean synthesized
   Vietnamese measures a 0.55–0.60 voice-band share and passes the gate
   easily, but real meeting recordings measure 0.05–0.45 — *below* the gate's
   0.5 bar — so the live mic would reject much of the material the file path
   transcribes fine. The thresholds were calibrated on mic speech; decide
   whether recorded/noisy material should pass too. The `cảm ơn các bạn`
   marker trade also still needs a live-speaker verdict.

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
- The live mic can be exercised without a person: install BlackHole
  (`brew install blackhole-2ch`) and run `dev/loopback_check.py` with a real
  recording — it plays into the loopback while the app's own engine captures.
  `dev/mic_check.py` diagnoses a physical microphone link by link.
- `dev/bump_version.py` bumps the patch, commits and tags, and refuses to run on
  a dirty tree. Commit your change first, then bump.
- `DICH_LOG_LEVEL=INFO` restores the startup output that was silenced.
- `DICH_HOME` relocates `~/.config/dich`, which is how the tests should isolate
  themselves.
- Deleting a directory under `~/.config/dich/models` frees space; it
  re-downloads when that model is next asked for.
