# Dịch — Roadmap to a stand-alone, fully featured application

Current version: **0.1.x** (document translation). The milestone that moves the
minor version to **0.2.0** is starting speech-to-text.

This document tracks what is left, ordered by dependency rather than by date.
Each item is roughly one patch bump.

---

## Milestone 0.2 — Speech-to-text

The version bumps to **0.2.0** when STT development starts, then patch bumps per
change until it is done.

### Speech capture
- [x] Mic capture module (`speech_engine.py`), 16 kHz mono, buffered windows
- [x] `faster-whisper` Vietnamese transcription, wired to the UI
- [ ] **End-to-end validation with real Vietnamese speech** — mic → transcript →
      translation, verified on-device
- [ ] Microphone permission / device selection in Settings (currently uses the
      default input device)
- [ ] Graceful handling when no microphone is present

### Sentence handling
- [x] Sentences emitted only on a final trigger (silence, end punctuation, or a
      long window), so translation sees whole units rather than fragments
- [ ] Tune the silence threshold and minimum-sentence length against real speech
      — tune the parameters against actual speakers
- [ ] Push-to-talk as an alternative to always-on listening
- [ ] Live partial transcript so the user sees text while they speak, not just
      after the sentence closes

### Translation integration
- [ ] Combine consecutive short utterances into one request where possible, to
      reduce per-sentence overhead
- [ ] Streaming the translated text in token by token (the engine already
      supports it; the UI does not yet)
- [ ] Conversation history so pronouns resolve across sentences

---

## Milestone 0.3 — Application quality

### Packaging (the "stand-alone" part)
- [ ] Package as a macOS `.app` with `pyinstaller` or `briefcase`, so the user
      double-clicks an icon instead of running Python
- [ ] Bundle the model weights in the app, or download them on first run into
      `~/Library/Application Support/Dịch/`
- [ ] App icon and Finder metadata
- [ ] Codesign and notarize so the app opens without Gatekeeper complaints
- [ ] Optional: ship a `.dmg`

### Startup and lifecycle
- [x] Load the model when the app starts (Settings checkbox, on by default)
- [x] Status indicator reflects loading / ready / failed rather than a red dot
- [ ] Remember window size and position between runs
- [ ] Resume the last document on launch
- [ ] Recent files list
- [ ] Proper quit handling for in-flight requests

### Model management
- [ ] Model picker that lists locally cached models, not just free-text entry
- [ ] Show model size and estimated RAM before switching
- [ ] Download progress for a model that is not yet cached
- [ ] Unload the model when idle, so background RAM is freed
- [ ] Cache an active translation so re-opening the same file is instant

---

## Milestone 0.4 — Translation features

### Text translation
- [x] PDF, DOCX, DOC, RTF, Markdown, plain text
- [x] Batching for long documents
- [x] Progress, tokens/sec, word counts
- [x] `/tmp` cache for unchanged documents
- [ ] Preserve PDF layout instead of extracting a flat text stream
- [ ] Preserve headings, tables, and lists in DOCX output rather than plain text
- [ ] Side-by-side aligned view (source paragraph ↔ translated paragraph)
- [ ] Compare view against the Google Translate reference, with a diff
- [ ] Export to the same format it was read from (translate a `.docx` and save
      a `.docx`)
- [ ] Batch-translate a folder
- [ ] Clipboard translation / translate selected text

### Quality and languages
- [ ] Quality scoring against the reference, so weak translations are flagged
- [ ] Support languages beyond Vietnamese → English; config is already shaped for
      it but the prompt and STT are Vietnamese-only
- [ ] Handle diagrams and OCR'd images inside PDFs
- [ ] Glossary support for proper nouns that must stay consistent

---

## Milestone 0.5 — Testing and release

- [ ] Unit tests for the cache key (a changed character must invalidate)
- [ ] Unit tests for the stop-token fix, so the `<end_of_turn>` runaway cannot
      silently return
- [ ] Tests for each file-format extractor
- [ ] Tests for the prompt strategy switch, including the `content` list shape
- [ ] Golden-file translation tests, to catch quality regressions
- [ ] CI that runs the tests and checks `py_compile`
- [ ] A written, reproducible install guide

---

## Known limitations, kept here deliberately

These are findings that cost real debugging time and should not be
rediscovered:

- **mlx-lm cannot load T5.** `VTSNLP/trans_model_vi_en` is a T5 encoder-decoder;
      the server dies with `Model type t5 not supported`. Encoder-decoder models
      are not supported at all.
- **plamo-2-translate is English↔Japanese only**, so it is not usable here.
- **mlx-lm.server cannot carry some prompt formats.** Its
      `process_message_content()` flattens a `content` list into a string, which
      discards the `source_lang_code`/`target_lang_code` TranslateGemma needs.
      That is why models load in-process instead of over HTTP.
- **Gemma-family models do not stop on their own.** mlx-lm stops on
      `tokenizer.eos_token_ids`, which does not include `<end_of_turn>`. Without
      adding it, a short sentence takes 10s and emits thousands of junk
      characters. Covered by `_align_stop_tokens()`.
- **Tkinter is not thread-safe.** Widget calls from the model's worker thread
      must go through the `Worker.post` queue.
