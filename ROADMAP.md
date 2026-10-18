# Dịch — Roadmap

Current version: **0.2.41**. Speech-to-text is working and is no longer the
constraint; testing is done (Milestone 0.4). The live microphone path was
repaired and verified via a BlackHole loopback (2026-10-18) — see
`BACKLOG.md` for the aliasing bug it exposed. What stands between this and a
finished app is **a mic session with a real speaker** (for tuning), and the
PDF-layout decision. The finished shape is **runnable from source**; the
double-clickable `.app` is deferred.

`BACKLOG.md` is the prioritised list of what to pick up next. This file keeps
the shape of where the project is going, and why.

---

## Milestone 0.2 — Speech-to-text — *done*

- [x] Mic capture, 16 kHz mono, numpy buffer with a configurable ceiling
- [x] `faster-whisper` Vietnamese transcription, mic and file
- [x] Audio decoded in-process: faster-whisper's `decode_audio()` passes
  
      `av.open(metadata_errors=...)`, which the installed PyAV rejects
- [x] Validated on real Vietnamese speech (150s sample, 22 utterances)
- [x] Sentences emitted on a final trigger rather than mid-clause
- [x] Context-accumulating translation: prior utterances as conversation turns,
  
      then a block re-translation every `refine_every`
- [x] `finalize()` so a trailing partial block is not left provisional
- [x] Input device selection, stale-device fallback, no-mic handling
- [x] Variable window (default 2s → first words ~1.2s after speech)
- [x] Buffer-memory readout, a countdown to the next transcript, and an
  
      explicit "Ns behind" / "Ns dropped" warning instead of a frozen 0.0s
- [x] Hallucination markers, in-sentence stripping, repetition detection
- [x] Silenced startup output; clean exit

**Carried out of this milestone since:** the GPU path landed (`mlx-audio`
runs Whisper on the Metal GPU, `7a2d319`), as did push-to-talk and the live
partial transcript. Streaming translated output is still open — it is the
one significant feature left (see Milestone 0.3.5).

Standing policy: **all AI and audio work should be GPU-accelerated.** STT
runs on the GPU via `mlx-audio`; translation runs in-process on MLX. If a new
AI feature cannot run on the GPU, say so explicitly when proposing it.

---

## Milestone 0.2.5 — Models and processing fixes — *done*

- [x] Only useful models offered, with a blocking progress bar while a model
  loads (`ed1968c`, `cd628cd`)
- [x] All MLX work on the GPU: STT via `mlx-audio` (`7a2d319`, `9034772`),
  translation in-process on MLX
- [x] The speech UI makes it clear audio is being processed: a live partial
  transcript appears while the speaker is still talking, translation starts
  after the first sentences and runs in parallel, and block re-translation
  looks back for context (`c625865`, `acd64d7`)
- [x] Noise and gibberish handled: speech-band filter, voice-only gate,
  silence floor, hallucination markers, disfluency cleanup (`0f43584`,
  `946c794`, `3fdb917`)
- [x] Input decibel meter with decaying peak (`2f29356`)
- [x] Transcription starts as soon as words are picked up (`c625865`)
- [x] Determinate flat progress bar replaces the aimless sweep (`b0c088c`,
  `5309ab9`, `caf5812`)
- [x] Models unload only when idle or under VRAM pressure (`ed1968c`,
  `cd628cd`)

## Milestone 0.3 — Translation fidelity — *done except PDF layout*

- [ ] Preserve PDF layout rather than a flat text stream — `pdftotext
  -layout` is a fallback; true columns and reading order are not done
- [x] Preserve headings, tables and lists in DOCX output (`cd628cd`)
- [x] Side-by-side aligned view: source paragraph ↔ translated paragraph
  (`715b264`)
- [x] Diff against the Google Translate reference (`cd97bdd`)
- [x] Export in the format it was read from (`caf5812`)
- [x] Batch-translate a folder; clipboard translation (`caf5812`,
  Cmd+Shift+V)
- [x] Quality scoring against the reference, so weak output is flagged
  (`6f23d74`, `dich/quality.py`)

## Milestone 0.3.5 — Streaming translated output — *done*

The last significant feature. `backend.py` now generates with
`mlx_lm.stream_generate`; the Translate pane appends tokens while the batch
runs, and the finished translation is re-rendered as Markdown when the run
completes.

- [x] `backend.py`: generate with `mlx_lm.stream_generate` and surface tokens
  as they arrive
- [x] `ui.py`: the Translate pane appends tokens while the batch runs
  (through `Worker.post` — Tkinter is not thread-safe), status keeps counting
  words and tokens/sec live
- [x] Streaming and batch paths produce identical final text (regression
  test with a fake engine; the model goldens pin the real output)

## Milestone 0.3.6 — Re-scan and the live-mic fix — *done*

- [x] Speech → Re-scan ▾: re-transcribe an archived recording with a chosen
  model, compare it beside the current transcript, Keep adopts it through the
  same path Open Audio uses
- [x] The live capture buffer copies each callback's audio instead of
  aliasing PortAudio's reused callback buffer — the bug that made live
  transcription loop garbage while archived recordings stayed clean (found
  via the BlackHole loopback, `dev/loopback_check.py`)
- [x] `dev/mic_check.py` diagnoses a physical microphone link by link

## Milestone 0.4 — Confidence — *done*

The largest gap, closed in 0.2.26–0.2.35: 80 tests under `tests/`, CI on every
push, and a golden-file quality gate. Writing them found two more shipped bugs —
the punctuation flush trigger had never fired, and `_trailing_silence(None)`
crashed — which is the whole argument for the suite.

- [x] Unit tests: cache key invalidation, `<end_of_turn>` stop handling, every
  
      extractor, the prompt-strategy switch, sentence assembly and its filters
- [x] `RoundedButton` pointer events must not loop (the UI freeze)
- [x] Golden-file translation tests for quality regressions
- [x] CI running the suite and `py_compile` on every push

## Milestone 0.5 — Stand-alone application — *app-shaped; bundle deferred*

The point at which this stops being "run Python in a terminal". The working
model is **runnable from source**; the double-clickable bundle is parked in
the deferred queue in `BACKLOG.md`.

- [ ] macOS `.app` via pyinstaller or briefcase — *deferred; the spec exists
  in `packaging/dich.spec` and is untested*
- [ ] Model weights bundled — *deferred with the bundle*
- [ ] App icon, Finder metadata, codesign + notarize — *deferred; needs an
  Apple Developer account and is manual*
- [x] Model picker listing what is already cached, with size/RAM estimates
  and a download-size warning for uncached models (`ed1968c`, `caf5812`)
- [x] Unload when idle to free RAM; unload under VRAM pressure (`ed1968c`,
  `cd628cd`)
- [x] Remember window size and position; resume the last document (`b3e52cb`)
- [x] Recent files; proper quit handling for in-flight requests
- [x] Written, reproducible install guide (`INSTALL.md`, 2026-10-18; includes
  the `dev/mic_check.py` rehearsal and `dev/loopback_check.py` loopback)

## Milestone 0.6 — Beyond Vietnamese → English — *descoped*

The app is strictly **Vietnamese → English** (decision recorded in
`BACKLOG.md`): other languages, per-language speech models, and
multi-language support will not be built.

- [ ] Configurable source and target language end to end — *descoped*
- [ ] A speech model per language — *descoped*
- [x] Glossary support so proper nouns stay consistent (`caf5812`)
- [x] OCR for diagrams and images inside PDFs (`abdf5f5`)

---

## Known limitations — constraints, not tasks

These cost real debugging time and are not bugs to be filed against:

- **mlx-lm cannot load T5.** `VTSNLP/trans_model_vi_en` is a T5 encoder-decoder;
  it dies with `Model type t5 not supported`. Encoder-decoder models are
  unsupported, which is why TranslateGemma is used. Settled — do not revisit.
- **plamo-2-translate is English↔Japanese only**, so it cannot do Vietnamese.
  Do not use this model. It is already gone from the config and the model
  cache (verified 2026-10-17).
- **mlx-lm.server cannot carry every prompt format.** Its
  `process_message_content()` flattens a `content` list into a plain string,
  discarding the `source_lang_code`/`target_lang_code` TranslateGemma needs —
  hence models load in-process rather than over HTTP.
- **Gemma-family models do not stop on their own.** mlx-lm stops on
  `tokenizer.eos_token_ids`, which excludes `<end_of_turn>`. Unhandled, a short
  sentence takes 10s and emits thousands of junk characters. Handled by
  `_align_stop_tokens()`.
- **Tkinter is not thread-safe.** Widget access from the model's worker thread
  must go through `Worker.post`.
- **faster-whisper cannot be handed a path either.** `decode_audio()` calls
  `av.open(metadata_errors=...)`, which the installed PyAV rejects, so audio is
  decoded here.
- **The hallucination filter drops genuine speech.** "cảm ơn các bạn" is removed
  deliberately: in YouTube-trained Whisper it is overwhelmingly an artefact.

## Bugs fixed after 0.2.17

Recorded because each one was invisible until it was looked for, which is the
argument for the tests in Milestone 0.4.

- **Markdown files were converted to HTML before translation.** `_read_markdown`
  ran them through `markdown.markdown()`, so the model translated HTML while the
  output pane renders Markdown — a translated `.md` arrived showing literal
  `<h1>` and `<strong>` tags. Markdown source is now fed as-is, and the renderer
  converts stray HTML back to Markdown so the failure cannot recur.
- **Dropped audio was silent.** The buffer ceiling discards the oldest audio when
  the transcriber falls behind, and nothing said so — speech was permanently lost
  with no indication. The status bar now reports "Ns dropped — raise the buffer
  ceiling".
- **The countdown lied.** `next_in` clamped to zero, so a lagging transcriber
  read "next transcript in 0.0s", which means *imminent* rather than *nothing is
  coming*. It now reports "Ns behind, catching up".
- **Opening a file while recording** fed both into the same buffer and clobbered
  the live transcript. Recording now stops first.
- **Clearing the speech panes while recording** stopped the buffer readout,
  removing the countdown from under someone still listening.
- **Opening an audio file did nothing at all.** The idle buffer report lacked
  `window_seconds`, so `_poll_memory`'s first tick — run synchronously inside
  `open_audio()` — raised `KeyError` before `worker.run(transcribe_file…)` was
  ever reached: no transcription, no buffer line, status stuck at
  "transcribing…". Introduced with the honest-countdown fix. The idle schema
  now mirrors the live engine's, the readout can no longer raise out of its
  caller, and both layers carry regression tests. Reported by the user after
  0.2.39; file transcription and the live mic were both re-verified end to end.
- **`markdown` was a dead dependency**, removed from the install line.
- The stored `mic_model` was `medium`, which falls behind at a 2-second window,
  while the README documented `small`. Aligned — and in 0.2.26 the same value
  was found still wrong in the seed `config.json` and in `DEFAULT_CONFIG`, so
  all three sources now agree.

### Found by the first test suite (0.2.20–0.2.35)

- **The punctuation sentence-flush trigger never fired.** `_is_complete` passed
  `.?!。！？…` to `endswith()` as one suffix, which checks for text ending in
  that entire sequence — never. Every sentence-ending period was invisible; only
  the Vietnamese end-word, segment-count and window-end triggers ever emitted.
- **`_on_finalized` crashed on every finished audio file** — a stray duplicated
  `worker.run(...)` block referenced locals that do not exist in its scope, and
  Tk's exception reporter swallowed the NameError.
- **`python backend.py` crashed on undefined `PORT`** — a leftover from the
  server era.
- **Documents with few paragraphs ignored `char_limit`** — the batching fast
  path returned the whole text as one prompt regardless of length, so a long
  single-paragraph document was truncated by the TranslateGemma context.
- **`_render_speech` called `root.after()` from a raw thread** — the exact
  pattern `Worker.post` exists to prevent.
- **`_trailing_silence(None)` crashed** when a pending utterance was judged
  before the first window was transcribed.
- **The buffer-overflow logic was unreachable** — trapped inside the `start()`
  closure, now extracted as `_on_audio` and covered by tests.
