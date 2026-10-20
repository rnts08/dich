# Changelog

All notable changes to Dịch. Versions are tagged `v<VERSION>` by
`dev/bump_version.py`; the format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The app is strictly Vietnamese → English. Version policy: a patch bump per
fix or feature, a minor bump when a milestone begins.

## 0.2.43 — 2026-10-20

### Added
- **Unsaved-settings protection**: Settings shows an *unsaved changes*
  marker whenever anything drifts from the last save (snapshot comparison,
  so programmatic refreshes never read as edits), and quitting asks Save /
  Discard / Cancel. Checkbox toggles no longer write config behind the Save
  button — Save is the single write path.
- **GPU-first speech models**: the speech-model default is now the machine's
  default (`large-v3-turbo` on Metal, `small` on CPU-only); GPU-only models
  (Qwen3-ASR) are offered in the picker only when a GPU is reachable, and a
  saved model the machine cannot run falls back to the default. Configs
  carrying the old CPU-era seed upgrade once at startup.
- **Active speech model in the title area**: a second line under the
  translation model shows `voice: <model> · idle/listening/transcribing ·
  GPU/CPU`.
- **Stats for nerds** (Settings → Statistics): a live panel with the last
  generation's encode speed (prompt tok/s), decode speed, tokens in → out
  and peak memory — captured from `stream_generate`'s final response — plus
  lifetime totals persisted in `config.json`: words translated, sentences
  transcribed, speech time recorded and transcribed, translations, files,
  recordings, cache hits. The status bar prefers the generation's own
  decode speed for its tok/s figure.

### Changed
- `TranslationResult` carries `prompt_tps`, `generation_tps` and
  `peak_memory_gb`; `backend.LLM_STATS` holds the last generation's numbers;
  `backend.bump_stats()` accumulates the lifetime counters under a lock.

## 0.2.42 — 2026-10-18

The live microphone path was repaired after the loopback work exposed a
shipped bug, and a recorded buffer can now get a second transcription pass.
See `BACKLOG.md` for the full aliasing investigation.

### Added
- **Re-scan** (Speech → Re-scan ▾): re-transcribe an archived recording — or
  any file — with a chosen speech model, compare it beside the current
  transcript, then Keep (adopt and re-translate) or Discard (change nothing).
  Adoption goes through the same path Open Audio uses; a scan refuses to run
  while another file job is active.
- `dev/mic_check.py` — diagnoses a physical microphone link by link: live
  level meter, then the app's own speech-band filter and voice gate over each
  captured window, with a verdict that names level, gate or model as the
  failing link. `--transcribe` shows what the engine would emit.
- `dev/loopback_check.py` — exercises the entire live path with no person and
  no speaker: plays a real recording into a BlackHole loopback while the
  app's own engine captures. Reports sentence count and first-sentence
  latency, archives the capture for Re-scan.
- `INSTALL.md` — reproducible clean-machine setup: Python and Tk, venv,
  dependencies, pre-commit gate, first-run downloads, microphone permission,
  data locations, troubleshooting.

### Fixed
- **Live capture aliasing** — `_on_audio` stored a *view* of sounddevice's
  callback buffer, which PortAudio reuses on the next invocation, so every
  buffered block silently showed the latest callback's audio. Windows became
  looped garbage and Whisper hallucinated over them, while archived
  recordings (which always copied) stayed clean. Capture now copies at
  arrival. Found via the BlackHole loopback: a real 150 s meeting recording
  through the repaired live path produces 36 correct sentences, first at
  7.4 s. Pinned by `tests/test_capture_aliasing.py`, which simulates the
  reused buffer.

### Changed
- `SpeechEngine.start()` split into `prepare()` (model, threads, archiving)
  and `open_capture()` (the device), so the loopback tool can drive
  `_on_audio` through its own full-duplex stream. macOS refuses two
  half-duplex streams on one device in one process (AUHAL −10863); one
  duplex stream works.
- `BACKLOG.md` and `ROADMAP.md` reconciled with what actually shipped:
  four done items struck from the remaining list, Milestone 0.2.5 checked
  off with its commits, Milestone 0.6 (other languages) descoped, the plamo
  cache note closed after verifying the model is gone from config and cache.

## 0.2.41 — 2026-10-17

### Added
- **Streaming translated output**: generation switched from blocking
  `mlx_generate()` to `mlx_lm.stream_generate()`. The Translate pane appends
  deltas while the batch runs (marshalled through `Worker.post`), and the
  finished text is re-rendered as Markdown on completion. The streamed path
  assembles exactly the text the batch path returns — chunk separators are
  emitted between batches, a fallback-model retry sends `None` so the client
  clears what it has, and cache hits never stream. Pinned by
  `tests/test_streaming.py` and by the model goldens, which now run through
  `stream_generate` with byte-identical output.

## 0.2.40 — 2026-10-16

The speech-quality burst: GPU transcription, the anti-hallucination stack,
and a long tail of fixes. Thirty-four commits between the 0.2.39 and 0.2.41
tags.

### Added
- Whisper on the Metal GPU via `mlx-audio` with `faster-whisper` CPU
  fallback (`large-v3-turbo`: 31× realtime on GPU vs 3.1× on CPU); the
  accurate model now runs live, with a 5 s window as the GPU default and
  `small` remaining the CPU default. Qwen3-ASR presets.
- Live partial transcript shown while the speaker is still talking.
- Voice-only mic gate (voice-band energy, spectral flatness, zero-crossing
  rate, learned noise floor), speech-band front-end (80 Hz–7 kHz), and
  voice-band noise-floor measurement — the GPU path's substitute for
  `vad_filter`.
- Rule-based disfluency cleanup: stutters, repeats, filler sounds, and the
  Whisper loop artefacts real meeting audio produces (a phrase repeated one
  segment per line, interleaved loops) — collapsed offline, then
  re-translated. Optional OpenAI-compatible LLM rewrite of transcript or
  translation (off by default).
- Microphone archive (raw WAV per session), transcript+translation saving,
  phase-by-phase file progress (decode → model → transcribe → translate),
  input decibel meter with decaying peak, push-to-talk, flat determinate
  progress bar, same-format export, folder batch, glossary, clipboard
  translation, side-by-side aligned view, resume-last-document on launch,
  recent files, OCR for image PDFs and for images embedded in text PDFs,
  offline quality scoring, diff vs the Google reference, DOCX structure
  preservation, blocking model-load progress, idle and VRAM-pressure model
  unload, cached-model picker with size/RAM estimates.
- `dev/pipeline_report.py` — the whole path (cleanup → contextual
  translation) over real transcripts.

### Fixed
- Opening an audio file did nothing at all (a `KeyError` in the idle buffer
  schema killed the worker before it started).
- A failed file transcription left the UI spinning; an empty transcription
  was reported as success; only one file or mic job could run at a time;
  opening audio while recording clobbered the buffer.
- File transcription showed a static label instead of real progress; the
  buffer readout claimed "0.0s" in file mode.
- The mic-model setting disagreed between config, seed and code.

### Changed
- Repackaged as the `dich/` package with dead code removed; backend config
  side effects deferred; CI deliberately disabled during the app-finishing
  work (restore from git history when shipping), replaced by ruff + a local
  pre-commit gate (`git config core.hooksPath .githooks`).
- Backlog decision recorded: strictly Vietnamese → English; packaging
  (`.app`, codesign, notarize, `.dmg`, bundled weights) deferred.

## 0.2.30–0.2.35 — 2026-10-12

The first test suite (Milestone 0.4, "Confidence"): unit tests for every
extractor, the prompt-strategy switch, stop-token handling, sentence
assembly, config and the contextual translator; a UI hover regression test;
and a golden-file translation quality gate whose review is the gate for
intended output changes. Writing them immediately found two shipped bugs:
the punctuation sentence-flush trigger had never fired, and
`_trailing_silence(None)` crashed. CI ran the suite on every push at this
point.

## Earlier

- **0.2.x before 0.2.30** — speech-to-text development: mic capture,
  faster-whisper Vietnamese transcription mic and file, in-process audio
  decoding (PyAV rejects faster-whisper's `decode_audio`), sentence
  assembly with completion triggers, context-accumulating translation with
  block refinement, buffer readout and honest lag reporting, hallucination
  markers, and the model-loading fixes (in-process MLX instead of
  `mlx_lm.server`, `<end_of_turn>` stop-token alignment for Gemma models).
- **0.1.x** — document translation: open a file (PDF, DOCX, DOC, RTF,
  Markdown, TXT), get a translation, batched for long documents and cached.
