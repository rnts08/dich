# Dịch — TODO

Last consolidated: 2026-10-21 · version **0.2.43** · branch `main`

The single list of what is left: bugs, todos, enhancements, deferred work,
and the constraints any change has to respect. This file replaces
`ROADMAP.md` and `BACKLOG.md` (history lives in `CHANGELOG.md`).

The app is strictly **Vietnamese → English** — other languages, per-language
speech models, and multi-language support will not be built (decision
recorded 2026-10-16). The finished shape is **runnable from source**.

Priority key — **P1** needed for a finished app, **P2** real improvement,
**P3** nice to have.

---

## Status

Nothing known broken. 349 tests pass (341 fast + 8 model goldens), the
pre-commit gate runs ruff + the fast suite on every commit, and the live
microphone path was repaired and verified end to end via a BlackHole
loopback (2026-10-18): real recordings through the app's own live engine
produce 36 correct sentences in 150 s, first at 7.4 s. What remains is a
human at the microphone for tuning, one threshold decision, and the
PDF-layout call.

Working today: document translation (PDF/DOCX/DOC/RTF/MD/TXT, batched,
cached, DOCX structure preserved, OCR fallback incl. embedded images),
streaming translations, speech transcription and contextual translation on
the GPU (mlx-audio) with CPU fallback, live partials, disfluency cleanup +
loop collapsing, optional LLM rewrite, re-scan of archived recordings,
side-by-side view, quality scoring, same-format export, folder batch,
glossary, explicit-save settings with unsaved-changes protection, a
machine-fitted speech-model picker (GPU default), the active speech model
under the translation model, and the Stats-for-nerds panel.

---

## Bugs

None open. (The last shipped bug was the live-capture aliasing fixed in
0.2.42: `_on_audio` kept a view of PortAudio's reused callback buffer, so
live windows were looped garbage while archives stayed clean — see
`CHANGELOG.md` 0.2.42 and `tests/test_capture_aliasing.py`.)

---

## Todos & enhancements

1. **Confirm the microphone with a real Vietnamese speaker** (P1) — the
   software path is verified via loopback; this is about gain, room noise
   and permission on the physical microphone.
   - [ ] Check the input device and System Settings → Microphone permission
   - [ ] Listen with a Vietnamese speaker and record what actually comes out
   - [ ] If accuracy is the problem, act on it; if latency is, tune the window
   - Tools: `dev/mic_check.py --list` then `--seconds 12` (names the failing
     link); `dev/loopback_check.py <file>` re-verifies the live path
     digitally (needs `brew install blackhole-2ch`).

2. **Tune the silence threshold and the voice gate against real speech**
   (P2) — evidence from the loopback work (2026-10-18): clean synthesized
   Vietnamese measures a 0.55–0.60 voice-band share and passes the gate
   easily, but real meeting recordings measure 0.05–0.45 — below the gate's
   0.5 bar — so the live mic would reject much of the material the file
   path transcribes fine. The thresholds were calibrated on mic speech;
   decide whether recorded/noisy material should pass too. Blocked on item 1.
   - [ ] Re-calibrate `VOICE_BAND_MIN_RATIO` / `SILENCE_FLOOR` against the
     live session's numbers
   - [ ] Decide the `cảm ơn các bạn` marker trade with a live-speaker verdict

3. **Perfect PDF layout** (P3) — `pdftotext -layout` is a fallback; true
   layout fidelity (columns, side-by-side text, reading order) is not done.
   - [ ] Decide: build it, or close it as a documented limitation

4. **Restore CI when the app ships** — the workflow was removed on purpose
   while the app was being finished; restore `.github/workflows/ci.yml`
   from git history at that point (do not re-add new CI before then).

---

## Deferred — packaging & distribution (not now)

- Build a macOS `.app` (pyinstaller or briefcase) so it double-clicks —
  the spec exists in `packaging/dich.spec`, untested (see
  `packaging/PACKAGING.md` and `packaging/build_app.sh`)
- Model weights bundled, or downloaded on first run with visible progress
- App icon and Finder metadata
- Codesign + notarize, so it opens without a Gatekeeper complaint
- `.dmg` optional

---

## Constraints — deliberate, not tasks

These cost real debugging time. They are the shape of the system, not
backlog items:

- **All AI and audio work should be GPU-accelerated** — STT via
  `mlx-audio` on Metal, translation in-process on MLX. If a new AI feature
  cannot run on the GPU, say so explicitly when proposing it.
- **mlx-lm cannot load T5.** `VTSNLP/trans_model_vi_en` dies with
  `Model type t5 not supported`; encoder-decoder models are unsupported
  outright. This is why TranslateGemma is used.
- **plamo-2-translate is English↔Japanese only.** Do not use it. It is
  already gone from the config and the model cache (verified 2026-10-17).
- **mlx-lm.server cannot carry every prompt format** — it flattens a
  `content` list, discarding the language tags TranslateGemma requires.
  Models therefore load in-process, not over HTTP.
- **Gemma-family models do not stop on their own** — `<end_of_turn>` is not
  in `tokenizer.eos_token_ids`. Handled by `_align_stop_tokens()`.
- **Tkinter is not thread-safe.** Anything touching a widget from a worker
  thread must go through `Worker.post`.
- **faster-whisper cannot be handed a path or a content list** — its
  `decode_audio()` calls `av.open(metadata_errors=...)`, which the
  installed PyAV rejects. Audio is decoded in-process instead.
- **Whisper hallucinates on silence on every backend.** The silence floor
  in `speech_engine` guards both paths, and the file path drops segments
  the model flags as non-speech. Do not remove it to "hear more".
- **The mlx-audio repo trap.** Weights-only repos such as
  `mlx-community/whisper-large-v3-turbo` have no processor, so
  `get_tokenizer()` raises `Processor not found`. Model ids map to the
  `-asr-fp16` repos, which carry the tokenizer and preprocessor.
- **The hallucination filter drops genuine speech.** "cảm ơn các bạn" is
  removed deliberately — in YouTube-trained Whisper it is overwhelmingly
  an artefact, but it is also a phrase a real person says. Revisit with a
  live speaker (see todo 2).

---

## Working notes

- The local gate is `.githooks/pre-commit` (ruff check + format check + the
  fast suite); enable once per clone with
  `git config core.hooksPath .githooks`, or run `./install.sh dev`.
- `dev/bump_version.py` bumps the patch, commits and tags, and refuses a
  dirty tree — commit first, then bump.
- `DICH_LOG_LEVEL=INFO` restores the silenced startup output; `DICH_HOME`
  relocates `~/.config/dich` (how the tests isolate themselves).
- Deleting a directory under `~/.config/dich/models` frees space; it
  re-downloads when that model is next asked for.
