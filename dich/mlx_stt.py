"""
GPU speech-to-text via mlx-audio (Apple Silicon).

Why this exists: faster-whisper runs on the CPU, where `small` is the fastest
model that keeps up with a live speaker and `large-v3-turbo` — the accurate one
— falls behind. MLX runs the same Whisper weights on the GPU. Measured on this
machine against the same Vietnamese audio:

    large-v3-turbo, 150s file   CPU int8 3.14x realtime -> GPU 31x
    live 2s window             CPU small 1009ms -> GPU turbo 354ms
                                                     GPU Qwen3 98ms

The repo mapping matters, and getting it wrong is the entire "Processor not
found" failure: `mlx-community/whisper-large-v3-turbo` ships weights only, so
mlx-audio cannot build a WhisperProcessor and get_tokenizer() raises. The
`-asr-fp16` repos carry preprocessor_config.json and the tokenizer too.

Requires: pip install mlx-audio
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

REPO_TEMPLATE = "mlx-community/whisper-{model}-asr-fp16"
DEFAULT_LANGUAGE = "vi"

# Non-Whisper GPU models, by the short id the config and the picker use. Kept
# separate from the Whisper template because their repos do not follow it.
GPU_MODELS = {
    "qwen3-0.6b": "mlx-community/Qwen3-ASR-0.6B-4bit",
    "qwen3-1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
}

# Qwen3-ASR wants the spelled-out name in its prompt; Whisper wants the code.
_QWEN_LANGUAGE = {"vi": "Vietnamese", "en": "English"}


def is_gpu_only(model: str) -> bool:
    """True for models only this backend can load (no faster-whisper equivalent)."""
    return model in GPU_MODELS


def repo_for(model: str) -> str:
    """MLX repo id for a model id, or the input when it already is one."""
    if model in GPU_MODELS:
        return GPU_MODELS[model]
    if "/" in model:
        return model
    return REPO_TEMPLATE.format(model=model)


def available() -> bool:
    """True when mlx-audio is installed and can reach a GPU."""
    try:
        import mlx.core as mx
    except Exception:  # noqa: BLE001 - absence must read as "not available"
        return False
    try:
        from mlx_audio.stt.utils import load  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    try:
        return bool(mx.metal.is_available())
    except Exception:  # noqa: BLE001 - very old mlx has no probe
        import platform

        return platform.machine() == "arm64"


@dataclass
class Segment:
    """The part of a faster-whisper segment the app actually reads."""

    text: str
    start: float
    end: float
    no_speech_prob: float = 0.0


def _load_repo(repo: str):
    from mlx_audio.stt.utils import load

    return load(repo)


class MLXWhisper:
    """
    Whisper on the GPU, wearing faster-whisper's transcribe() shape.

    SpeechEngine and transcribe_file call `model.transcribe(...)` and read
    `seg.text` / `seg.end`, so matching that signature keeps them unchanged and
    lets the two backends be swapped by configuration alone.
    """

    def __init__(self, model_name: str = "large-v3-turbo"):
        self.model_name = model_name
        self._model = None

    def _loaded(self):
        if self._model is None:
            repo = repo_for(self.model_name)
            logger.info(f"Loading {repo} on the GPU via mlx-audio")
            self._model = _load_repo(repo)
        return self._model

    def transcribe(
        self,
        audio,
        language: str = DEFAULT_LANGUAGE,
        **kwargs,
    ) -> Tuple[List[Segment], SimpleNamespace]:
        """
        Transcribe one buffer.

        faster-whisper-only arguments (beam_size, vad_filter, …) are accepted
        and dropped rather than forwarded: mlx-audio raises on unknown keywords.
        Silence is not special-cased here — the engine must keep its own
        SILENCE_FLOOR guard, because Whisper on any backend invents text when
        handed a quiet buffer.
        """
        model = self._loaded()
        options = {
            "language": language or DEFAULT_LANGUAGE,
            "condition_on_previous_text": bool(
                kwargs.get("condition_on_previous_text", False)
            ),
        }
        # Forward the caller's silence threshold when it names one, so the model
        # itself can drop no-speech segments instead of only relying on the
        # engine's own guard.
        no_speech = kwargs.get("no_speech_threshold")
        if no_speech is not None:
            options["no_speech_threshold"] = float(no_speech)
        out = model.generate(audio, **options)
        segments = [
            Segment(
                text=str(s.get("text", "")).strip(),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                no_speech_prob=float(s.get("no_speech_prob", 0.0) or 0.0),
            )
            for s in (out.segments or [])
        ]
        info = SimpleNamespace(
            language=getattr(out, "language", language) or language,
            language_probability=0.0,  # mlx-audio does not report one
            duration=0.0,  # unknown; the caller already has the audio length
        )
        return segments, info


def is_gpu_model(model: str) -> bool:
    """True when `model` names something this backend can load."""
    return bool(model) and "faster-whisper" not in model


class MLXQwen3:
    """
    Qwen3-ASR on the GPU, wearing the same transcribe() shape as the Whisper
    adapter. It is the fastest live option measured here (98 ms for a 2 s
    window versus 354 ms for turbo) at the cost of a few more mishearings, so
    it is a selectable preset rather than the default.
    """

    def __init__(self, model_name: str = "qwen3-0.6b"):
        self.model_name = model_name
        self._model = None

    def _loaded(self):
        if self._model is None:
            repo = repo_for(self.model_name)
            logger.info(f"Loading {repo} on the GPU via mlx-audio")
            self._model = _load_repo(repo)
        return self._model

    def transcribe(
        self,
        audio,
        language: str = DEFAULT_LANGUAGE,
        **kwargs,
    ) -> Tuple[List[Segment], SimpleNamespace]:
        """Transcribe one buffer. See MLXWhisper.transcribe for the contract."""
        _ = kwargs  # beam_size/vad_filter are faster-whisper's, not this model's
        model = self._loaded()
        wanted = _QWEN_LANGUAGE.get((language or DEFAULT_LANGUAGE).lower(), language)
        out = model.generate(audio, language=wanted)

        raw = list(out.segments or [])
        if raw:
            segments = [
                Segment(
                    text=str(s.get("text", "")).strip(),
                    start=float(s.get("start", 0.0)),
                    end=float(s.get("end", 0.0)),
                )
                for s in raw
            ]
        else:
            # No timestamps: keep the text, but end=0 will not trip the
            # engine's near-window-end flush, so punctuation must carry it.
            segments = [
                Segment(text=str(getattr(out, "text", "")).strip(), start=0.0, end=0.0)
            ]

        info = SimpleNamespace(
            language=language or DEFAULT_LANGUAGE,
            language_probability=0.0,
            duration=0.0,
        )
        return segments, info


def make_model(model: str):
    """The GPU adapter for `model`, chosen by repo family."""
    repo = repo_for(model)
    if model in GPU_MODELS or "Qwen3" in repo:
        return MLXQwen3(model)
    return MLXWhisper(model)


def describe(model: str) -> Optional[str]:
    """One line naming the repo that would be loaded, for logs and UI."""
    return repo_for(model) if is_gpu_model(model) else None
