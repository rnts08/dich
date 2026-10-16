"""
Optional external-LLM rewrite.

The app's own pipeline is local and deterministic: nothing it translates leaves
the machine. This module is the deliberate exception. When the user turns it on
and invokes a rewrite, it sends one piece of text — the Vietnamese transcript or
its English translation — to an OpenAI-compatible chat endpoint so a larger
model can turn a literal, disfluent passage into clean prose without dropping
any of the meaning.

It speaks the `/chat/completions` shape, which is what OpenAI, Ollama, LM
Studio, llama.cpp's server, vLLM, OpenRouter and most hosted providers expose,
so the same settings work against a local model or a cloud one. Configuration
lives in config.json; this module never reads it directly, which keeps it a pure
function of the arguments and easy to test without a network.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# How long to wait for the endpoint. Local models can be slow on the first
# request while they load, so the default is generous.
DEFAULT_TIMEOUT = 120.0

TRANSCRIPT_INSTRUCTION = (
    "You are a careful transcript editor working in Vietnamese. Clean up the "
    "transcript you are given: remove stutters, repeated words, filler sounds "
    '("ừm", "ờ", "à"), false starts and other non-functional speech. Fix '
    "punctuation and capitalisation. Keep every piece of meaning, names, "
    "numbers and the speaker's intent — do not lose any context. Do NOT "
    "translate, summarise or add information that was not said. Reply with the "
    "cleaned Vietnamese text only, with no commentary."
)

TRANSLATION_INSTRUCTION = (
    "You are a careful editor of English prose. What you are given is an "
    "English translation of Vietnamese speech. Rewrite it so it reads as "
    "natural, idiomatic English, while preserving all meaning and detail "
    "exactly: do not add or omit anything, do not summarise, and do not "
    "translate it back. Fix disfluencies, repetitions, awkward phrasing and "
    "punctuation. Reply with the rewritten English text only, with no "
    "commentary."
)

_INSTRUCTIONS = {
    "transcript": TRANSCRIPT_INSTRUCTION,
    "translation": TRANSLATION_INSTRUCTION,
}


class LLMError(RuntimeError):
    """Anything that stops a rewrite: bad config or a bad response."""


def endpoint(base_url: str) -> str:
    """
    The chat-completions URL for a configured base URL.

    Accepts either a bare base ("https://api.openai.com/v1") or a full URL that
    already names the path, so a pasted endpoint works too.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise LLMError("No base URL is set (Settings → External LLM).")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def instruction_for(kind: str, override: Optional[str] = None) -> str:
    """The system prompt for a rewrite target, honouring a user override."""
    if override and override.strip():
        return override.strip()
    return _INSTRUCTIONS.get(kind, TRANSLATION_INSTRUCTION)


def build_payload(
    text: str,
    kind: str,
    model: str,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """The request body for an OpenAI-compatible chat completion."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": instruction_for(kind, instruction)},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
        "stream": False,
    }


def rewrite(
    text: str,
    *,
    kind: str = "translation",
    instruction: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    transport: Optional[Callable] = None,
) -> str:
    """
    Rewrite `text` with the configured chat endpoint and return its reply.

    Args:
        text: The transcript or translation to rewrite.
        kind: "transcript" (Keeps Vietnamese) or "translation" (keeps English).
        instruction: Overrides the built-in system prompt.
        config: Settings mapping; defaults to an empty mapping.
        transport: The POST callable, injectable so tests need no network.

    Raises:
        LLMError: on a missing setting, a transport failure or an unexpected
            response shape.
    """
    if not (text or "").strip():
        raise LLMError("There is nothing to rewrite.")

    cfg = config or {}
    base = str(cfg.get("llm_base_url") or "")
    model = str(cfg.get("llm_model") or "").strip()
    if not model:
        raise LLMError("No model is set (Settings → External LLM).")

    headers = {"Content-Type": "application/json"}
    key = str(cfg.get("llm_api_key") or "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # An explicit instruction wins; otherwise fall back to the one saved in
    # config, so a user who customised the prompt does not have to pass it in.
    if instruction is None:
        instruction = cfg.get("llm_instruction")
    payload = build_payload(text, kind, model, instruction)
    post = transport or httpx.post
    try:
        response = post(
            endpoint(base),
            json=payload,
            headers=headers,
            timeout=float(cfg.get("llm_timeout") or DEFAULT_TIMEOUT),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise LLMError(
            f"The endpoint returned {exc.response.status_code}: "
            f"{exc.response.text[:300]}"
        ) from exc
    except (httpx.HTTPError, OSError) as exc:
        raise LLMError(f"Could not reach the endpoint: {exc}") from exc

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError(f"The response was not a chat completion: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise LLMError("The endpoint returned an empty completion.")
    # Some providers wrap the text in a code fence; strip it so the panes do not
    # fill with backticks.
    return _strip_fence(content).strip()


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:-1] if len(lines) >= 2 else []
        return "\n".join(inner)
    return text


def check(
    *,
    config: Optional[Dict[str, Any]] = None,
    transport: Optional[Callable] = None,
) -> str:
    """Round-trip a tiny text so the Settings panel can prove a connection."""
    return rewrite(
        "Xin chào, đây là một thử nghiệm.",
        kind="transcript",
        config=config,
        transport=transport,
    )
