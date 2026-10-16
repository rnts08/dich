"""
Vietnamese disfluency cleanup.

A faithful speech transcript keeps the mechanics of talking, not just the
words: hesitation sounds ("ừm", "ờ"), immediate stutters ("cái cái"), repeated
phrases and false starts. Whisper transcribes them all, so they arrive in the
transcript verbatim and then get translated verbatim. This module strips that
noise with plain rules — no model, no network — so cleanup is instant and works
even while the machine is offline, and the translation that follows reads as the
speaker meant it to.

The rules are deliberately conservative. A word is removed only when it is a
known hesitation sound or an exact immediate repetition, and Vietnamese
reduplication ("người người" = everyone) is spared. The module is pure: the same
input always yields the same output, so it is safe to run twice and easy to
test. The ambiguity that rules cannot resolve — a false start that only makes
sense with the sentence around it — is left to the optional external-LLM
rewrite.
"""

from __future__ import annotations

import re
from typing import Dict

# Hesitation and non-lexical sounds, removed wherever they stand as a whole
# word. These are never content words in Vietnamese, so removing them cannot
# change the meaning of a sentence. The English forms are included because
# Whisper occasionally emits them for Vietnamese speech.
HESITATIONS = frozenset(
    {
        "ừm",
        "ờm",
        "ưm",
        "hừm",
        "hửm",
        "hmm",
        "hm",
        "mmm",
        "mm",
        "uhm",
        "uh",
        "um",
        "erm",
        "er",
    }
)

# Interjections that are only removed when they stand alone between pauses. Mid
# clause they can carry meaning — "ừ" is also "yes", "à" is also "oh/and" — so
# they are kept there. A filler wedged between commas ("vâng, ừ, tôi…") is the
# case this catches.
ISOLATED_HESITATIONS = frozenset({"ừ", "ờ", "à", "hả", "hử", "nhỉ", "nhé", "nha"})

# Vietnamese reduplication is meaningful: "người người" (everyone), "ngày ngày"
# (day after day). An exact repeat of one of these is kept even though it looks
# like a stutter.
REDUPLICATIVES = frozenset(
    {
        "người",
        "ngày",
        "đêm",
        "nhà",
        "ai",
        "nào",
        "đâu",
        "gì",
        "sao",
        "lúc",
        "khi",
        "thôi",
        "vui",
        "buồn",
    }
)

# A word: anything that is not whitespace. Vietnamese letters are word
# characters under re.UNICODE, so a plain \w+ already spans "người".
_WORD = re.compile(r"\w+", re.UNICODE)
_HESITATION_RE = {
    word: re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE) for word in HESITATIONS
}


def _build_isolated_pattern() -> re.Pattern:
    alt = "|".join(
        re.escape(w) for w in sorted(ISOLATED_HESITATIONS, key=len, reverse=True)
    )
    # Match a filler that is fenced by pauses: preceded by the start of the
    # string or a punctuation mark, and followed by punctuation or the end. The
    # lookahead keeps the following mark; the leading mark is captured and
    # re-emitted so only the filler disappears.
    return re.compile(
        rf"(?P<pre>(?:[,\;:.!?…\n]\s*|\A))(?P<word>{alt})(?=\s*(?:[,\;:.!?…\n]|\Z))",
        re.IGNORECASE | re.UNICODE,
    )


_ISOLATED_RE = _build_isolated_pattern()


def _remove_hesitations(text: str) -> str:
    for pattern in _HESITATION_RE.values():
        text = pattern.sub(" ", text)
    return _ISOLATED_RE.sub(lambda m: m.group("pre"), text)


def _collapse_repetitions(text: str) -> str:
    """
    Collapse an immediately repeated word to one copy.

    "tôi tôi nghĩ" -> "tôi nghĩ", "rất rất tốt" -> "rất tốt". A word in
    REDUPLICATIVES is left alone, and a repeat separated by anything other than
    whitespace ("cảm ơn, cảm ơn") is treated as deliberate.
    """

    def repl(match: re.Match) -> str:
        word = match.group("word")
        if word.lower() in REDUPLICATIVES:
            return match.group(0)
        return word

    return re.sub(
        r"\b(?P<word>\w+)(?:\s+\1\b)+",
        repl,
        text,
        flags=re.IGNORECASE | re.UNICODE,
    )


def _tidy(text: str) -> str:
    """Tidy the spans and punctuation the removals leave behind."""
    text = re.sub(r"[^\S\n]+", " ", text)  # collapse runs of spaces/tabs
    text = re.sub(r"\s+([,;:!?…])", r"\1", text)  # no space before punctuation
    text = re.sub(r"([,;:])(?:\s*\1)+", r"\1", text)  # ",," -> ","
    text = re.sub(r",\s*(?=[,;:])", "", text)  # ", ," -> ","
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"^[\s,;:]+", "", text)  # leading punctuation left by removal
    text = re.sub(r"[\s,;:]+$", "", text)  # trailing punctuation left behind
    return text.strip()


def _cleanup_line(line: str) -> str:
    if not line.strip():
        return ""
    line = _remove_hesitations(line)
    line = _collapse_repetitions(line)
    return _tidy(line)


def cleanup(text: str) -> str:
    """
    Remove stutters, repetitions and filler sounds from a Vietnamese transcript.

    Whitespace and blank lines are preserved, so a multi-paragraph transcript
    keeps its shape. A line that was nothing but filler disappears entirely.
    """
    if not text:
        return ""
    return "\n".join(_cleanup_line(line) for line in text.split("\n")).strip()


def cleanup_words(text: str) -> int:
    """Word count, for the before/after figures the UI reports."""
    return len(_WORD.findall(text or ""))


def cleanup_stats(original: str, cleaned: str) -> Dict[str, object]:
    """
    Describe what a cleanup changed, for the status line.

    `removed` counts words that disappeared, `saved` counts characters; both are
    clamped at zero so a rule that only re-spaces text cannot report a negative
    removal.
    """
    before_words = cleanup_words(original)
    after_words = cleanup_words(cleaned)
    return {
        "removed": max(0, before_words - after_words),
        "words_before": before_words,
        "words_after": after_words,
        "saved_chars": max(0, len(original) - len(cleaned)),
        "changed": original.strip() != cleaned.strip(),
    }


def cleanup_report(original: str) -> Dict[str, object]:
    """Convenience: clean once and describe the change."""
    cleaned = cleanup(original)
    stats = cleanup_stats(original, cleaned)
    stats["cleaned"] = cleaned
    return stats


def summarize(stats: Dict[str, object]) -> str:
    """A one-line, human description of a cleanup result."""
    removed = int(stats.get("removed", 0) or 0)
    if not stats.get("changed"):
        return "nothing to clean"
    if removed <= 0:
        return "cleaned up spacing and punctuation"
    return f"removed {removed} filler{'s' if removed != 1 else ''}"
