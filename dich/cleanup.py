"""
Vietnamese disfluency cleanup.

A faithful speech transcript keeps the mechanics of talking, not just the
words: hesitation sounds ("ừm", "ờ"), immediate stutters ("cái cái"), repeated
phrases and false starts. Whisper transcribes them all, so they arrive in the
transcript verbatim and then get translated verbatim. This module strips that
noise with plain rules — no model, no network — so cleanup is instant and works
even while the machine is offline, and the translation that follows reads as the
speaker meant it to.

The single biggest real-world artefact is not a stutter but a **loop**: on
meeting-room audio Whisper sometimes gets stuck and emits the same short phrase
over and over ("Tuy nhiên là" twenty times, "Thì anh" nine times, a company name
a dozen times). Real recordings make this plain, so the repeat rule works on
three levels: an immediate run of words repeats inside a line, a whole line
repeats the one directly above it, and a short line recurs with a different
fragment wedged between copies (the last is the same loop, just not consecutive).

The rules are deliberately conservative. A word is removed only when it is a
known hesitation sound or an exact immediate repetition, and Vietnamese
reduplication ("người người" = everyone) is spared. Punctuation is part of a
word for comparison, so a deliberate repeat kept apart by a comma
("cảm ơn, cảm ơn") does not collapse. The module is pure: the same input always
yields the same output, so it is safe to run twice and easy to test. The
ambiguity that rules cannot resolve — a false start that only makes sense with
the sentence around it — is left to the optional external-LLM rewrite.
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

# Vietnamese reduplication is meaningful: "người người" (everyone), "từ từ"
# (slowly), "chung chung" (vaguely). An exact repeat of one of these is kept even
# though it looks like a stutter. This is a curated allowlist of the reduplicative
# forms (từ láy) a meeting transcript actually meets; it cannot be exhaustive, so
# it errs toward keeping a repeat when the doubled form is a real word.
REDUPLICATIVES = frozenset(
    {
        # people, time and place
        "người",
        "ngày",
        "đêm",
        "nhà",
        "ai",
        "lúc",
        "khi",
        "sáng",
        "chiều",
        "tối",
        # degree and manner adverbs
        "từ",
        "chung",
        "dần",
        "lâu",
        "nhanh",
        "chậm",
        "mau",
        "nhẹ",
        "khẽ",
        "xa",
        "gần",
        # feelings, and the wh- words
        "vui",
        "buồn",
        "nào",
        "đâu",
        "gì",
        "sao",
        "thôi",
    }
)

# A word: anything that is not whitespace. Vietnamese letters are word
# characters under re.UNICODE, so a plain \w+ already spans "người".
_WORD = re.compile(r"\w+", re.UNICODE)
# How many words a repeated block may span. A stutter repeats one word; a
# Whisper loop repeats a short phrase ("thế nên cái lúc này"). Ten covers every
# loop seen in real meeting audio without scanning whole paragraphs per word.
MAX_REPEAT_BLOCK = 10

# A line loop that is not consecutive: the model repeats a short phrase with a
# different fragment wedged between each copy. Only lines this short, recurring
# this often within this many lines, are treated as the same loop.
LOOP_WINDOW = 8
LOOP_MIN_REPEATS = 3
LOOP_MAX_WORDS = 4

# Punctuation that may cling to a word. Stripped before two words are compared,
# so "vô," and "vô" are the same word; kept when the word is emitted.
_PUNCTUATION = ",.!?…;:\"'“”‘’()[]"


def _bare(token: str) -> str:
    """A word reduced to what it means: no clinging punctuation, casefolded."""
    return token.strip(_PUNCTUATION).casefold()


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


def _collapse_once(tokens: list) -> list:
    """One greedy left-to-right pass; see `_collapse_repetitions`."""
    bare = [_bare(token) for token in tokens]
    out: list = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        size = min(MAX_REPEAT_BLOCK, (n - i) // 2)
        while size >= 1:
            block = bare[i : i + size]
            j = i + size
            while bare[j : j + size] == block:
                j += size
            repeats = (j - i) // size
            # Punctuation on the last word means the copies are held apart by a
            # comma rather than being a stutter, so demand one more repeat.
            punctuated = _bare(tokens[i + size - 1]) != tokens[i + size - 1].casefold()
            needed = 3 if punctuated else 2
            if repeats >= needed and not (size == 1 and bare[i] in REDUPLICATIVES):
                out.extend(tokens[i : i + size])
                i = j
                collapsed = True
                break
            size -= 1
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return out


def _collapse_repetitions(text: str) -> str:
    """
    Collapse an immediately repeated run of words to a single copy.

    "tôi tôi nghĩ" -> "tôi nghĩ", and likewise a whole phrase a Whisper loop
    repeats ("thế nên cái lúc này thế nên cái lúc này" -> one copy). Both are
    the same shape — a block of one to `MAX_REPEAT_BLOCK` words that follows
    itself. Words are compared with punctuation stripped, so a run does not break
    just because the last copy lacks a comma, but a repeat held apart by commas
    ("cảm ơn, cảm ơn") is treated as deliberate and must happen a third time
    before it collapses. A single word in REDUPLICATIVES is spared, because
    Vietnamese reduplication is meaningful.

    The pass repeats until it stops changing the text: the first pass prefers the
    longest block, so "a a a a" collapses to "a a" and the next pass finishes the
    job ("a"). A loop that repeats a short phrase still collapses in one pass.
    """
    tokens = text.split()
    if len(tokens) < 2:
        return text
    previous = None
    current = tokens
    while current != previous:
        previous = current
        current = _collapse_once(current)
    return " ".join(current)


def _dedupe_consecutive_lines(lines: list) -> list:
    """
    Collapse a repeated line, consecutive or interleaved with short fragments.

    This is where a Whisper loop is undone. Two shapes appear in real audio:

      * **Consecutive** — the same phrase segment after segment, one per line
        ("Vtp", "Vtp", "Vtp"). Always collapsed to one.
      * **Interleaved** — the model gets stuck on a short phrase but keeps
        emitting a different fragment between copies ("Tuy nhiên là", "Phải
        bán", "Tuy nhiên là", "Thấy", "Tuy nhiên là"). Consecutive matching never
        fires here. A short line (up to `LOOP_MAX_WORDS`) that already appears
        `LOOP_MIN_REPEATS - 1` times in the previous `LOOP_WINDOW` lines is
        treated as part of the loop and dropped; a longer line, or a phrase that
        only recurs now and then, is left alone, because it may be deliberate.

    Blank lines are kept and break a run, so paragraph structure survives.
    """
    out: list = []
    previous = ""
    for line in lines:
        key = line.strip().casefold()
        if key:
            if key == previous:
                continue
            if len(key.split()) <= LOOP_MAX_WORDS:
                recent = [kept.strip().casefold() for kept in out[-LOOP_WINDOW:]]
                if recent.count(key) >= LOOP_MIN_REPEATS - 1:
                    continue
        out.append(line)
        previous = key
    return out


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
    keeps its shape. A line that was nothing but filler disappears entirely, and
    a run of identical lines — a Whisper loop — collapses to one.
    """
    if not text:
        return ""
    lines = [_cleanup_line(line) for line in text.split("\n")]
    return "\n".join(_dedupe_consecutive_lines(lines)).strip()


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
