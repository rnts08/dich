"""Offline translation quality metrics vs a reference (gold standard).

No API keys are needed: BLEU, a chrF-style character n-gram F1 and ROUGE-L are
all computed locally from the candidate translation and a reference translation.
All metrics run from 0.0 (no overlap) to 1.0 (identical).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List


def _tokens(text: str) -> List[str]:
    return text.split()


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def bleu(reference: str, candidate: str, max_order: int = 4) -> float:
    """BLEU with smoothing, 0..1.

    Geometric mean of clipped n-gram precisions times the brevity penalty.
    Returns 0 when the candidate is empty or shares no n-gram with the
    reference.
    """
    ref_tokens = _tokens(reference)
    cand_tokens = _tokens(candidate)
    if not cand_tokens:
        return 0.0

    precisions: List[float] = []
    for n in range(1, max_order + 1):
        ref_ngrams = Counter(_ngrams(ref_tokens, n))
        cand_ngrams = Counter(_ngrams(cand_tokens, n))
        overlap = sum((cand_ngrams & ref_ngrams).values())
        total = max(1, sum(cand_ngrams.values()))
        precisions.append(overlap / total)

    if not any(precisions):
        return 0.0
    eps = 1e-9
    geo = math.exp(sum(math.log(max(p, eps)) for p in precisions) / len(precisions))

    r, c = len(ref_tokens), len(cand_tokens)
    bp = 1.0 if c >= r else math.exp(1 - r / c)
    return bp * geo


def _char_ngrams(text: str, n: int) -> List[str]:
    if len(text) < n:
        return []
    return [text[i : i + n] for i in range(len(text) - n + 1)]


def chrf(reference: str, candidate: str, max_order: int = 6) -> float:
    """Character n-gram F1 (chrF-style), 0..1.

    Averages precision and recall over character n-grams of order 1..max_order,
    then combines them with the harmonic mean.
    """
    if not candidate:
        return 0.0

    total_prec = total_rec = 0.0
    orders = 0
    for n in range(1, max_order + 1):
        ref_ng = Counter(_char_ngrams(reference, n))
        cand_ng = Counter(_char_ngrams(candidate, n))
        if not cand_ng:
            continue
        overlap = sum((cand_ng & ref_ng).values())
        total_prec += overlap / sum(cand_ng.values())
        total_rec += overlap / max(1, sum(ref_ng.values()))
        orders += 1

    if not orders:
        return 0.0
    precision = total_prec / orders
    recall = total_rec / orders
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_len(a: str, b: str) -> int:
    """Length of the longest common subsequence (space-optimised DP)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j] + 1)
            else:
                cur.append(max(cur[-1], prev[j]))
        prev = cur
    return prev[-1]


def rouge_l(reference: str, candidate: str) -> float:
    """ROUGE-L (longest common subsequence F1), 0..1."""
    if not candidate or not reference:
        return 0.0
    lcs = _lcs_len(reference, candidate)
    precision = lcs / len(candidate)
    recall = lcs / len(reference)
    if not precision or not recall:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score(reference: str, candidate: str) -> Dict[str, float]:
    """All metrics plus a 0-100 composite quality score.

    The composite is the plain average of BLEU, chrF and ROUGE-L, so a perfect
    match scores 100 and unrelated text scores 0.
    """
    b = bleu(reference, candidate)
    f = chrf(reference, candidate)
    r = rouge_l(reference, candidate)
    return {
        "bleu": round(b, 4),
        "chrf": round(f, 4),
        "rouge_l": round(r, 4),
        "quality": round((b + f + r) / 3 * 100, 1),
    }
