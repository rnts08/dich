"""Offline quality metrics vs a reference (gold standard)."""

import pytest

from dich import quality


def test_identical_text_scores_perfect():
    text = "Tôi đang học tiếng Việt mỗi ngày."
    result = quality.score(text, text)
    assert result["bleu"] == 1.0
    assert result["chrf"] == 1.0
    assert result["rouge_l"] == 1.0
    assert result["quality"] == 100.0


def test_empty_candidate_scores_zero():
    result = quality.score("xin chào", "")
    assert result["bleu"] == 0.0
    assert result["chrf"] == 0.0
    assert result["rouge_l"] == 0.0
    assert result["quality"] == 0.0


def test_empty_reference_scores_zero():
    result = quality.score("", "xin chào")
    assert result["quality"] == 0.0


def test_unrelated_text_scores_low():
    result = quality.score("tôi yêu bạn", "the weather is nice today")
    assert 0.0 < result["quality"] < 40.0  # some char overlap, but low


def test_partial_overlap_is_between_zero_and_one():
    # Same words plus one extra: not perfect, but clearly not zero.
    result = quality.score("xin chào các bạn", "xin chào các bạn thân")
    assert 0.0 < result["quality"] < 100.0


def test_chrf_is_sensitive_to_characters():
    assert quality.chrf("abc", "abc") == 1.0
    assert quality.chrf("abc", "def") == 0.0


def test_rouge_l_rewards_subsequence_overlap():
    # LCS of "abcde" and "ace" is "ace": precision 3/3, recall 3/5
    assert quality.rouge_l("abcde", "ace") == pytest.approx(0.75)


def test_bleu_brevity_penalty_penalizes_short_candidates():
    full = "the cat is on the mat"
    short = "the cat"
    assert quality.bleu(full, short) < quality.bleu(full, full)
