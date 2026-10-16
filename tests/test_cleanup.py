"""The deterministic disfluency cleaner, as pure functions.

A rules-only pass has no model to hide behind: every case here is a decision
about what counts as noise. The tests pin the conservative choices — filler
sounds between pauses go, but a word that merely contains a hesitation vowel
("giờ") and a meaningful Vietnamese reduplication ("người người") stay.
"""

from dich import cleanup


def test_whole_word_hesitations_are_removed():
    assert cleanup.cleanup("ừm tôi nghĩ vậy") == "tôi nghĩ vậy"
    assert cleanup.cleanup("hmm ờm không biết") == "không biết"


def test_hesitation_letters_inside_a_word_are_kept():
    # "giờ" contains "ờ", but \b keeps the needle out of the haystack.
    assert cleanup.cleanup("bây giờ chúng ta bắt đầu") == "bây giờ chúng ta bắt đầu"


def test_isolated_interjection_between_commas_is_removed():
    assert cleanup.cleanup("Vâng, ừ, tôi đồng ý") == "Vâng, tôi đồng ý"


def test_interjection_at_the_edges_of_a_clause_is_removed():
    assert cleanup.cleanup("Ừ, đúng rồi") == "đúng rồi"
    assert cleanup.cleanup("đúng rồi, ờ") == "đúng rồi"


def test_mid_clause_interjection_is_kept():
    # Not fenced by pauses, so it could be the "yes" sense of "ừ"; left alone.
    assert cleanup.cleanup("ừ đúng rồi") == "ừ đúng rồi"


def test_immediate_word_repetition_collapses():
    assert cleanup.cleanup("tôi tôi nghĩ") == "tôi nghĩ"
    assert cleanup.cleanup("rất rất tốt") == "rất tốt"


def test_reduplication_is_preserved():
    assert cleanup.cleanup("người người nhà nhà") == "người người nhà nhà"
    assert cleanup.cleanup("ngày ngày trôi qua") == "ngày ngày trôi qua"


def test_deliberate_repeat_after_punctuation_is_kept():
    assert cleanup.cleanup("cảm ơn, cảm ơn các bạn") == "cảm ơn, cảm ơn các bạn"


def test_spacing_and_punctuation_are_tidied():
    assert cleanup.cleanup("trời đẹp , , chúng ta đi") == "trời đẹp, chúng ta đi"


def test_a_line_of_only_filler_disappears():
    assert cleanup.cleanup("ừm") == ""
    assert cleanup.cleanup("  ừm  ") == ""


def test_text_with_nothing_to_clean_is_unchanged():
    text = "Xin chào các bạn. Hôm nay trời rất đẹp."
    assert cleanup.cleanup(text) == text


def test_blank_lines_between_paragraphs_survive():
    assert cleanup.cleanup("xin chào\n\nkết thúc") == "xin chào\n\nkết thúc"


def test_a_filler_only_line_leaves_a_blank_gap():
    assert cleanup.cleanup("xin chào\nừm\nkết thúc") == "xin chào\n\nkết thúc"


def test_empty_input_is_empty():
    assert cleanup.cleanup("") == ""
    assert cleanup.cleanup(None) == ""


# --- the report helpers the UI leans on -------------------------------------


def test_stats_count_removed_words():
    stats = cleanup.cleanup_stats("ừm tôi tôi nghĩ", "tôi nghĩ")
    assert stats["removed"] == 2
    assert stats["words_before"] == 4
    assert stats["words_after"] == 2
    assert stats["changed"] is True


def test_stats_are_never_negative():
    stats = cleanup.cleanup_stats("tôi nghĩ", "tôi nghĩ")
    assert stats["removed"] == 0
    assert stats["saved_chars"] == 0
    assert stats["changed"] is False


def test_report_includes_the_cleaned_text():
    report = cleanup.cleanup_report("ừm xin chào")
    assert report["cleaned"] == "xin chào"
    assert report["changed"] is True


def test_summarize_names_the_work():
    assert cleanup.summarize({"changed": False}) == "nothing to clean"
    assert cleanup.summarize({"changed": True, "removed": 1}) == "removed 1 filler"
    assert cleanup.summarize({"changed": True, "removed": 3}) == "removed 3 fillers"
    assert (
        cleanup.summarize({"changed": True, "removed": 0})
        == "cleaned up spacing and punctuation"
    )
