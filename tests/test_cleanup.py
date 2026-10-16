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


def test_reduplication_found_in_real_meeting_audio_is_preserved():
    # "từ từ" (slowly) and "chung chung" (vaguely) are words, not stutters.
    assert cleanup.cleanup("từ từ như một") == "từ từ như một"
    assert cleanup.cleanup("nó là chung chung") == "nó là chung chung"


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


# --- repetition and loop shapes seen in real meeting audio ------------------


def test_an_immediately_repeated_phrase_collapses():
    # A Whisper loop inside one segment, not just a one-word stutter.
    assert (
        cleanup.cleanup("thế nên cái lúc này thế nên cái lúc này")
        == "thế nên cái lúc này"
    )


def test_a_one_word_loop_on_one_line_collapses_to_one():
    assert cleanup.cleanup("Vtp Vtp Vtp Vtp") == "Vtp"
    assert cleanup.cleanup("tôi tôi tôi tôi nghĩ") == "tôi nghĩ"


def test_consecutive_identical_lines_collapse_to_one():
    # The dominant real-world artefact: one segment per line, all identical.
    assert cleanup.cleanup("Thì anh\nThì anh\nThì anh") == "Thì anh"
    assert cleanup.cleanup("Vtp\nVtp\nVtp\nVtp\nVtp") == "Vtp"


def test_a_long_run_of_identical_lines_collapses():
    text = "\n".join(["Tuy nhiên là"] * 20)
    assert cleanup.cleanup(text) == "Tuy nhiên là"


def test_duplicate_lines_are_compared_case_insensitively():
    assert cleanup.cleanup("VTP\nVtp\nVTP") == "VTP"


def test_a_repeat_that_is_not_adjacent_survives():
    # Recurring later in the transcript may be deliberate, so it is kept.
    assert cleanup.cleanup("hai ngày\nvà rồi\nhai ngày") == "hai ngày\nvà rồi\nhai ngày"


def test_a_phrase_repeated_thrice_through_commas_collapses():
    # Three verbatim copies is a loop, even when commas separate them.
    assert cleanup.cleanup("1 có thể vô, 1 có thể vô, 1 có thể vô") == "1 có thể vô"


def test_missing_punctuation_on_the_last_copy_does_not_block_collapse():
    # The loop's final copy often loses its comma; the words are still equal.
    assert cleanup.cleanup("cái này, cái này, cái này") == "cái này"


def test_reduplicated_single_words_are_still_spared():
    assert cleanup.cleanup("người người nhà nhà") == "người người nhà nhà"
    assert cleanup.cleanup("ngày ngày\nngày ngày") == "ngày ngày"


def test_cleanup_is_idempotent():
    # The UI re-cleans the whole transcript on every pass, so cleaning twice must
    # equal cleaning once.
    text = "ừm Thì anh\nThì anh\n1 có thể vô, 1 có thể vô, 1 có thể vô"
    once = cleanup.cleanup(text)
    assert cleanup.cleanup(once) == once


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
