"""The prompt strategy switch: content list for TranslateGemma, system+string otherwise.

build_messages and _context_messages need no model load — the shape decision is
pure, and the bug class it guards (mlx-lm.server flattening the content list)
was only visible from the outside.
"""


def _gemma(backend, monkeypatch):
    monkeypatch.setattr(backend, "SOURCE_LANGUAGE", "Vietnamese")
    monkeypatch.setattr(backend, "TARGET_LANGUAGE", "English")
    return backend.LocalEngine("mlx-community/translategemma-4b-it-4bit")


def _qwen(backend, monkeypatch):
    monkeypatch.setattr(backend, "SOURCE_LANGUAGE", "Vietnamese")
    monkeypatch.setattr(backend, "TARGET_LANGUAGE", "English")
    return backend.LocalEngine("mlx-community/Qwen3-4B-Instruct-2507-4bit")


def test_translategemma_uses_a_content_list(backend, monkeypatch):
    messages = _gemma(backend, monkeypatch).build_messages("Xin chào")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["source_lang_code"] == "vi"
    assert content[0]["target_lang_code"] == "en"
    assert "Xin chào" in content[0]["text"]


def test_translategemma_text_carries_the_markdown_request(backend, monkeypatch):
    messages = _gemma(backend, monkeypatch).build_messages("Xin chào")
    assert "markdown" in messages[0]["content"][0]["text"].lower()


def test_markdown_request_is_not_duplicated(backend, monkeypatch):
    chunk = "Xin chào. " + backend.MARKDOWN_INSTRUCTION.strip()
    text = _gemma(backend, monkeypatch).build_messages(chunk)[0]["content"][0]["text"]
    assert text.count("Format the translation as Markdown") == 1


def test_chat_style_uses_system_turn_and_plain_string(backend, monkeypatch):
    messages = _qwen(backend, monkeypatch).build_messages("Xin chào")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert isinstance(messages[1]["content"], str)
    assert messages[1]["content"] == "Xin chào"
    assert "Vietnamese" in messages[0]["content"]
    assert "tiếng Việt" in messages[0]["content"]


def test_lang_code_falls_back_to_two_letters(backend):
    assert backend._lang_code("Vietnamese") == "vi"
    assert backend._lang_code("Klingon") == "kl"


def test_profile_override_only_applies_to_the_configured_model(backend, monkeypatch):
    monkeypatch.setattr(backend, "CONFIG", {**backend.CONFIG, "prompt_style": "chat"})
    monkeypatch.setattr(backend, "MODEL_NAME", "some/model")
    # The configured model picks up the override...
    assert backend.get_profile("some/model")["prompt_style"] == "chat"
    # ...while a TranslateGemma model that is not configured keeps its own style.
    assert (backend.get_profile("mlx-community/translategemma-4b-it-4bit")
            ["prompt_style"] == "translategemma")


def test_context_messages_keep_only_the_last_three_turns(backend, monkeypatch):
    _gemma(backend, monkeypatch)          # sets the language globals
    history = [(f"câu {i}", f"line {i}") for i in range(5)]
    messages = backend._context_messages("câu mới", history, "translategemma")

    assert [m["role"] for m in messages] == [
        "user", "assistant", "user", "assistant", "user", "assistant", "user",
    ]
    user_texts = [
        turn["content"][0]["text"]
        for turn in messages if turn["role"] == "user"
    ]
    assert user_texts[0] == "câu 2"       # CONTEXT_TURNS = 3, oldest dropped
    assert user_texts[-1] == "câu mới"


def test_context_messages_chat_style_keeps_plain_strings(backend, monkeypatch):
    _qwen(backend, monkeypatch)           # sets the language globals
    messages = backend._context_messages(
        "câu mới", [("câu một", "line one")], "chat")
    assert messages[0] == {"role": "user", "content": "câu một"}
    assert messages[1] == {"role": "assistant", "content": "line one"}
    assert messages[2] == {"role": "user", "content": "câu mới"}
