"""External-LLM rewrite: request shape and failure modes, with no network.

This is the one path that sends text off the machine, so the request is pinned
exactly — endpoint, auth header, model and the system instruction that keeps the
source language intact — and every failure becomes an LLMError the UI can show
rather than a raw httpx exception.
"""

import httpx
import pytest

from dich import llm


class FakeResponse:
    """The slice of httpx.Response the module actually touches."""

    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=self)

    def json(self):
        return self._payload


def _reply(content):
    return {"choices": [{"message": {"content": content}}]}


def _config(**over):
    base = {"llm_base_url": "http://localhost:1234/v1", "llm_model": "some-model"}
    base.update(over)
    return base


# --- endpoint shaping --------------------------------------------------------


def test_endpoint_appends_the_completions_path():
    assert (
        llm.endpoint("https://api.openai.com/v1")
        == "https://api.openai.com/v1/chat/completions"
    )
    assert (
        llm.endpoint("http://localhost:11434/v1/")
        == "http://localhost:11434/v1/chat/completions"
    )


def test_endpoint_accepts_a_full_url():
    url = "https://example.test/v1/chat/completions"
    assert llm.endpoint(url) == url


def test_endpoint_rejects_a_blank_base():
    with pytest.raises(llm.LLMError):
        llm.endpoint("")


# --- the request itself ------------------------------------------------------


def test_rewrite_sends_model_instruction_and_text():
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen.update(url=url, json=json, headers=headers, timeout=timeout)
        return FakeResponse(_reply("Đã sửa lại."))

    out = llm.rewrite(
        "ừm xin chào",
        kind="transcript",
        config=_config(llm_api_key="k"),
        transport=post,
    )

    assert out == "Đã sửa lại."
    assert seen["url"] == "http://localhost:1234/v1/chat/completions"
    assert seen["json"]["model"] == "some-model"
    assert seen["json"]["stream"] is False
    assert seen["headers"]["Authorization"] == "Bearer k"
    system, user = seen["json"]["messages"]
    assert system["role"] == "system"
    assert "Vietnamese" in system["content"]
    assert user == {"role": "user", "content": "ừm xin chào"}


def test_no_api_key_sends_no_authorization_header():
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen["headers"] = headers
        return FakeResponse(_reply("ok"))

    llm.rewrite("x", config=_config(), transport=post)
    assert "Authorization" not in seen["headers"]


def test_translation_kind_asks_for_english():
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen["json"] = json
        return FakeResponse(_reply("ok"))

    llm.rewrite("Hello", kind="translation", config=_config(), transport=post)
    assert "English" in seen["json"]["messages"][0]["content"]


def test_a_custom_instruction_overrides_the_default():
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen["json"] = json
        return FakeResponse(_reply("ok"))

    llm.rewrite("x", instruction="  ONLY THIS  ", config=_config(), transport=post)
    assert seen["json"]["messages"][0]["content"] == "ONLY THIS"


def test_a_configured_instruction_is_used_when_none_is_passed():
    seen = {}

    def post(url, json=None, headers=None, timeout=None):
        seen["json"] = json
        return FakeResponse(_reply("ok"))

    llm.rewrite("x", config=_config(llm_instruction="FROM CONFIG"), transport=post)
    assert seen["json"]["messages"][0]["content"] == "FROM CONFIG"


# --- failures become LLMError ------------------------------------------------


def test_blank_text_is_refused():
    with pytest.raises(llm.LLMError):
        llm.rewrite("   ", config=_config())


def test_missing_base_url_is_refused():
    with pytest.raises(llm.LLMError):
        llm.rewrite("x", config={"llm_model": "m"}, transport=lambda *a, **k: None)


def test_missing_model_is_refused():
    with pytest.raises(llm.LLMError):
        llm.rewrite("x", config={"llm_base_url": "http://x/v1"})


def test_http_error_becomes_llm_error():
    def post(url, json=None, headers=None, timeout=None):
        return FakeResponse({}, status=401, text="unauthorized")

    with pytest.raises(llm.LLMError) as exc:
        llm.rewrite("x", config=_config(), transport=post)
    assert "401" in str(exc.value)


def test_connection_error_becomes_llm_error():
    def post(url, json=None, headers=None, timeout=None):
        raise httpx.ConnectError("no route to host")

    with pytest.raises(llm.LLMError):
        llm.rewrite("x", config=_config(), transport=post)


def test_unexpected_response_shape_becomes_llm_error():
    def post(url, json=None, headers=None, timeout=None):
        return FakeResponse({"nope": True})

    with pytest.raises(llm.LLMError):
        llm.rewrite("x", config=_config(), transport=post)


def test_empty_completion_is_refused():
    def post(url, json=None, headers=None, timeout=None):
        return FakeResponse(_reply("   "))

    with pytest.raises(llm.LLMError):
        llm.rewrite("x", config=_config(), transport=post)


# --- output tidying ----------------------------------------------------------


def test_code_fences_are_stripped():
    def post(url, json=None, headers=None, timeout=None):
        return FakeResponse(_reply("```\ncleaned text\n```"))

    out = llm.rewrite("x", config=_config(), transport=post)
    assert out == "cleaned text"


# --- availability gate -------------------------------------------------------


def test_llm_available_needs_the_switch_and_the_settings(backend, monkeypatch):
    monkeypatch.setitem(backend.CONFIG, "llm_enabled", True)
    monkeypatch.setitem(backend.CONFIG, "llm_base_url", "http://x/v1")
    monkeypatch.setitem(backend.CONFIG, "llm_model", "m")
    assert backend.llm_available() is True

    monkeypatch.setitem(backend.CONFIG, "llm_enabled", False)
    assert backend.llm_available() is False

    monkeypatch.setitem(backend.CONFIG, "llm_enabled", True)
    monkeypatch.setitem(backend.CONFIG, "llm_model", "")
    assert backend.llm_available() is False
