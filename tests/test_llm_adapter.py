"""Chat Completions client: bearer header, tool parsing, and redacted logs."""

import json
import urllib.request

from agent_engine.adapters.llm.openai_chat import (
    CannedTransport,
    OpenAIChatLlm,
    UrllibTransport,
    completion_body,
)
from agent_engine.domain.models import ToolCall


class _Capture:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.url = ""
        self.headers: dict[str, str] = {}
        self.body = b""
        self.timeout_s = 0.0

    def post(self, url: str, headers: dict[str, str], body: bytes, timeout_s: float) -> bytes:
        self.url = url
        self.headers = headers
        self.body = body
        self.timeout_s = timeout_s
        return json.dumps(self.payload).encode("utf-8")


def test_llm_posts_chat_completions_with_bearer_and_parses_tool_call() -> None:
    """The client POSTs /chat/completions with a bearer token and parses tool_calls."""
    secret = "super-secret-token"
    arguments = {"path": "adder.py", "note": "object-args"}
    payload = completion_body([("read_file", json.dumps({"path": "adder.py"}))])
    message = payload["choices"][0]["message"]  # type: ignore[index]
    message["tool_calls"].append(  # type: ignore[index]
        {
            "id": "extra",
            "type": "function",
            "function": {"name": "run_checks", "arguments": {"unused": True}},
        }
    )
    transport = _Capture(payload)
    llm = OpenAIChatLlm("https://example.test/v1/", secret, "demo-model", transport)
    calls = llm.complete([{"role": "user", "content": "fix"}])
    assert transport.url == "https://example.test/v1/chat/completions"
    assert transport.headers["Authorization"] == f"Bearer {secret}"
    assert transport.timeout_s == 60
    body = json.loads(transport.body)
    assert body["model"] == "demo-model"
    names = [item["function"]["name"] for item in body["tools"]]
    assert names == [
        "read_file",
        "replace_text",
        "run_checks",
        "finish",
        "list_files",
        "search_text",
        "remember",
    ]
    assert calls[0] == ToolCall("read_file", json.dumps({"path": "adder.py"}), "canned-0")
    assert calls[1].name == "run_checks"
    assert json.loads(calls[1].arguments_json) == {"unused": True}
    assert llm.post_count() == 1


def test_llm_log_redacts_bearer_token() -> None:
    """The request log keeps the URL and replaces the bearer token."""
    secret = "example.test"
    transport = _Capture(completion_body([("finish", "{}")]))
    llm = OpenAIChatLlm("https://example.test/v1", secret, "demo-model", transport)
    llm.complete([{"role": "user", "content": secret}])
    assert llm.logs
    assert secret not in llm.logs[0]
    assert "Bearer [redacted]" in llm.logs[0]
    assert "chat/completions" in llm.logs[0]


def test_llm_rejects_a_body_without_a_message() -> None:
    """A response that is not a chat completion raises ValueError."""
    llm = OpenAIChatLlm(
        "https://example.test/v1",
        "token",
        "demo-model",
        _Capture({"choices": []}),
    )
    try:
        llm.complete([])
    except ValueError as exc:
        assert "message" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_canned_transport_records_the_request_and_pops_a_body() -> None:
    """The canned transport returns scripted JSON and does not open a socket."""
    body = completion_body([("finish", "{}")])
    transport = CannedTransport([body])
    raw = transport.post("http://canned.invalid/chat/completions", {"Authorization": "x"}, b"{}", 60)
    assert json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "finish"
    assert transport.requests[0][0].endswith("/chat/completions")
    try:
        CannedTransport([]).post("http://canned.invalid/chat/completions", {}, b"", 1)
    except ValueError as exc:
        assert "no body" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_urllib_transport_posts_with_the_timeout(monkeypatch) -> None:
    """UrllibTransport passes the socket timeout through to urlopen."""

    class _Response:
        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    seen: dict[str, object] = {}

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["data"] = request.data
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    raw = UrllibTransport().post("https://example.test/v1/chat/completions", {}, b"{}", 60)
    assert raw == b"{}"
    assert seen["timeout"] == 60
    assert seen["url"] == "https://example.test/v1/chat/completions"
