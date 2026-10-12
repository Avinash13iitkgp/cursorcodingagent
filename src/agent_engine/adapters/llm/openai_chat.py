"""OpenAI-compatible Chat Completions.

complete POSTs {base_url}/chat/completions with Authorization: Bearer and the
tools read_file, replace_text, run_checks, and finish. The socket timeout is
60 seconds. The request log stores Bearer [redacted] and never the token.
Parsing reads tool_calls off the first choice. This module does not choose
an MCP server.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Protocol

from agent_engine.domain.models import ToolCall

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file in the work tree.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": "Replace one snippet when the preimage hash matches.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "expected_sha256", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_checks",
            "description": "Compile and test with the operator command list.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Ask the verifier to accept or reject the task.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class Transport(Protocol):
    """HTTP POST used by OpenAIChatLlm. Tests pass a canned body."""

    def post(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> bytes:
        """Send one request.

        Args:
            url: Absolute chat completions URL.
            headers: Request headers, including the bearer token.
            body: JSON body.
            timeout_s: Socket timeout in seconds.

        Returns:
            The response body.
        """
        ...  # pragma: no cover


class UrllibTransport:
    """Stdlib POST. The product worker and the live path use this class."""

    def post(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> bytes:
        """POST body and return the response bytes.

        Args:
            url: Absolute chat completions URL.
            headers: Request headers.
            body: JSON body.
            timeout_s: Socket timeout passed to urlopen.

        Returns:
            The response body.

        Raises:
            urllib.error.URLError: The socket failed or the status is an error.
        """
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()


class CannedTransport:
    """Return scripted chat bodies and record posts. It does not use the network."""

    def __init__(self, bodies: list[dict[str, object]]) -> None:
        """Store the response sequence.

        Args:
            bodies: Chat completion objects, consumed in order.
        """
        self._bodies = list(bodies)
        self.requests: list[tuple[str, dict[str, str], bytes]] = []

    def post(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> bytes:
        """Pop the next canned body.

        Args:
            url: Recorded and otherwise ignored.
            headers: Recorded and otherwise ignored.
            body: Recorded and otherwise ignored.
            timeout_s: Ignored. Canned replies do not wait.

        Returns:
            The next body encoded as JSON.

        Raises:
            ValueError: The script has no body left.
        """
        del timeout_s
        self.requests.append((url, headers, body))
        if not self._bodies:
            raise ValueError("canned transport has no body left")
        return json.dumps(self._bodies.pop(0)).encode("utf-8")


class OpenAIChatLlm:
    """Chat Completions client. Logs redact the bearer token."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        transport: Transport,
        timeout_s: float = 60,
    ) -> None:
        """Bind the client to one endpoint.

        Args:
            base_url: Origin plus optional prefix, without the final path.
            api_key: Bearer token. Stored for the header and redacted in logs.
            model: Model name sent in the JSON body.
            transport: POST implementation. Live code passes UrllibTransport.
            timeout_s: Socket timeout. The default is 60 seconds.
        """
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._transport = transport
        self._timeout_s = timeout_s
        self._posts = 0
        self.logs: list[str] = []

    def post_count(self) -> int:
        """Return how many completions were requested.

        Returns:
            The number of complete calls on this object.
        """
        return self._posts

    def complete(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """POST one turn and parse tool calls.

        Args:
            messages: Transcript in Chat Completions shape.

        Returns:
            Tool calls from the first choice, in order.

        Raises:
            ValueError: The body is not a chat completion with a message.
        """
        self._posts += 1
        payload = {"model": self._model, "messages": messages, "tools": _TOOLS}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._log(headers, body)
        raw = self._transport.post(self._url, headers, body, self._timeout_s)
        return _parse_calls(raw)

    def _log(self, headers: dict[str, str], body: bytes) -> None:
        redacted = dict(headers)
        redacted["Authorization"] = "Bearer [redacted]"
        text = json.dumps({"url": self._url, "headers": redacted})
        if self._api_key and self._api_key in text:
            text = text.replace(self._api_key, "[redacted]")
        decoded = body.decode("utf-8")
        if self._api_key and self._api_key in decoded:
            decoded = decoded.replace(self._api_key, "[redacted]")
        self.logs.append(text + decoded)


def _parse_calls(raw: bytes) -> tuple[ToolCall, ...]:
    try:
        payload = json.loads(raw)
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("chat completion is missing a message") from exc
    calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:
        function = item["function"]
        arguments = function["arguments"]
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(ToolCall(str(function["name"]), arguments, str(item.get("id", ""))))
    return tuple(calls)


def completion_body(calls: list[tuple[str, str]]) -> dict[str, object]:
    """Build one canned chat completion for the given tool calls.

    Args:
        calls: Pairs of tool name and arguments JSON.

    Returns:
        A chat completion object with those tool_calls on the first choice.
    """
    tool_calls = []
    for index, (name, arguments) in enumerate(calls):
        tool_calls.append(
            {
                "id": f"canned-{index}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                }
            }
        ]
    }
