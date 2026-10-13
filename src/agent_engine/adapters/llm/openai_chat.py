"""OpenAI-compatible Chat Completions.

complete POSTs {base_url}/chat/completions with Authorization: Bearer and the
tools read_file, replace_text, write_file, run_checks, and finish. A caller
may append tools it already discovered. plan POSTs the same endpoint with
submit_plan as the only tool, so a planning turn cannot answer with a file
tool. When a text listener is set and the transport can stream, the request
sets stream and reasoning tokens are handed to that listener as they arrive.
The socket timeout is 60 seconds. The request log stores Bearer [redacted]
and never the token. Parsing reads tool_calls off the first choice. This
module does not choose an MCP server or its argv.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from collections.abc import Callable, Iterator
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
            "name": "write_file",
            "description": "Create one new file, with its parent directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
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
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List relative paths under the work root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_text",
            "description": "Find a literal substring under the work root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store one knowledge note. It does not write the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "quote": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["path", "quote", "note"],
            },
        },
    },
]

_LOOK_NAMES = ("list_files", "search_text", "read_file")
_LOOK_TOOLS = [
    tool for tool in _TOOLS if tool["function"]["name"] in _LOOK_NAMES  # type: ignore[index]
]

_PLAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_plan",
            "description": "Submit the execution plan for this prompt. It writes nothing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "approach": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "detail": {"type": "string"},
                            },
                            "required": ["title", "detail"],
                        },
                    },
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "action": {"type": "string", "enum": ["create", "edit"]},
                                "detail": {"type": "string"},
                            },
                            "required": ["path", "action", "detail"],
                        },
                    },
                },
                "required": ["summary", "approach", "steps", "files"],
            },
        },
    }
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

    def __init__(self) -> None:
        """Remember the in-flight response so close can drop the socket."""
        self._lock = threading.Lock()
        self._response: object | None = None

    def close(self) -> None:
        """Close the in-flight response, if a POST is waiting on the socket.

        Returns:
            None.
        """
        with self._lock:
            response = self._response
        if response is not None:
            response.close()  # type: ignore[attr-defined]

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
        response = urllib.request.urlopen(request, timeout=timeout_s)
        with self._lock:
            self._response = response
        try:
            return response.read()
        finally:
            with self._lock:
                self._response = None
            closer = getattr(response, "close", None)
            if closer is not None:
                closer()

    def iter_lines(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> Iterator[str]:
        """POST body and yield each response line.

        Args:
            url: Absolute chat completions URL.
            headers: Request headers.
            body: JSON body. The caller sets stream on it.
            timeout_s: Socket timeout passed to urlopen.

        Returns:
            An iterator of decoded lines, including the line ending the
            server sent.

        Raises:
            urllib.error.URLError: The socket failed or the status is an error.
        """
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        response = urllib.request.urlopen(request, timeout=timeout_s)
        with self._lock:
            self._response = response
        try:
            while True:
                raw = response.readline()
                if not raw:
                    break
                yield raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        finally:
            with self._lock:
                self._response = None
            closer = getattr(response, "close", None)
            if closer is not None:
                closer()


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

    def iter_lines(
        self, url: str, headers: dict[str, str], body: bytes, timeout_s: float
    ) -> Iterator[str]:
        """Yield the next canned body as one SSE data line.

        Args:
            url: Recorded by post.
            headers: Recorded by post.
            body: Recorded by post.
            timeout_s: Ignored. Canned replies do not wait.

        Returns:
            The next body wrapped as ``data:``, then ``data: [DONE]``.
        """
        raw = self.post(url, headers, body, timeout_s).decode("utf-8")
        yield f"data: {raw}\n"
        yield "data: [DONE]\n"

    def close(self) -> None:
        """No socket to close. The canned transport does not dial out.

        Returns:
            None.
        """
        return None


class OpenAIChatLlm:
    """Chat Completions client. Logs redact the bearer token."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        transport: Transport,
        timeout_s: float = 60,
        extra_tools: list[dict[str, object]] | None = None,
        on_text: Callable[[str], None] | None = None,
        explore: bool = False,
    ) -> None:
        """Bind the client to one endpoint.

        Args:
            base_url: Origin plus optional prefix, without the final path.
            api_key: Bearer token. Stored for the header and redacted in logs.
            model: Model name sent in the JSON body.
            transport: POST implementation. Live code passes UrllibTransport.
            timeout_s: Socket timeout. The default is 60 seconds.
            extra_tools: Tools appended after the four built-ins. The caller
                discovered them. This client does not choose an MCP argv.
            on_text: Receives reasoning and assistant text. When set, and the
                transport has iter_lines, the request streams. None leaves the
                call on a single response body.
            explore: When true, plan also offers list_files, search_text, and
                read_file and does not force submit_plan. A new project leaves
                this false.
        """
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._api_key = api_key
        self._model = model
        self._transport = transport
        self._timeout_s = timeout_s
        self._extra_tools = extra_tools or []
        self._on_text = on_text
        self._explore = explore
        self._posts = 0
        self.logs: list[str] = []
        self.tool_choice: str | None = None

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
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "tools": [*_TOOLS, *self._extra_tools] if self._extra_tools else _TOOLS,
        }
        if self.tool_choice:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": self.tool_choice},
            }
        return self._post_turn(payload)

    def plan(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """POST one planning turn. An existing tree may be searched first.

        Args:
            messages: Planning prompt in Chat Completions shape.

        Returns:
            Tool calls from the first choice. write_file, replace_text,
            run_checks, and finish are not sent. explore adds list_files,
            search_text, and read_file, and requires some tool instead of
            forcing submit_plan.

        Raises:
            ValueError: The body is not a chat completion with a message.
        """
        payload: dict[str, object] = {"model": self._model, "messages": messages}
        if self._explore:
            payload["tools"] = [*_LOOK_TOOLS, *_PLAN_TOOLS]
            payload["tool_choice"] = "required"
        else:
            payload["tools"] = _PLAN_TOOLS
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": "submit_plan"},
            }
        return self._post_turn(payload)

    def _post_turn(self, payload: dict[str, object]) -> tuple[ToolCall, ...]:
        self._posts += 1
        streamer = getattr(self._transport, "iter_lines", None)
        if self._on_text is not None and streamer is not None:
            payload = dict(payload)
            payload["stream"] = True
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._log(headers, body)
        if self._on_text is not None and streamer is not None:
            return _read_stream(
                streamer(self._url, headers, body, self._timeout_s), self._on_text
            )
        raw = self._transport.post(self._url, headers, body, self._timeout_s)
        text, calls = _parse_message(raw)
        if self._on_text is not None and text:
            self._on_text(text)
        return calls

    def summarize(self, messages: list[dict[str, object]]) -> str:
        """POST one tool-free completion and return the assistant text.

        Args:
            messages: Prompt to compress. The bearer token is redacted in the log.

        Returns:
            The assistant content. An empty content is an empty string.

        Raises:
            ValueError: The body is not a chat completion with a message.
        """
        self._posts += 1
        payload: dict[str, object] = {"model": self._model, "messages": messages}
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        self._log(headers, body)
        raw = self._transport.post(self._url, headers, body, self._timeout_s)
        try:
            message = json.loads(raw)["choices"][0]["message"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("chat completion is missing a message") from exc
        content = message.get("content")
        return "" if content is None else str(content)

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

    def close(self) -> None:
        """Close the transport socket when a lost lease must not start another POST.

        Returns:
            None.
        """
        closer = getattr(self._transport, "close", None)
        if closer is not None:
            closer()


def _parse_message(raw: bytes) -> tuple[str, tuple[ToolCall, ...]]:
    try:
        payload = json.loads(raw)
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("chat completion is missing a message") from exc
    if not isinstance(message, dict):
        raise ValueError("chat completion is missing a message")
    return _message_text(message), _calls_from_message(message)


def _message_text(message: dict[str, object]) -> str:
    parts: list[str] = []
    for key in ("reasoning_content", "content"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    return "\n".join(parts)


def _calls_from_message(message: dict[str, object]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for item in message.get("tool_calls") or []:  # type: ignore[union-attr]
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append(ToolCall(str(function.get("name", "")), arguments, str(item.get("id", ""))))
    return tuple(calls)


def _read_stream(lines: Iterator[str], on_text: Callable[[str], None]) -> tuple[ToolCall, ...]:
    calls: dict[int, dict[str, str]] = {}
    order: list[int] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line == "[DONE]":
            break
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            continue
        choice = choices[0]
        message = choice.get("message")
        if isinstance(message, dict) and "delta" not in choice:
            text = _message_text(message)
            if text:
                on_text(text)
            return _calls_from_message(message)
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        for key in ("reasoning_content", "content"):
            piece = delta.get(key)
            if isinstance(piece, str) and piece:
                on_text(piece)
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            index = int(item.get("index", 0))
            slot = calls.get(index)
            if slot is None:
                slot = {"id": "", "name": "", "arguments": ""}
                calls[index] = slot
                order.append(index)
            if item.get("id"):
                slot["id"] = str(item["id"])
            function = item.get("function")
            if isinstance(function, dict):
                if function.get("name"):
                    slot["name"] += str(function["name"])
                if function.get("arguments"):
                    slot["arguments"] += str(function["arguments"])
    return tuple(
        ToolCall(calls[index]["name"], calls[index]["arguments"], calls[index]["id"])
        for index in order
    )


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
