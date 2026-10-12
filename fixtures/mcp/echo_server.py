"""Stdio echo server for the MCP fixture.

Reads newline-delimited JSON-RPC until stdin closes, then exits. The reply
for tools/call reports whether AGENT_LLM_API_KEY is set and does not print
the value.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> None:
    """Answer initialize, tools/list, and tools/call on stdout.

    Returns:
        None. The process exits when stdin reaches EOF.
    """
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            result: dict[str, object] = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "0.1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "ping",
                        "description": "Echo the arguments.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                        },
                    }
                ]
            }
        elif method == "tools/call":
            result = _call(message.get("params"))
        else:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "error": {"code": -32601, "message": "method not found"},
                }
            )
            continue
        _write({"jsonrpc": "2.0", "id": message["id"], "result": result})


def _call(params: object) -> dict[str, object]:
    arguments: dict[str, object] = {}
    if isinstance(params, dict) and isinstance(params.get("arguments"), dict):
        arguments = params["arguments"]
    text = json.dumps(
        {
            "has_api_key": "AGENT_LLM_API_KEY" in os.environ,
            "text": arguments.get("text", ""),
        },
        sort_keys=True,
    )
    return {"content": [{"type": "text", "text": text}]}


def _write(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
