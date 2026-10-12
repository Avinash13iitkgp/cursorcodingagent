"""Stdio MCP client.

Servers are started with Jail.run. The operator owns mcp.json beside the
data root. The model cannot supply the server argv.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from agent_engine.adapters.exec.jail import Jail, tool_env
from agent_engine.domain.errors import ToolTimeout

_TIMEOUT_S = 30.0
_MANAGERS = ("apt-get", "apt", "brew", "pip3", "pip", "yum", "dnf")
_IGNORED_ARGUMENT_KEYS = ("argv", "server_argv")


@dataclass(frozen=True)
class _Entry:
    argv: tuple[str, ...]
    env: tuple[str, ...]


class McpClient:
    """Call operator-listed MCP servers over stdio inside the jail."""

    def __init__(self, data_root: Path, jail: Jail, work_root: Path) -> None:
        """Bind the client to one data root and one task work tree.

        Args:
            data_root: Directory that contains mcp.json and tenants/.
            jail: Process-group runner. The model does not choose its argv.
            work_root: Child working directory. It is the task work tree.

        Returns:
            None.
        """
        self._data_root = data_root
        self._jail = jail
        self._work_root = work_root

    def server_names(self) -> tuple[str, ...]:
        """Return the server names listed in mcp.json.

        Returns:
            Names in file order. An absent or unreadable file is empty.
        """
        return tuple(self._catalog())

    def list_tools(self, server: str) -> tuple[dict[str, object], ...]:
        """Return Chat Completions tools named mcp__{server}__{tool}.

        Args:
            server: Operator server name from mcp.json.

        Returns:
            One function tool per listed MCP tool. An unknown server or a
            protocol error is an empty tuple.
        """
        entry = self._catalog().get(server)
        if entry is None:
            return ()
        result, error = self._exchange(entry, "tools/list", {})
        if error is not None or result is None:
            return ()
        listed = result.get("tools")
        if not isinstance(listed, list):
            return ()
        tools: list[dict[str, object]] = []
        for item in listed:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            tool_name = str(item["name"])
            schema = item.get("inputSchema")
            if not isinstance(schema, dict):
                schema = {"type": "object"}
            description = item.get("description")
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"mcp__{server}__{tool_name}",
                        "description": description if isinstance(description, str) else tool_name,
                        "parameters": schema,
                    },
                }
            )
        return tuple(tools)

    def chat_tools(self) -> list[dict[str, object]]:
        """Return every namespaced tool from mcp.json.

        Returns:
            Chat Completions tools. Servers that fail to list are skipped.
        """
        tools: list[dict[str, object]] = []
        for name in self.server_names():
            tools.extend(self.list_tools(name))
        return tools

    def call(self, server: str, tool: str, arguments: dict[str, object]) -> str:
        """Run one tools/call in the jail and return its text.

        Args:
            server: Operator server name. A name that is not in mcp.json
                does not start a process.
            tool: Tool name inside that server, without the mcp__ prefix.
            arguments: Model arguments. argv keys are dropped. A
                package-manager install of g++, python, or go does not start
                a process.

        Returns:
            The tool text, or a JSON error object. This method does not
            write tasks.status.

        Raises:
            OSError: The work root cannot be created.
        """
        if not isinstance(arguments, dict):
            return _error("protocol error", "arguments must be an object")
        if _toolchain_install(arguments):
            return _error("toolchain install is not allowed", "the model cannot add a toolchain")
        entry = self._catalog().get(server)
        if entry is None:
            return _error("unknown server", server)
        forwarded = {
            key: value for key, value in arguments.items() if key not in _IGNORED_ARGUMENT_KEYS
        }
        result, error = self._exchange(
            entry,
            "tools/call",
            {"name": tool, "arguments": forwarded},
        )
        if error is not None or result is None:
            return _error("protocol error", error or "missing result")
        return _tool_text(result)

    def _catalog(self) -> dict[str, _Entry]:
        path = self._data_root / "mcp.json"
        if not path.is_file():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        servers = raw.get("servers") if isinstance(raw, dict) else None
        if not isinstance(servers, list):
            return {}
        catalog: dict[str, _Entry] = {}
        for item in servers:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            argv = item.get("argv")
            if not isinstance(name, str) or not name or not isinstance(argv, list) or not argv:
                continue
            env_names = item.get("env") if isinstance(item.get("env"), list) else []
            catalog[name] = _Entry(
                tuple(str(part) for part in argv),
                tuple(str(part) for part in env_names if isinstance(part, str)),
            )
        return catalog

    def _exchange(
        self, entry: _Entry, method: str, params: dict[str, object]
    ) -> tuple[dict[str, object] | None, str | None]:
        self._work_root.mkdir(parents=True, exist_ok=True)
        stdin = _session(method, params)
        try:
            completed = self._jail.run(
                _resolve_argv(entry.argv),
                cwd=self._work_root,
                timeout_s=_TIMEOUT_S,
                env=_child_env(entry.env),
                stdin=stdin,
            )
        except ToolTimeout as exc:
            return None, f"ToolTimeout: {exc}"
        except ValueError as exc:
            return None, str(exc)
        responses = _responses(completed.stdout)
        opened = responses.get(1)
        if opened is not None and "error" in opened:
            return None, json.dumps(opened["error"])
        message = responses.get(2)
        if message is None:
            detail = completed.stderr.strip() or completed.stdout.strip() or f"exit {completed.returncode}"
            return None, detail[:500]
        if "error" in message:
            return None, json.dumps(message["error"])
        result = message.get("result")
        if not isinstance(result, dict):
            return None, "missing result"
        return result, None


def _child_env(names: tuple[str, ...]) -> dict[str, str]:
    env = tool_env()
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            env[name] = value
    return env


def _resolve_argv(argv: tuple[str, ...]) -> list[str]:
    resolved: list[str] = []
    for index, token in enumerate(argv):
        if index == 0 and token in {"python", "python3"}:
            resolved.append(sys.executable)
            continue
        resolved.append(_resolve_token(token))
    return resolved


def _resolve_token(token: str) -> str:
    path = Path(token)
    if path.is_absolute():
        return token
    candidates = [Path.cwd() / token]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / token)
        if len(candidates) > 6:
            break
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return token


def _session(method: str, params: dict[str, object]) -> str:
    messages = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agent_engine", "version": "0.1.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": method, "params": params},
    ]
    return "".join(json.dumps(message) + "\n" for message in messages)


def _responses(stdout: str) -> dict[int, dict[str, object]]:
    found: dict[int, dict[str, object]] = {}
    for line in stdout.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, dict) or "id" not in message:
            continue
        try:
            found[int(message["id"])] = message
        except (TypeError, ValueError):
            continue
    return found


def _tool_text(result: dict[str, object]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        if parts:
            return "\n".join(parts)
    return json.dumps(result)


def _toolchain_install(arguments: dict[str, object]) -> bool:
    text = f" {_flatten(arguments).lower()} "
    if "install" not in text:
        return False
    if not any(manager in text for manager in _MANAGERS):
        return False
    if "g++" in text or "python" in text or "golang" in text:
        return True
    return " go " in text


def _flatten(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(item) for item in value)
    return ""


def _error(code: str, detail: str) -> str:
    return json.dumps({"error": code, "detail": detail})
