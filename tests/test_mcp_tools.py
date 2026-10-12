"""MCP tools stay inside one fenced step and the operator's mcp.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_engine.adapters.exec.jail import Jail
from agent_engine.adapters.mcp.client import McpClient
from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.tenant_store import get_task, provision
from agent_engine.application.react import idempotency_key
from agent_engine.domain.errors import ToolTimeout
from agent_engine.domain.models import TenantId
from tests.step4_support import FixedChecks, ScriptedLlm, StatusGate, call, lease_task, loop

ECHO = Path(__file__).resolve().parents[1] / "fixtures" / "mcp" / "echo_server.py"


class RecordingJail(Jail):
    """Record Jail.run arguments, then run the real process group."""

    def __init__(self) -> None:
        """Start with an empty call list."""
        self.calls: list[tuple[list[str], Path, float, dict[str, str], str | None]] = []

    def run(self, argv, *, cwd, timeout_s, env, stdin=None):  # type: ignore[no-untyped-def]
        """Record the call and delegate to Jail.

        Args:
            argv: Command the client chose from mcp.json.
            cwd: Task work root.
            timeout_s: Jail timeout.
            env: Exact child environment.
            stdin: JSON-RPC session text.

        Returns:
            The completed process.
        """
        self.calls.append((list(argv), cwd, timeout_s, dict(env), stdin))
        return super().run(argv, cwd=cwd, timeout_s=timeout_s, env=env, stdin=stdin)


class TimeoutJail(Jail):
    """Fail before a process starts."""

    def run(self, argv, *, cwd, timeout_s, env, stdin=None):  # type: ignore[no-untyped-def]
        """Raise ToolTimeout without starting argv.

        Args:
            argv: Ignored.
            cwd: Ignored.
            timeout_s: Ignored.
            env: Ignored.
            stdin: Ignored.

        Returns:
            Never.

        Raises:
            ToolTimeout: Always.
        """
        del argv, cwd, timeout_s, env, stdin
        raise ToolTimeout("command exceeded 30s")


class ProbeJail(Jail):
    """Read the intent row at the moment the server process would start."""

    def __init__(self, connection) -> None:  # type: ignore[no-untyped-def]
        """Bind the probe to the loop's database connection.

        Args:
            connection: Open tenant connection.
        """
        self._connection = connection
        self.status: tuple[str, str] | None = None

    def run(self, argv, *, cwd, timeout_s, env, stdin=None):  # type: ignore[no-untyped-def]
        """Assert T1 is committed, then start the server.

        Args:
            argv: Echo server command.
            cwd: Work root.
            timeout_s: Jail timeout.
            env: Child environment.
            stdin: JSON-RPC session.

        Returns:
            The completed process.
        """
        row = self._connection.execute(
            "SELECT status, tool_name FROM step_intents"
        ).fetchone()
        self.status = None if row is None else (str(row["status"]), str(row["tool_name"]))
        return super().run(argv, cwd=cwd, timeout_s=timeout_s, env=env, stdin=stdin)


def test_list_tools_namespaces_names_as_mcp_server_tool(tmp_path: Path) -> None:
    """list_tools returns mcp__{server}__{tool}."""
    _write_mcp(tmp_path)
    client = McpClient(tmp_path, Jail(), tmp_path / "work")
    tools = client.list_tools("echo")
    assert tools[0]["function"]["name"] == "mcp__echo__ping"  # type: ignore[index]
    assert client.chat_tools()[0]["function"]["name"] == "mcp__echo__ping"
    assert client.list_tools("missing") == ()
    assert client.server_names() == ("echo",)
    (tmp_path / "mcp.json").write_text("not-json", encoding="utf-8")
    assert client.list_tools("echo") == ()
    (tmp_path / "mcp.json").write_text("{}", encoding="utf-8")
    assert client.server_names() == ()


def test_call_uses_jail_and_returns_the_tool_result(tmp_path: Path) -> None:
    """tools/call goes through Jail.run and returns the echo text."""
    _write_mcp(tmp_path, argv=["python3", "fixtures/mcp/echo_server.py"])
    jail = RecordingJail()
    work = tmp_path / "work"
    client = McpClient(tmp_path, jail, work)
    result = client.call("echo", "ping", {"text": "hello"})
    assert "hello" in result
    argv, cwd, timeout_s, _env, stdin = jail.calls[0]
    assert argv[0] == sys.executable
    assert any(part.endswith("echo_server.py") for part in argv)
    assert cwd == work
    assert timeout_s == 30
    assert stdin is not None and "tools/call" in stdin and "initialize" in stdin
    timed = McpClient(tmp_path, TimeoutJail(), work)
    timeout = timed.call("echo", "ping", {"text": "hello"})
    assert "ToolTimeout" in timeout


def test_child_env_omits_api_key_unless_the_entry_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AGENT_LLM_API_KEY reaches the child only when mcp.json lists it."""
    secret = "super-secret-key"
    monkeypatch.setenv("AGENT_LLM_API_KEY", secret)
    _write_mcp(tmp_path)
    jail = RecordingJail()
    client = McpClient(tmp_path, jail, tmp_path / "work")
    hidden = client.call("echo", "ping", {"text": "hidden"})
    assert secret not in hidden
    assert "AGENT_LLM_API_KEY" not in jail.calls[0][3]
    assert '"has_api_key": false' in hidden
    _write_mcp(tmp_path, env=["AGENT_LLM_API_KEY"])
    shown = client.call("echo", "ping", {"text": "shown"})
    assert jail.calls[1][3]["AGENT_LLM_API_KEY"] == secret
    assert secret not in shown
    assert '"has_api_key": true' in shown


def test_model_supplied_server_argv_is_ignored(tmp_path: Path) -> None:
    """A model argv is not the process command."""
    _write_mcp(tmp_path)
    jail = RecordingJail()
    client = McpClient(tmp_path, jail, tmp_path / "work")
    result = client.call(
        "echo",
        "ping",
        {"argv": ["python", "-c", "print('pwned')"], "text": "safe"},
    )
    assert "safe" in result
    assert "pwned" not in result
    assert "-c" not in jail.calls[0][0]
    assert any(part.endswith("echo_server.py") for part in jail.calls[0][0])


def test_model_cannot_add_a_toolchain_install(tmp_path: Path) -> None:
    """A package-manager install does not start the server."""
    _write_mcp(tmp_path)
    jail = RecordingJail()
    client = McpClient(tmp_path, jail, tmp_path / "work")
    refused = client.call("echo", "ping", {"command": "brew install g++"})
    assert "toolchain install is not allowed" in refused
    assert jail.calls == []
    unknown = client.call("gcc", "compile", {"files": ["main.cpp"]})
    assert "unknown server" in unknown
    assert jail.calls == []
    listed = client.call("echo", "ping", {"argv": ["apt-get", "install", "python"]})
    assert "toolchain install is not allowed" in listed
    assert jail.calls == []


def test_mcp_call_is_one_fenced_step_and_does_not_set_status(tmp_path: Path) -> None:
    """T1 is committed before the server starts, and status stays running."""
    env = lease_task(tmp_path, "mcp-echo", broken=False)
    try:
        _write_mcp(tmp_path)
        probe = ProbeJail(env.connection)
        gate = StatusGate(env, None)
        llm = ScriptedLlm(
            [
                (call("mcp__echo__ping", {"text": "ping"}),),
                (),
            ]
        )
        result = loop(env, llm, gate, FixedChecks(), mcp=McpClient(tmp_path, probe, env.work)).run(
            1_700_000_000.0
        )
        assert probe.status == ("intent_logged", "mcp__echo__ping")
        assert result.status == "running"
        assert result.executions == 1
        assert "ping" in result.detail
        assert gate.calls == 0
        row = env.connection.execute(
            "SELECT status, idempotency_key, args_json, tool_name FROM step_intents"
        ).fetchone()
        assert row["status"] == "committed"
        assert row["tool_name"] == "mcp__echo__ping"
        assert row["idempotency_key"] == idempotency_key(
            "acme", "mcp-echo", 0, "mcp__echo__ping", json.loads(row["args_json"])
        )
        assert get_task(tmp_path, TenantId("acme"), "mcp-echo").status == "running"
    finally:
        env.connection.close()


def test_mcp_error_leaves_the_task_running(tmp_path: Path) -> None:
    """A protocol error is the step result and the task stays running."""
    env = lease_task(tmp_path, "mcp-echo", broken=False)
    try:
        _write_mcp(tmp_path, argv=[sys.executable, "-c", "print('nope')"])
        llm = ScriptedLlm([(call("mcp__echo__ping", {"text": "ping"}),), ()])
        gate = StatusGate(env, None)
        result = loop(env, llm, gate, FixedChecks(), mcp=McpClient(tmp_path, Jail(), env.work)).run(
            1_700_000_000.0
        )
        assert result.status == "running"
        assert "protocol error" in result.detail
        assert "nope" in result.detail
        assert gate.calls == 0
        assert get_task(tmp_path, TenantId("acme"), "mcp-echo").status == "running"
        row = env.connection.execute("SELECT status FROM step_intents").fetchone()
        assert row["status"] == "committed"
    finally:
        env.connection.close()


def test_acme_mcp_call_does_not_write_globex(tmp_path: Path) -> None:
    """An acme MCP call writes no globex transcript row."""
    provision(tmp_path, TenantId("globex"), "Globex")
    env = lease_task(tmp_path, "mcp-echo", broken=False)
    try:
        _write_mcp(tmp_path)
        llm = ScriptedLlm([(call("mcp__echo__ping", {"text": "ping"}),), ()])
        result = loop(env, llm, StatusGate(env, None), FixedChecks(), mcp=McpClient(tmp_path, Jail(), env.work)).run(
            1_700_000_000.0
        )
        assert result.status == "running"
        globex = open_tenant_db(tmp_path, TenantId("globex"), read_only=True)
        try:
            steps = globex.execute("SELECT COUNT(*) AS n FROM step_intents").fetchone()["n"]
            messages = globex.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
        finally:
            globex.close()
        assert steps == 0
        assert messages == 0
    finally:
        env.connection.close()


def _write_mcp(
    data_root: Path,
    *,
    env: list[str] | None = None,
    argv: list[str] | None = None,
) -> None:
    catalog = {
        "servers": [
            {
                "name": "echo",
                "argv": argv or ["python", str(ECHO)],
                "env": env or [],
            }
        ]
    }
    (data_root / "mcp.json").write_text(json.dumps(catalog), encoding="utf-8")
