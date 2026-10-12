"""Lease a task without importing the lease service.

Step 4 unit tests stamp the fence with SQL so application coverage stays on
the loop and the gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.step_store import SqliteStepStore
from agent_engine.adapters.sqlite.tenant_store import init_task, provision
from agent_engine.adapters.workspace.edits import OptimisticEditor
from agent_engine.adapters.workspace.plain import PlainWorkspace
from agent_engine.application.react import ReactLoop
from agent_engine.domain.models import TenantId, ToolCall
from agent_engine.domain.ports import CheckRunner

NOW = 1_700_000_000.0
TTL = 3600.0
WORKER = "33333333-3333-4333-8333-333333333333"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "broken_adder" / "adder.py"


@dataclass
class Leased:
    """One open tenant connection and its work tree."""

    data_root: Path
    tenant: TenantId
    task_id: str
    connection: Any
    store: SqliteStepStore
    work: Path
    token: int


class ScriptedLlm:
    """Return scripted tool calls and count posts."""

    def __init__(self, turns: list[tuple[ToolCall, ...]]) -> None:
        """Store one tuple of tool calls per complete call.

        Args:
            turns: Consumed in order. An empty tuple is a turn with no tool.
        """
        self._turns = list(turns)
        self.posts = 0
        self.seen: list[list[dict[str, object]]] = []

    def post_count(self) -> int:
        """Return how many times complete was called.

        Returns:
            The post count.
        """
        return self.posts

    def complete(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """Pop the next scripted turn.

        Args:
            messages: Recorded so a test can see what the model was told.

        Returns:
            The next tuple of tool calls.

        Raises:
            AssertionError: The script has no turn left.
        """
        self.posts += 1
        self.seen.append(messages)
        if not self._turns:
            raise AssertionError("model called too often")
        return self._turns.pop(0)


class FixedChecks:
    """Check runner that does not start a process."""

    def __init__(self, ok: bool = True, stdout: str = "", stderr: str = "") -> None:
        """Store the result run will return.

        Args:
            ok: Whether the checks pass.
            stdout: Text returned to the tool.
            stderr: Text returned to the tool.
        """
        self.ok = ok
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[Path] = []

    def run(self, work_root: Path) -> tuple[bool, str, str]:
        """Record the work root and return the stored result.

        Args:
            work_root: Ignored except for the call list.

        Returns:
            The stored ok, stdout, and stderr.
        """
        self.calls.append(work_root)
        return self.ok, self.stdout, self.stderr


class StatusGate:
    """Test gate. It writes terminal status only when asked."""

    def __init__(self, env: Leased, status: str | None) -> None:
        """Bind the gate to one leased task.

        Args:
            env: Store and fence to write.
            status: completed or rejected. None leaves the status unchanged.
        """
        self._env = env
        self._status = status
        self.calls = 0

    def finish(self, now: float) -> str:
        """Optionally write a terminal status.

        Args:
            now: Instant compared with the lease.

        Returns:
            The status string, or running when this gate does not write.
        """
        self.calls += 1
        if self._status is None:
            return "running"
        self._env.store.set_terminal(
            self._env.tenant,
            self._env.task_id,
            self._status,
            WORKER,
            self._env.token,
            now,
        )
        return self._status


def call(name: str, args: dict[str, object]) -> ToolCall:
    """Build one tool call.

    Args:
        name: Tool name.
        args: JSON object the tool will parse.

    Returns:
        A tool call with an empty id. The loop assigns call ids.
    """
    return ToolCall(name, json.dumps(args), "")


def lease_task(tmp_path: Path, task_id: str = "fix-adder", *, broken: bool = True) -> Leased:
    """Provision acme, init the task, and hold a fence at NOW.

    Args:
        tmp_path: Pytest directory used as the data root.
        task_id: Task to create.
        broken: When true, copy the broken adder into the work tree.

    Returns:
        The open connection and workspace. The caller closes the connection.
    """
    tenant = TenantId("acme")
    provision(tmp_path, tenant, "Acme")
    init_task(tmp_path, tenant, task_id)
    work = tmp_path / "tenants" / "acme" / "tasks" / task_id
    work.mkdir(parents=True, exist_ok=True)
    if broken:
        (work / "adder.py").write_bytes(FIXTURE.read_bytes())
    connection = open_tenant_db(tmp_path, tenant)
    connection.execute("BEGIN IMMEDIATE")
    connection.execute(
        """
        UPDATE tasks
        SET fencing_token = 1, lease_owner = ?, lease_expires_at = ?
        WHERE tenant_id = ? AND id = ?
        """,
        (WORKER, NOW + TTL, tenant.value, task_id),
    )
    connection.execute("COMMIT")
    return Leased(tmp_path, tenant, task_id, connection, SqliteStepStore(connection), work, 1)


def loop(
    env: Leased,
    llm: ScriptedLlm,
    gate: StatusGate | Any,
    checks: CheckRunner,
    workspace: PlainWorkspace | None = None,
) -> ReactLoop:
    """Wire a loop around an already leased task.

    Args:
        env: Open lease.
        llm: Scripted or other model.
        gate: Finish gate.
        checks: Check runner.
        workspace: Work root. The default is a plain workspace.

    Returns:
        A loop that uses OptimisticEditor.
    """
    return ReactLoop(
        env.tenant,
        env.task_id,
        WORKER,
        env.token,
        env.work,
        env.store,
        llm,
        workspace or PlainWorkspace(env.work),
        OptimisticEditor(),
        checks,
        gate,
    )
