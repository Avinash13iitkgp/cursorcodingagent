"""ReAct loop for one leased task.

run reads the latest step first. A terminal task is a cache hit: it does not
insert and it does not call the model. intent_logged resumes that row and does
not call the model. No row calls the model, then T1 inserts exactly one tool
call and commits before any file write or command. Extra tool calls in one
message become later steps. T2 then writes the result, the cache row, and the
transcript under the fence. The loop stops at 8 agent steps. finish does not
set completed or rejected; VerifyGate is the only writer of those statuses.
This module does not import sqlite or adapters.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_engine.domain.errors import StaleContent
from agent_engine.domain.models import StepView, TenantId, TextEdit, ToolCall
from agent_engine.domain.ports import CheckRunner, Clock, Editor, Llm, StepLog, Workspace

_MAX_STEPS = 8
_TERMINAL = ("completed", "rejected")
_PROMPT = (
    "Fix adder.py so add(a, b) returns a + b. "
    "Use read_file, replace_text, run_checks, and finish. "
    "Do not modify checks or add a compiler install."
)


@dataclass(frozen=True)
class RunResult:
    """What one loop run printed for the caller."""

    status: str
    cache_hit: bool
    executions: int
    posts: int


class FinishGate(Protocol):
    """Gate shape the loop calls. VerifyGate is the production implementation."""

    def finish(self, now: float) -> str:
        """Return the terminal status after checks.

        Args:
            now: Instant compared with the lease expiry.

        Returns:
            completed or rejected.
        """
        ...  # pragma: no cover


def idempotency_key(
    tenant_id: str, task_id: str, step_index: int, tool: str, args: dict[str, object]
) -> str:
    """Hash the canonical identity of one step.

    Args:
        tenant_id: Tenant that owns the step.
        task_id: Task primary key.
        step_index: Index of this step.
        tool: Tool name.
        args: Model arguments. Key order does not change the digest.

    Returns:
        SHA256 hex of the canonical JSON object.
    """
    payload = {
        "args": args,
        "step_index": step_index,
        "task_id": task_id,
        "tenant_id": tenant_id,
        "tool": tool,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ReactLoop:
    """One tool call per step, with T1 before the side effect and fenced T2 after."""

    def __init__(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        work_root: Path,
        store: StepLog,
        llm: Llm,
        workspace: Workspace,
        editor: Editor,
        checks: CheckRunner,
        gate: FinishGate,
    ) -> None:
        """Bind the loop to one lease.

        Args:
            tenant_id: Tenant file this loop writes.
            task_id: Task primary key.
            worker_id: Lease owner.
            token: Fencing token for T1, T2, and the gate.
            work_root: Directory passed to checks. File tools use workspace.
            store: Intent, cache, and transcript port.
            llm: Model port. A cache hit does not call it.
            workspace: File port. Paths cannot leave the work root.
            editor: Optimistic edit port.
            checks: run_checks runner. It does not set task status.
            gate: Writer of completed and rejected. The loop does not set them.
        """
        self._tenant_id = tenant_id
        self._task_id = task_id
        self._worker_id = worker_id
        self._token = token
        self._work_root = work_root
        self._store = store
        self._llm = llm
        self._workspace = workspace
        self._editor = editor
        self._checks = checks
        self._gate = gate
        self._instant = lambda: 0.0

    def run(self, now: float, *, clock: Clock | None = None) -> RunResult:
        """Drive steps until a terminal status, an empty model turn, or 8 steps.

        Args:
            now: Instant compared with lease_expires_at when clock is omitted.
            A clock supplies a fresh instant for each fenced write.

        Returns:
            The task status, whether this run was a cache hit, how many tools
            ran, and how many model posts this llm object has made.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: A fenced write lost the lease. The task stays running
                when the gate has not committed a terminal status.
        """
        self._instant = (lambda: now) if clock is None else clock.now
        self._editor.resume(self._workspace)
        executions = 0
        while True:
            open_step = self._store.open_intent(self._tenant_id, self._task_id)
            if open_step is not None:
                executions += 1
                self._execute(open_step)
                status = self._store.task_status(self._tenant_id, self._task_id)
                if status in _TERMINAL:
                    return self._result(status, False, executions)
                continue
            status = self._store.task_status(self._tenant_id, self._task_id)
            if status in _TERMINAL:
                return self._result(status, True, executions)
            if self._store.agent_count(self._tenant_id, self._task_id) >= _MAX_STEPS:
                return self._result(status, False, executions)
            if not self._prepare_next():
                return self._result(status, False, executions)

    def _prepare_next(self) -> bool:
        queued = self._queued_call()
        messages: tuple[tuple[str, str], ...] = ()
        if queued is None:
            calls = self._llm.complete(self._history())
            if not calls:
                return False
            queued, messages = self._lead_call(calls)
        now = self._instant()
        self._store.mark_running(
            self._tenant_id, self._task_id, self._worker_id, self._token, now
        )
        step_index = self._store.next_index(self._tenant_id, self._task_id)
        args = json.loads(queued.arguments_json)
        key = idempotency_key(
            self._tenant_id.value, self._task_id, step_index, queued.name, args
        )
        self._store.insert_intent(
            self._tenant_id,
            self._task_id,
            step_index,
            key,
            queued.name,
            queued.arguments_json,
            self._worker_id,
            self._token,
            now,
            messages,
        )
        return True

    def _lead_call(
        self, calls: tuple[ToolCall, ...]
    ) -> tuple[ToolCall, tuple[tuple[str, str], ...]]:
        start = self._store.next_index(self._tenant_id, self._task_id)
        numbered: list[ToolCall] = []
        for offset, call in enumerate(calls):
            numbered.append(
                ToolCall(call.name, call.arguments_json, f"call-{start + offset}")
            )
        assistant = {
            "content": None,
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in numbered
            ],
        }
        rows: list[tuple[str, str]] = []
        if not self._store.list_messages(self._tenant_id, self._task_id):
            rows.append(("user", json.dumps({"content": _PROMPT})))
        rows.append(("assistant", json.dumps(assistant)))
        return numbered[0], tuple(rows)

    def _execute(self, step: StepView) -> None:
        now = self._instant()
        self._store.mark_running(
            self._tenant_id, self._task_id, self._worker_id, self._token, now
        )
        args = json.loads(step.args_json)
        if step.tool_name == "read_file":
            result = self._read_file(str(args["path"]))
        elif step.tool_name == "replace_text":
            result = self._replace(args)
        elif step.tool_name == "run_checks":
            ok, stdout, stderr = self._checks.run(self._work_root)
            result = json.dumps({"ok": ok, "stdout": stdout, "stderr": stderr})
        elif step.tool_name == "finish":
            status = self._gate.finish(now)
            result = json.dumps({"status": status})
        else:
            result = json.dumps({"error": "unknown tool"})
        body = json.dumps({"tool_call_id": f"call-{step.step_index}", "content": result})
        self._store.commit_step(
            self._tenant_id,
            self._task_id,
            step.step_index,
            step.idempotency_key,
            result,
            self._worker_id,
            self._token,
            now,
            (("tool", body),),
        )

    def _read_file(self, path: str) -> str:
        data = self._workspace.read(path)
        return json.dumps(
            {
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "content": data.decode("utf-8"),
            }
        )

    def _replace(self, args: dict[str, object]) -> str:
        edit = TextEdit(
            str(args["path"]),
            str(args["expected_sha256"]),
            str(args["old"]),
            str(args["new"]),
        )
        try:
            outcome = self._editor.apply(self._workspace, edit)
        except StaleContent:
            current = hashlib.sha256(self._workspace.read(edit.path)).hexdigest()
            return json.dumps({"error": "StaleContent", "sha256": current})
        return json.dumps({"status": outcome.status})

    def _queued_call(self) -> ToolCall | None:
        rows = self._store.list_messages(self._tenant_id, self._task_id)
        last_body: dict[str, object] | None = None
        last_at = -1
        for index, (_seq, role, body) in enumerate(rows):
            if role == "assistant":
                last_body = json.loads(body)
                last_at = index
        if last_body is None:
            return None
        tool_calls = last_body.get("tool_calls")
        if not isinstance(tool_calls, list):
            return None
        following = sum(1 for row in rows[last_at + 1 :] if row[1] == "tool")
        if following >= len(tool_calls):
            return None
        item = tool_calls[following]
        if not isinstance(item, dict):
            return None
        function = item.get("function")
        if not isinstance(function, dict):
            return None
        return ToolCall(
            str(function["name"]),
            str(function["arguments"]),
            str(item.get("id", "")),
        )

    def _history(self) -> list[dict[str, object]]:
        rows = self._store.list_messages(self._tenant_id, self._task_id)
        if not rows:
            return [{"role": "user", "content": _PROMPT}]
        history: list[dict[str, object]] = []
        for _seq, role, body in rows:
            payload = json.loads(body)
            payload["role"] = role
            history.append(payload)
        return history

    def _result(self, status: str, cache_hit: bool, executions: int) -> RunResult:
        return RunResult(status, cache_hit, executions, self._llm.post_count())
