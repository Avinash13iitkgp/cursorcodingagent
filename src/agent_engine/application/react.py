"""ReAct loop for one leased task.

run reads the latest step first. A terminal task is a cache hit: it does not
insert and it does not call the model. intent_logged resumes that row and does
not call the model. No row calls the model, then T1 inserts exactly one tool
call and commits before any file write or command. Extra tool calls in one
message become later steps. T2 then writes the result, the cache row, and the
transcript under the fence. The loop stops at max_steps agent steps, 8 by
default. write_file creates a new path and its parent directories; an
existing file stays on replace_text. finish does not set completed or
rejected; VerifyGate is the only writer of those statuses.
This module does not import sqlite or adapters. A name that starts with
mcp__ is one step on the MCP port when that port is bound.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_engine.application.context import (
    ContextBuilder,
    list_files,
    redact,
    remember_note,
    search_text,
)
from agent_engine.domain.errors import StaleContent
from agent_engine.domain.models import StepView, TenantId, TextEdit, ToolCall
from agent_engine.domain.ports import (
    CheckRunner,
    Clock,
    Editor,
    Llm,
    McpCaller,
    StepLog,
    Workspace,
)

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
    detail: str = ""


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
        mcp: McpCaller | None = None,
        context: ContextBuilder | None = None,
        secret: str = "",
        prompt: str | None = None,
        max_steps: int = _MAX_STEPS,
        trace: Callable[[str], None] | None = None,
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
            mcp: MCP port. None leaves mcp__ names as an unknown tool.
            context: Packer used immediately before the model call. None packs
                the raw history under the default budget.
            secret: Bearer token redacted from notes and trajectory bodies.
            prompt: First user message. None keeps the adder sentence, which
                run-task and the demos use.
            max_steps: Agent steps this run may reach. apply passes a larger
                cap so a small tree fits in one run.
            trace: Receives one log line per model turn and per tool result.
                None stays quiet, which run-task's library callers and the
                demos rely on.
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
        self._mcp = mcp
        self._context = context or ContextBuilder()
        self._secret = secret
        self._prompt = _PROMPT if prompt is None else prompt
        self._max_steps = max_steps
        self._trace = trace
        self._detail = ""
        self._instant = lambda: 0.0

    def run(
        self, now: float, *, clock: Clock | None = None, resume_only: bool = False
    ) -> RunResult:
        """Drive steps until a terminal status, an empty model turn, or the cap.

        Args:
            now: Instant compared with lease_expires_at when clock is omitted.
            A clock supplies a fresh instant for each fenced write.
            resume_only: After the open intent is executed, return without
                another model call. The successor uses this so a crashed step
                is committed once.

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
                if status in _TERMINAL or resume_only:
                    return self._result(status, False, executions)
                continue
            status = self._store.task_status(self._tenant_id, self._task_id)
            if status in _TERMINAL:
                return self._result(status, True, executions)
            if self._store.agent_count(self._tenant_id, self._task_id) >= self._max_steps:
                self._note(f"model: stopped at {self._max_steps} steps")
                return self._result(status, False, executions)
            if not self._prepare_next():
                return self._result(status, False, executions)

    def _prepare_next(self) -> bool:
        queued = self._queued_call()
        messages: tuple[tuple[str, str], ...] = ()
        if queued is None:
            history = self._context.pack(
                self._tenant_id, self._task_id, self._history()
            )
            done = self._store.agent_count(self._tenant_id, self._task_id)
            self._note(f"model: asking for the next action ({done} steps done)")
            calls = self._llm.complete(history)
            if not calls:
                self._note("model: no tool call, stopping")
                return False
            for call in calls:
                self._note(f"model chose {call.name}\n{_shown(call.arguments_json, self._secret)}")
            queued, messages = self._lead_call(calls)
        else:
            self._note(f"model: queued {queued.name}\n{_shown(queued.arguments_json, self._secret)}")
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
            rows.append(("user", json.dumps({"content": self._prompt})))
        rows.append(("assistant", json.dumps(assistant)))
        return numbered[0], tuple(rows)

    def _execute(self, step: StepView) -> None:
        now = self._instant()
        self._store.mark_running(
            self._tenant_id, self._task_id, self._worker_id, self._token, now
        )
        self._note(
            f"step {step.step_index}: {step.tool_name}\n"
            f"{_shown(step.args_json, self._secret)}"
        )
        args = json.loads(step.args_json)
        knowledge = None
        if step.tool_name == "read_file":
            result = self._read_file(str(args["path"]))
        elif step.tool_name == "replace_text":
            result = self._replace(args)
        elif step.tool_name == "write_file":
            result = self._write_file(args)
        elif step.tool_name == "run_checks":
            ok, stdout, stderr = self._checks.run(self._work_root)
            result = json.dumps({"ok": ok, "stdout": stdout, "stderr": stderr})
        elif step.tool_name == "finish":
            status = self._gate.finish(now)
            result = json.dumps({"status": status})
        elif step.tool_name.startswith("mcp__"):
            result = self._mcp_call(step.tool_name, args)
        elif step.tool_name == "list_files":
            result = self._search(list_files, args)
        elif step.tool_name == "search_text":
            result = self._search(search_text, args, query=str(args.get("query", "")))
        elif step.tool_name == "remember":
            result, knowledge = self._remember(args, step.step_index)
        else:
            result = json.dumps({"error": "unknown tool"})
        result = redact(result, self._secret)
        self._note(f"step {step.step_index} result:\n{result}")
        safe_args = args if isinstance(args, dict) else {}
        safe_args = json.loads(redact(json.dumps(safe_args), self._secret))
        trajectory = list(_trajectory(step.tool_name, step.step_index, safe_args, result))
        if knowledge is not None:
            path, quote, body, _source = knowledge
            trajectory.append(
                (
                    "knowledge",
                    step.step_index,
                    json.dumps({"path": path, "quote": quote, "body": body}),
                )
            )
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
            tuple(trajectory),
            knowledge,
        )

    def _search(self, fn, args: dict[str, object], query: str | None = None) -> str:  # type: ignore[no-untyped-def]
        if not isinstance(args, dict):
            return json.dumps({"error": "unknown tool"})
        path = str(args.get("path", "."))
        try:
            if query is None:
                return str(fn(self._workspace, path))
            return str(fn(self._workspace, query, path))
        except ValueError as exc:
            return json.dumps({"error": "ValueError", "detail": str(exc)})

    def _remember(
        self, args: dict[str, object], step_index: int
    ) -> tuple[str, tuple[str, str, str, int] | None]:
        if not isinstance(args, dict):
            return json.dumps({"error": "unknown tool"}), None
        try:
            note = remember_note(
                self._workspace,
                str(args.get("path", "")),
                str(args.get("quote", "")),
                str(args.get("note", "")),
                step_index,
                self._secret,
            )
        except ValueError as exc:
            return json.dumps({"error": "ValueError", "detail": str(exc)}), None
        return json.dumps({"status": "remembered", "path": note[0]}), note

    def _mcp_call(self, name: str, args: dict[str, object]) -> str:
        if self._mcp is None or not isinstance(args, dict):
            return json.dumps({"error": "unknown tool"})
        parts = name.split("__", 2)
        if len(parts) < 3 or not parts[1] or not parts[2]:
            return json.dumps({"error": "unknown tool"})
        result = self._mcp.call(parts[1], parts[2], args)
        self._detail = result
        return result

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

    def _write_file(self, args: dict[str, object]) -> str:
        if not isinstance(args, dict):
            return json.dumps({"error": "unknown tool"})
        path = str(args.get("path", ""))
        try:
            _guard_new_path(path)
            target = self._workspace.resolve(path)
        except ValueError as exc:
            return json.dumps({"error": "ValueError", "detail": str(exc)})
        if target.exists():
            return json.dumps({"error": "FileExists", "path": path})
        self._workspace.write(path, str(args.get("content", "")).encode("utf-8"))
        return json.dumps({"status": "written", "path": path})

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
            return [{"role": "user", "content": self._prompt}]
        history: list[dict[str, object]] = []
        for _seq, role, body in rows:
            payload = json.loads(body)
            payload["role"] = role
            history.append(payload)
        return history

    def _note(self, text: str) -> None:
        if self._trace is not None:
            self._trace(text)

    def _result(self, status: str, cache_hit: bool, executions: int) -> RunResult:
        return RunResult(
            status, cache_hit, executions, self._llm.post_count(), self._detail
        )


def _shown(args_json: str, secret: str) -> str:
    text = redact(args_json, secret)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return json.dumps(parsed, indent=2)


def _guard_new_path(relative_path: str) -> None:
    path = Path(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError("path must stay under the work root")


def _trajectory(
    tool: str, step_index: int, args: dict[str, object], result: str
) -> tuple[tuple[str, int, str], ...]:
    try:
        parsed: object = json.loads(result)
    except json.JSONDecodeError:
        parsed = result
    return (
        ("model_inference", step_index, json.dumps({"text": "", "tool": tool})),
        ("tool_call", step_index, json.dumps({"tool": tool, "arguments": args})),
        ("tool_result", step_index, json.dumps({"tool": tool, "result": parsed})),
    )
