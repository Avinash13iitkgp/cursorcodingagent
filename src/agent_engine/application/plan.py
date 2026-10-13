"""Plan one task before any file is written.

A plan call offers submit_plan, so the model cannot answer with write_file or
replace_text and this module never writes the work tree. When a codebase is
bound, the same call also offers list_files, search_text, and read_file, and
the session runs those lookups before it accepts submit_plan.
validate reads summary, approach, steps[{title, detail}], and
files[{path, action, detail}] with an action of create or edit, and rejects
an absolute path or one that walks out of the work root. render_plan prints
that as an execution plan: approach, numbered steps, then the file tree.
pinned packs the whole plan for the apply loop. This module does not import
sqlite or adapters.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_engine.application.context import list_files, search_text
from agent_engine.domain.models import ToolCall
from agent_engine.domain.ports import Workspace

_ACTIONS = ("create", "edit")
_SLUG_WORD = re.compile(r"[a-z0-9]+")
_SLUG_CHARS = 40
_PLAN_SHAPE = (
    "summary is one sentence. approach says how the work will be done. steps is "
    "the ordered execution plan: every step has a title and a detail that names "
    "the tool (write_file, replace_text, read_file, run_checks, or finish), the "
    "path it touches, and what that step changes. Cover creating or editing every "
    "file, then running the operator checks and finish. files lists every path "
    "from the work root down, including nested packages and tests. Each file has "
    "action create or edit and a detail of what the file will contain. Paths are "
    "relative to the work root."
)
_PLAN_RULES = (
    "Write the execution plan for the user prompt and call submit_plan once. "
    "This call cannot read or write files. " + _PLAN_SHAPE
)
_EXPLORE_RULES = (
    "The work tree already has files. Inspect it with list_files, search_text, "
    "and read_file, then call submit_plan once. Do not call write_file, "
    "replace_text, run_checks, or finish during planning. " + _PLAN_SHAPE
)
_LOOK = frozenset({"list_files", "search_text", "read_file"})
_PLAN_TURNS = 8


@dataclass(frozen=True)
class PlannedFile:
    """One path the apply loop is expected to create or edit."""

    path: str
    action: str
    detail: str


@dataclass(frozen=True)
class PlanStep:
    """One ordered action in the execution plan."""

    title: str
    detail: str


@dataclass(frozen=True)
class TaskPlan:
    """The execution plan the operator accepts or rejects."""

    summary: str
    approach: str
    steps: tuple[PlanStep, ...]
    files: tuple[PlannedFile, ...]


class Planner(Protocol):
    """One completion. A bound codebase also offers the read-only tools."""

    def plan(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """Ask the model for the next plan tool call.

        Args:
            messages: Prompt in Chat Completions shape, including lookup results.

        Returns:
            The tool calls from that message. write_file and replace_text are
            not offered.

        Raises:
            ValueError: The response is not a chat completion.
        """
        ...  # pragma: no cover


class PlanSession:
    """Model calls that return a validated plan and write nothing."""

    def __init__(
        self,
        planner: Planner,
        trace: Callable[[str], None] | None = None,
        codebase: Workspace | None = None,
    ) -> None:
        """Bind the session to one planning completion.

        Args:
            planner: Port that sends the planning tools.
            trace: Receives one line per model call and lookup. None stays quiet.
            codebase: Existing tree the model may list, search, and read. None
                means a new project, so the only accepted tool is submit_plan.
        """
        self._planner = planner
        self._trace = trace
        self._codebase = codebase

    def propose(self, prompt: str) -> TaskPlan:
        """Ask for one plan and validate it.

        Args:
            prompt: User sentence. The model plans the work for it.

        Returns:
            The validated plan. No lease is taken and no file is written. An
            existing codebase may be read first.

        Raises:
            ValueError: The prompt is empty, the model named a write tool on a
                new project, the lookups never reached submit_plan, or the
                submit_plan arguments are not a valid plan.
        """
        if not prompt.strip():
            raise ValueError("plan requires a prompt")
        rules = _EXPLORE_RULES if self._codebase is not None else _PLAN_RULES
        messages: list[dict[str, object]] = [
            {"role": "system", "content": rules},
            {"role": "user", "content": prompt},
        ]
        turns = _PLAN_TURNS if self._codebase is not None else 1
        for turn in range(turns):
            self._note("planning: calling the model" if turn == 0 else "planning: calling the model again")
            calls = self._planner.plan(messages)
            if not calls:
                raise ValueError("plan call named no tool")
            if self._codebase is None or all(call.name == "submit_plan" for call in calls):
                return _accepted_plan(calls[0])
            messages.append(_assistant_message(calls, turn))
            for index, call in enumerate(calls):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id or f"plan-{turn}-{index}",
                        "content": self._lookup(call),
                    }
                )
        raise ValueError("plan did not call submit_plan")

    def _lookup(self, call: ToolCall) -> str:
        self._note(f"planning: {call.name} {call.arguments_json}")
        if call.name == "submit_plan":
            result = json.dumps({"error": "call submit_plan after the lookups"})
        else:
            result = _inspect(self._codebase, call)
        self._note(f"planning result: {result}")
        return result

    def _note(self, text: str) -> None:
        if self._trace is not None:
            self._trace(text)


def validate_plan(payload: object) -> TaskPlan:
    """Check one submit_plan object.

    Args:
        payload: Parsed submit_plan arguments.

    Returns:
        The plan with its files in the order the model listed them.

    Raises:
        ValueError: summary or approach is missing, steps is empty, a step has
            no title or detail, files is empty, an action is not create or
            edit, a file detail is missing, or a path is absolute, empty,
            repeated, or walks out of the work root.
    """
    if not isinstance(payload, dict):
        raise ValueError("plan must be an object")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("plan needs a summary")
    approach = payload.get("approach")
    if not isinstance(approach, str) or not approach.strip():
        raise ValueError("plan needs an approach")
    steps = _steps(payload.get("steps"))
    raw = payload.get("files")
    if not isinstance(raw, list) or not raw:
        raise ValueError("plan needs at least one file")
    files: list[PlannedFile] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("a plan file must be an object")
        path = _plan_path(item.get("path"))
        if path in seen:
            raise ValueError(f"{path} is listed twice")
        seen.add(path)
        action = item.get("action")
        if action not in _ACTIONS:
            raise ValueError(f"action for {path} must be create or edit")
        detail = item.get("detail")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError(f"{path} needs a detail of what the file will contain")
        files.append(PlannedFile(path, str(action), detail.strip()))
    return TaskPlan(summary.strip(), approach.strip(), steps, tuple(files))


def plan_body(plan: TaskPlan) -> str:
    """Render one plan as the stored body.

    Args:
        plan: Validated plan.

    Returns:
        JSON with summary, approach, steps, and files. validate_plan reads it back.
    """
    return json.dumps(
        {
            "summary": plan.summary,
            "approach": plan.approach,
            "steps": [{"title": step.title, "detail": step.detail} for step in plan.steps],
            "files": [
                {"path": entry.path, "action": entry.action, "detail": entry.detail}
                for entry in plan.files
            ],
        }
    )


def load_plan(body_json: str) -> TaskPlan:
    """Read one stored plan body.

    Args:
        body_json: Body written by plan_body.

    Returns:
        The validated plan.

    Raises:
        ValueError: The body is not JSON or is not a valid plan.
    """
    try:
        payload = json.loads(body_json)
    except json.JSONDecodeError as exc:
        raise ValueError("stored plan is not JSON") from exc
    return validate_plan(payload)


def render_tree(plan: TaskPlan) -> str:
    """Draw the planned paths as a directory tree.

    Args:
        plan: Validated plan.

    Returns:
        One line per directory and file, two spaces per level, sorted with
        directories before files.
    """
    root: dict[str, dict[str, object]] = {}
    for entry in plan.files:
        node = root
        parts = entry.path.split("/")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            node = child  # type: ignore[assignment]
        node.setdefault(parts[-1], {})
    lines: list[str] = []
    _draw(root, 0, lines)
    return "\n".join(lines)


def render_plan(plan: TaskPlan) -> str:
    """Render the execution plan, including the file tree.

    Args:
        plan: Validated plan.

    Returns:
        A markdown document: summary, approach, numbered steps, the tree, and
        one block per file. Nothing is truncated.
    """
    lines = ["# Plan", "", plan.summary, "", "## Approach", "", plan.approach, "", "## Steps", ""]
    for index, step in enumerate(plan.steps, start=1):
        lines.append(f"{index}. {step.title}")
        for detail_line in step.detail.splitlines():
            lines.append(f"   {detail_line}")
        lines.append("")
    lines.extend(["## Files", "", render_tree(plan), ""])
    for entry in plan.files:
        lines.append(f"- {entry.action} {entry.path}")
        for detail_line in entry.detail.splitlines():
            lines.append(f"  {detail_line}")
    return "\n".join(lines).rstrip()


def pinned_plan(plan: TaskPlan) -> str:
    """Render the accepted plan for the packed system rules.

    Args:
        plan: Accepted plan. It is packed whole and is not truncated.

    Returns:
        The summary, the approach, every step, and one line per planned path.
    """
    lines = [f"Plan: {plan.summary}", f"Approach: {plan.approach}", "Steps:"]
    for index, step in enumerate(plan.steps, start=1):
        lines.append(f"{index}. {step.title}")
        lines.append(f"   {step.detail}")
    lines.append("Planned files:")
    for entry in plan.files:
        lines.append(f"{entry.action} {entry.path} - {entry.detail}")
    return "\n".join(lines)


def apply_prompt(prompt: str) -> str:
    """Render the task sentence the apply loop stores first.

    Args:
        prompt: User sentence stored with the plan.

    Returns:
        The sentence and the tools apply offers for a new or existing path.
    """
    return (
        f"{prompt}\n"
        "Execute the accepted plan in order. Use write_file for a new path "
        "and replace_text with the hash from read_file for a path that "
        "already exists. Do not list or search the tree. Call run_checks and "
        "finish when the tree is written. Do not modify checks."
    )


def task_slug(prompt: str) -> str:
    """Derive a task id from a prompt.

    Args:
        prompt: User sentence. Only letters and digits are kept.

    Returns:
        A lowercase slug of at most 40 characters that matches the task id
        shape. Whole words are kept, so the caller can add a suffix.

    Raises:
        ValueError: The prompt has no letter or digit.
    """
    words = _SLUG_WORD.findall(prompt.lower())
    if not words:
        raise ValueError("prompt has no word to name a task")
    parts: list[str] = [words[0][:_SLUG_CHARS]]
    length = len(parts[0])
    for word in words[1:]:
        if length + 1 + len(word) > _SLUG_CHARS:
            break
        parts.append(word)
        length += 1 + len(word)
    return "-".join(parts)


def _draw(node: dict[str, dict[str, object]], depth: int, lines: list[str]) -> None:
    for name in sorted(node, key=lambda key: (not node[key], key)):
        child = node[name]
        indent = "  " * depth
        if child:
            lines.append(f"{indent}{name}/")
            _draw(child, depth + 1, lines)  # type: ignore[arg-type]
        else:
            lines.append(f"{indent}{name}")


def _accepted_plan(call: ToolCall) -> TaskPlan:
    if call.name != "submit_plan":
        raise ValueError(f"plan call returned {call.name}")
    try:
        payload = json.loads(call.arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError("submit_plan arguments are not JSON") from exc
    return validate_plan(payload)


def _assistant_message(calls: tuple[ToolCall, ...], turn: int) -> dict[str, object]:
    tool_calls: list[dict[str, object]] = []
    for index, call in enumerate(calls):
        tool_calls.append(
            {
                "id": call.call_id or f"plan-{turn}-{index}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
        )
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _inspect(codebase: Workspace | None, call: ToolCall) -> str:
    if codebase is None or call.name not in _LOOK:
        return json.dumps({"error": "planning cannot write or run checks"})
    try:
        args = json.loads(call.arguments_json)
    except json.JSONDecodeError:
        return json.dumps({"error": "arguments are not JSON"})
    if not isinstance(args, dict):
        return json.dumps({"error": "arguments must be an object"})
    path = str(args.get("path", "." if call.name != "read_file" else ""))
    try:
        if call.name == "list_files":
            return list_files(codebase, path)
        if call.name == "search_text":
            return search_text(codebase, str(args.get("query", "")), path)
        return _read_for_plan(codebase, path)
    except (ValueError, FileNotFoundError, OSError) as exc:
        return json.dumps({"error": type(exc).__name__, "detail": str(exc)})


def _read_for_plan(codebase: Workspace, path: str) -> str:
    data = codebase.read(path)
    return json.dumps(
        {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "content": data.decode("utf-8", errors="replace"),
        }
    )


def _steps(value: object) -> tuple[PlanStep, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("plan needs at least one step")
    steps: list[PlanStep] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("a plan step must be an object")
        title = item.get("title")
        detail = item.get("detail")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("a plan step needs a title")
        if not isinstance(detail, str) or not detail.strip():
            raise ValueError(f"step {title!r} needs a detail")
        steps.append(PlanStep(title.strip(), detail.strip()))
    return tuple(steps)


def _plan_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a plan file needs a path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise ValueError(f"{value} must stay under the work root")
    return path.as_posix()
