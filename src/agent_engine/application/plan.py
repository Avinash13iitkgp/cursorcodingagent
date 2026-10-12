"""Plan one task before any file is written.

A plan call offers one tool, submit_plan, so the model cannot answer with
write_file or replace_text and this module never touches the work tree.
validate reads summary and files[{path, action, detail}] with an action of
create or edit, and rejects an absolute path or one that walks out of the work
root. tree renders the stored paths from the work root down. pinned packs the
user sentence and those paths for the apply loop. This module does not import
sqlite or adapters.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from agent_engine.domain.models import ToolCall

_ACTIONS = ("create", "edit")
_PLAN_RULES = (
    "Plan the work tree for the user prompt. Call submit_plan once. List every "
    "file path from the work root down, including nested packages and test "
    "files. Paths are relative to the work root. This call cannot read or "
    "write files."
)


@dataclass(frozen=True)
class PlannedFile:
    """One path the apply loop is expected to create or edit."""

    path: str
    action: str
    detail: str


@dataclass(frozen=True)
class TaskPlan:
    """A summary and the file list the operator accepts or rejects."""

    summary: str
    files: tuple[PlannedFile, ...]


class Planner(Protocol):
    """One completion that offers submit_plan and no file tool."""

    def plan(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """Ask the model for one plan.

        Args:
            messages: Prompt in Chat Completions shape.

        Returns:
            The tool calls from that message. Only submit_plan is offered.

        Raises:
            ValueError: The response is not a chat completion.
        """
        ...  # pragma: no cover


class PlanSession:
    """One model call that returns a validated plan and writes nothing."""

    def __init__(self, planner: Planner) -> None:
        """Bind the session to one planning completion.

        Args:
            planner: Port that sends submit_plan as the only tool.
        """
        self._planner = planner

    def propose(self, prompt: str) -> TaskPlan:
        """Ask for one plan and validate it.

        Args:
            prompt: User sentence. The model plans the tree for it.

        Returns:
            The validated plan. No lease is taken and no file is written.

        Raises:
            ValueError: The prompt is empty, the model named another tool, or
                the submit_plan arguments are not a valid plan.
        """
        if not prompt.strip():
            raise ValueError("plan requires a prompt")
        calls = self._planner.plan(
            [
                {"role": "system", "content": _PLAN_RULES},
                {"role": "user", "content": prompt},
            ]
        )
        if not calls:
            raise ValueError("plan call named no tool")
        first = calls[0]
        if first.name != "submit_plan":
            raise ValueError(f"plan call returned {first.name}")
        try:
            payload = json.loads(first.arguments_json)
        except json.JSONDecodeError as exc:
            raise ValueError("submit_plan arguments are not JSON") from exc
        return validate_plan(payload)


def validate_plan(payload: object) -> TaskPlan:
    """Check one submit_plan object.

    Args:
        payload: Parsed submit_plan arguments.

    Returns:
        The plan with its files in the order the model listed them.

    Raises:
        ValueError: summary is missing, files is empty, an action is not
            create or edit, or a path is absolute, empty, repeated, or walks
            out of the work root.
    """
    if not isinstance(payload, dict):
        raise ValueError("plan must be an object")
    summary = payload.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("plan needs a summary")
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
        files.append(PlannedFile(path, str(action), "" if detail is None else str(detail)))
    return TaskPlan(summary.strip(), tuple(files))


def plan_body(plan: TaskPlan) -> str:
    """Render one plan as the stored body.

    Args:
        plan: Validated plan.

    Returns:
        JSON with summary and files. validate_plan reads it back.
    """
    return json.dumps(
        {
            "summary": plan.summary,
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


def pinned_plan(plan: TaskPlan) -> str:
    """Render the accepted plan for the packed system rules.

    Args:
        plan: Accepted plan. It is packed whole and is not truncated.

    Returns:
        The summary and one line per planned path.
    """
    lines = [f"Plan: {plan.summary}", "Planned files:"]
    for entry in plan.files:
        suffix = f" - {entry.detail}" if entry.detail else ""
        lines.append(f"{entry.action} {entry.path}{suffix}")
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
        "Use write_file for a new path and replace_text with the hash from "
        "read_file for a path that already exists. Call run_checks and finish "
        "when the tree is written. Do not modify checks."
    )


def _draw(node: dict[str, dict[str, object]], depth: int, lines: list[str]) -> None:
    for name in sorted(node, key=lambda key: (not node[key], key)):
        child = node[name]
        indent = "  " * depth
        if child:
            lines.append(f"{indent}{name}/")
            _draw(child, depth + 1, lines)  # type: ignore[arg-type]
        else:
            lines.append(f"{indent}{name}")


def _plan_path(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("a plan file needs a path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise ValueError(f"{value} must stay under the work root")
    return path.as_posix()
