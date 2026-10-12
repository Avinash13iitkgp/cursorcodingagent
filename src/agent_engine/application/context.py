"""Context packer for one model call.

Pack order is fixed: system rules and the task prompt, then knowledge notes
newest first, then the newest trajectory steps in full. Size is len(text) // 4.
When that estimate exceeds the budget, older tool results shrink to an outcome
line: tool name, StaleContent or exit code, and the sha256. File bodies are
not repeated. If the pack is still over, one tool-free summary call compresses
the dropped middle into a context_summary row. The original rows stay stored.
The prompt keeps the newest summary plus the newest steps. This builder does
not set completed or rejected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from agent_engine.domain.models import TenantId
from agent_engine.domain.ports import Workspace

_RULES = (
    "Use read_file, replace_text, run_checks, finish, list_files, search_text, and remember. "
    "Do not modify checks or add a compiler install. Search is not a shell."
)
_TASK = "Fix adder.py so add(a, b) returns a + b."
_KINDS = ("model_inference", "tool_call", "tool_result")


class ContextLog(Protocol):
    """Trajectory and knowledge reads. The builder does not open SQLite."""

    def list_trajectory(
        self, tenant_id: TenantId, task_id: str
    ) -> list[tuple[int, str, int | None, str]]:
        """Return seq, kind, step_index, and body_json.

        Args:
            tenant_id: Tenant whose file holds the log.
            task_id: Task primary key.

        Returns:
            Rows in seq order.
        """
        ...  # pragma: no cover

    def list_knowledge(
        self, tenant_id: TenantId, task_id: str
    ) -> list[tuple[int, str, str, str]]:
        """Return note_id, path, quote, and body, newest first.

        Args:
            tenant_id: Tenant whose file holds the notes.
            task_id: Task primary key.

        Returns:
            Notes in note_id order, newest first.
        """
        ...  # pragma: no cover

    def write_summary(self, tenant_id: TenantId, task_id: str, summary: str) -> None:
        """Store one context_summary without changing task status.

        Args:
            tenant_id: Tenant whose file holds the log.
            task_id: Task primary key.
            summary: Redacted summary text.

        Returns:
            None.
        """
        ...  # pragma: no cover


class ContextBuilder:
    """Pack a model prompt from the trajectory log and the knowledge notes."""

    def __init__(
        self,
        log: ContextLog | None = None,
        budget: int = 8000,
        summarize=None,  # type: ignore[no-untyped-def]
        secret: str = "",
    ) -> None:
        """Bind the packer to one log and one budget.

        Args:
            log: Trajectory and knowledge port. None packs the raw history.
            budget: Maximum len(text) // 4. The default is 8000.
            summarize: Tool-free completion. None skips the summary call.
            secret: Bearer token removed from notes, trajectory text, and the
                summary request.

        Returns:
            None.
        """
        self._log = log
        self._budget = budget
        self._summarize = summarize
        self._secret = secret
        self.last_messages: list[dict[str, object]] = []

    def pack(
        self,
        tenant_id: TenantId,
        task_id: str,
        history: list[dict[str, object]] | None = None,
    ) -> list[dict[str, object]]:
        """Return the messages for one model call.

        Args:
            tenant_id: Tenant whose notes and trajectory are packed.
            task_id: Task primary key.
            history: Raw transcript used when this builder has no log.

        Returns:
            Chat messages. System rules and the task prompt stay even when
            the estimate exceeds the budget.
        """
        if self._log is None:
            messages = list(history or [])
            self.last_messages = messages
            if _tokens(messages) <= self._budget:
                return messages
            shrunk = _shrink_history(messages)
            self.last_messages = shrunk
            return shrunk
        notes = self._log.list_knowledge(tenant_id, task_id)
        rows = self._log.list_trajectory(tenant_id, task_id)
        full = _render(notes, rows, shrink=False, summary="")
        if _tokens(full) <= self._budget:
            self.last_messages = full
            return full
        shrunk = _render(notes, rows, shrink=True, summary="")
        if _tokens(shrunk) <= self._budget:
            self.last_messages = shrunk
            return shrunk
        middle = _middle_text(rows)
        summary = self._summary(middle)
        if summary:
            self._log.write_summary(tenant_id, task_id, redact(summary, self._secret))
        packed = _render(notes, rows, shrink=True, summary=summary)
        self.last_messages = packed
        return packed

    def _summary(self, middle: str) -> str:
        cleaned = redact(middle, self._secret)
        if not cleaned.strip() or self._summarize is None:
            return ""
        text = self._summarize([{"role": "user", "content": cleaned}])
        return redact(str(text), self._secret)


def list_files(workspace: Workspace, relative_path: str = ".") -> str:
    """List relative file paths under the work root.

    Args:
        workspace: Work-root port. Paths cannot leave it.
        relative_path: Directory under the work root. The default is the root.

    Returns:
        JSON with a paths array.

    Raises:
        ValueError: The path is absolute or leaves the work root. Nothing is written.
    """
    paths = _files(workspace, relative_path)
    return json.dumps({"paths": paths})


def search_text(workspace: Workspace, query: str, relative_path: str = ".") -> str:
    """Find a literal substring and return at most 20 hits.

    Args:
        workspace: Work-root port. Paths cannot leave it.
        query: Literal text. This is not a shell command and not argv.
        relative_path: Directory under the work root. The default is the root.

    Returns:
        JSON hits with path, line, and a short snippet.

    Raises:
        ValueError: The path is absolute or leaves the work root. Nothing is written.
    """
    hits: list[dict[str, object]] = []
    for rel in _files(workspace, relative_path):
        text = workspace.read(rel).decode("utf-8", errors="replace")
        for number, line in enumerate(text.splitlines(), start=1):
            if query not in line:
                continue
            hits.append({"path": rel, "line": number, "snippet": line.strip()[:80]})
            if len(hits) >= 20:
                return json.dumps({"hits": hits})
    return json.dumps({"hits": hits})


def remember_note(
    workspace: Workspace,
    path: str,
    quote: str,
    note: str,
    source_step: int,
    secret: str,
) -> tuple[str, str, str, int]:
    """Check the path and redact the bearer token from a knowledge note.

    Args:
        workspace: Work-root port. The note does not write the file.
        path: Path relative to the work root.
        quote: Quoted source text.
        note: The model's own note.
        source_step: Step index stored with the note.
        secret: Bearer token replaced with ``[redacted]``.

    Returns:
        Path, quote, body, and source step.

    Raises:
        ValueError: The path is absolute or leaves the work root. Nothing is written.
    """
    _guard(path)
    workspace.resolve(path)
    return path, redact(quote, secret), redact(note, secret), source_step


def redact(text: str, secret: str) -> str:
    """Replace the bearer token in text.

    Args:
        text: Note, result, or summary input.
        secret: Token value. An empty secret leaves text unchanged.

    Returns:
        Text with the token replaced by ``[redacted]``.
    """
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


def outcome_line(body: dict[str, object]) -> str:
    """Shrink one tool result to tool name, outcome, and sha256.

    Args:
        body: Stored tool_result object.

    Returns:
        One outcome line. File bodies are omitted.
    """
    tool = str(body.get("tool", "tool"))
    result = body.get("result")
    blob = json.dumps(body)
    sha = ""
    if isinstance(result, dict) and result.get("sha256"):
        sha = str(result["sha256"])
    elif body.get("sha256"):
        sha = str(body["sha256"])
    if "StaleContent" in blob:
        return f"{tool} StaleContent {sha}".rstrip()
    if isinstance(result, dict) and "ok" in result:
        code = 0 if result["ok"] else 1
        return f"{tool} exit {code} {sha}".rstrip()
    return f"{tool} ok {sha}".rstrip()


def _files(workspace: Workspace, relative_path: str) -> list[str]:
    _guard(relative_path)
    root = workspace.resolve(".").resolve()
    start = workspace.resolve(relative_path)
    if start.is_file():
        return [start.resolve().relative_to(root).as_posix()]
    found: list[str] = []
    if not start.exists():
        return found
    for path in start.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            rel = path.resolve().relative_to(root)
        except ValueError:
            continue
        if path.suffix == ".tmp":
            continue
        found.append(rel.as_posix())
    return sorted(found)


def _guard(relative_path: str) -> None:
    path = Path(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts:
        raise ValueError("path must stay under the work root")


def _render(
    notes: list[tuple[int, str, str, str]],
    rows: list[tuple[int, str, int | None, str]],
    *,
    shrink: bool,
    summary: str,
) -> list[dict[str, object]]:
    messages: list[dict[str, object]] = [
        {"role": "system", "content": _RULES + "\n" + _TASK}
    ]
    for _note_id, path, quote, body in notes:
        messages.append({"role": "user", "content": f"note {path}: {quote}\n{body}"})
    if summary:
        messages.append({"role": "user", "content": summary})
    else:
        stored = [row for row in rows if row[1] == "context_summary"]
        if stored:
            messages.append({"role": "user", "content": stored[-1][3]})
    indexes = _indexes(rows)
    if not indexes:
        return messages
    newest = indexes[-1]
    for step_index in indexes:
        messages.append(
            {
                "role": "user",
                "content": _step_text(rows, step_index, shrink=shrink and step_index != newest),
            }
        )
    return messages


def _middle_text(rows: list[tuple[int, str, int | None, str]]) -> str:
    indexes = _indexes(rows)
    if len(indexes) < 2:
        return ""
    parts: list[str] = []
    for step_index in indexes[:-1]:
        parts.append(_step_text(rows, step_index, shrink=True))
    return "\n".join(parts)


def _indexes(rows: list[tuple[int, str, int | None, str]]) -> list[int]:
    found: list[int] = []
    for _seq, kind, step_index, _body in rows:
        if kind not in _KINDS or step_index is None or step_index in found:
            continue
        found.append(step_index)
    return found


def _step_text(
    rows: list[tuple[int, str, int | None, str]], step_index: int, *, shrink: bool
) -> str:
    parts: list[str] = []
    for _seq, kind, row_step, body in rows:
        if row_step != step_index or kind not in _KINDS:
            continue
        if shrink and kind == "tool_result":
            parts.append(outcome_line(json.loads(body)))
        else:
            parts.append(body)
    return "\n".join(parts)


def _shrink_history(messages: list[dict[str, object]]) -> list[dict[str, object]]:
    shrunk: list[dict[str, object]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and len(content) > 160:
            text = content
            if "StaleContent" in text:
                line = "tool StaleContent"
            else:
                line = "tool ok"
            copied = dict(message)
            copied["content"] = line
            shrunk.append(copied)
            continue
        shrunk.append(message)
    return shrunk


def _tokens(messages: list[dict[str, object]]) -> int:
    return len(json.dumps(messages)) // 4
