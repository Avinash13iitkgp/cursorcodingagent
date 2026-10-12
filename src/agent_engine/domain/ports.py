"""Ports the adapters implement. This module performs no I/O.

Methods take a tenant id and raise UnknownTenant or TenantMismatch when the
tenant file cannot be used. They do not open SQLite themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agent_engine.domain.models import EditResult, StepView, Task, Tenant, TenantId, TextEdit, ToolCall


class TenantDirectory(Protocol):
    """Creates and reads tenant databases under one data root."""

    def provision(
        self, data_root: Path, tenant_id: TenantId, display_name: str
    ) -> Path:
        """Create the tenant directory and engine.db.

        Args:
            data_root: Directory that contains tenants/.
            tenant_id: Tenant whose directory and tenant.id row must match.
            display_name: Name stored on the tenant row.

        Returns:
            Path to that tenant's engine.db.

        Raises:
            TenantMismatch: The resolved path leaves the data root, or an
                existing tenant.id row disagrees with tenant_id.
        """
        ...  # pragma: no cover

    def read(self, data_root: Path, tenant_id: TenantId) -> Tenant:
        """Return the tenant row from that tenant's database.

        Args:
            data_root: Directory that contains tenants/.
            tenant_id: Tenant file to open.

        Returns:
            The single tenant row.

        Raises:
            UnknownTenant: The tenant directory or engine.db is missing.
            TenantMismatch: tenant.id does not equal tenant_id.
        """
        ...  # pragma: no cover


class TaskLog(Protocol):
    """Pending tasks inside one tenant database."""

    def init_task(self, data_root: Path, tenant_id: TenantId, task_id: str) -> Task:
        """Insert a pending task whose workspace_root is relative.

        Args:
            data_root: Directory that contains tenants/.
            tenant_id: Tenant that owns the new row.
            task_id: Task primary key inside that tenant file.

        Returns:
            The stored task. A second call with the same id returns the existing row.

        Raises:
            UnknownTenant: The tenant database does not exist.
            TenantMismatch: The tenant file disagrees with tenant_id.
            ValueError: task_id is not a safe relative path segment.
        """
        ...  # pragma: no cover

    def get_task(self, data_root: Path, tenant_id: TenantId, task_id: str) -> Task:
        """Read one task from the tenant file.

        Args:
            data_root: Directory that contains tenants/.
            tenant_id: Tenant file to open.
            task_id: Task id inside that file.

        Returns:
            The stored task.

        Raises:
            UnknownTenant: The tenant database does not exist.
            TenantMismatch: The tenant file disagrees with tenant_id.
            TaskNotFound: The tenant file has no such task.
        """
        ...  # pragma: no cover


class Clock(Protocol):
    """Supplies the wall-clock instant used for lease expiry."""

    def now(self) -> float:
        """Return seconds since the Unix epoch.

        Returns:
            The current instant. Tests supply a manual value. Production uses
            time.time. This clock is not comparable across machines as a fence.
        """
        ...  # pragma: no cover


class LeaseStore(Protocol):
    """Lease rows inside the one tenant database already opened for this store."""

    def steal(
        self, tenant_id: TenantId, task_id: str, worker_id: str, now: float, expiry: float
    ) -> int:
        """Bump fencing_token when the lease is free or expired.

        The predicate is lease_owner IS NULL OR lease_expires_at <= now, inside
        BEGIN IMMEDIATE.

        Args:
            tenant_id: Tenant bound to this connection.
            task_id: Task row inside that file.
            worker_id: Worker that will own the new token.
            now: Instant compared with lease_expires_at.
            expiry: New lease_expires_at.

        Returns:
            The new fencing token.

        Raises:
            TaskNotFound: No task row.
            LeaseNotHeld: Zero rows because another owner is still unexpired.
        """
        ...  # pragma: no cover

    def renew(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
        expiry: float,
    ) -> None:
        """Extend lease_expires_at without changing fencing_token.

        The predicate is lease_owner, fencing_token, and lease_expires_at > now.

        Args:
            tenant_id: Tenant bound to this connection.
            task_id: Task row inside that file.
            worker_id: Current owner.
            token: Token the caller still holds.
            now: Instant compared with lease_expires_at.
            expiry: New lease_expires_at.

        Returns:
            None.

        Raises:
            TaskNotFound: No task row.
            FenceRejected: Zero rows because the token, owner, or expiry does not match.
        """
        ...  # pragma: no cover

    def release(
        self, tenant_id: TenantId, task_id: str, worker_id: str, token: int
    ) -> int:
        """Bump fencing_token and clear the owner.

        The predicate is lease_owner and fencing_token. The next acquire does
        not wait for the old expiry.

        Args:
            tenant_id: Tenant bound to this connection.
            task_id: Task row inside that file.
            worker_id: Current owner.
            token: Token the caller still holds.

        Returns:
            The token stored after the bump.

        Raises:
            TaskNotFound: No task row.
            FenceRejected: Zero rows because the token or owner does not match.
        """
        ...  # pragma: no cover

    def commit_marker(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Write a committed marker only while the fence still matches.

        The predicate is fencing_token, lease_owner, and lease_expires_at > now.
        Zero rows is FenceRejected when the task exists, and TaskNotFound when
        it does not.

        Args:
            tenant_id: Tenant bound to this connection.
            task_id: Task row inside that file.
            worker_id: Current owner.
            token: Token the caller still holds.
            now: Instant compared with lease_expires_at.

        Returns:
            None.

        Raises:
            TaskNotFound: No task row.
            FenceRejected: Zero rows because the fence does not match.
        """
        ...  # pragma: no cover


class Workspace(Protocol):
    """Files under one work root. The editor uses this port and not a raw base path."""

    def resolve(self, relative_path: str) -> Path:
        """Return the absolute path of a relative work-root entry.

        Args:
            relative_path: Path relative to the work root.

        Returns:
            The resolved path inside the work root.

        Raises:
            ValueError: The path is absolute or resolves outside the work root.
        """
        ...  # pragma: no cover

    def read(self, relative_path: str) -> bytes:
        """Read one file inside the work root.

        Args:
            relative_path: Path relative to the work root.

        Returns:
            The file bytes.

        Raises:
            ValueError: The path escapes the work root.
            FileNotFoundError: The file does not exist.
        """
        ...  # pragma: no cover

    def write(self, relative_path: str, data: bytes) -> None:
        """Replace one file inside the work root.

        Args:
            relative_path: Path relative to the work root.
            data: Bytes to store.

        Returns:
            None.

        Raises:
            ValueError: The path escapes the work root.
        """
        ...  # pragma: no cover

    def cleanup_tmp(self) -> None:
        """Delete leftover ``*.tmp`` files under the work root.

        Returns:
            None.
        """
        ...  # pragma: no cover

    def remove_file(self, relative_path: str) -> None:
        """Remove one work-tree file. Copy-on-write records a tombstone.

        Args:
            relative_path: Path relative to the work root.

        Returns:
            None.

        Raises:
            ValueError: The path escapes the work root.
            FileNotFoundError: The file does not exist.
        """
        ...  # pragma: no cover


class Editor(Protocol):
    """Optimistic edits. The loop calls this port and does not hash files itself."""

    def apply(self, workspace: Workspace, edit: TextEdit) -> EditResult:
        """Replace edit.old once when the preimage hash still matches.

        Args:
            workspace: Work root that owns the file.
            edit: Relative path, expected preimage hash, and the two snippets.

        Returns:
            applied or already_applied.

        Raises:
            ValueError: The path escapes the work root, or old is empty.
            StaleContent: The hash does not match, or old matches zero times.
            AmbiguousEdit: old matches more than once.
        """
        ...  # pragma: no cover

    def resume(self, workspace: Workspace) -> None:
        """Delete leftover temp files in the work root.

        Args:
            workspace: Work root that may contain ``*.tmp`` files from a crash.

        Returns:
            None.
        """
        ...  # pragma: no cover


class CheckRunner(Protocol):
    """Operator-owned checks. The model does not choose the argv."""

    def run(self, work_root: Path) -> tuple[bool, str, str]:
        """Run checks/commands inside the jail.

        Args:
            work_root: Task work tree. PYTHONPATH for the child points here.

        Returns:
            ok, stdout, and stderr. This method does not change tasks.status.

        Raises:
            None. A timeout or a non-zero exit is ok False.
        """
        ...  # pragma: no cover


class Llm(Protocol):
    """One Chat Completions turn. Offline tests supply a canned implementation."""

    def post_count(self) -> int:
        """Return how many completions this object has requested.

        Returns:
            The number of complete calls.
        """
        ...  # pragma: no cover

    def complete(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """Ask the model for the next tool calls.

        Args:
            messages: Transcript in Chat Completions shape.

        Returns:
            The tool calls from this message, in order. Empty when the model
            names no tool.

        Raises:
            ValueError: The response is not a chat completion.
        """
        ...  # pragma: no cover


class StepLog(Protocol):
    """Intent, cache, and transcript rows inside one tenant database."""

    def task_status(self, tenant_id: TenantId, task_id: str) -> str:
        """Return tasks.status for this task.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.

        Returns:
            The status text.

        Raises:
            TaskNotFound: The task row does not exist.
        """
        ...  # pragma: no cover

    def open_intent(self, tenant_id: TenantId, task_id: str) -> StepView | None:
        """Return the newest intent_logged agent step.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.

        Returns:
            The open step, or None. commit_marker rows are ignored.
        """
        ...  # pragma: no cover

    def next_index(self, tenant_id: TenantId, task_id: str) -> int:
        """Return the next step_index, including marker rows.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.

        Returns:
            One past the highest step_index, or 0 when the task has no steps.
        """
        ...  # pragma: no cover

    def agent_count(self, tenant_id: TenantId, task_id: str) -> int:
        """Count agent steps, excluding commit_marker.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.

        Returns:
            The number of agent step rows.
        """
        ...  # pragma: no cover

    def list_messages(self, tenant_id: TenantId, task_id: str) -> list[tuple[int, str, str]]:
        """Return the transcript in seq order.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.

        Returns:
            Tuples of seq, role, and body_json.
        """
        ...  # pragma: no cover

    def mark_running(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Set tasks.status to running while the fence matches.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.
            worker_id: Current lease owner.
            token: Fencing token the caller holds.
            now: Instant compared with lease_expires_at.

        Returns:
            None. Already running is a no-op when the fence matches.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
        """
        ...  # pragma: no cover

    def insert_intent(
        self,
        tenant_id: TenantId,
        task_id: str,
        step_index: int,
        idempotency_key: str,
        tool_name: str,
        args_json: str,
        worker_id: str,
        token: int,
        now: float,
        messages: tuple[tuple[str, str], ...],
    ) -> None:
        """Commit one intent_logged row before any tool side effect.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.
            step_index: Primary key inside the task.
            idempotency_key: SHA256 of the canonical step identity.
            tool_name: Tool the step will run.
            args_json: Model arguments.
            worker_id: Current lease owner.
            token: Fencing token the caller holds.
            now: Instant compared with lease_expires_at.
            messages: Role and body_json rows appended in this transaction.

        Returns:
            None. The transaction is committed before this method returns.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The fence does not match.
        """
        ...  # pragma: no cover

    def commit_step(
        self,
        tenant_id: TenantId,
        task_id: str,
        step_index: int,
        idempotency_key: str,
        result_json: str,
        worker_id: str,
        token: int,
        now: float,
        messages: tuple[tuple[str, str], ...],
    ) -> None:
        """Commit the step result, the cache row, and the transcript under the fence.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.
            step_index: Intent row to move from intent_logged to committed.
            idempotency_key: Cache key stored on the intent.
            result_json: Tool result.
            worker_id: Current lease owner.
            token: Fencing token the caller holds.
            now: Instant compared with lease_expires_at.
            messages: Role and body_json rows appended in this transaction.

        Returns:
            None.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
            ValueError: The fence matches and the intent row is absent.
        """
        ...  # pragma: no cover

    def cache_get(self, tenant_id: TenantId, idempotency_key: str) -> str | None:
        """Return a cached result from this tenant file.

        Args:
            tenant_id: Tenant bound to this store.
            idempotency_key: SHA256 key.

        Returns:
            result_json, or None when this file has no such key.
        """
        ...  # pragma: no cover

    def set_terminal(
        self,
        tenant_id: TenantId,
        task_id: str,
        status: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Write completed or rejected while the fence matches.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.
            status: completed or rejected.
            worker_id: Current lease owner.
            token: Fencing token the caller holds.
            now: Instant compared with lease_expires_at.

        Returns:
            None. The same terminal status already stored is a no-op.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
            ValueError: status is not completed or rejected.
        """
        ...  # pragma: no cover


class McpCaller(Protocol):
    """One stdio MCP call. The loop does not start the server itself."""

    def call(self, server: str, tool: str, arguments: dict[str, object]) -> str:
        """Run one namespaced tool and return its text.

        Args:
            server: Operator server name.
            tool: Tool name inside that server.
            arguments: Model arguments. The caller does not supply argv.

        Returns:
            Tool text, or a JSON error object. This method does not write
            tasks.status.
        """
        ...  # pragma: no cover
