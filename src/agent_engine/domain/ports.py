"""Ports the adapters implement. This module performs no I/O.

Methods take a tenant id and raise UnknownTenant or TenantMismatch when the
tenant file cannot be used. They do not open SQLite themselves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from agent_engine.domain.models import Task, Tenant, TenantId


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
