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
