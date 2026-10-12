"""Provision tenants and pending tasks. Each function opens one tenant file.

This module does not read another tenant's database and does not import the
application layer. UnknownTenant means the file is missing. TenantMismatch
means the path or tenant.id row does not match. TaskNotFound means the task
row is absent. ValueError means the task id is not a safe path segment.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from agent_engine.adapters.sqlite.db import (
    apply_pragmas,
    apply_schema,
    open_tenant_db,
    resolve_tenant_dir,
    tenant_db_path,
)
from agent_engine.domain.errors import TaskNotFound
from agent_engine.domain.models import Task, Tenant, TenantId


def provision(data_root: Path, tenant_id: TenantId, display_name: str) -> Path:
    """Create tenants/{id}/ at mode 0700 and engine.db, or return an existing match.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Directory name and tenant.id value.
        display_name: Stored on the tenant row. Not used as a path.

    Returns:
        Path to engine.db. A second call for the same tenant returns that path
        when the existing tenant.id row matches.

    Raises:
        TenantMismatch: The directory resolves outside the data root, or the
            existing tenant.id row disagrees with tenant_id.
    """
    tenant_dir = resolve_tenant_dir(data_root, tenant_id)
    database = tenant_db_path(data_root, tenant_id)
    if database.is_file():
        connection = open_tenant_db(data_root, tenant_id)
        connection.close()
        return database
    tenant_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(tenant_dir, 0o700)
    connection = sqlite3.connect(database, isolation_level=None)
    try:
        apply_pragmas(connection)
        apply_schema(connection)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO tenant (id, display_name, llm_api_key_env) VALUES (?, ?, ?)",
            (tenant_id.value, display_name, "AGENT_LLM_API_KEY"),
        )
        connection.execute("COMMIT")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:
        connection.close()
        if database.exists():
            database.unlink()
        raise
    else:
        connection.close()
    return database


def read_tenant(data_root: Path, tenant_id: TenantId) -> Tenant:
    """Read the one tenant row.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant file to open.

    Returns:
        The stored tenant.

    Raises:
        UnknownTenant: engine.db does not exist.
        TenantMismatch: tenant.id disagrees with tenant_id.
    """
    connection = open_tenant_db(data_root, tenant_id, read_only=True)
    try:
        row = connection.execute(
            "SELECT id, display_name, llm_api_key_env FROM tenant"
        ).fetchone()
    finally:
        connection.close()
    return Tenant(
        id=TenantId(row["id"]),
        display_name=row["display_name"],
        llm_api_key_env=row["llm_api_key_env"],
    )


def init_task(data_root: Path, tenant_id: TenantId, task_id: str) -> Task:
    """Insert a pending task, or return the existing row with the same id.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant that owns the row.
        task_id: Primary key inside that tenant file.

    Returns:
        The task. workspace_root is tasks/{task_id}, relative to the tenant directory.

    Raises:
        UnknownTenant: The tenant database does not exist.
        TenantMismatch: The tenant file disagrees with tenant_id.
        ValueError: task_id is not a safe relative path segment.
    """
    draft = Task(
        tenant_id=tenant_id,
        id=task_id,
        status="pending",
        fencing_token=0,
        lease_owner=None,
        lease_expires_at=None,
        workspace_root=f"tasks/{task_id}",
    )
    connection = open_tenant_db(data_root, tenant_id)
    try:
        existing = _select_task(connection, tenant_id, task_id)
        if existing is not None:
            return existing
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO tasks (
                tenant_id, id, status, fencing_token, lease_owner,
                lease_expires_at, workspace_root
            ) VALUES (?, ?, 'pending', 0, NULL, NULL, ?)
            """,
            (tenant_id.value, task_id, draft.workspace_root),
        )
        connection.execute("COMMIT")
        stored = _select_task(connection, tenant_id, task_id)
    finally:
        connection.close()
    if stored is None:
        raise TaskNotFound(task_id)
    return stored


def get_task(data_root: Path, tenant_id: TenantId, task_id: str) -> Task:
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
        ValueError: task_id is not a safe relative path segment.
    """
    Task(
        tenant_id=tenant_id,
        id=task_id,
        status="pending",
        fencing_token=0,
        lease_owner=None,
        lease_expires_at=None,
        workspace_root=f"tasks/{task_id}",
    )
    connection = open_tenant_db(data_root, tenant_id, read_only=True)
    try:
        stored = _select_task(connection, tenant_id, task_id)
    finally:
        connection.close()
    if stored is None:
        raise TaskNotFound(task_id)
    return stored


def read_pragmas(data_root: Path, tenant_id: TenantId) -> tuple[str, int, int]:
    """Read journal_mode, synchronous, and busy_timeout from a writable connection.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant file to open.

    Returns:
        journal_mode text, synchronous integer (2 is FULL), and busy_timeout
        milliseconds.

    Raises:
        UnknownTenant: The tenant database does not exist.
        TenantMismatch: The tenant file disagrees with tenant_id.
    """
    connection = open_tenant_db(data_root, tenant_id)
    try:
        journal = connection.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        busy = connection.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        connection.close()
    return str(journal), int(synchronous), int(busy)


def _select_task(
    connection: sqlite3.Connection, tenant_id: TenantId, task_id: str
) -> Task | None:
    row = connection.execute(
        """
        SELECT tenant_id, id, status, fencing_token, lease_owner,
               lease_expires_at, workspace_root
        FROM tasks
        WHERE tenant_id = ? AND id = ?
        """,
        (tenant_id.value, task_id),
    ).fetchone()
    if row is None:
        return None
    return Task(
        tenant_id=TenantId(row["tenant_id"]),
        id=row["id"],
        status=row["status"],
        fencing_token=row["fencing_token"],
        lease_owner=row["lease_owner"],
        lease_expires_at=row["lease_expires_at"],
        workspace_root=row["workspace_root"],
    )
