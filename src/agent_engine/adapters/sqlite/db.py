"""Open one SQLite file for one tenant.

This module owns path checks and the three connection pragmas: journal_mode=WAL,
synchronous=FULL, and busy_timeout=5000. It does not import the application
layer. Callers receive a connection they must close. UnknownTenant means the
file is missing. TenantMismatch means the resolved path or the tenant.id row
does not match the requested tenant.
"""

from __future__ import annotations

import sqlite3
from importlib.resources import files
from pathlib import Path

from agent_engine.domain.errors import TenantMismatch, UnknownTenant
from agent_engine.domain.models import TenantId

_SCHEMA = files("agent_engine.adapters.sqlite").joinpath("schema.sql").read_text(
    encoding="utf-8"
)


def resolve_tenant_dir(data_root: Path, tenant_id: TenantId) -> Path:
    """Return data_root/tenants/{tenant_id} after resolving symlinks.

    Args:
        data_root: Directory that contains tenants/. It may not exist yet.
        tenant_id: Directory name under tenants/.

    Returns:
        The resolved tenant directory. This function does not create it.

    Raises:
        TenantMismatch: The resolved tenants directory or tenant directory
            leaves data_root.
    """
    root = data_root.resolve()
    tenants = (root / "tenants").resolve()
    try:
        tenants.relative_to(root)
    except ValueError as exc:
        raise TenantMismatch(f"{tenants} is outside {root}") from exc
    tenant_dir = (tenants / tenant_id.value).resolve()
    try:
        tenant_dir.relative_to(tenants)
    except ValueError as exc:
        raise TenantMismatch(f"{tenant_dir} is outside {tenants}") from exc
    return tenant_dir


def tenant_db_path(data_root: Path, tenant_id: TenantId) -> Path:
    """Return the engine.db path for one tenant.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant whose file is engine.db inside its directory.

    Returns:
        Path to engine.db. The file is not created.

    Raises:
        TenantMismatch: The tenant directory or the database path resolves
            outside that tenant directory.
    """
    tenant_dir = resolve_tenant_dir(data_root, tenant_id)
    database = tenant_dir / "engine.db"
    if database.exists() or database.is_symlink():
        resolved = database.resolve()
        try:
            resolved.relative_to(tenant_dir)
        except ValueError as exc:
            raise TenantMismatch(f"{resolved} is outside {tenant_dir}") from exc
        return resolved
    return database


def apply_pragmas(connection: sqlite3.Connection) -> None:
    """Set WAL, synchronous=FULL, and a 5 second busy timeout on this connection.

    Args:
        connection: Open tenant database. The caller keeps ownership.

    Returns:
        None.
    """
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA busy_timeout=5000")


def apply_schema(connection: sqlite3.Connection) -> None:
    """Create the tenant tables and stamp triggers.

    Args:
        connection: Empty database connection. The caller commits.

    Returns:
        None.
    """
    connection.executescript(_SCHEMA)


def open_tenant_db(
    data_root: Path,
    tenant_id: TenantId,
    *,
    read_only: bool = False,
) -> sqlite3.Connection:
    """Open engine.db for tenant_id and check the tenant.id row before other SQL.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant file to open. No other tenant file is attached.
        read_only: When true, open with mode=ro and query_only=ON so the
            inspection itself does not write. Pragmas that change the file
            are applied only on a writable connection.

    Returns:
        An open connection with row_factory set to sqlite3.Row. The caller
        closes it. isolation_level is None so callers control transactions.

    Raises:
        UnknownTenant: engine.db does not exist.
        TenantMismatch: The path escapes the tenant directory, or tenant.id
            disagrees with tenant_id.
    """
    database = tenant_db_path(data_root, tenant_id)
    if not database.is_file():
        raise UnknownTenant(tenant_id.value)
    if read_only:
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro",
            uri=True,
            isolation_level=None,
        )
    else:
        connection = sqlite3.connect(database, isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        _assert_tenant_row(connection, tenant_id)
    except TenantMismatch:
        connection.close()
        raise
    if read_only:
        connection.execute("PRAGMA query_only=ON")
    else:
        apply_pragmas(connection)
    return connection


def _assert_tenant_row(connection: sqlite3.Connection, tenant_id: TenantId) -> None:
    try:
        row = connection.execute("SELECT id FROM tenant").fetchone()
    except sqlite3.Error as exc:
        raise TenantMismatch("tenant table is unreadable") from exc
    found = None if row is None else row["id"]
    if found != tenant_id.value:
        raise TenantMismatch(f"tenant.id is {found}, expected {tenant_id.value}")
