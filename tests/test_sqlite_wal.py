"""Per-tenant SQLite files: WAL, isolation, and the stamp trigger."""

import shutil
import sqlite3
from pathlib import Path

import pytest

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.tenant_store import (
    get_task,
    init_task,
    provision,
    read_pragmas,
    read_tenant,
)
from agent_engine.domain.errors import TaskNotFound, TenantMismatch, UnknownTenant
from agent_engine.domain.models import TenantId


def test_provision_creates_two_distinct_engine_db_files(tmp_path: Path) -> None:
    """acme and globex each get their own engine.db, and provision is repeatable."""
    acme = provision(tmp_path, TenantId("acme"), "Acme")
    globex = provision(tmp_path, TenantId("globex"), "Globex")
    again = provision(tmp_path, TenantId("acme"), "Acme")
    assert acme == again
    assert acme.is_file() and globex.is_file()
    assert acme.resolve() != globex.resolve()
    assert acme.stat().st_ino != globex.stat().st_ino
    assert (tmp_path / "tenants" / "acme").stat().st_mode & 0o777 == 0o700
    assert read_tenant(tmp_path, TenantId("acme")).display_name == "Acme"
    assert read_tenant(tmp_path, TenantId("acme")).llm_api_key_env == "AGENT_LLM_API_KEY"


def test_connection_sets_wal_full_and_busy_timeout(tmp_path: Path) -> None:
    """A tenant connection reports WAL, synchronous=FULL (2), and busy_timeout=5000."""
    provision(tmp_path, TenantId("acme"), "Acme")
    journal, synchronous, busy = read_pragmas(tmp_path, TenantId("acme"))
    assert journal == "wal"
    assert synchronous == 2
    assert busy == 5000


def test_same_task_id_is_independent_across_tenant_files(tmp_path: Path) -> None:
    """Updating fix-adder in acme leaves globex's fix-adder at fencing token 0."""
    _provision_pair(tmp_path)
    init_task(tmp_path, TenantId("acme"), "fix-adder")
    init_task(tmp_path, TenantId("globex"), "fix-adder")
    connection = open_tenant_db(tmp_path, TenantId("acme"))
    try:
        connection.execute("UPDATE tasks SET fencing_token = 9 WHERE id = 'fix-adder'")
    finally:
        connection.close()
    assert get_task(tmp_path, TenantId("acme"), "fix-adder").fencing_token == 9
    readonly = open_tenant_db(tmp_path, TenantId("globex"), read_only=True)
    try:
        token = readonly.execute(
            "SELECT fencing_token FROM tasks WHERE id = 'fix-adder'"
        ).fetchone()["fencing_token"]
        with pytest.raises(sqlite3.OperationalError):
            readonly.execute("UPDATE tasks SET fencing_token = 3")
    finally:
        readonly.close()
    assert token == 0


def test_second_connection_reads_committed_task(tmp_path: Path) -> None:
    """A task committed on one connection is visible to the next connection."""
    provision(tmp_path, TenantId("acme"), "Acme")
    init_task(tmp_path, TenantId("acme"), "fix-adder")
    first = open_tenant_db(tmp_path, TenantId("acme"))
    first.close()
    second = open_tenant_db(tmp_path, TenantId("acme"))
    try:
        row = second.execute(
            "SELECT id, status, workspace_root FROM tasks"
        ).fetchone()
    finally:
        second.close()
    assert row["id"] == "fix-adder"
    assert row["status"] == "pending"
    assert row["workspace_root"] == "tasks/fix-adder"
    again = init_task(tmp_path, TenantId("acme"), "fix-adder")
    assert again.fencing_token == 0
    with pytest.raises(TaskNotFound):
        get_task(tmp_path, TenantId("acme"), "missing-task")


def test_open_rejects_path_outside_tenant_dir(tmp_path: Path) -> None:
    """A tenant path that resolves outside the data root creates no database."""
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "data"
    (root / "tenants").mkdir(parents=True)
    (root / "tenants" / "acme").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TenantMismatch):
        provision(root, TenantId("acme"), "Acme")
    assert not (outside / "engine.db").exists()
    escaped_root = tmp_path / "escaped-root"
    escaped_root.mkdir()
    (escaped_root / "tenants").symlink_to(outside, target_is_directory=True)
    with pytest.raises(TenantMismatch):
        provision(escaped_root, TenantId("acme"), "Acme")
    assert not (outside / "engine.db").exists()


def test_copied_file_with_wrong_tenant_row_raises(tmp_path: Path) -> None:
    """A database copied into another tenant directory fails the tenant.id check."""
    source = provision(tmp_path, TenantId("acme"), "Acme")
    globex_dir = tmp_path / "tenants" / "globex"
    globex_dir.mkdir()
    shutil.copy(source, globex_dir / "engine.db")
    with pytest.raises(TenantMismatch):
        open_tenant_db(tmp_path, TenantId("globex"))
    outside_db = tmp_path / "escaped.db"
    shutil.move(source, outside_db)
    source.symlink_to(outside_db)
    with pytest.raises(TenantMismatch):
        open_tenant_db(tmp_path, TenantId("acme"))


def test_missing_directory_raises_unknown_tenant(tmp_path: Path) -> None:
    """init-task before provision raises UnknownTenant and creates no database."""
    with pytest.raises(UnknownTenant):
        init_task(tmp_path, TenantId("acme"), "fix-adder")
    assert not (tmp_path / "tenants" / "acme" / "engine.db").exists()


def test_stamp_trigger_rolls_back_mismatched_tenant_id(tmp_path: Path) -> None:
    """An insert whose tenant_id disagrees with the tenant row is rolled back."""
    provision(tmp_path, TenantId("acme"), "Acme")
    connection = open_tenant_db(tmp_path, TenantId("acme"))
    try:
        with pytest.raises(sqlite3.IntegrityError, match="tenant mismatch"):
            connection.execute(
                """
                INSERT INTO tasks (tenant_id, id, status, workspace_root)
                VALUES ('globex', 'fix-adder', 'pending', 'tasks/fix-adder')
                """
            )
        count = connection.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
    finally:
        connection.close()
    assert count == 0
    empty = tmp_path / "tenants" / "blank"
    empty.mkdir()
    bare = sqlite3.connect(empty / "engine.db")
    bare.execute("CREATE TABLE other (id TEXT)")
    bare.commit()
    bare.close()
    with pytest.raises(TenantMismatch):
        open_tenant_db(tmp_path, TenantId("blank"))


def test_provision_deletes_database_when_setup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure while creating the tenant row removes the partial engine.db."""

    def fail_schema(connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("schema failed")

    monkeypatch.setattr(
        "agent_engine.adapters.sqlite.tenant_store.apply_schema",
        fail_schema,
    )
    with pytest.raises(sqlite3.OperationalError, match="schema failed"):
        provision(tmp_path, TenantId("acme"), "Acme")
    assert not (tmp_path / "tenants" / "acme" / "engine.db").exists()


def test_init_task_raises_when_inserted_row_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """init_task raises TaskNotFound when the insert is not visible afterward."""
    provision(tmp_path, TenantId("acme"), "Acme")

    def hide_row(connection: sqlite3.Connection, tenant_id: TenantId, task_id: str) -> None:
        return None

    monkeypatch.setattr(
        "agent_engine.adapters.sqlite.tenant_store._select_task",
        hide_row,
    )
    with pytest.raises(TaskNotFound):
        init_task(tmp_path, TenantId("acme"), "fix-adder")


def _provision_pair(data_root: Path) -> None:
    provision(data_root, TenantId("acme"), "Acme")
    provision(data_root, TenantId("globex"), "Globex")
