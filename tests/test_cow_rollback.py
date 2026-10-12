"""Copy-on-write hardlinks, tombstones, and fenced rollback."""

import errno
import json
import os
import time
from pathlib import Path

import pytest

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.step_store import SqliteStepStore
from agent_engine.adapters.workspace.cow import CowWorkspace
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.demo import _commit_read, prepare_task
from agent_engine.domain.errors import FenceRejected
from agent_engine.domain.models import TenantId

_WORKER = "33333333-3333-4333-8333-333333333333"


def test_seed_then_write_changes_only_work_inode(tmp_path: Path) -> None:
    """Seeding hardlinks base and work. A later write replaces only the work inode."""
    workspace = prepare_task(tmp_path, "fix-adder")
    base = workspace.work_root.parent / "base" / "adder.py"
    work = workspace.work_root / "adder.py"
    assert base.stat().st_ino == work.stat().st_ino
    workspace.write("adder.py", b"changed\n")
    assert work.stat().st_ino != base.stat().st_ino
    assert base.read_bytes() != work.read_bytes()


def test_copy_break_does_not_modify_base_bytes(tmp_path: Path) -> None:
    """Copy-break leaves the base inode bytes and link count alone."""
    workspace = prepare_task(tmp_path, "fix-adder")
    base = workspace.work_root.parent / "base" / "adder.py"
    original = base.read_bytes()
    workspace.write("adder.py", original + b"\n")
    assert base.read_bytes() == original
    assert base.stat().st_nlink == 1


def test_remove_file_tombstones_and_keeps_base(tmp_path: Path) -> None:
    """remove_file drops the work entry, records a tombstone, and keeps base."""
    workspace = prepare_task(tmp_path, "fix-adder")
    base = workspace.work_root.parent / "base" / "adder.py"
    original = base.read_bytes()
    workspace.remove_file("adder.py")
    assert not (workspace.work_root / "adder.py").exists()
    assert base.read_bytes() == original
    workspace.snapshot(1)
    tombstones = json.loads(
        (workspace.work_root.parent / "snapshots" / "1" / "tombstones.json").read_text(
            encoding="utf-8"
        )
    )
    assert tombstones == ["adder.py"]
    workspace.write("adder.py", b"recreated\n")
    workspace.relink(1)
    assert not (workspace.work_root / "adder.py").exists()
    assert base.read_bytes() == original


def test_rollback_restores_bytes_and_marks_steps_rolled_back(tmp_path: Path) -> None:
    """Rollback restores the snapshot and marks later steps and cache rows rolled_back."""
    workspace = prepare_task(tmp_path, "fix-adder")
    original = (workspace.work_root / "adder.py").read_bytes()
    tenant_id = TenantId("acme")
    connection = open_tenant_db(tmp_path, tenant_id)
    try:
        now = time.time()
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        lease = service.acquire("fix-adder", _WORKER, now, 60)
        store = SqliteStepStore(connection)
        _commit_read(store, tenant_id, "fix-adder", _WORKER, lease.fencing_token, now, 0)
        workspace.snapshot(0)
        workspace.write("adder.py", b"later\n")
        _commit_read(store, tenant_id, "fix-adder", _WORKER, lease.fencing_token, time.time(), 1)
        connection.execute("ALTER TABLE trajectory ADD COLUMN status TEXT")
        connection.execute(
            """
            INSERT INTO trajectory (tenant_id, task_id, seq, step_index, kind, body_json, status)
            VALUES ('acme', 'fix-adder', 1, 1, 'tool_result', '{}', 'committed')
            """
        )
        connection.execute(
            """
            INSERT INTO trajectory (tenant_id, task_id, seq, step_index, kind, body_json, status)
            VALUES ('acme', 'fix-adder', 0, 0, 'tool_call', '{}', 'committed')
            """
        )
        workspace.rollback(
            0, connection, "acme", "fix-adder", _WORKER, lease.fencing_token, time.time()
        )
        with pytest.raises(FenceRejected):
            workspace.rollback(0, connection, "acme", "fix-adder", _WORKER, 99, time.time())
    finally:
        connection.close()
    assert (workspace.work_root / "adder.py").read_bytes() == original
    connection = open_tenant_db(tmp_path, tenant_id, read_only=True)
    try:
        steps = {
            int(row["step_index"]): row["status"]
            for row in connection.execute("SELECT step_index, status FROM step_intents")
        }
        cache = {
            row["step_index"]: row["status"]
            for row in connection.execute(
                """
                SELECT step_intents.step_index AS step_index, idempotency_cache.status AS status
                FROM step_intents
                JOIN idempotency_cache
                  ON idempotency_cache.idempotency_key = step_intents.idempotency_key
                """
            )
        }
        trajectory = {
            int(row["step_index"]): row["status"]
            for row in connection.execute("SELECT step_index, status FROM trajectory")
        }
    finally:
        connection.close()
    assert steps == {0: "committed", 1: "rolled_back"}
    assert cache[0] == "committed"
    assert cache[1] == "rolled_back"
    assert trajectory == {0: "committed", 1: "rolled_back"}
    assert store_cache_misses_rolled_back(tmp_path)


def test_rollback_relink_is_under_5ms(tmp_path: Path) -> None:
    """The relink itself, not the database update, finishes in under 5 ms."""
    workspace = prepare_task(tmp_path, "fix-adder")
    workspace.snapshot(2)
    workspace.write("adder.py", b"dirty\n")
    started = time.perf_counter()
    workspace.relink(2)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 5
    assert (workspace.work_root / "adder.py").read_bytes() != b"dirty\n"


def test_os_link_exdev_is_not_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """os.link raising EXDEV fails the snapshot instead of copying bytes."""
    workspace = prepare_task(tmp_path, "fix-adder")

    def _cross_device(*_args: object, **_kwargs: object) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", _cross_device)
    with pytest.raises(OSError) as caught:
        workspace.snapshot(4)
    assert caught.value.errno == errno.EXDEV


def test_escape_and_tmp_stay_inside_work(tmp_path: Path) -> None:
    """Paths outside work raise, and cleanup deletes only in-root temp files."""
    workspace = prepare_task(tmp_path, "fix-adder")
    with pytest.raises(ValueError):
        workspace.read("../base/adder.py")
    leftover = workspace.work_root / "note.tmp"
    leftover.write_bytes(b"tmp")
    workspace.cleanup_tmp()
    assert not leftover.exists()


def store_cache_misses_rolled_back(tmp_path: Path) -> bool:
    """Return whether cache_get hides the rolled_back step."""
    from agent_engine.application.react import idempotency_key

    tenant_id = TenantId("acme")
    connection = open_tenant_db(tmp_path, tenant_id, read_only=True)
    try:
        store = SqliteStepStore(connection)
        key = idempotency_key("acme", "fix-adder", 1, "read_file", {"path": "adder.py"})
        return store.cache_get(tenant_id, key) is None
    finally:
        connection.close()
