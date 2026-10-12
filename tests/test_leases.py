"""Fenced leases: the token is the commit capability and expiry is only liveness."""

import threading
from pathlib import Path

import pytest

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.tenant_store import get_task, init_task, provision
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.domain.errors import FenceRejected, LeaseNotHeld, TaskNotFound, TenantMismatch
from agent_engine.domain.models import Lease, TenantId


class ManualClock:
    """Fixed instant. Lease calls pass now explicitly, so this clock stays put."""

    def now(self) -> float:
        """Return the fixed instant.

        Returns:
            Zero. Tests do not read a moving clock.
        """
        return 0.0


def test_acquire_idle_task_returns_token_1(tmp_path: Path) -> None:
    """The first acquire of an idle task bumps fencing_token from 0 to 1."""
    tenant_id, connection, service = _open(tmp_path)
    try:
        assert isinstance(SystemClock().now(), float)
        lease = service.acquire("fix-adder", "worker-a", 0, 10)
        assert lease.fencing_token == 1
        assert lease.expires_at == 10
        stored = get_task(tmp_path, tenant_id, "fix-adder")
        assert stored.fencing_token == 1
        assert stored.lease_owner == "worker-a"
        with pytest.raises(ValueError):
            service.acquire("fix-adder", "", 0, 10)
        with pytest.raises(ValueError):
            service.acquire("fix-adder", "worker-a", 0, 0)
    finally:
        connection.close()


def test_renew_keeps_token_and_extends_expiry(tmp_path: Path) -> None:
    """Renew keeps fencing_token and sets expiry to now plus ttl."""
    tenant_id, connection, service = _open(tmp_path)
    other, other_connection, other_service = _open(tmp_path)
    try:
        lease = service.acquire("fix-adder", "worker-a", 0, 10)
        renewed = service.renew(lease, 5, 10)
        assert renewed.fencing_token == 1
        assert renewed.expires_at == 15
        stored = get_task(tmp_path, tenant_id, "fix-adder")
        assert stored.fencing_token == 1
        assert stored.lease_expires_at == 15
        with pytest.raises(LeaseNotHeld):
            other_service.acquire("fix-adder", "worker-b", 11, 10)
        assert get_task(tmp_path, tenant_id, "fix-adder").fencing_token == 1
    finally:
        connection.close()
        other_connection.close()


def test_second_worker_inside_ttl_raises_lease_not_held(tmp_path: Path) -> None:
    """A second worker before expiry gets LeaseNotHeld and the token stays put."""
    tenant_id, connection, service = _open(tmp_path)
    try:
        service.acquire("fix-adder", "worker-a", 0, 10)
        with pytest.raises(LeaseNotHeld):
            service.acquire("fix-adder", "worker-b", 9, 10)
        stored = get_task(tmp_path, tenant_id, "fix-adder")
        assert stored.fencing_token == 1
        assert stored.lease_owner == "worker-a"
    finally:
        connection.close()


def test_second_worker_at_expiry_receives_token_2(tmp_path: Path) -> None:
    """At expiry the next worker steals the lease and receives token 2."""
    tenant_id, connection, service = _open(tmp_path)
    try:
        service.acquire("fix-adder", "worker-a", 0, 10)
        stolen = service.acquire("fix-adder", "worker-b", 10, 10)
        assert stolen.fencing_token == 2
        stored = get_task(tmp_path, tenant_id, "fix-adder")
        assert stored.lease_owner == "worker-b"
        assert stored.lease_expires_at == 20
    finally:
        connection.close()


def test_stale_token_commit_raises_fence_rejected(tmp_path: Path) -> None:
    """A commit or renew with a token the row does not hold raises FenceRejected."""
    tenant_id, connection, service = _open(tmp_path)
    try:
        lease = service.acquire("fix-adder", "worker-a", 0, 10)
        stale = Lease(tenant_id, "fix-adder", "worker-a", 99, lease.expires_at)
        with pytest.raises(FenceRejected):
            service.commit_marker(stale, 1)
        with pytest.raises(FenceRejected):
            service.renew(stale, 1, 10)
        assert get_task(tmp_path, tenant_id, "fix-adder").fencing_token == 1
        assert _marker_count(tmp_path, tenant_id) == 0
    finally:
        connection.close()


def test_holder_token_commit_succeeds_after_steal_only_for_new_owner(tmp_path: Path) -> None:
    """After a steal, the old token cannot commit and the new owner's marker can."""
    tenant_id, connection, service = _open(tmp_path)
    try:
        original = service.acquire("fix-adder", "worker-a", 0, 10)
        stolen = service.acquire("fix-adder", "worker-b", 10, 10)
        with pytest.raises(FenceRejected):
            service.commit_marker(original, 10)
        assert _marker_count(tmp_path, tenant_id) == 0
        service.commit_marker(stolen, 10)
        assert _marker_count(tmp_path, tenant_id) == 1
        assert get_task(tmp_path, tenant_id, "fix-adder").fencing_token == 2
    finally:
        connection.close()


def test_release_lets_new_uuid_acquire_without_waiting(tmp_path: Path) -> None:
    """Release bumps the token and clears the owner so a new UUID need not wait."""
    tenant_id, connection, service = _open(tmp_path)
    try:
        lease = service.acquire("fix-adder", "worker-a", 0, 10)
        service.release(lease, 1)
        stored = get_task(tmp_path, tenant_id, "fix-adder")
        assert stored.lease_owner is None
        assert stored.fencing_token == 2
        with pytest.raises(FenceRejected):
            service.release(lease, 1)
        nxt = service.acquire("fix-adder", "worker-b", 1, 10)
        assert nxt.fencing_token == 3
        assert get_task(tmp_path, tenant_id, "fix-adder").lease_owner == "worker-b"
    finally:
        connection.close()


def test_globex_connection_cannot_change_acme_fence(tmp_path: Path) -> None:
    """A globex service cannot commit or rewrite the acme fencing token."""
    acme, acme_connection, acme_service = _open(tmp_path, "acme")
    globex, globex_connection, globex_service = _open(tmp_path, "globex")
    try:
        held = acme_service.acquire("fix-adder", "worker-a", 0, 10)
        with pytest.raises(TenantMismatch):
            globex_service.commit_marker(held, 1)
        with pytest.raises(TenantMismatch):
            globex_service.renew(held, 1, 10)
        with pytest.raises(TenantMismatch):
            globex_service.release(held, 1)
        globex_service.acquire("fix-adder", "worker-g", 0, 10)
        assert get_task(tmp_path, acme, "fix-adder").fencing_token == 1
        assert get_task(tmp_path, acme, "fix-adder").lease_owner == "worker-a"
        assert get_task(tmp_path, globex, "fix-adder").fencing_token == 1
    finally:
        acme_connection.close()
        globex_connection.close()


def test_same_task_id_has_independent_tokens_per_file(tmp_path: Path) -> None:
    """fix-adder in acme and globex each start at token 1 when acquired."""
    acme, acme_connection, acme_service = _open(tmp_path, "acme")
    globex, globex_connection, globex_service = _open(tmp_path, "globex")
    try:
        assert acme_service.acquire("fix-adder", "worker-a", 0, 10).fencing_token == 1
        assert globex_service.acquire("fix-adder", "worker-g", 0, 10).fencing_token == 1
        assert get_task(tmp_path, acme, "fix-adder").fencing_token == 1
        assert get_task(tmp_path, globex, "fix-adder").fencing_token == 1
    finally:
        acme_connection.close()
        globex_connection.close()


def test_missing_task_raises_task_not_found(tmp_path: Path) -> None:
    """Acquire, renew, release, and commit of an absent task raise TaskNotFound."""
    tenant_id, connection, service = _open(tmp_path, init=False)
    missing = Lease(tenant_id, "fix-adder", "worker-a", 1, 10)
    try:
        with pytest.raises(TaskNotFound):
            service.acquire("fix-adder", "worker-a", 0, 10)
        with pytest.raises(TaskNotFound):
            service.renew(missing, 0, 10)
        with pytest.raises(TaskNotFound):
            service.release(missing, 0)
        with pytest.raises(TaskNotFound):
            service.commit_marker(missing, 0)
    finally:
        connection.close()


def test_two_connections_begin_immediate_serialize_token_bump(tmp_path: Path) -> None:
    """Two connections serialize the token bump so only one acquire wins."""
    tenant_id = TenantId("acme")
    provision(tmp_path, tenant_id, "Acme")
    init_task(tmp_path, tenant_id, "fix-adder")
    barrier = threading.Barrier(2)
    won: list[int] = []
    refused: list[str] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def attempt(worker_id: str) -> None:
        connection = open_tenant_db(tmp_path, tenant_id)
        try:
            service = LeaseService(tenant_id, SqliteLeaseStore(connection), ManualClock())
            barrier.wait(timeout=5)
            lease = service.acquire("fix-adder", worker_id, 0, 10)
            with lock:
                won.append(lease.fencing_token)
        except LeaseNotHeld:
            with lock:
                refused.append(worker_id)
        except BaseException as exc:
            with lock:
                failures.append(exc)
        finally:
            connection.close()

    threads = [
        threading.Thread(target=attempt, args=(worker,))
        for worker in (
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
        )
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []
    assert won == [1]
    assert len(refused) == 1
    assert get_task(tmp_path, tenant_id, "fix-adder").fencing_token == 1


def _open(
    tmp_path: Path, tenant: str = "acme", *, init: bool = True
) -> tuple[TenantId, object, LeaseService]:
    tenant_id = TenantId(tenant)
    provision(tmp_path, tenant_id, tenant)
    if init:
        init_task(tmp_path, tenant_id, "fix-adder")
    connection = open_tenant_db(tmp_path, tenant_id)
    service = LeaseService(tenant_id, SqliteLeaseStore(connection), ManualClock())
    return tenant_id, connection, service


def _marker_count(tmp_path: Path, tenant_id: TenantId) -> int:
    connection = open_tenant_db(tmp_path, tenant_id, read_only=True)
    try:
        row = connection.execute(
            "SELECT COUNT(*) FROM step_intents WHERE tenant_id = ? AND task_id = ?",
            (tenant_id.value, "fix-adder"),
        ).fetchone()
    finally:
        connection.close()
    return int(row[0])
