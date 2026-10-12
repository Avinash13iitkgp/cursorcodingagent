"""Idempotency keys stay inside one tenant file."""

from pathlib import Path

import pytest

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.step_store import SqliteStepStore
from agent_engine.adapters.sqlite.tenant_store import init_task, provision
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.application.react import idempotency_key
from agent_engine.domain.errors import TenantMismatch
from agent_engine.domain.models import Lease, TenantId
from tests.step4_support import NOW, ScriptedLlm, StatusGate, call, lease_task, loop


def test_same_key_on_globex_file_is_a_miss(tmp_path: Path) -> None:
    """A cache row in acme is invisible in globex, including the same key text."""
    env = lease_task(tmp_path)
    try:
        llm = ScriptedLlm(
            (
                (call("read_file", {"path": "adder.py"}),),
                (),
            )
        )
        result = loop(env, llm, StatusGate(env, None), _Checks()).run(NOW)
        assert result.executions == 1
        key = idempotency_key("acme", "fix-adder", 0, "read_file", {"path": "adder.py"})
        assert env.store.cache_get(env.tenant, key) is not None
        provision(tmp_path, TenantId("globex"), "Globex")
        init_task(tmp_path, TenantId("globex"), "fix-adder")
        globex = open_tenant_db(tmp_path, TenantId("globex"), read_only=True)
        try:
            other = SqliteStepStore(globex)
            assert other.cache_get(TenantId("globex"), key) is None
            count = globex.execute("SELECT COUNT(*) AS n FROM step_intents").fetchone()
            assert int(count["n"]) == 0
        finally:
            globex.close()
        again = idempotency_key("globex", "fix-adder", 0, "read_file", {"path": "adder.py"})
        assert again != key
    finally:
        env.connection.close()


def test_idempotency_key_ignores_argument_order() -> None:
    """Canonical JSON sorts keys, so argument order does not change the digest."""
    left = idempotency_key("acme", "fix-adder", 1, "replace_text", {"path": "adder.py", "old": "a"})
    right = idempotency_key("acme", "fix-adder", 1, "replace_text", {"old": "a", "path": "adder.py"})
    assert left == right
    assert len(left) == 64


class _Checks:
    def run(self, work_root: Path) -> tuple[bool, str, str]:
        del work_root
        return True, "", ""


def test_fenced_lease_service_acquire_renew_release_and_marker(tmp_path: Path) -> None:
    """The loop's lease service still bumps, renews, marks, and releases one token."""
    provision(tmp_path, TenantId("acme"), "Acme")
    init_task(tmp_path, TenantId("acme"), "fix-adder")
    connection = open_tenant_db(tmp_path, TenantId("acme"))
    try:
        service = LeaseService(TenantId("acme"), SqliteLeaseStore(connection), SystemClock())
        assert SystemClock().now() > 0
        with pytest.raises(ValueError):
            service.acquire("fix-adder", "", NOW, 10)
        with pytest.raises(ValueError):
            service.acquire("fix-adder", "worker", NOW, 0)
        lease = service.acquire("fix-adder", "worker", NOW, 10)
        renewed = service.renew(lease, NOW + 1, 10)
        foreign = Lease(TenantId("globex"), "fix-adder", "worker", renewed.fencing_token, NOW)
        with pytest.raises(TenantMismatch):
            service.renew(foreign, NOW + 1, 10)
        with pytest.raises(TenantMismatch):
            service.release(foreign, NOW)
        with pytest.raises(TenantMismatch):
            service.commit_marker(foreign, NOW)
        service.commit_marker(renewed, NOW + 1)
        service.release(renewed, NOW + 1)
    finally:
        connection.close()
