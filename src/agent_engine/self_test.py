"""Scripted self-test procedures for the CLI gate.

This module owns the fixed tenants, task id, worker ids, and timestamps used
to check steps 1 and 2. It is not the product command path: acquire, renew,
and release stay in the process entrypoint and take their arguments from the
user. A failed check raises. run_self_test turns that into PASS or FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.tenant_store import get_task, init_task, provision, read_pragmas
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.domain.errors import (
    FenceRejected,
    LeaseNotHeld,
    TaskNotFound,
    TenantMismatch,
    UnknownTenant,
)
from agent_engine.domain.models import TenantId

_WORKER_A = "11111111-1111-4111-8111-111111111111"
_WORKER_B = "22222222-2222-4222-8222-222222222222"
_TASK_ID = "fix-adder"


def run_self_test(data_root: Path, step: int | None, through: int | None) -> int:
    """Run one self-test step, or every step through a number.

    Args:
        data_root: Directory that contains tenants/. The procedure writes there.
        step: A single step number. Ignored as an upper bound when through is set.
        through: Run steps 1 through this number.

    Returns:
        0 when the procedure prints PASS. 1 when it prints FAIL.
    """
    if step is None and through is None:
        print("FAIL", file=sys.stderr)
        print("self-test requires --step or --through", file=sys.stderr)
        return 1
    if through is not None:
        if through < 1 or through > 2 or (step is not None and (step < 1 or step > through)):
            print("FAIL")
            print(f"step {through} is not implemented", file=sys.stderr)
            return 1
        steps = range(1, through + 1)
    elif step not in (1, 2):
        print("FAIL")
        print(f"step {step} is not implemented", file=sys.stderr)
        return 1
    else:
        steps = (step,)
    try:
        for number in steps:
            if number == 1:
                _run_step1(data_root)
            else:
                _run_step2(data_root)
    except (
        UnknownTenant,
        TenantMismatch,
        TaskNotFound,
        FenceRejected,
        LeaseNotHeld,
        ValueError,
        OSError,
    ) as exc:
        print("FAIL")
        print(exc, file=sys.stderr)
        return 1
    print("PASS")
    return 0


def _run_step1(data_root: Path) -> None:
    """Provision acme and globex, store fix-adder in each, and check WAL."""
    provision(data_root, TenantId("acme"), "Acme")
    provision(data_root, TenantId("globex"), "Globex")
    init_task(data_root, TenantId("acme"), _TASK_ID)
    init_task(data_root, TenantId("globex"), _TASK_ID)
    journal, synchronous, _busy = read_pragmas(data_root, TenantId("acme"))
    if journal != "wal" or synchronous != 2:
        raise OSError(f"expected wal and 2, got {journal} and {synchronous}")
    acme = get_task(data_root, TenantId("acme"), _TASK_ID)
    globex = get_task(data_root, TenantId("globex"), _TASK_ID)
    if acme.id != _TASK_ID or globex.id != _TASK_ID:
        raise OSError("fix-adder missing from a tenant file")
    acme_db = data_root / "tenants" / "acme" / "engine.db"
    globex_db = data_root / "tenants" / "globex" / "engine.db"
    if not acme_db.is_file() or not globex_db.is_file():
        raise OSError("expected two engine.db files")
    if acme_db.resolve() == globex_db.resolve():
        raise OSError("tenant databases resolved to one file")


def _run_step2(data_root: Path) -> None:
    """Acquire, renew, reject, and steal a lease without touching globex."""
    _run_step1(data_root)
    acme = TenantId("acme")
    first = _lease_call(
        data_root,
        acme,
        lambda service: service.acquire(_TASK_ID, _WORKER_A, 0, 10),
    )
    if first.fencing_token != 1 or first.worker_id != _WORKER_A:
        raise OSError(f"expected token 1 for worker A, got {first.fencing_token}")
    held = get_task(data_root, acme, _TASK_ID)
    if held.fencing_token != 1 or held.lease_owner != _WORKER_A:
        raise OSError("show-lease after acquire did not keep token 1")
    renewed = _lease_call(
        data_root,
        acme,
        lambda service: service.renew(first, 5, 10),
    )
    if renewed.fencing_token != 1:
        raise OSError("renew changed the fencing token")
    try:
        _lease_call(
            data_root,
            acme,
            lambda service: service.acquire(_TASK_ID, _WORKER_B, 5, 10),
        )
    except LeaseNotHeld:
        pass
    else:
        raise OSError("expected LeaseNotHeld inside the TTL")
    if get_task(data_root, acme, _TASK_ID).fencing_token != 1:
        raise OSError("rejected acquire changed the token")
    # Renew at now=5 with ttl=10 moves expiry to 15, so the steal waits until then.
    stolen = _lease_call(
        data_root,
        acme,
        lambda service: service.acquire(_TASK_ID, _WORKER_B, 15, 10),
    )
    if stolen.fencing_token != 2:
        raise OSError(f"expected token 2 after expiry, got {stolen.fencing_token}")
    globex = get_task(data_root, TenantId("globex"), _TASK_ID)
    if globex.fencing_token != 0 or globex.lease_owner is not None:
        raise OSError("globex lease changed during the acme procedure")


def _lease_call(data_root: Path, tenant_id: TenantId, fn):
    connection = open_tenant_db(data_root, tenant_id)
    try:
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        return fn(service)
    finally:
        connection.close()
