"""Scripted self-test procedures for the CLI gate.

This module owns the fixed tenants, task id, worker ids, timestamps, the
jail and edit checks, and the canned ReAct procedure for steps 1 through 4.
Step 5 delegates to the demos. It is not the product command path. A failed
check raises. run_self_test turns that into PASS or FAIL. --live on step 4
or step 5 calls the real model.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

from agent_engine.composition import format_report, run_task

from agent_engine.adapters.exec.jail import Jail, tool_env
from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.tenant_store import get_task, init_task, provision, read_pragmas
from agent_engine.adapters.workspace.edits import FlatWorkspace, OptimisticEditor
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.domain.errors import (
    AmbiguousEdit,
    FenceRejected,
    LeaseNotHeld,
    StaleContent,
    TaskNotFound,
    TenantMismatch,
    ToolTimeout,
    UnknownTenant,
)
from agent_engine.domain.models import TenantId, TextEdit

_WORKER_A = "11111111-1111-4111-8111-111111111111"
_WORKER_B = "22222222-2222-4222-8222-222222222222"
_TASK_ID = "fix-adder"


def run_self_test(
    data_root: Path, step: int | None, through: int | None, *, live: bool = False
) -> int:
    """Run one self-test step, or every step through a number.

    Args:
        data_root: Directory that contains tenants/. The procedure writes there.
        step: A single step number. Ignored as an upper bound when through is set.
        through: Run steps 1 through this number. Those procedures stay offline.
        live: When step 4 or step 5 is selected by itself, call the real model.

    Returns:
        0 when the procedure prints PASS. 1 when it prints FAIL.
    """
    if step is None and through is None:
        print("FAIL", file=sys.stderr)
        print("self-test requires --step or --through", file=sys.stderr)
        return 1
    if through is not None:
        if through < 1 or through > 5 or (step is not None and (step < 1 or step > through)):
            print("FAIL")
            print(f"step {through} is not implemented", file=sys.stderr)
            return 1
        steps = range(1, through + 1)
    elif step not in (1, 2, 3, 4, 5):
        print("FAIL")
        print(f"step {step} is not implemented", file=sys.stderr)
        return 1
    else:
        steps = (step,)
    try:
        for number in steps:
            if number == 1:
                _run_step1(data_root)
            elif number == 2:
                _run_step2(data_root)
            elif number == 3:
                _run_step3(data_root)
            elif number == 4:
                _run_step4(data_root, live=live and through is None)
            else:
                _run_step5(data_root, live=live and through is None)
    except (
        UnknownTenant,
        TenantMismatch,
        TaskNotFound,
        FenceRejected,
        LeaseNotHeld,
        StaleContent,
        AmbiguousEdit,
        ToolTimeout,
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


def _run_step3(data_root: Path) -> None:
    """Kill a process group on timeout, then reject a stale edit."""
    data_root.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[2] / "fixtures" / "scripts" / "spawn_tree.py"
    pid_file = data_root / "jail-pids"
    jail = Jail()
    try:
        jail.run(
            [sys.executable, str(script), str(pid_file)],
            cwd=data_root,
            timeout_s=1,
            env=tool_env(),
        )
    except ToolTimeout:
        pass
    else:
        raise OSError("expected ToolTimeout from the sleeper tree")
    pids = [int(line) for line in pid_file.read_text(encoding="utf-8").split()]
    if len(pids) != 2:
        raise OSError("spawn_tree did not write two pids")
    for pid in pids:
        _wait_dead(pid)
    printed = jail.run(
        [sys.executable, "-c", "print(123)"],
        cwd=data_root,
        timeout_s=5,
        env=tool_env(),
    )
    if printed.returncode != 0 or "123" not in printed.stdout:
        raise OSError(f"expected stdout 123, got {printed.stdout!r}")
    original = b"return a - b\n"
    adder = data_root / "adder.py"
    adder.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    workspace = FlatWorkspace(data_root)
    editor = OptimisticEditor()
    applied = editor.apply(
        workspace,
        TextEdit("adder.py", digest, "return a - b", "return a + b"),
    )
    if applied.status != "applied" or adder.read_bytes() != b"return a + b\n":
        raise OSError("replace did not swap the line")
    adder.write_bytes(adder.read_bytes() + b"x")
    try:
        editor.apply(
            workspace,
            TextEdit("adder.py", digest, "return a - b", "return a + b"),
        )
    except StaleContent:
        pass
    else:
        raise OSError("expected StaleContent after the file changed")
    if not adder.read_bytes().endswith(b"x"):
        raise OSError("stale replace changed the file")


def _run_step4(data_root: Path, *, live: bool) -> None:
    """Repair the adder with a canned model, then reject a forced finish.

    A live run calls the real model on a fresh broken copy and expects completed.
    """
    if live:
        work = _prepare_adder(data_root, "acme", "fix-adder-live")
        _prepare_tenant(data_root, "globex")
        report = run_task(data_root, TenantId("acme"), "fix-adder-live", llm="live")
        print(format_report(report))
        if report.status != "completed":
            raise OSError(f"live run status is {report.status}")
        if b"return a + b" not in (work / "adder.py").read_bytes():
            raise OSError("live run left the broken adder in place")
        return
    work = _prepare_adder(data_root, "acme", _TASK_ID)
    _prepare_tenant(data_root, "globex")
    init_task(data_root, TenantId("globex"), _TASK_ID)
    first = run_task(data_root, TenantId("acme"), _TASK_ID, llm="canned")
    print(format_report(first))
    if first.status != "completed" or first.cache_hit:
        raise OSError(f"canned repair status is {first.status}")
    if b"return a + b" not in (work / "adder.py").read_bytes():
        raise OSError("canned repair did not change the adder")
    if _step_count(data_root, "globex") != 0:
        raise OSError("acme run wrote globex step_intents")
    second = run_task(data_root, TenantId("acme"), _TASK_ID, llm="canned")
    print(format_report(second))
    if not second.cache_hit or second.executions != 0 or second.posts != 0:
        raise OSError("replay was not a cache hit")
    if second.status != "completed":
        raise OSError("cache hit did not keep completed")
    rejected_work = _prepare_adder(data_root, "acme", "reject-adder")
    rejected = run_task(
        data_root,
        TenantId("acme"),
        "reject-adder",
        llm="canned",
        tool_choice="finish",
    )
    print(format_report(rejected))
    if rejected.status != "rejected":
        raise OSError(f"forced finish status is {rejected.status}")
    if get_task(data_root, TenantId("acme"), "reject-adder").status != "rejected":
        raise OSError("tasks.status was not rejected")
    if b"return a - b" not in (rejected_work / "adder.py").read_bytes():
        raise OSError("forced finish changed the broken adder")


def _run_step5(data_root: Path, *, live: bool) -> None:
    """Run the offline suspend and rollback demos, or the live demos.

    Args:
        data_root: Directory that contains tenants/.
        live: When true, run scenarios 1, 2, 3, and 5 against the real model.
            Otherwise run scenarios 4 and 6 on this machine.
    """
    from agent_engine.demo import run_scenarios

    if live:
        run_scenarios(data_root, (1, 2, 3, 5), llm="live")
    else:
        run_scenarios(data_root, (4, 6), llm="canned")


def _prepare_tenant(data_root: Path, tenant: str) -> None:
    provision(data_root, TenantId(tenant), tenant.capitalize())


def _prepare_adder(data_root: Path, tenant: str, task_id: str) -> Path:
    _prepare_tenant(data_root, tenant)
    init_task(data_root, TenantId(tenant), task_id)
    work = data_root / "tenants" / tenant / "tasks" / task_id
    base = work / "base"
    base.mkdir(parents=True, exist_ok=True)
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "broken_adder"
    if not (base / "adder.py").exists():
        shutil.copyfile(fixture / "adder.py", base / "adder.py")
    from agent_engine.adapters.workspace.cow import CowWorkspace

    CowWorkspace(work).seed()
    tree = work / "work"
    checks = data_root / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture / "test_adder.py", checks / "test_adder.py")
    copied = tree / "test_adder.py"
    if copied.exists():
        copied.unlink()
    return tree


def _step_count(data_root: Path, tenant: str) -> int:
    connection = open_tenant_db(data_root, TenantId(tenant), read_only=True)
    try:
        row = connection.execute("SELECT COUNT(*) AS n FROM step_intents").fetchone()
    finally:
        connection.close()
    return int(row["n"])


def _wait_dead(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    raise OSError(f"pid {pid} still alive")


def _lease_call(data_root: Path, tenant_id: TenantId, fn):
    connection = open_tenant_db(data_root, tenant_id)
    try:
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        return fn(service)
    finally:
        connection.close()
