"""Six demos. Scenarios run as acme. globex keeps a sentinel and an empty task table.

1. Live repair of the broken adder.
2. SIGKILL after the edit and before T2. The successor's fence is 2, it commits
   the one open intent, and it does not write.
3. A second run is a cache hit: zero side effects and zero extra HTTP posts.
4. SIGSTOP, wait until expiry, the other worker steals, SIGCONT. The pre-stop
   token is FenceRejected.
5. One out-of-band byte makes the next replace stale. A forced finish while
   the checks fail is rejected, and the status is never completed.
6. Copy-on-write rollback. rollback_ms is the relink only and stays under 5.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from agent_engine.adapters.sqlite.db import open_tenant_db, resolve_tenant_dir
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.step_store import SqliteStepStore
from agent_engine.adapters.sqlite.tenant_store import get_task, init_task, provision
from agent_engine.adapters.workspace.cow import CowWorkspace
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.application.react import idempotency_key
from agent_engine.domain.models import TenantId

_WORKER_B = "22222222-2222-4222-8222-222222222222"
_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "broken_adder"
_SENTINEL = b"globex-sentinel\n"


def run_scenarios(data_root: Path, scenarios: tuple[int, ...], *, llm: str) -> None:
    """Run the numbered scenarios in order.

    Args:
        data_root: Directory that contains tenants/. The demos write there.
        scenarios: Scenario numbers. 1, 2, 3, and 5 call the model when llm
            is live. 4 and 6 stay on this machine.
        llm: canned or live. Offline self-test passes canned.

    Returns:
        None.

    Raises:
        OSError: A scenario did not meet its PASS condition.
        ValueError: A live scenario is missing a credential.
    """
    for number in scenarios:
        if number == 1:
            _scenario_repair(data_root, "fix-adder", llm=llm)
        elif number == 2:
            _scenario_crash(data_root, llm=llm, crash_ttl=2.0)
        elif number == 3:
            _scenario_cache(data_root, llm=llm)
        elif number == 4:
            _scenario_suspend(data_root)
        elif number == 5:
            _scenario_stale(data_root, llm=llm)
        elif number == 6:
            _scenario_rollback(data_root)
        else:
            raise OSError(f"unknown scenario {number}")
        if _globex_steps(data_root) != 0:
            raise OSError("acme scenario wrote globex step_intents")
        print(f"scenario {number} PASS")


def seed_world(data_root: Path) -> str:
    """Provision acme and globex and write the globex sentinel.

    Args:
        data_root: Directory that contains tenants/.

    Returns:
        SHA256 hex of the sentinel bytes.
    """
    provision(data_root, TenantId("acme"), "Acme")
    provision(data_root, TenantId("globex"), "Globex")
    sentinel = resolve_tenant_dir(data_root, TenantId("globex")) / "sentinel"
    if not sentinel.exists():
        sentinel.write_bytes(_SENTINEL)
    digest = hashlib.sha256(sentinel.read_bytes()).hexdigest()
    print(f"sentinel {digest}")
    checks = data_root / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    target = checks / "test_adder.py"
    if not target.exists():
        target.write_bytes((_FIXTURE / "test_adder.py").read_bytes())
    return digest


def prepare_task(data_root: Path, task_id: str) -> CowWorkspace:
    """Create an acme task whose base holds the broken adder, then seed work.

    Args:
        data_root: Directory that contains tenants/.
        task_id: Task primary key. globex is not given a task row.

    Returns:
        The copy-on-write workspace for that task.
    """
    seed_world(data_root)
    init_task(data_root, TenantId("acme"), task_id)
    task_dir = resolve_tenant_dir(data_root, TenantId("acme")) / "tasks" / task_id
    base = task_dir / "base"
    base.mkdir(parents=True, exist_ok=True)
    adder = base / "adder.py"
    if not adder.exists():
        adder.write_bytes((_FIXTURE / "adder.py").read_bytes())
    workspace = CowWorkspace(task_dir)
    workspace.seed()
    return workspace


def _scenario_repair(data_root: Path, task_id: str, *, llm: str) -> None:
    prepare_task(data_root, task_id)
    report = _run_worker(data_root, task_id, llm=llm, ttl=180, extra={})
    if "status=completed" not in report:
        raise OSError(report)
    work = _work(data_root, task_id)
    if b"return a + b" not in (work / "adder.py").read_bytes():
        raise OSError("repair left the broken adder in place")
    print(report)


def _scenario_cache(data_root: Path, *, llm: str) -> None:
    _scenario_repair(data_root, "cache-adder", llm=llm)
    report = _run_worker(data_root, "cache-adder", llm=llm, ttl=180, extra={})
    if "cache hit" not in report or "side_effects=0" not in report or "posts=0" not in report:
        raise OSError(report)
    print(report)


def _scenario_crash(data_root: Path, *, llm: str, crash_ttl: float) -> None:
    task_id = "crash-adder"
    workspace = prepare_task(data_root, task_id)
    adder = workspace.work_root / "adder.py"
    proc = _popen(
        data_root,
        task_id,
        llm=llm,
        ttl=180,
        extra={
            "AGENT_CRASH_AFTER": "replace_text",
            "AGENT_CRASH_TTL": str(crash_ttl),
            "AGENT_CANNED": "replace" if llm == "canned" else "repair",
        },
    )
    try:
        out, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired as exc:
        _kill(proc)
        raise OSError("crash worker timed out") from exc
    if proc.returncode is None or proc.returncode >= 0:
        raise OSError(f"expected SIGKILL, got {proc.returncode}: {out}")
    if b"return a + b" not in adder.read_bytes():
        raise OSError("the edit was not on disk before the kill")
    expiry = _lease_expiry(data_root, task_id)
    while time.time() < expiry:
        time.sleep(0.02)
    before = adder.stat().st_ino
    successor = _run_worker(
        data_root,
        task_id,
        llm="canned",
        ttl=30,
        extra={"AGENT_CANNED": "stop", "AGENT_RESUME_ONLY": "1"},
    )
    if adder.stat().st_ino != before:
        raise OSError("worker B wrote the adder")
    if "token=2" not in successor or "writes=0" not in successor or "executions=1" not in successor:
        raise OSError(successor)
    if _committed_count(data_root, task_id) != 1 and llm == "canned":
        raise OSError("expected one committed intent")
    print("fence 1 -> 2")
    print("one committed intent")
    print("writes_b=0")
    print(successor)


def _scenario_suspend(data_root: Path) -> None:
    task_id = "suspend-adder"
    prepare_task(data_root, task_id)
    ready = data_root / "suspend.ready"
    release = data_root / "suspend.release"
    ready.unlink(missing_ok=True)
    release.unlink(missing_ok=True)
    proc = _popen(
        data_root,
        task_id,
        llm="canned",
        ttl=1,
        extra={
            "AGENT_CANNED": "read",
            "AGENT_HOLD_SEC": "30",
            "AGENT_READY_FILE": str(ready),
            "AGENT_RELEASE_FILE": str(release),
        },
    )
    try:
        deadline = time.time() + 10
        while not ready.exists():
            if proc.poll() is not None:
                raise OSError(f"worker exited before the hold: {proc.stdout.read() if proc.stdout else ''}")
            if time.time() > deadline:
                raise OSError("worker did not reach the hold")
            time.sleep(0.02)
        os.kill(proc.pid, signal.SIGSTOP)
        print("stop")
        expiry = _lease_expiry(data_root, task_id)
        while time.time() < expiry:
            time.sleep(0.02)
        print("expiry")
        _steal(data_root, task_id)
        print("steal")
        release.write_text("go", encoding="utf-8")
        os.kill(proc.pid, signal.SIGCONT)
        print("continue")
        out, _ = proc.communicate(timeout=10)
    finally:
        if proc.poll() is None:
            _kill(proc)
    if "FenceRejected" not in out:
        raise OSError(out)
    print("FenceRejected")


def _scenario_stale(data_root: Path, *, llm: str) -> None:
    task_id = "stale-adder"
    workspace = prepare_task(data_root, task_id)
    report = _run_worker(
        data_root,
        task_id,
        llm=llm,
        ttl=180,
        extra={
            "AGENT_MUTATE_AFTER": "read_file",
            "AGENT_CANNED": "stale",
            "AGENT_FORCE_FINISH_AFTER_STALE": "1",
        },
    )
    body = (workspace.work_root / "adder.py").read_bytes()
    if b"return a - b" not in body or not body.endswith(b"x"):
        raise OSError(f"background byte was not kept: {body!r}")
    status = get_task(data_root, TenantId("acme"), task_id).status
    if status != "rejected":
        raise OSError(f"status is {status}")
    if "status=completed" in report:
        raise OSError(report)
    print(report)


def _scenario_rollback(data_root: Path) -> None:
    task_id = "rollback-adder"
    workspace = prepare_task(data_root, task_id)
    original = (workspace.work_root / "adder.py").read_bytes()
    tenant_id = TenantId("acme")
    connection = open_tenant_db(data_root, tenant_id)
    try:
        now = time.time()
        worker_id = "44444444-4444-4444-8444-444444444444"
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        lease = service.acquire(task_id, worker_id, now, 60)
        store = SqliteStepStore(connection)
        _commit_read(store, tenant_id, task_id, worker_id, lease.fencing_token, now, 0)
        workspace.snapshot(0)
        workspace.write("adder.py", original.replace(b"return a - b", b"return a + b"))
        _commit_read(store, tenant_id, task_id, worker_id, lease.fencing_token, time.time(), 1)
        elapsed = workspace.rollback(
            0,
            connection,
            tenant_id.value,
            task_id,
            worker_id,
            lease.fencing_token,
            time.time(),
        )
    finally:
        connection.close()
    if elapsed >= 5:
        raise OSError(f"rollback_ms={elapsed}")
    if (workspace.work_root / "adder.py").read_bytes() != original:
        raise OSError("rollback did not restore the snapshot bytes")
    if (workspace.work_root.parent / "base" / "adder.py").read_bytes() != original:
        raise OSError("rollback changed base")
    connection = open_tenant_db(data_root, tenant_id, read_only=True)
    try:
        rows = connection.execute(
            "SELECT step_index, status FROM step_intents WHERE task_id = ? ORDER BY step_index",
            (task_id,),
        ).fetchall()
    finally:
        connection.close()
    statuses = {int(row["step_index"]): row["status"] for row in rows}
    if statuses.get(0) != "committed" or statuses.get(1) != "rolled_back":
        raise OSError(f"step statuses {statuses}")
    print(f"rollback_ms={elapsed:.3f}")


def _commit_read(
    store: SqliteStepStore,
    tenant_id: TenantId,
    task_id: str,
    worker_id: str,
    token: int,
    now: float,
    step_index: int,
) -> None:
    args = {"path": "adder.py"}
    key = idempotency_key(tenant_id.value, task_id, step_index, "read_file", args)
    store.mark_running(tenant_id, task_id, worker_id, token, now)
    store.insert_intent(
        tenant_id,
        task_id,
        step_index,
        key,
        "read_file",
        json.dumps(args),
        worker_id,
        token,
        now,
        (),
    )
    store.commit_step(
        tenant_id,
        task_id,
        step_index,
        key,
        json.dumps({"ok": True}),
        worker_id,
        token,
        now,
        (),
    )


def _run_worker(
    data_root: Path,
    task_id: str,
    *,
    llm: str,
    ttl: float,
    extra: dict[str, str],
) -> str:
    proc = _popen(data_root, task_id, llm=llm, ttl=ttl, extra=extra)
    try:
        out, _ = proc.communicate(timeout=180)
    except subprocess.TimeoutExpired as exc:
        _kill(proc)
        raise OSError(f"worker {task_id} timed out") from exc
    if proc.returncode != 0:
        raise OSError(f"worker {task_id} exited {proc.returncode}: {out}")
    return out


def _popen(
    data_root: Path,
    task_id: str,
    *,
    llm: str,
    ttl: float,
    extra: dict[str, str],
) -> subprocess.Popen[str]:
    src = str(Path(__file__).resolve().parents[1])
    env = os.environ.copy()
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    env.update(extra)
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_engine.worker",
            "--data-root",
            str(data_root),
            "--tenant",
            "acme",
            "--task",
            task_id,
            "--llm",
            llm,
            "--ttl",
            str(ttl),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _kill(proc: subprocess.Popen[str]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def _work(data_root: Path, task_id: str) -> Path:
    return resolve_tenant_dir(data_root, TenantId("acme")) / "tasks" / task_id / "work"


def _lease_expiry(data_root: Path, task_id: str) -> float:
    connection = open_tenant_db(data_root, TenantId("acme"), read_only=True)
    try:
        row = connection.execute(
            "SELECT lease_expires_at FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None or row["lease_expires_at"] is None:
        raise OSError("lease has no expiry")
    return float(row["lease_expires_at"])


def _steal(data_root: Path, task_id: str) -> None:
    tenant_id = TenantId("acme")
    connection = open_tenant_db(data_root, tenant_id)
    try:
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        service.acquire(task_id, _WORKER_B, time.time(), 30)
    finally:
        connection.close()


def _committed_count(data_root: Path, task_id: str) -> int:
    connection = open_tenant_db(data_root, TenantId("acme"), read_only=True)
    try:
        row = connection.execute(
            """
            SELECT COUNT(*) AS n FROM step_intents
            WHERE task_id = ? AND status = 'committed'
            """,
            (task_id,),
        ).fetchone()
    finally:
        connection.close()
    return int(row["n"])


def _globex_steps(data_root: Path) -> int:
    connection = open_tenant_db(data_root, TenantId("globex"), read_only=True)
    try:
        row = connection.execute("SELECT COUNT(*) AS n FROM step_intents").fetchone()
    finally:
        connection.close()
    return int(row["n"])
