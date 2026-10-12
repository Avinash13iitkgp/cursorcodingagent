"""Crash resume: SIGKILL after the edit and before T2."""

import threading
import time
from pathlib import Path

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.demo import _scenario_crash, prepare_task, run_scenarios
from agent_engine.domain.models import TenantId
from agent_engine.worker import WorkerConfig, main, serve


def test_sigkill_after_edit_before_t2_resumes_with_token_2_and_zero_writes(
    tmp_path: Path,
) -> None:
    """The killed worker leaves the edit uncommitted. Token 2 resumes with no write."""
    _scenario_crash(tmp_path, llm="canned", crash_ttl=0.4)


def test_crash_hook_leaves_the_intent_logged(tmp_path: Path, monkeypatch) -> None:
    """AGENT_CRASH_AFTER raises SIGKILL after the write and before T2."""
    prepare_task(tmp_path, "crash-hook")

    def _die(_pid: int, _sig: int) -> None:
        raise SystemExit(9)

    monkeypatch.setattr("agent_engine.worker.os.kill", _die)
    try:
        serve(
            WorkerConfig(
                data_root=tmp_path,
                tenant_id="acme",
                task_id="crash-hook",
                llm="canned",
                ttl=30,
                canned="replace",
                crash_after="replace_text",
                crash_ttl=30,
            )
        )
    except SystemExit as exc:
        assert exc.code == 9
    else:
        raise AssertionError("crash hook returned")
    connection = open_tenant_db(tmp_path, TenantId("acme"), read_only=True)
    try:
        row = connection.execute(
            "SELECT status FROM step_intents WHERE task_id = 'crash-hook'"
        ).fetchone()
    finally:
        connection.close()
    assert row["status"] == "intent_logged"
    body = (
        tmp_path / "tenants" / "acme" / "tasks" / "crash-hook" / "work" / "adder.py"
    ).read_bytes()
    assert b"return a + b" in body


def test_stale_byte_then_forced_finish_is_rejected(tmp_path: Path) -> None:
    """An out-of-band byte stays, and a forced finish is rejected."""
    run_scenarios(tmp_path, (5,), llm="canned")
    connection = open_tenant_db(tmp_path, TenantId("acme"), read_only=True)
    try:
        status = connection.execute(
            "SELECT status FROM tasks WHERE id = 'stale-adder'"
        ).fetchone()["status"]
    finally:
        connection.close()
    assert status == "rejected"


def test_canned_replay_prints_a_cache_hit(tmp_path: Path, capsys) -> None:
    """Scenario 3 replays a finished task with zero posts and zero side effects."""
    run_scenarios(tmp_path, (3,), llm="canned")
    captured = capsys.readouterr().out
    assert "cache hit" in captured
    assert "side_effects=0" in captured
    assert "posts=0" in captured


def test_worker_main_repairs_with_the_canned_model(tmp_path: Path, capsys, monkeypatch) -> None:
    """main runs the canned repair in this process and the second call is a cache hit."""
    monkeypatch.delenv("AGENT_CRASH_AFTER", raising=False)
    monkeypatch.delenv("AGENT_MUTATE_AFTER", raising=False)
    monkeypatch.delenv("AGENT_HOLD_SEC", raising=False)
    prepare_task(tmp_path, "fix-adder")
    args = [
        "--data-root",
        str(tmp_path),
        "--tenant",
        "acme",
        "--task",
        "fix-adder",
        "--llm",
        "canned",
        "--ttl",
        "30",
    ]
    assert main(args) == 0
    assert main(args) == 0
    output = capsys.readouterr().out
    assert "status=completed" in output
    assert "cache hit" in output


def test_heartbeat_zero_rows_does_not_start_t2(tmp_path: Path) -> None:
    """A renew that matches zero rows closes the turn and leaves the intent logged."""
    prepare_task(tmp_path, "hold-adder")
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    box: dict[str, object] = {}

    def _run() -> None:
        box["report"] = serve(
            WorkerConfig(
                data_root=tmp_path,
                tenant_id="acme",
                task_id="hold-adder",
                llm="canned",
                ttl=30,
                canned="read",
                hold_sec=5,
                ready_file=ready,
                release_file=release,
                heartbeat_interval=0.05,
            )
        )

    thread = threading.Thread(target=_run)
    thread.start()
    deadline = time.time() + 5
    while not ready.exists():
        if time.time() > deadline:
            release.write_text("go", encoding="utf-8")
            thread.join(timeout=5)
            raise AssertionError("worker did not hold")
        time.sleep(0.02)
    connection = open_tenant_db(tmp_path, TenantId("acme"))
    connection.execute(
        "UPDATE tasks SET lease_expires_at = ? WHERE id = ?",
        (time.time() - 1, "hold-adder"),
    )
    connection.close()
    time.sleep(0.3)
    release.write_text("go", encoding="utf-8")
    thread.join(timeout=5)
    report = box["report"]
    assert getattr(report, "fence_rejected") is True
    connection = open_tenant_db(tmp_path, TenantId("acme"), read_only=True)
    try:
        row = connection.execute(
            "SELECT status FROM step_intents WHERE task_id = 'hold-adder'"
        ).fetchone()
    finally:
        connection.close()
    assert row["status"] == "intent_logged"
