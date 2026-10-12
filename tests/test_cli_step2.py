"""CLI lease commands and self-test --step 2 / --through 2."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

_WORKER_A = "11111111-1111-4111-8111-111111111111"
_WORKER_B = "22222222-2222-4222-8222-222222222222"


def test_self_test_step2_prints_pass_and_leaves_globex_token_at_zero(tmp_path: Path) -> None:
    """self-test --step 2 and --through 2 print PASS and leave globex at token 0."""
    for flag, root in (("--step", tmp_path / "step"), ("--through", tmp_path / "through")):
        result = _run(root, "self-test", flag, "2")
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout
        assert _lease(root, "globex") == (0, None)
        token, owner = _lease(root, "acme")
        assert token == 2
        assert owner == _WORKER_B


def test_lease_cli_follows_acquire_renew_and_steal(tmp_path: Path) -> None:
    """The lease CLI prints token 1, rejects the second worker, then prints token 2."""
    for tenant, name in (("acme", "Acme"), ("globex", "Globex")):
        assert _run(tmp_path, "provision", "--tenant", tenant, "--display-name", name).returncode == 0
        assert _run(tmp_path, "init-task", "--tenant", tenant, "--task", "fix-adder").returncode == 0
    acquired = _run(
        tmp_path,
        "acquire",
        "--tenant",
        "acme",
        "--task",
        "fix-adder",
        "--worker-id",
        _WORKER_A,
        "--now",
        "0",
        "--ttl",
        "10",
    )
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout.strip() == "1"
    shown = _run(tmp_path, "show-lease", "--tenant", "acme", "--task", "fix-adder")
    assert "fencing_token=1" in shown.stdout
    assert f"lease_owner={_WORKER_A}" in shown.stdout
    assert "lease_expires_at=10" in shown.stdout
    renewed = _run(
        tmp_path,
        "renew",
        "--tenant",
        "acme",
        "--task",
        "fix-adder",
        "--worker-id",
        _WORKER_A,
        "--token",
        "1",
        "--now",
        "5",
        "--ttl",
        "10",
    )
    assert renewed.stdout.strip() == "1"
    denied = _run(
        tmp_path,
        "acquire",
        "--tenant",
        "acme",
        "--task",
        "fix-adder",
        "--worker-id",
        _WORKER_B,
        "--now",
        "5",
        "--ttl",
        "10",
    )
    assert denied.returncode != 0
    assert "LeaseNotHeld" in denied.stderr
    assert "fencing_token=1" in _run(tmp_path, "show-lease", "--tenant", "acme", "--task", "fix-adder").stdout
    stolen = _run(
        tmp_path,
        "acquire",
        "--tenant",
        "acme",
        "--task",
        "fix-adder",
        "--worker-id",
        _WORKER_B,
        "--now",
        "15",
        "--ttl",
        "10",
    )
    assert stolen.returncode == 0, stolen.stderr
    assert stolen.stdout.strip() == "2"
    idle = _run(tmp_path, "show-lease", "--tenant", "globex", "--task", "fix-adder")
    assert "fencing_token=0" in idle.stdout
    assert "lease_owner=null" in idle.stdout
    released = _run(
        tmp_path,
        "release",
        "--tenant",
        "acme",
        "--task",
        "fix-adder",
        "--worker-id",
        _WORKER_B,
        "--token",
        "2",
    )
    assert released.returncode == 0, released.stderr
    assert "released" in released.stdout
    cleared = _run(tmp_path, "show-lease", "--tenant", "acme", "--task", "fix-adder")
    assert "fencing_token=3" in cleared.stdout
    assert "lease_owner=null" in cleared.stdout


def _run(data_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    data_root.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "agent_engine", "--data-root", str(data_root), *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _lease(data_root: Path, tenant: str) -> tuple[int, str | None]:
    database = data_root / "tenants" / tenant / "engine.db"
    connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    try:
        row = connection.execute(
            "SELECT fencing_token, lease_owner FROM tasks WHERE id = ?",
            ("fix-adder",),
        ).fetchone()
    finally:
        connection.close()
    return int(row[0]), row[1]
