"""CLI jail, replace, and self-test --step 3."""

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path


def test_self_test_step3_rejects_stale_hash_and_kills_process_group(tmp_path: Path) -> None:
    """self-test --step 3 and --through 3 print PASS after the jail and stale edit."""
    for flag, root in (("--step", tmp_path / "step"), ("--through", tmp_path / "through")):
        result = _run(root, "self-test", flag, "3")
        assert result.returncode == 0, result.stderr
        assert "PASS" in result.stdout
        adder = (root / "adder.py").read_bytes()
        assert adder.endswith(b"x")
        assert b"return a + b" in adder


def test_jail_timeout_and_replace_cli(tmp_path: Path) -> None:
    """jail prints ToolTimeout, and replace writes only when the hash matches."""
    script = Path(__file__).resolve().parents[1] / "fixtures" / "scripts" / "spawn_tree.py"
    pid_file = tmp_path / "pids"
    timed_out = _run(
        tmp_path,
        "jail",
        "--timeout",
        "1",
        "--",
        sys.executable,
        str(script),
        str(pid_file),
    )
    assert timed_out.returncode != 0
    assert "ToolTimeout" in timed_out.stderr
    leader, child = (int(line) for line in pid_file.read_text(encoding="utf-8").split())
    assert _dead(leader)
    assert _dead(child)
    printed = _run(tmp_path, "jail", "--timeout", "5", "--", sys.executable, "-c", "print(123)")
    assert printed.returncode == 0, printed.stderr
    assert printed.stdout.strip() == "123"
    work = tmp_path / "work"
    work.mkdir()
    original = b"return a - b\n"
    adder = work / "adder.py"
    adder.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()
    replaced = _run(
        tmp_path,
        "replace",
        "--root",
        str(work),
        "--path",
        "adder.py",
        "--expected-sha256",
        digest,
        "--old",
        "return a - b",
        "--new",
        "return a + b",
    )
    assert replaced.returncode == 0, replaced.stderr
    assert replaced.stdout.strip() == "applied"
    assert adder.read_bytes() == b"return a + b\n"
    adder.write_bytes(adder.read_bytes() + b"x")
    stale = _run(
        tmp_path,
        "replace",
        "--root",
        str(work),
        "--path",
        "adder.py",
        "--expected-sha256",
        digest,
        "--old",
        "return a - b",
        "--new",
        "return a + b",
    )
    assert stale.returncode != 0
    assert "StaleContent" in stale.stderr
    assert adder.read_bytes().endswith(b"x")


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


def _dead(pid: int) -> bool:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False
