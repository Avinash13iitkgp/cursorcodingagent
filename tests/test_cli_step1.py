"""CLI self-test for step 1 creates two tenant files and prints PASS."""

import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def test_self_test_step1_exits_zero_and_prints_pass(tmp_path: Path) -> None:
    """self-test --step 1 exits 0, prints PASS, and stores fix-adder in both files."""
    src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "agent_engine",
            "self-test",
            "--step",
            "1",
            "--data-root",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "PASS" in result.stdout
    for tenant in ("acme", "globex"):
        database = tmp_path / "tenants" / tenant / "engine.db"
        assert database.is_file()
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT id FROM tasks").fetchone()
        finally:
            connection.close()
        assert row[0] == "fix-adder"
