"""CLI gate for the offline step 5 demos and the README command index."""

import os
import subprocess
import sys
from pathlib import Path


def test_self_test_step5_offline_covers_suspend_and_rollback(tmp_path: Path) -> None:
    """self-test --step 5 prints the suspend order, a sub-5ms rollback, and PASS."""
    result = _run(tmp_path, "self-test", "--step", "5")
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout
    assert output.index("stop") < output.index("expiry")
    assert output.index("expiry") < output.index("steal")
    assert output.index("steal") < output.index("continue")
    assert output.index("continue") < output.index("FenceRejected")
    marker = "rollback_ms="
    assert marker in output
    elapsed = float(output.split(marker, 1)[1].split()[0])
    assert elapsed < 5
    assert "PASS" in output


def test_readme_lists_install_env_self_test_and_demo() -> None:
    """The setup guide lists install, the API key variable, self-test, and demo."""
    text = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    assert "pip install" in text
    assert "AGENT_LLM_API_KEY" in text
    assert "self-test" in text
    assert "demo --tenant acme" in text


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
