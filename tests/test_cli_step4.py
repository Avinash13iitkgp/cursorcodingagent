"""CLI self-test for the canned ReAct procedure and the missing live token."""

import os
import subprocess
import sys
from pathlib import Path


def test_self_test_step4_canned_prints_completed_then_cache_hit(tmp_path: Path) -> None:
    """self-test --step 4 and --through 4 print completed, then a cache hit."""
    for flag, root in (("--step", tmp_path / "step"), ("--through", tmp_path / "through")):
        result = _run(root, "self-test", flag, "4")
        assert result.returncode == 0, result.stderr + result.stdout
        assert "completed" in result.stdout
        assert "cache hit" in result.stdout
        assert "executions=0" in result.stdout
        assert "posts=0" in result.stdout
        assert "PASS" in result.stdout


def test_self_test_live_without_token_exits_nonzero(tmp_path: Path) -> None:
    """self-test --step 4 --live names AGENT_LLM_API_KEY and exits non-zero when it is unset."""
    env = os.environ.copy()
    env.pop("AGENT_LLM_API_KEY", None)
    result = _run(tmp_path, "self-test", "--step", "4", "--live", env=env)
    assert result.returncode != 0
    assert "AGENT_LLM_API_KEY" in result.stderr


def _run(
    data_root: Path, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    data_root.mkdir(parents=True, exist_ok=True)
    src = Path(__file__).resolve().parents[1] / "src"
    child = os.environ.copy() if env is None else env
    child["PYTHONPATH"] = str(src) + os.pathsep + child.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "agent_engine", "--data-root", str(data_root), *args],
        check=False,
        capture_output=True,
        text=True,
        env=child,
    )
