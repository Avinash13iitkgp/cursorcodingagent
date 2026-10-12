"""The jail kills the whole process group and passes only the given environment."""

import os
import sys
import time
from pathlib import Path

import pytest

from agent_engine.adapters.exec.jail import Jail, tool_env
from agent_engine.domain.errors import ToolTimeout

_SCRIPT = Path(__file__).resolve().parents[1] / "fixtures" / "scripts" / "spawn_tree.py"


def test_timeout_kills_leader_and_child(tmp_path: Path) -> None:
    """A timeout SIGKILLs the session leader and the child it forked."""
    pid_file = tmp_path / "pids"
    with pytest.raises(ToolTimeout):
        Jail().run(
            [sys.executable, str(_SCRIPT), str(pid_file)],
            cwd=tmp_path,
            timeout_s=1,
            env=tool_env(),
        )
    leader, child = (int(line) for line in pid_file.read_text(encoding="utf-8").split())
    assert _dead(leader)
    assert _dead(child)


def test_successful_command_returns_stdout(tmp_path: Path) -> None:
    """A command that finishes in time returns its stdout and status."""
    result = Jail().run(
        [sys.executable, "-c", "print(123)"],
        cwd=tmp_path,
        timeout_s=5,
        env=tool_env(),
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "123"
    with pytest.raises(ValueError):
        Jail().run([], cwd=tmp_path, timeout_s=5, env=tool_env())
    with pytest.raises(ValueError):
        Jail().run([sys.executable, "-c", "print(1)"], cwd=tmp_path, timeout_s=0, env=tool_env())


def test_child_env_excludes_agent_llm_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The child sees the passed mapping and not the parent's API key."""
    monkeypatch.setenv("AGENT_LLM_API_KEY", "secret")
    env = tool_env()
    assert "AGENT_LLM_API_KEY" not in env
    result = Jail().run(
        [sys.executable, "-c", "import os; print(repr(os.environ.get('AGENT_LLM_API_KEY')))"],
        cwd=tmp_path,
        timeout_s=5,
        env=env,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "None"


def _dead(pid: int) -> bool:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False
