"""VerifyGate is the only writer of completed and rejected."""

import shlex
import sys
from pathlib import Path

import pytest

from agent_engine.adapters.exec.jail import Jail
from agent_engine.adapters.verifier.external import ExternalVerifier
from agent_engine.application.verify_gate import VerifyGate
from agent_engine.domain.errors import FenceRejected, ToolTimeout
from tests.step4_support import FIXTURE, NOW, WORKER, lease_task


def test_verifier_pass_sets_completed(tmp_path: Path) -> None:
    """A fixed adder passes the Python fixture and the gate writes completed."""
    env = lease_task(tmp_path)
    try:
        (env.work / "adder.py").write_text(
            (env.work / "adder.py").read_text(encoding="utf-8").replace(
                "return a - b", "return a + b"
            ),
            encoding="utf-8",
        )
        _install_checks(tmp_path)
        gate = _gate(env, tmp_path)
        assert gate.finish(NOW) == "completed"
        assert env.store.task_status(env.tenant, env.task_id) == "completed"
    finally:
        env.connection.close()


def test_verifier_fail_sets_rejected(tmp_path: Path) -> None:
    """A broken adder fails the fixture and the gate writes rejected without returning the traceback."""
    env = lease_task(tmp_path)
    try:
        _install_checks(tmp_path)
        gate = _gate(env, tmp_path)
        assert gate.finish(NOW) == "rejected"
        assert env.store.task_status(env.tenant, env.task_id) == "rejected"
    finally:
        env.connection.close()


def test_modified_check_file_fails_the_gate(tmp_path: Path) -> None:
    """A work-tree test_adder.py that differs from checks/ fails even when the adder is fixed."""
    env = lease_task(tmp_path)
    try:
        (env.work / "adder.py").write_text(
            "def add(a: int, b: int) -> int:\n    return a + b\n",
            encoding="utf-8",
        )
        _install_checks(tmp_path)
        (env.work / "test_adder.py").write_text(
            "import unittest\n\nclass T(unittest.TestCase):\n    def test_ok(self):\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        gate = _gate(env, tmp_path)
        assert gate.finish(NOW) == "rejected"
        assert env.store.task_status(env.tenant, env.task_id) == "rejected"
    finally:
        env.connection.close()


def test_verifier_child_env_has_no_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The jailed check process does not receive AGENT_LLM_API_KEY."""
    sentinel = "secret-not-for-the-child"
    monkeypatch.setenv("AGENT_LLM_API_KEY", sentinel)
    checks = tmp_path / "checks"
    checks.mkdir()
    code = "import os; print(repr(os.environ.get('AGENT_LLM_API_KEY')))"
    (checks / "commands").write_text(
        "\n" + f"{sys.executable} -c {shlex.quote(code)}\n",
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    ok, stdout, stderr = ExternalVerifier(checks, Jail()).run(work)
    assert ok, stderr
    assert "None" in stdout
    assert sentinel not in stdout
    assert sentinel not in stderr


def test_verifier_timeout_is_a_failed_result(tmp_path: Path) -> None:
    """A jailed timeout is a failed check and does not escape as an uncaught error."""
    checks = tmp_path / "checks"
    checks.mkdir()
    (checks / "commands").write_text("python -c 'print(1)'\n", encoding="utf-8")

    class _Slow:
        def run(self, argv: list[str], *, cwd: Path, timeout_s: float, env: dict[str, str]):
            del argv, cwd, timeout_s, env
            raise ToolTimeout("slow")

    ok, _stdout, stderr = ExternalVerifier(checks, _Slow()).run(tmp_path / "work")
    assert ok is False
    assert "slow" in stderr


def test_fence_rejected_leaves_the_task_running(tmp_path: Path) -> None:
    """A stale token cannot write completed, and the task stays running."""
    env = lease_task(tmp_path)
    try:
        env.store.mark_running(env.tenant, env.task_id, WORKER, env.token, NOW)
        gate = VerifyGate(
            env.tenant,
            env.task_id,
            WORKER,
            env.token + 1,
            env.work,
            _Ok(),
            env.store,
        )
        with pytest.raises(FenceRejected):
            gate.finish(NOW)
        assert env.store.task_status(env.tenant, env.task_id) == "running"
    finally:
        env.connection.close()


def _install_checks(data_root: Path) -> None:
    checks = data_root / "checks"
    checks.mkdir(parents=True, exist_ok=True)
    source = FIXTURE.parent / "test_adder.py"
    (checks / "test_adder.py").write_bytes(source.read_bytes())


def _gate(env, data_root: Path) -> VerifyGate:
    return VerifyGate(
        env.tenant,
        env.task_id,
        WORKER,
        env.token,
        env.work,
        ExternalVerifier(data_root / "checks", Jail()),
        env.store,
    )


class _Ok:
    def run(self, work_root: Path) -> tuple[bool, str, str]:
        del work_root
        return True, "", ""
