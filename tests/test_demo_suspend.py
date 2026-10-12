"""SIGSTOP, lease steal, and the globex sentinel."""

import hashlib
from pathlib import Path

from agent_engine.demo import _scenario_suspend, run_scenarios, seed_world


def test_sigstop_then_steal_then_sigcont_rejects_old_token(
    tmp_path: Path, capsys
) -> None:
    """The log order is stop, expiry, steal, continue, then FenceRejected."""
    _scenario_suspend(tmp_path)
    output = capsys.readouterr().out
    assert output.index("stop") < output.index("expiry")
    assert output.index("expiry") < output.index("steal")
    assert output.index("steal") < output.index("continue")
    assert output.index("continue") < output.index("FenceRejected")


def test_globex_sentinel_unchanged_after_acme_scenario(tmp_path: Path, capsys) -> None:
    """An acme rollback leaves the globex sentinel hash and an empty step table."""
    digest = seed_world(tmp_path)
    run_scenarios(tmp_path, (6,), llm="canned")
    sentinel = tmp_path / "tenants" / "globex" / "sentinel"
    assert hashlib.sha256(sentinel.read_bytes()).hexdigest() == digest
    output = capsys.readouterr().out
    assert f"sentinel {digest}" in output
    assert "rollback_ms=" in output
    from agent_engine.demo import _globex_steps

    assert _globex_steps(tmp_path) == 0
