"""CLI gate for the canned MCP echo tool."""

from pathlib import Path

import pytest

from agent_engine.self_test import run_self_test


def test_self_test_step6_echo_tool_prints_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """self-test --step 6 prints the echo text and PASS."""
    monkeypatch.setenv("AGENT_LLM_API_KEY", "parent-secret-key")
    code = run_self_test(tmp_path, 6, None)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "ping" in output
    assert "PASS" in output
    assert "parent-secret-key" not in output
    assert "mcp-step6-sentinel" not in output
    assert "completed" not in output
    text = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    assert "mcp.json" in text
    assert "mcp__{server}__{tool}" in text
    assert "self-test --step 6" in text
