"""CLI gate for canned search, remember, and the context summary."""

from pathlib import Path

import pytest

from agent_engine.self_test import run_self_test


def test_self_test_step7_canned_search_prints_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """self-test --step 7 prints the remembered note, the summary, and PASS."""
    code = run_self_test(tmp_path, 7, None)
    output = capsys.readouterr().out
    assert code == 0, output
    assert "remembered-helper" in output
    assert "search_text ok" in output
    assert "compressed the middle" in output
    assert "PASS" in output
    assert "completed" not in output
    text = Path(__file__).resolve().parents[1].joinpath("README.md").read_text(encoding="utf-8")
    assert "list_files" in text
    assert "search_text" in text
    assert "remember" in text
    assert "context_token_budget" in text
    assert "self-test --step 7" in text
