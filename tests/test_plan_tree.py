"""Plan, accept, apply, and the write_file path jail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_engine.__main__ import main
from agent_engine.adapters.sqlite.tenant_store import init_task, provision
from agent_engine.application.plan import (
    PlanSession,
    apply_prompt,
    load_plan,
    pinned_plan,
    plan_body,
    render_tree,
    validate_plan,
)
from agent_engine.domain.models import TenantId, ToolCall
from step4_support import NOW, FixedChecks, ScriptedLlm, StatusGate, call, lease_task, loop

PROMPT = "Build a small api package with tests."
TASK = "api-tree"
APP = "services/api/app.py"
TEST = "services/api/tests/test_app.py"


def test_canned_plan_stores_a_tree_and_leaves_the_work_tree_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """plan prints the tree, stores it as proposed, and writes no file."""
    _prepare(tmp_path)
    assert _cli(tmp_path, "plan", "--prompt", PROMPT, "--llm", "canned") == 0
    printed = capsys.readouterr().out
    assert "services/" in printed
    assert "app.py" in printed
    assert _work_files(tmp_path) == []
    assert _cli(tmp_path, "show-plan") == 0
    shown = capsys.readouterr().out
    assert f"prompt={PROMPT}" in shown
    assert "status=proposed" in shown
    assert TEST.rsplit("/", 1)[1] in shown


def test_apply_before_accept_exits_nonzero_and_creates_no_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """apply refuses a proposed plan, and a missing plan, without writing."""
    _prepare(tmp_path)
    assert _cli(tmp_path, "apply", "--llm", "canned") == 1
    assert "no plan" in capsys.readouterr().err
    assert _cli(tmp_path, "plan", "--prompt", PROMPT, "--llm", "canned") == 0
    capsys.readouterr()
    assert _cli(tmp_path, "apply", "--llm", "canned") == 1
    assert "not accepted" in capsys.readouterr().err
    assert _work_files(tmp_path) == []


def test_accepted_plan_writes_the_nested_tree_and_the_check_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After accept, canned write_file calls create both planned paths."""
    _prepare(tmp_path)
    assert _cli(tmp_path, "plan", "--prompt", PROMPT, "--llm", "canned") == 0
    assert _cli(tmp_path, "accept") == 0
    assert "accepted" in capsys.readouterr().out
    assert _cli(tmp_path, "apply", "--llm", "canned", "--checks-cmd", "python -m pytest -q") == 0
    printed = capsys.readouterr().out
    assert printed.startswith("running")
    work = _work_root(tmp_path)
    assert _work_files(tmp_path) == [APP, TEST]
    assert b"return a + b" in (work / APP).read_bytes()
    commands = tmp_path / "checks" / "commands"
    assert commands.read_text(encoding="utf-8") == "python -m pytest -q\n"
    assert _cli(tmp_path, "show-plan") == 0
    assert "status=accepted" in capsys.readouterr().out


def test_apply_packs_the_user_sentence_and_the_planned_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The packed prompt names the prompt and the plan, not only the adder."""
    _prepare(tmp_path)
    assert _cli(tmp_path, "plan", "--prompt", PROMPT, "--llm", "canned") == 0
    assert _cli(tmp_path, "accept") == 0
    capsys.readouterr()
    assert _cli(tmp_path, "apply", "--llm", "canned") == 0
    packed = json.loads(capsys.readouterr().out.splitlines()[1])
    system = str(packed[0]["content"])
    assert PROMPT in system
    assert APP in system
    assert TEST in system
    assert "Fix adder.py" not in system


def test_plan_refuses_a_second_proposal_after_accept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An accepted plan is not replaced, and accept needs a stored plan."""
    _prepare(tmp_path)
    assert _cli(tmp_path, "accept") == 1
    assert "no plan to accept" in capsys.readouterr().err
    assert _cli(tmp_path, "plan", "--prompt", PROMPT, "--llm", "canned") == 0
    assert _cli(tmp_path, "plan", "--prompt", "something else", "--llm", "canned") == 0
    capsys.readouterr()
    assert _cli(tmp_path, "show-plan") == 0
    assert "prompt=something else" in capsys.readouterr().out
    assert _cli(tmp_path, "accept") == 0
    capsys.readouterr()
    assert _cli(tmp_path, "plan", "--prompt", PROMPT, "--llm", "canned") == 1
    assert "already accepted" in capsys.readouterr().err


def test_write_file_jails_the_path_and_keeps_an_existing_file(tmp_path: Path) -> None:
    """Dot-dot, absolute, and existing paths write nothing. A new path nests."""
    env = lease_task(tmp_path, TASK, broken=False)
    try:
        llm = ScriptedLlm(
            [
                (call("write_file", {"path": "../secret.txt", "content": "leak"}),),
                (call("write_file", {"path": "/tmp/secret.txt", "content": "leak"}),),
                (call("write_file", {"path": "", "content": "leak"}),),
                (call("write_file", {"path": APP, "content": "first\n"}),),
                (call("write_file", {"path": APP, "content": "second\n"}),),
                (),
            ]
        )
        runner = loop(env, llm, StatusGate(env, None), FixedChecks())
        report = runner.run(NOW)
        assert report.status == "running"
        assert report.executions == 5
        results = _results(env)
        assert [json.loads(row)["error"] for row in results[:3]] == [
            "ValueError",
            "ValueError",
            "ValueError",
        ]
        assert json.loads(results[3])["status"] == "written"
        assert json.loads(results[4])["error"] == "FileExists"
        assert (env.work / APP).read_bytes() == b"first\n"
        assert not (env.work.parent / "secret.txt").exists()
        assert list(env.work.rglob("secret.txt")) == []
    finally:
        env.connection.close()


def test_write_file_rejects_a_non_object_argument(tmp_path: Path) -> None:
    """A JSON array for write_file is an unknown tool and writes nothing."""
    env = lease_task(tmp_path, TASK, broken=False)
    try:
        llm = ScriptedLlm([(ToolCall("write_file", json.dumps(["x"]), ""),), ()])
        runner = loop(env, llm, StatusGate(env, None), FixedChecks())
        runner.run(NOW)
        assert json.loads(_results(env)[0])["error"] == "unknown tool"
    finally:
        env.connection.close()


def test_validate_plan_rejects_bad_bodies() -> None:
    """A plan needs a summary, at least one file, a safe path, and an action."""
    good = {
        "summary": "Create the api package.",
        "files": [{"path": APP, "action": "create", "detail": "add"}],
    }
    plan = validate_plan(good)
    assert plan.files[0].path == APP
    assert load_plan(plan_body(plan)) == plan
    for bad in (
        ["not", "an", "object"],
        {"files": good["files"]},
        {"summary": "  ", "files": good["files"]},
        {"summary": "ok", "files": []},
        {"summary": "ok", "files": ["not an object"]},
        {"summary": "ok", "files": [{"path": "", "action": "create"}]},
        {"summary": "ok", "files": [{"path": "../x.py", "action": "create"}]},
        {"summary": "ok", "files": [{"path": "/x.py", "action": "create"}]},
        {"summary": "ok", "files": [{"path": "pkg/", "action": "create"}]},
        {"summary": "ok", "files": [{"path": APP, "action": "delete"}]},
        {
            "summary": "ok",
            "files": [
                {"path": APP, "action": "create"},
                {"path": APP, "action": "edit"},
            ],
        },
    ):
        with pytest.raises(ValueError):
            validate_plan(bad)
    with pytest.raises(ValueError):
        load_plan("{not json")


def test_render_tree_nests_directories_before_files() -> None:
    """The tree prints one line per level with directories first."""
    plan = validate_plan(
        {
            "summary": "ok",
            "files": [
                {"path": "README.md", "action": "create"},
                {"path": APP, "action": "create", "detail": "add"},
                {"path": TEST, "action": "edit"},
            ],
        }
    )
    assert render_tree(plan).splitlines() == [
        "services/",
        "  api/",
        "    tests/",
        "      test_app.py",
        "    app.py",
        "README.md",
    ]
    pinned = pinned_plan(plan)
    assert "create services/api/app.py - add" in pinned
    assert f"edit {TEST}" in pinned
    assert PROMPT in apply_prompt(PROMPT)
    assert "write_file" in apply_prompt(PROMPT)


def test_plan_session_offers_only_submit_plan() -> None:
    """The session refuses an empty prompt, another tool, and bad arguments."""
    body = json.dumps({"summary": "ok", "files": [{"path": APP, "action": "create"}]})
    planner = _Planner([(ToolCall("submit_plan", body, "plan-0"),)])
    plan = PlanSession(planner).propose(PROMPT)
    assert plan.summary == "ok"
    assert planner.seen[0][1]["content"] == PROMPT
    with pytest.raises(ValueError):
        PlanSession(planner).propose("  ")
    for turn in (
        (),
        (ToolCall("write_file", "{}", "plan-0"),),
        (ToolCall("submit_plan", "{not json", "plan-0"),),
    ):
        with pytest.raises(ValueError):
            PlanSession(_Planner([turn])).propose(PROMPT)


class _Planner:
    """Scripted planning port. It does not post to the network."""

    def __init__(self, turns: list[tuple[ToolCall, ...]]) -> None:
        """Store one tuple of tool calls per plan call.

        Args:
            turns: Consumed in order.
        """
        self._turns = list(turns)
        self.seen: list[list[dict[str, object]]] = []

    def plan(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
        """Pop the next scripted turn.

        Args:
            messages: Recorded so a test can read the planning prompt.

        Returns:
            The next tuple of tool calls.
        """
        self.seen.append(messages)
        return self._turns.pop(0)


def _prepare(data_root: Path) -> None:
    tenant = TenantId("acme")
    provision(data_root, tenant, "Acme")
    init_task(data_root, tenant, TASK)


def _cli(data_root: Path, command: str, *args: str) -> int:
    return main(
        [
            "--data-root",
            str(data_root),
            command,
            "--tenant",
            "acme",
            "--task",
            TASK,
            *args,
        ]
    )


def _work_root(data_root: Path) -> Path:
    return data_root / "tenants" / "acme" / "tasks" / TASK / "work"


def _work_files(data_root: Path) -> list[str]:
    root = _work_root(data_root)
    if not root.is_dir():
        return []
    return sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    )


def _results(env) -> list[str]:  # type: ignore[no-untyped-def]
    rows = env.connection.execute(
        """
        SELECT result_json FROM step_intents
        WHERE tenant_id = ? AND task_id = ?
        ORDER BY step_index
        """,
        (env.tenant.value, env.task_id),
    ).fetchall()
    return [str(row["result_json"]) for row in rows]
