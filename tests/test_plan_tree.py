"""Plan, accept, apply, and the write_file path jail."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_engine.__main__ import main
from agent_engine.adapters.sqlite.tenant_store import init_task, provision
from agent_engine.adapters.workspace.plain import PlainWorkspace
from agent_engine.application.plan import (
    PlanSession,
    apply_prompt,
    load_plan,
    pinned_plan,
    plan_body,
    render_tree,
    task_slug,
    validate_plan,
)
from agent_engine.composition import apply_plan, format_report, start_project
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
    report = apply_plan(tmp_path, TenantId("acme"), TASK, llm="canned")
    packed = json.loads(report.detail)
    system = str(packed[0]["content"])
    assert PROMPT in system
    assert APP in system
    assert TEST in system
    assert "Fix adder.py" not in system
    assert "Do not list, search, or remember" in system
    rendered = format_report(report)
    assert rendered.startswith("running")
    assert "system" in rendered
    assert not rendered.splitlines()[1].startswith("[{")


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


def test_new_provisions_plans_and_builds_in_one_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """new --yes names the task after the prompt and writes the tree."""
    code = main(
        [
            "--data-root",
            str(tmp_path),
            "new",
            PROMPT,
            "--llm",
            "canned",
            "--yes",
            "--checks-cmd",
            "python -m pytest -q",
        ]
    )
    printed = capsys.readouterr().out
    assert code == 0
    slug = "build-a-small-api-package-with-tests"
    assert f"task={slug}" in printed
    assert "status=running" in printed
    assert "services/" in printed
    work = tmp_path / "tenants" / "acme" / "tasks" / slug / "work"
    assert (work / APP).is_file()
    assert (work / TEST).is_file()
    assert (tmp_path / "checks" / "commands").read_text(encoding="utf-8") == (
        "python -m pytest -q\n"
    )


def test_new_without_a_yes_leaves_the_plan_proposed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A declined plan is stored, prints how to resume, and writes no file."""
    base = ["--data-root", str(tmp_path), "new", PROMPT, "--llm", "canned"]
    assert main([*base, "--plan-only"]) == 0
    printed = capsys.readouterr().out
    assert "plan is proposed" in printed
    assert "agent accept" in printed
    slug = "build-a-small-api-package-with-tests"
    work = tmp_path / "tenants" / "acme" / "tasks" / slug / "work"
    assert not work.is_dir() or list(work.rglob("*")) == []
    assert not (tmp_path / "checks" / "commands").exists()
    assert main([*base, "--yes"]) == 0
    assert f"task={slug}-2" in capsys.readouterr().out


def test_start_project_takes_an_explicit_task_and_a_refusal(tmp_path: Path) -> None:
    """A named task is used as given, and a false confirm writes nothing."""
    refused = start_project(
        tmp_path,
        TenantId("acme"),
        PROMPT,
        task_id=TASK,
        llm="canned",
        confirm=lambda tree: False,
    )
    assert refused.task_id == TASK
    assert refused.accepted is False
    assert refused.report is None
    assert "app.py" in refused.tree
    assert _work_files(tmp_path) == []
    built = start_project(
        tmp_path,
        TenantId("acme"),
        PROMPT,
        task_id=TASK,
        llm="canned",
        checks_cmd=("python -m pytest -q",),
    )
    assert built.accepted is True
    assert built.report is not None
    assert built.work_root == _work_root(tmp_path)
    assert _work_files(tmp_path) == [APP, TEST]


def test_task_slug_keeps_whole_words_and_needs_one_word() -> None:
    """The slug is a short lowercase id built from the prompt words."""
    assert task_slug("Build a FastAPI service!") == "build-a-fastapi-service"
    assert task_slug("  Add   health  endpoint  ") == "add-health-endpoint"
    long = task_slug("build an extremely detailed inventory reporting service today")
    assert len(long) <= 40
    assert not long.endswith("-")
    assert task_slug("x" * 60) == "x" * 40
    with pytest.raises(ValueError):
        task_slug("!!! ???")


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
    """A plan needs a summary, steps, at least one file, a safe path, and an action."""
    good = _plan_payload()
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
        _plan_payload(
            files=[
                {"path": "README.md", "action": "create", "detail": "what this repo is"},
                {"path": APP, "action": "create", "detail": "add"},
                {"path": TEST, "action": "edit", "detail": "cover add"},
            ]
        )
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
    assert "Do not list or search the tree" in apply_prompt(PROMPT)


def test_plan_session_offers_only_submit_plan() -> None:
    """The session refuses an empty prompt, another tool, and bad arguments."""
    body = json.dumps(_plan_payload(summary="ok"))
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


def test_plan_session_reads_an_existing_tree_and_does_not_write(tmp_path: Path) -> None:
    """Lookups run against the tree. write_file is refused and creates nothing."""
    (tmp_path / "app.py").write_text("marker-in-tree\n", encoding="utf-8")
    planner = _Planner(
        [
            (
                call("list_files", {"path": "."}),
                call("search_text", {"query": "marker-in-tree", "path": "."}),
                call("write_file", {"path": "extra.py", "content": "nope"}),
            ),
            (ToolCall("submit_plan", json.dumps(_plan_payload()), "plan-final"),),
        ]
    )
    notes: list[str] = []
    plan = PlanSession(planner, trace=notes.append, codebase=PlainWorkspace(tmp_path)).propose(
        PROMPT
    )
    sent = json.dumps(planner.seen[1])
    assert plan.files[0].path == APP
    assert "app.py" in sent
    assert "marker-in-tree" in sent
    assert "planning cannot write" in sent
    assert not (tmp_path / "extra.py").exists()
    assert planner.seen[0][0]["content"].startswith("The work tree already")
    assert any(line.startswith("planning: list_files") for line in notes)
    assert any(line.startswith("planning: search_text") for line in notes)


def test_plan_lookup_rejects_a_path_outside_the_tree(tmp_path: Path) -> None:
    """A lookup outside the tree is an error result and writes nothing."""
    (tmp_path / "app.py").write_text("keep\n", encoding="utf-8")
    planner = _Planner(
        [
            (call("read_file", {"path": "../secret.txt"}),),
            (ToolCall("submit_plan", json.dumps(_plan_payload()), "plan-final"),),
        ]
    )
    plan = PlanSession(planner, codebase=PlainWorkspace(tmp_path)).propose(PROMPT)
    assert plan.summary == "Create the api package."
    assert "ValueError" in json.dumps(planner.seen[1])
    assert not (tmp_path.parent / "secret.txt").exists()


def _plan_payload(
    files: list[dict[str, str]] | None = None, summary: str = "Create the api package."
) -> dict[str, object]:
    return {
        "summary": summary,
        "approach": "Create the module with write_file, then the test, then run checks.",
        "steps": [
            {
                "title": "Create the module",
                "detail": "write_file services/api/app.py with add.",
            }
        ],
        "files": files
        if files is not None
        else [{"path": APP, "action": "create", "detail": "add returns a + b"}],
    }


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
