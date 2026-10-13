"""Search, remember, the context budget, and the trajectory log."""

from __future__ import annotations

import json
from pathlib import Path

from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.tenant_store import get_task, provision
from agent_engine.adapters.sqlite.trajectory_store import TrajectoryStore
from agent_engine.adapters.workspace.plain import PlainWorkspace
from agent_engine.application.context import (
    ContextBuilder,
    list_files,
    outcome_line,
    search_text,
)
from agent_engine.domain.models import TenantId
from tests.step4_support import NOW, WORKER, FixedChecks, ScriptedLlm, StatusGate, call, lease_task, loop


def test_search_text_stays_inside_the_work_root(tmp_path: Path) -> None:
    """A path that leaves the work root raises and writes nothing."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        (env.work / "adder.py").write_text("return a - b\n" * 20, encoding="utf-8")
        outside = tmp_path / "secret.txt"
        outside.write_text("top-secret-marker\n", encoding="utf-8")
        found = json.loads(search_text(PlainWorkspace(env.work), "return", "adder.py"))
        assert len(found["hits"]) == 20
        assert search_text(PlainWorkspace(env.work), "return", "missing-dir") == '{"hits": []}'
        assert list_files(PlainWorkspace(env.work))
        for escaped in ("../secret.txt", str(outside), ""):
            try:
                search_text(PlainWorkspace(env.work), "top-secret-marker", escaped)
            except ValueError:
                pass
            else:
                raise AssertionError(escaped)
        assert outside.read_text(encoding="utf-8") == "top-secret-marker\n"
        llm = ScriptedLlm([(call("search_text", {"query": "top-secret-marker", "path": ".."}),), ()])
        result = loop(env, llm, StatusGate(env, None), FixedChecks()).run(NOW)
        assert result.status == "running"
        assert "top-secret-marker" not in result.detail
        assert "ValueError" in _step_result(env)
    finally:
        env.connection.close()


def test_remember_is_visible_on_the_next_model_call(tmp_path: Path) -> None:
    """The packed prompt after remember contains the note."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        (env.work / "adder.py").write_text("return a - b\n", encoding="utf-8")
        builder = _builder(env)
        llm = ScriptedLlm(
            [
                (
                    call(
                        "remember",
                        {"path": "adder.py", "quote": "return a - b", "note": "remembered-helper"},
                    ),
                ),
                (call("read_file", {"path": "adder.py"}),),
                (),
            ]
        )
        result = loop(env, llm, StatusGate(env, None), FixedChecks(), context=builder).run(NOW)
        assert result.status == "running"
        assert "remembered-helper" in json.dumps(llm.seen[1])
    finally:
        env.connection.close()


def test_acme_knowledge_is_invisible_in_globex(tmp_path: Path) -> None:
    """An acme note and trajectory row are absent from globex."""
    provision(tmp_path, TenantId("globex"), "Globex")
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        (env.work / "adder.py").write_text("return a - b\n", encoding="utf-8")
        llm = ScriptedLlm(
            [
                (
                    call(
                        "remember",
                        {"path": "adder.py", "quote": "return a - b", "note": "only-acme"},
                    ),
                ),
                (),
            ]
        )
        loop(env, llm, StatusGate(env, None), FixedChecks(), context=_builder(env)).run(NOW)
        globex = open_tenant_db(tmp_path, TenantId("globex"), read_only=True)
        try:
            notes = globex.execute("SELECT COUNT(*) AS n FROM knowledge").fetchone()["n"]
            rows = globex.execute("SELECT COUNT(*) AS n FROM trajectory").fetchone()["n"]
        finally:
            globex.close()
        assert notes == 0
        assert rows == 0
    finally:
        env.connection.close()


def test_over_budget_truncates_old_tool_bodies_and_keeps_the_outcome(tmp_path: Path) -> None:
    """An old tool body shrinks to the outcome line and stays full in the log."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        _seed_steps(env, huge="HUGE-TOOL-BODY")
        packed = ContextBuilder(_store(env), budget=80).pack(env.tenant, env.task_id)
        text = json.dumps(packed)
        assert "search_text ok" in text
        assert "HUGE-TOOL-BODY" not in text
        assert any("HUGE-TOOL-BODY" in body for body in _bodies(env))
        history = [
            {"role": "user", "content": "x" * 200},
            {"role": "user", "content": "StaleContent " + ("y" * 200)},
        ]
        shrunk_history = ContextBuilder(budget=1).pack(env.tenant, env.task_id, history)
        assert "tool ok" in json.dumps(shrunk_history)
        assert "tool StaleContent" in json.dumps(shrunk_history)
        assert outcome_line(
            {"tool": "run_checks", "result": {"ok": False, "sha256": "abc"}}
        ) == "run_checks exit 1 abc"
        assert outcome_line(
            {"tool": "replace_text", "result": {"error": "StaleContent", "sha256": "abc"}}
        ) == "replace_text StaleContent abc"
        assert outcome_line({"tool": "read_file", "sha256": "abc"}) == "read_file ok abc"
        assert outcome_line(
            {
                "tool": "read_file",
                "result": {
                    "sha256": "abc",
                    "content": "except StaleContent:\n    pass\n",
                },
            }
        ) == "read_file ok abc"
    finally:
        env.connection.close()


def test_summary_replaces_the_middle_and_stores_a_context_summary(tmp_path: Path) -> None:
    """A pack that stays over budget stores one context_summary."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        _seed_steps(env, huge="HUGE-TOOL-BODY")
        seen: list[str] = []

        def summarize(messages: list[dict[str, object]]) -> str:
            seen.append(json.dumps(messages))
            return "compressed the middle"

        before = get_task(tmp_path, env.tenant, env.task_id).status
        packed = ContextBuilder(_store(env), budget=40, summarize=summarize).pack(
            env.tenant, env.task_id
        )
        assert seen
        assert "HUGE-TOOL-BODY" not in seen[0]
        assert "compressed the middle" in json.dumps(packed)
        assert "context_summary" in _kinds(env)
        assert get_task(tmp_path, env.tenant, env.task_id).status == before
    finally:
        env.connection.close()


def test_summary_does_not_set_status(tmp_path: Path) -> None:
    """write_summary leaves tasks.status pending."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        assert get_task(tmp_path, env.tenant, env.task_id).status == "pending"
        _store(env).write_summary(env.tenant, env.task_id, "compressed the middle")
        assert get_task(tmp_path, env.tenant, env.task_id).status == "pending"
        packed = ContextBuilder(_store(env), budget=8000).pack(env.tenant, env.task_id)
        assert "compressed the middle" in json.dumps(packed)
    finally:
        env.connection.close()


def test_t2_appends_inference_tool_call_and_result(tmp_path: Path) -> None:
    """One search step commits model_inference, tool_call, and tool_result together."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        (env.work / "adder.py").write_text("return a - b\n", encoding="utf-8")
        llm = ScriptedLlm([(call("search_text", {"query": "return", "path": "."}),), ()])
        loop(env, llm, StatusGate(env, None), FixedChecks(), context=_builder(env)).run(NOW)
        kinds = _kinds(env)
        assert {"model_inference", "tool_call", "tool_result"} <= kinds
    finally:
        env.connection.close()


def test_bearer_token_is_absent_from_trajectory_and_notes(tmp_path: Path) -> None:
    """A note that contains the bearer token is stored redacted."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        (env.work / "adder.py").write_text("return a - b\n", encoding="utf-8")
        secret = "super-secret-token"
        llm = ScriptedLlm(
            [
                (
                    call(
                        "remember",
                        {
                            "path": "adder.py",
                            "quote": f"return {secret}",
                            "note": secret,
                        },
                    ),
                ),
                (),
            ]
        )
        loop(
            env,
            llm,
            StatusGate(env, None),
            FixedChecks(),
            context=_builder(env, secret=secret),
            secret=secret,
        ).run(NOW)
        blob = "\n".join(_bodies(env))
        notes = "\n".join(_notes(env))
        assert secret not in blob
        assert secret not in notes
        assert "[redacted]" in notes
    finally:
        env.connection.close()


def _builder(env, secret: str = "") -> ContextBuilder:  # type: ignore[no-untyped-def]
    return ContextBuilder(_store(env), budget=8000, secret=secret)


def _store(env) -> TrajectoryStore:  # type: ignore[no-untyped-def]
    return TrajectoryStore(env.connection, WORKER, env.token, lambda: NOW)


def _seed_steps(env, huge: str) -> None:  # type: ignore[no-untyped-def]
    env.connection.execute("BEGIN IMMEDIATE")
    env.connection.execute(
        """
        INSERT INTO trajectory (tenant_id, task_id, seq, step_index, kind, body_json)
        VALUES (?, ?, 0, 0, 'tool_result', ?)
        """,
        (
            env.tenant.value,
            env.task_id,
            json.dumps({"tool": "search_text", "result": {"text": huge * 80}}),
        ),
    )
    env.connection.execute(
        """
        INSERT INTO trajectory (tenant_id, task_id, seq, step_index, kind, body_json)
        VALUES (?, ?, 1, 1, 'tool_result', ?)
        """,
        (
            env.tenant.value,
            env.task_id,
            json.dumps({"tool": "read_file", "result": {"status": "short"}}),
        ),
    )
    env.connection.execute("COMMIT")


def _step_result(env) -> str:  # type: ignore[no-untyped-def]
    row = env.connection.execute("SELECT result_json FROM step_intents").fetchone()
    return "" if row is None else str(row["result_json"])


def _bodies(env) -> list[str]:  # type: ignore[no-untyped-def]
    rows = env.connection.execute("SELECT body_json FROM trajectory").fetchall()
    return [str(row["body_json"]) for row in rows]


def _kinds(env) -> set[str]:  # type: ignore[no-untyped-def]
    rows = env.connection.execute("SELECT kind FROM trajectory").fetchall()
    return {str(row["kind"]) for row in rows}


def _notes(env) -> list[str]:  # type: ignore[no-untyped-def]
    rows = env.connection.execute("SELECT path, quote, body FROM knowledge").fetchall()
    return [f"{row['path']} {row['quote']} {row['body']}" for row in rows]
