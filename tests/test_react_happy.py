"""ReAct loop: T1 before the tool, one call per step, cache, and stale bytes."""

import hashlib
import json
import shlex
import sys
from pathlib import Path

import pytest

from agent_engine.adapters.exec.jail import Jail
from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.verifier.external import ExternalVerifier
from agent_engine.adapters.sqlite.trajectory_store import TrajectoryStore
from agent_engine.adapters.workspace.plain import PlainWorkspace
from agent_engine.application.context import (
    ContextBuilder,
    list_files,
    outcome_line,
    remember_note,
    search_text,
)
from agent_engine.application.react import idempotency_key
from agent_engine.domain.models import ToolCall
from tests.step4_support import (
    NOW,
    WORKER,
    FixedChecks,
    ScriptedLlm,
    StatusGate,
    call,
    lease_task,
    loop,
)


def test_intent_row_is_durable_before_tool_execution(tmp_path: Path) -> None:
    """T1 is committed before the editor opens the file for write."""
    env = lease_task(tmp_path)
    probe = open_tenant_db(tmp_path, env.tenant)
    try:
        digest = hashlib.sha256((env.work / "adder.py").read_bytes()).hexdigest()
        guard = _Guard(env.work, probe, env.task_id)
        llm = ScriptedLlm(
            (
                (
                    call(
                        "replace_text",
                        {
                            "path": "adder.py",
                            "expected_sha256": digest,
                            "old": "return a - b",
                            "new": "return a + b",
                        },
                    ),
                ),
            )
        )
        with pytest.raises(RuntimeError, match="stop"):
            loop(env, llm, StatusGate(env, None), FixedChecks(), guard).run(NOW)
        assert guard.seen
        row = probe.execute(
            "SELECT status FROM step_intents WHERE task_id = ?",
            (env.task_id,),
        ).fetchone()
        assert row["status"] == "intent_logged"
    finally:
        probe.close()
        env.connection.close()


def test_resume_of_intent_logged_skips_model(tmp_path: Path) -> None:
    """An intent_logged finish runs without asking the model for that step."""
    env = lease_task(tmp_path)
    try:
        args = {}
        env.store.insert_intent(
            env.tenant,
            env.task_id,
            0,
            idempotency_key("acme", env.task_id, 0, "finish", args),
            "finish",
            json.dumps(args),
            WORKER,
            env.token,
            NOW,
            (),
        )
        llm = ScriptedLlm(())
        result = loop(env, llm, StatusGate(env, "completed"), FixedChecks()).run(NOW)
        assert llm.post_count() == 0
        assert result.status == "completed"
        assert result.executions == 1
        assert env.store.open_intent(env.tenant, env.task_id) is None
    finally:
        env.connection.close()


def test_committed_step_is_cache_hit_with_zero_executions_and_zero_posts(tmp_path: Path) -> None:
    """A completed task replays from cache and does not call the model or a tool."""
    env = lease_task(tmp_path)
    try:
        first = loop(
            env,
            ScriptedLlm(((call("finish", {}),),)),
            StatusGate(env, "completed"),
            FixedChecks(),
        ).run(NOW)
        assert first.status == "completed"
        assert first.cache_hit is False
        second_llm = ScriptedLlm(())
        second = loop(env, second_llm, StatusGate(env, "completed"), FixedChecks()).run(NOW)
        assert second.cache_hit is True
        assert second.executions == 0
        assert second.posts == 0
        assert second.status == "completed"
    finally:
        env.connection.close()


def test_replay_of_applied_edit_is_already_applied_and_writes_nothing(tmp_path: Path) -> None:
    """Resuming a replace whose postimage is on disk does not open the file for write."""
    env = lease_task(tmp_path, broken=False)
    try:
        (env.work / "adder.py").write_bytes(b"return a + b\n")
        preimage = hashlib.sha256(b"return a - b\n").hexdigest()
        args = {
            "path": "adder.py",
            "expected_sha256": preimage,
            "old": "return a - b",
            "new": "return a + b",
        }
        env.store.insert_intent(
            env.tenant,
            env.task_id,
            0,
            idempotency_key("acme", env.task_id, 0, "replace_text", args),
            "replace_text",
            json.dumps(args),
            WORKER,
            env.token,
            NOW,
            (),
        )
        workspace = _Counting(env.work)
        llm = ScriptedLlm(((),))
        loop(env, llm, StatusGate(env, None), FixedChecks(), workspace).run(NOW)
        assert workspace.writes == 0
        assert (env.work / "adder.py").read_bytes() == b"return a + b\n"
        step = env.connection.execute(
            "SELECT status, result_json FROM step_intents WHERE step_index = 0"
        ).fetchone()
        assert step["status"] == "committed"
        assert json.loads(step["result_json"])["status"] == "already_applied"
    finally:
        env.connection.close()


def test_stale_replace_keeps_background_bytes_and_feeds_the_error_to_the_next_step(
    tmp_path: Path,
) -> None:
    """A stale hash leaves the background byte and the next model call sees the new hash."""
    env = lease_task(tmp_path, broken=False)
    try:
        original = b"return a - b\n"
        (env.work / "adder.py").write_bytes(original + b"x")
        stale = hashlib.sha256(original).hexdigest()
        current = hashlib.sha256(original + b"x").hexdigest()
        llm = ScriptedLlm(
            (
                (_replace(stale),),
                (_replace(current),),
                (),
            )
        )
        result = loop(env, llm, StatusGate(env, None), FixedChecks()).run(NOW)
        assert result.status == "running"
        body = (env.work / "adder.py").read_bytes()
        assert body == b"return a + b\nx"
        told = json.dumps(llm.seen[1])
        assert "StaleContent" in told
        assert current in told
    finally:
        env.connection.close()


def test_one_message_with_two_tool_calls_creates_two_steps(tmp_path: Path) -> None:
    """Extra tool calls in one message become the next step before another model post."""
    env = lease_task(tmp_path)
    try:
        digest = hashlib.sha256((env.work / "adder.py").read_bytes()).hexdigest()

        class _Watch(ScriptedLlm):
            def complete(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
                self.posts += 1
                self.seen.append(messages)
                if self.posts == 1:
                    return (
                        call("read_file", {"path": "adder.py"}),
                        call(
                            "replace_text",
                            {
                                "path": "adder.py",
                                "expected_sha256": digest,
                                "old": "return a - b",
                                "new": "return a + b",
                            },
                        ),
                    )
                assert env.store.agent_count(env.tenant, env.task_id) == 2
                return ()

        result = loop(env, _Watch([]), StatusGate(env, None), FixedChecks()).run(NOW)
        assert result.executions == 2
        assert env.store.agent_count(env.tenant, env.task_id) == 2
        assert b"return a + b" in (env.work / "adder.py").read_bytes()
    finally:
        env.connection.close()


def test_finish_does_not_set_completed(tmp_path: Path) -> None:
    """finish records a step and leaves tasks.status unset when the gate does not write it."""
    env = lease_task(tmp_path)
    try:
        gate = StatusGate(env, None)
        result = loop(
            env,
            ScriptedLlm(((call("finish", {}),), ())),
            gate,
            FixedChecks(),
        ).run(NOW)
        assert gate.calls == 1
        assert result.status == "running"
        assert env.store.task_status(env.tenant, env.task_id) == "running"
    finally:
        env.connection.close()


def test_run_checks_is_jailed_and_does_not_set_status(tmp_path: Path) -> None:
    """run_checks executes checks/commands in the jail and does not write completed or rejected."""
    env = lease_task(tmp_path)
    try:
        checks = tmp_path / "checks"
        checks.mkdir()
        code = "print('marker')"
        (checks / "commands").write_text(
            f"{sys.executable} -c {shlex.quote(code)}\n",
            encoding="utf-8",
        )
        llm = ScriptedLlm(
            (
                (call("run_checks", {"argv": "echo pwned"}),),
                (),
            )
        )
        result = loop(
            env,
            llm,
            StatusGate(env, None),
            ExternalVerifier(checks, Jail()),
        ).run(NOW)
        assert result.status == "running"
        text = env.store.list_messages(env.tenant, env.task_id)[-1][2]
        assert "marker" in text
        assert "pwned" not in text
        assert env.store.task_status(env.tenant, env.task_id) == "running"
    finally:
        env.connection.close()


def test_t2_persists_messages(tmp_path: Path) -> None:
    """T2 writes the user, assistant, and tool rows in the tenant file."""
    env = lease_task(tmp_path)
    try:
        loop(
            env,
            ScriptedLlm(((call("read_file", {"path": "adder.py"}),), ())),
            StatusGate(env, None),
            FixedChecks(),
        ).run(NOW)
        roles = [role for _seq, role, _body in env.store.list_messages(env.tenant, env.task_id)]
        assert roles == ["user", "assistant", "tool"]
        cached = env.connection.execute("SELECT COUNT(*) AS n FROM idempotency_cache").fetchone()
        assert int(cached["n"]) == 1
    finally:
        env.connection.close()


def test_loop_stops_at_eight_steps(tmp_path: Path) -> None:
    """The loop does not ask for a ninth step."""
    env = lease_task(tmp_path)
    try:

        class _Reads:
            def __init__(self) -> None:
                self.posts = 0

            def post_count(self) -> int:
                return self.posts

            def complete(self, messages: list[dict[str, object]]) -> tuple[ToolCall, ...]:
                del messages
                self.posts += 1
                return (call("read_file", {"path": "adder.py"}),)

        llm = _Reads()
        result = loop(env, llm, StatusGate(env, None), FixedChecks()).run(NOW)
        assert result.executions == 8
        assert llm.posts == 8
        assert result.status == "running"
        assert env.store.agent_count(env.tenant, env.task_id) == 8
    finally:
        env.connection.close()


def test_plain_paths_stay_in_the_work_root(tmp_path: Path) -> None:
    """PlainWorkspace rejects escape paths and deletes leftover tmp files inside the root."""
    root = tmp_path / "work"
    workspace = PlainWorkspace(root)
    workspace.cleanup_tmp()
    with pytest.raises(ValueError):
        workspace.resolve("../outside")
    with pytest.raises(ValueError):
        workspace.resolve("/tmp/outside")
    workspace.write("adder.py", b"ok")
    assert workspace.read("adder.py") == b"ok"
    (root / "orphan.tmp").write_bytes(b"t")
    (root / "dir.tmp").mkdir()
    workspace.cleanup_tmp()
    assert not (root / "orphan.tmp").exists()
    assert (root / "dir.tmp").is_dir()


def test_empty_model_turn_leaves_the_task_pending(tmp_path: Path) -> None:
    """A model message with no tool call does not insert a step."""
    env = lease_task(tmp_path)
    try:
        result = loop(env, ScriptedLlm(((),)), StatusGate(env, None), FixedChecks()).run(NOW)
        assert result.executions == 0
        assert result.status == "pending"
        assert env.store.agent_count(env.tenant, env.task_id) == 0
    finally:
        env.connection.close()


def test_unknown_tool_is_committed_as_an_error(tmp_path: Path) -> None:
    """A tool name the loop does not know becomes a committed error and the task stays running."""
    env = lease_task(tmp_path)
    try:
        args = {}
        env.store.insert_intent(
            env.tenant,
            env.task_id,
            0,
            idempotency_key("acme", env.task_id, 0, "nope", args),
            "nope",
            json.dumps(args),
            WORKER,
            env.token,
            NOW,
            (),
        )
        loop(env, ScriptedLlm(((),)), StatusGate(env, None), FixedChecks()).run(NOW)
        row = env.connection.execute("SELECT status, result_json FROM step_intents").fetchone()
        assert row["status"] == "committed"
        assert json.loads(row["result_json"])["error"] == "unknown tool"
    finally:
        env.connection.close()


def test_context_pack_shrinks_searches_and_redacts(tmp_path: Path) -> None:
    """Step 4 still executes the context packer, search, and remember."""
    env = lease_task(tmp_path, "context-demo", broken=False)
    try:
        work = env.work
        (work / "adder.py").write_text("return a - b\n" * 20, encoding="utf-8")
        (work / "scratch.tmp").write_text("skip\n", encoding="utf-8")
        (work / "linked").symlink_to(work / "adder.py")
        workspace = PlainWorkspace(work)
        capped = json.loads(search_text(workspace, "return", "adder.py"))
        assert len(capped["hits"]) == 20
        assert search_text(workspace, "return", "missing-dir") == '{"hits": []}'
        listed = json.loads(list_files(workspace))
        assert "adder.py" in listed["paths"]
        assert "scratch.tmp" not in listed["paths"]
        for escaped in ("../secret.txt", "/etc/passwd", ""):
            with pytest.raises(ValueError):
                search_text(workspace, "return", escaped)
        secret = "super-secret-token"
        note = remember_note(workspace, "adder.py", secret, f"note {secret}", 0, secret)
        assert note[1] == "[redacted]"
        assert outcome_line(
            {"tool": "replace_text", "result": {"error": "StaleContent", "sha256": "abc"}}
        ) == "replace_text StaleContent abc"
        assert outcome_line(
            {"tool": "run_checks", "result": {"ok": False, "sha256": "abc"}}
        ) == "run_checks exit 1 abc"
        assert outcome_line({"tool": "read_file", "sha256": "abc"}) == "read_file ok abc"
        store = TrajectoryStore(env.connection, WORKER, env.token, lambda: NOW)
        env.connection.execute("BEGIN IMMEDIATE")
        env.connection.execute(
            """
            INSERT INTO trajectory (tenant_id, task_id, seq, step_index, kind, body_json)
            VALUES (?, ?, 0, 0, 'tool_result', ?)
            """,
            (
                env.tenant.value,
                env.task_id,
                json.dumps({"tool": "search_text", "result": {"text": "HUGE-TOOL-BODY" * 80}}),
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
        shrunk = ContextBuilder(store, budget=200).pack(env.tenant, env.task_id)
        assert "search_text ok" in json.dumps(shrunk)
        assert "HUGE-TOOL-BODY" not in json.dumps(shrunk)
        seen: list[str] = []

        def summarize(messages: list[dict[str, object]]) -> str:
            seen.append(json.dumps(messages))
            return "compressed the middle"

        packed = ContextBuilder(store, budget=40, summarize=summarize, secret=secret).pack(
            env.tenant, env.task_id
        )
        assert seen
        assert "HUGE-TOOL-BODY" not in seen[0]
        assert "compressed the middle" in json.dumps(packed)
        history = [
            {"role": "user", "content": "short"},
            {"role": "user", "content": "x" * 200},
            {"role": "user", "content": "StaleContent " + ("y" * 200)},
        ]
        plain = ContextBuilder(budget=1).pack(env.tenant, env.task_id, history)
        assert "tool ok" in json.dumps(plain)
        assert "tool StaleContent" in json.dumps(plain)
        llm = ScriptedLlm(
            [
                (call("list_files", {"path": "."}),),
                (call("search_text", {"query": "return", "path": ".."}),),
                (
                    call(
                        "remember",
                        {"path": "adder.py", "quote": secret, "note": secret},
                    ),
                ),
                (),
            ]
        )
        result = loop(
            env,
            llm,
            StatusGate(env, None),
            FixedChecks(),
            context=ContextBuilder(store, secret=secret),
            secret=secret,
        ).run(NOW)
        assert result.status == "running"
        assert secret not in "\n".join(
            str(row[0])
            for row in env.connection.execute(
                "SELECT body FROM knowledge WHERE task_id = ?",
                (env.task_id,),
            ).fetchall()
        )
    finally:
        env.connection.close()


@pytest.mark.skipif(
    not (
        __import__("os").environ.get("AGENT_LLM_API_KEY")
        and __import__("os").environ.get("AGENT_LLM_BASE_URL")
        and __import__("os").environ.get("AGENT_LLM_MODEL")
    ),
    reason="needs AGENT_LLM_API_KEY, AGENT_LLM_BASE_URL, and AGENT_LLM_MODEL",
)
def test_react_happy_live(tmp_path: Path) -> None:
    """The live adapter repairs the broken adder and the gate writes completed."""
    from agent_engine.composition import run_task
    from agent_engine.domain.models import TenantId
    from agent_engine.self_test import _prepare_adder

    _prepare_adder(tmp_path, "acme", "fix-adder-live")
    report = run_task(tmp_path, TenantId("acme"), "fix-adder-live", llm="live")
    assert report.status == "completed"


class _Guard(PlainWorkspace):
    """Fail the write after checking that the intent row is already committed."""

    def __init__(self, root: Path, probe, task_id: str) -> None:
        super().__init__(root)
        self._probe = probe
        self._task_id = task_id
        self.seen = False

    def write(self, relative_path: str, data: bytes) -> None:
        row = self._probe.execute(
            "SELECT status FROM step_intents WHERE task_id = ?",
            (self._task_id,),
        ).fetchone()
        assert row is not None and row["status"] == "intent_logged"
        self.seen = True
        raise RuntimeError("stop")


class _Counting(PlainWorkspace):
    """Count write calls and forward the others."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.writes = 0

    def write(self, relative_path: str, data: bytes) -> None:
        self.writes += 1
        super().write(relative_path, data)


def _replace(digest: str) -> ToolCall:
    return call(
        "replace_text",
        {
            "path": "adder.py",
            "expected_sha256": digest,
            "old": "return a - b",
            "new": "return a + b",
        },
    )
