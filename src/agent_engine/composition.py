"""Wiring root for one run-task, plan, accept, or apply process.

This module opens the tenant database, the copy-on-write workspace, the jail,
the verifier, and the LLM. application/ does not import it. A live command
reads the tenant's llm_api_key_env and fails at startup when that variable,
the base URL, or the model name is missing. --llm canned uses a scripted
transport and does not read the secret. Model-backed leases last 180 seconds.
propose_plan takes no lease and writes no file. accept_plan calls no model.
apply_plan refuses a plan that is not accepted, writes the operator check
commands, then runs the loop with that plan pinned in the pack and without
list_files, search_text, or remember. start_project chains those calls for
one prompt and asks the caller's confirm before the accept, so a declined
plan still writes nothing. format_report prints a readable pack, not one
JSON line.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_engine.adapters.exec.jail import Jail
from agent_engine.adapters.mcp.client import McpClient
from agent_engine.adapters.sqlite.trajectory_store import TrajectoryStore
from agent_engine.application.context import ContextBuilder
from agent_engine.adapters.llm.openai_chat import (
    CannedTransport,
    OpenAIChatLlm,
    UrllibTransport,
    completion_body,
)
from agent_engine.adapters.sqlite.db import open_tenant_db, resolve_tenant_dir
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.plan_store import SqlitePlanStore
from agent_engine.adapters.sqlite.step_store import SqliteStepStore
from agent_engine.adapters.sqlite.tenant_store import (
    get_task,
    init_task,
    provision,
    read_tenant,
)
from agent_engine.adapters.verifier.external import ExternalVerifier
from agent_engine.adapters.workspace.cow import CowWorkspace
from agent_engine.adapters.workspace.edits import OptimisticEditor
from agent_engine.adapters.workspace.plain import PlainWorkspace
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.application.plan import (
    PlanSession,
    apply_prompt,
    load_plan,
    pinned_plan,
    plan_body,
    render_plan,
    task_slug,
)
from agent_engine.application.react import ReactLoop, RunResult
from agent_engine.application.verify_gate import VerifyGate
from agent_engine.domain.errors import FenceRejected, TaskNotFound
from agent_engine.domain.models import TenantId

_TERMINAL = ("completed", "rejected")
_TTL = 180.0
_APP_PATH = "services/api/app.py"
_TEST_PATH = "services/api/tests/test_app.py"
_CANNED_PLAN = json.dumps(
    {
        "summary": "Create the api package and its test.",
        "approach": (
            "Create services/api/app.py with write_file, then the pytest module "
            "with write_file, then run the operator checks and finish."
        ),
        "steps": [
            {
                "title": "Create the api module",
                "detail": (
                    "Call write_file on services/api/app.py. Define add(a, b) "
                    "and return a + b. Parent directories are created with the file."
                ),
            },
            {
                "title": "Create the test module",
                "detail": (
                    "Call write_file on services/api/tests/test_app.py. Import add "
                    "from services.api.app and assert add(1, 2) == 3."
                ),
            },
            {
                "title": "Run checks and finish",
                "detail": (
                    "Call run_checks. The operator command list is the gate. "
                    "Then call finish so the verifier can mark the task."
                ),
            },
        ],
        "files": [
            {"path": _APP_PATH, "action": "create", "detail": "add(a, b) returns a + b"},
            {"path": _TEST_PATH, "action": "create", "detail": "test add(1, 2) == 3"},
        ],
    }
)
_CANNED_APP = "def add(a, b):\n    return a + b\n"
_CANNED_TEST = (
    "from services.api.app import add\n"
    "\n"
    "\n"
    "def test_add() -> None:\n"
    "    assert add(1, 2) == 3\n"
)


@dataclass(frozen=True)
class ProjectResult:
    """What one start_project call stored, built, and left on disk."""

    task_id: str
    tree: str
    accepted: bool
    work_root: Path
    report: RunResult | None


def format_report(report: RunResult) -> str:
    """Render the loop result for the CLI.

    Args:
        report: Status, cache hit, executions, and posts from one run.

    Returns:
        Status, and on a cache hit the execution and post counts. A packed
        prompt is one block per message, not a single JSON line.
    """
    lines = [report.status]
    if report.cache_hit:
        lines.append("cache hit")
        lines.append(f"executions={report.executions}")
        lines.append(f"posts={report.posts}")
    if report.detail:
        lines.append(_readable_detail(report.detail))
    return "\n".join(lines)


def run_task(
    data_root: Path,
    tenant_id: TenantId,
    task_id: str,
    *,
    llm: str = "live",
    tool_choice: str | None = None,
    context_token_budget: int | None = None,
    prompt: str | None = None,
    plan_text: str = "",
    max_steps: int = 8,
    trace: Callable[[str], None] | None = None,
) -> RunResult:
    """Acquire a lease, run the loop, and release the lease.

    Args:
        data_root: Directory that contains tenants/ and checks/.
        tenant_id: Tenant file to open. No other tenant file is attached.
        task_id: Task whose work root is tasks/{task_id}.
        llm: canned uses the scripted transport. Any other value is the live adapter.
        tool_choice: When llm is canned and this is finish, the scripted model
            calls finish immediately. context runs search, remember, and read_file.
            tree writes a nested package with write_file.
        context_token_budget: Pack budget. None reads AGENT_CONTEXT_TOKEN_BUDGET,
            then 8000.
        prompt: First user message and the packed task sentence. None keeps the
            adder sentence run-task and the demos use.
        plan_text: Accepted plan packed with the system rules, whole. A
            non-empty plan drops list_files, search_text, and remember.
        max_steps: Agent steps one run may reach.
        trace: Receives one log line per model turn and tool result. None
            stays quiet.

    Returns:
        The loop result. A task that is already terminal is a cache hit and
        does not post.

    Raises:
        UnknownTenant: The tenant database does not exist.
        TaskNotFound: The task row does not exist.
        ValueError: A live run is missing the API key variable, the base URL,
            or the model name.
        LeaseNotHeld: Another worker holds an unexpired lease.
    """
    task = get_task(data_root, tenant_id, task_id)
    task_dir = resolve_tenant_dir(data_root, tenant_id) / task.workspace_root
    workspace = CowWorkspace(task_dir)
    if (task_dir / "base").is_dir():
        if not (workspace.work_root / "adder.py").exists():
            workspace.seed()
    else:
        workspace.work_root.mkdir(parents=True, exist_ok=True)
    work = workspace.work_root
    if task.status in _TERMINAL:
        workspace.cleanup_tmp()
        return RunResult(task.status, True, 0, 0)
    connection = open_tenant_db(data_root, tenant_id)
    try:
        now = time.time()
        worker_id = str(uuid.uuid4())
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        lease = service.acquire(task_id, worker_id, now, _TTL)
        try:
            mcp = McpClient(data_root, Jail(), work)
            model = _llm(
                data_root,
                tenant_id,
                work,
                llm,
                tool_choice,
                mcp.chat_tools(),
                follow_plan=bool(plan_text),
            )
            budget = (
                int(os.environ.get("AGENT_CONTEXT_TOKEN_BUDGET", "8000"))
                if context_token_budget is None
                else context_token_budget
            )
            log = TrajectoryStore(connection, worker_id, lease.fencing_token, lambda: time.time())
            builder = ContextBuilder(
                log,
                budget=budget,
                summarize=model.summarize,
                secret=os.environ.get("AGENT_LLM_API_KEY", ""),
                task=prompt,
                plan=plan_text,
            )
            store = SqliteStepStore(connection)
            checks = ExternalVerifier(data_root / "checks", Jail())
            loop = ReactLoop(
                tenant_id,
                task_id,
                worker_id,
                lease.fencing_token,
                work,
                store,
                model,
                workspace,
                OptimisticEditor(),
                checks,
                VerifyGate(
                    tenant_id,
                    task_id,
                    worker_id,
                    lease.fencing_token,
                    work,
                    checks,
                    store,
                ),
                mcp,
                builder,
                os.environ.get("AGENT_LLM_API_KEY", ""),
                prompt,
                max_steps,
                trace,
            )
            result = loop.run(now, clock=SystemClock())
            if result.detail or not builder.last_messages:
                return result
            return RunResult(
                result.status,
                result.cache_hit,
                result.executions,
                result.posts,
                json.dumps(builder.last_messages),
            )
        finally:
            try:
                service.release(lease, time.time())
            except FenceRejected:
                pass
    finally:
        connection.close()


def propose_plan(
    data_root: Path,
    tenant_id: TenantId,
    task_id: str,
    prompt: str,
    *,
    llm: str = "live",
) -> str:
    """Store one proposed plan for a prompt and return its tree.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant file to open.
        task_id: Task the plan belongs to.
        prompt: User sentence the model plans for.
        llm: canned returns the fixed plan. Any other value is the live adapter.

    Returns:
        The execution plan: approach, steps, and the file tree. No lease is
        taken and no file is written. An existing tree may be read.

    Raises:
        UnknownTenant: The tenant database does not exist.
        TaskNotFound: The task row does not exist.
        ValueError: The task is terminal, the stored plan is accepted, a live
            run is missing its credentials, or the model did not submit a
            valid plan.
    """
    task = get_task(data_root, tenant_id, task_id)
    if task.status in _TERMINAL:
        raise ValueError(f"{task_id} is {task.status}")
    codebase = _existing_code(data_root, tenant_id, task.workspace_root)
    model = _plan_llm(data_root, tenant_id, llm, explore=codebase is not None)
    plan = PlanSession(model, codebase=codebase).propose(prompt)
    connection = open_tenant_db(data_root, tenant_id)
    try:
        SqlitePlanStore(connection).propose(tenant_id, task_id, prompt, plan_body(plan))
    finally:
        connection.close()
    return render_plan(plan)


def accept_plan(data_root: Path, tenant_id: TenantId, task_id: str) -> None:
    """Mark the stored plan accepted.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant file to open.
        task_id: Task whose plan is accepted.

    Returns:
        None. No model call and no file write.

    Raises:
        UnknownTenant: The tenant database does not exist.
        ValueError: The task has no plan row.
    """
    connection = open_tenant_db(data_root, tenant_id)
    try:
        SqlitePlanStore(connection).accept(tenant_id, task_id)
    finally:
        connection.close()


def show_plan(data_root: Path, tenant_id: TenantId, task_id: str) -> str:
    """Render the stored prompt, status, and tree.

    Args:
        data_root: Directory that contains tenants/.
        tenant_id: Tenant file to open.
        task_id: Task whose plan is printed.

    Returns:
        The prompt line, the status line, and the tree.

    Raises:
        UnknownTenant: The tenant database does not exist.
        ValueError: The task has no plan row, or the stored body is not a
            valid plan.
    """
    prompt, body, status = _read_plan(data_root, tenant_id, task_id)
    plan = load_plan(body)
    return "\n".join([f"prompt={prompt}", f"status={status}", render_plan(plan)])


def apply_plan(
    data_root: Path,
    tenant_id: TenantId,
    task_id: str,
    *,
    llm: str = "live",
    max_steps: int = 32,
    checks_cmd: tuple[str, ...] = (),
    trace: Callable[[str], None] | None = None,
) -> RunResult:
    """Run the loop with the accepted plan pinned in the pack.

    Args:
        data_root: Directory that contains tenants/ and checks/.
        tenant_id: Tenant file to open.
        task_id: Task whose accepted plan is applied.
        llm: canned writes the fixed tree. Any other value is the live adapter.
        max_steps: Agent steps one run may reach. A tree that hits this cap
            stays running and a later apply resumes it.
        checks_cmd: Operator commands written to {data_root}/checks/commands
            before the loop. The model does not choose them and cannot write
            that directory.
        trace: Receives one log line per model turn and tool result. None
            stays quiet.

    Returns:
        The loop result.

    Raises:
        UnknownTenant: The tenant database does not exist.
        TaskNotFound: The task row does not exist.
        ValueError: The plan is missing or still proposed, or a live run is
            missing its credentials.
        LeaseNotHeld: Another worker holds an unexpired lease.
    """
    prompt, body, status = _read_plan(data_root, tenant_id, task_id)
    if status != "accepted":
        raise ValueError(f"plan for {task_id} is {status}, not accepted")
    plan = load_plan(body)
    if checks_cmd:
        checks = data_root / "checks"
        checks.mkdir(parents=True, exist_ok=True)
        (checks / "commands").write_text(
            "".join(f"{line}\n" for line in checks_cmd), encoding="utf-8"
        )
    return run_task(
        data_root,
        tenant_id,
        task_id,
        llm=llm,
        tool_choice="tree" if llm == "canned" else None,
        prompt=apply_prompt(prompt),
        plan_text=pinned_plan(plan),
        max_steps=max_steps,
        trace=trace,
    )


def start_project(
    data_root: Path,
    tenant_id: TenantId,
    prompt: str,
    *,
    task_id: str = "",
    display_name: str = "",
    llm: str = "live",
    max_steps: int = 32,
    checks_cmd: tuple[str, ...] = (),
    confirm: Callable[[str], bool] | None = None,
    trace: Callable[[str], None] | None = None,
) -> ProjectResult:
    """Provision, plan, confirm, accept, and apply in one call.

    Args:
        data_root: Directory that contains tenants/ and checks/. It is created
            when it does not exist.
        tenant_id: Tenant to provision or reuse.
        prompt: User sentence the model plans for.
        task_id: Task to create. An empty value names the task after the
            prompt and adds a numeric suffix when that id is taken.
        display_name: Tenant display name. Empty uses the tenant id.
        llm: canned returns the fixed plan and tree. Any other value is live.
        max_steps: Agent steps one apply may reach.
        checks_cmd: Operator commands written before the loop. The model does
            not choose them and cannot write that directory.
        confirm: Called with the tree before accept. None accepts without
            asking. A false answer leaves the plan proposed and writes no file.
        trace: Receives one log line per apply turn. None stays quiet.

    Returns:
        The task id, the tree, whether the plan was accepted, the work root,
        and the loop result. The result is None when the plan was not accepted.

    Raises:
        TenantMismatch: The resolved path leaves the data root.
        ValueError: The prompt has no word for a task id, a live run is
            missing its credentials, or the model did not submit a valid plan.
        LeaseNotHeld: Another worker holds an unexpired lease.
    """
    provision(data_root, tenant_id, display_name or tenant_id.value)
    chosen = task_id or _free_task(data_root, tenant_id, task_slug(prompt))
    init_task(data_root, tenant_id, chosen)
    tree = propose_plan(data_root, tenant_id, chosen, prompt, llm=llm)
    work_root = resolve_tenant_dir(data_root, tenant_id) / "tasks" / chosen / "work"
    if confirm is not None and not confirm(tree):
        return ProjectResult(chosen, tree, False, work_root, None)
    accept_plan(data_root, tenant_id, chosen)
    report = apply_plan(
        data_root,
        tenant_id,
        chosen,
        llm=llm,
        max_steps=max_steps,
        checks_cmd=checks_cmd,
        trace=trace,
    )
    return ProjectResult(chosen, tree, True, work_root, report)


def _free_task(data_root: Path, tenant_id: TenantId, base: str) -> str:
    candidate = base
    suffix = 2
    while True:
        try:
            get_task(data_root, tenant_id, candidate)
        except TaskNotFound:
            return candidate
        candidate = f"{base}-{suffix}"
        suffix += 1


def _read_plan(data_root: Path, tenant_id: TenantId, task_id: str) -> tuple[str, str, str]:
    connection = open_tenant_db(data_root, tenant_id)
    try:
        stored = SqlitePlanStore(connection).read(tenant_id, task_id)
    finally:
        connection.close()
    if stored is None:
        raise ValueError(f"{task_id} has no plan")
    return stored


def _existing_code(data_root: Path, tenant_id: TenantId, workspace_root: str) -> PlainWorkspace | None:
    task_dir = resolve_tenant_dir(data_root, tenant_id) / workspace_root
    for name in ("work", "base"):
        root = task_dir / name
        if _has_source(root):
            return PlainWorkspace(root)
    return None


def _has_source(root: Path) -> bool:
    if not root.is_dir():
        return False
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink() and path.suffix != ".tmp":
            return True
    return False


def _plan_llm(
    data_root: Path, tenant_id: TenantId, llm: str, *, explore: bool = False
) -> OpenAIChatLlm:
    if llm == "canned":
        return OpenAIChatLlm(
            "http://canned.invalid",
            "canned-token",
            "canned",
            CannedTransport([completion_body([("submit_plan", _CANNED_PLAN)])]),
            explore=explore,
        )
    tenant = read_tenant(data_root, tenant_id)
    api_key, base_url, model = _live_credentials(tenant.llm_api_key_env)
    return OpenAIChatLlm(base_url, api_key, model, UrllibTransport(), explore=explore)


def _llm(
    data_root: Path,
    tenant_id: TenantId,
    work: Path,
    llm: str,
    tool_choice: str | None,
    extra_tools: list[dict[str, object]] | None = None,
    *,
    follow_plan: bool = False,
) -> OpenAIChatLlm:
    if llm == "canned":
        return OpenAIChatLlm(
            "http://canned.invalid",
            "canned-token",
            "canned",
            CannedTransport(_canned_bodies(work, tool_choice)),
            extra_tools=extra_tools,
            follow_plan=follow_plan,
        )
    tenant = read_tenant(data_root, tenant_id)
    api_key, base_url, model = _live_credentials(tenant.llm_api_key_env)
    return OpenAIChatLlm(
        base_url,
        api_key,
        model,
        UrllibTransport(),
        extra_tools=extra_tools,
        follow_plan=follow_plan,
    )


def _live_credentials(env_name: str) -> tuple[str, str, str]:
    names = (env_name, "AGENT_LLM_BASE_URL", "AGENT_LLM_MODEL")
    for name in names:
        if not os.environ.get(name):
            raise ValueError(f"missing {name}")
    return os.environ[env_name], os.environ["AGENT_LLM_BASE_URL"], os.environ["AGENT_LLM_MODEL"]


def _canned_bodies(work: Path, tool_choice: str | None) -> list[dict[str, object]]:
    if tool_choice == "finish":
        return [completion_body([("finish", "{}")])]
    if tool_choice == "tree":
        return [
            completion_body(
                [("write_file", json.dumps({"path": _APP_PATH, "content": _CANNED_APP}))]
            ),
            completion_body(
                [("write_file", json.dumps({"path": _TEST_PATH, "content": _CANNED_TEST}))]
            ),
            completion_body([]),
        ]
    if tool_choice == "context":
        return [
            completion_body(
                [("search_text", json.dumps({"path": ".", "query": "marker-from-helper"}))]
            ),
            completion_body(
                [
                    (
                        "remember",
                        json.dumps(
                            {
                                "path": "helper.py",
                                "quote": "marker-from-helper",
                                "note": "remembered-helper",
                            }
                        ),
                    )
                ]
            ),
            completion_body([("read_file", json.dumps({"path": "helper.py"}))]),
            completion_body([]),
        ]
    if tool_choice == "budget":
        return [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "compressed the middle",
                        }
                    }
                ]
            },
            completion_body([]),
        ]
    if tool_choice and tool_choice.startswith("mcp__"):
        return [
            completion_body([(tool_choice, json.dumps({"text": "ping"}))]),
            completion_body([]),
        ]
    digest = hashlib.sha256((work / "adder.py").read_bytes()).hexdigest()
    replace = json.dumps(
        {
            "path": "adder.py",
            "expected_sha256": digest,
            "old": "return a - b",
            "new": "return a + b",
        }
    )
    return [
        completion_body([("read_file", json.dumps({"path": "adder.py"}))]),
        completion_body([("replace_text", replace)]),
        completion_body([("finish", "{}")]),
    ]


def _readable_detail(detail: str) -> str:
    try:
        parsed: object = json.loads(detail)
    except json.JSONDecodeError:
        return detail
    if not isinstance(parsed, list) or not parsed:
        return detail
    first = parsed[0]
    if not isinstance(first, dict) or "role" not in first:
        return detail
    blocks: list[str] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "message"))
        content = item.get("content", "")
        text = content if isinstance(content, str) else json.dumps(content)
        body = _pretty_step(text)
        if body:
            blocks.append(f"{role}\n{body}")
        else:
            blocks.append(role)
    return "\n\n".join(blocks)


def _pretty_step(text: str) -> str:
    parts: list[str] = []
    for line in text.splitlines():
        try:
            obj: object = json.loads(line)
        except json.JSONDecodeError:
            parts.append(line)
            continue
        shown = _pretty_obj(obj)
        if shown:
            parts.append(shown)
    return "\n".join(parts)


def _pretty_obj(obj: object) -> str:
    if not isinstance(obj, dict):
        return json.dumps(obj, indent=2)
    if obj.get("text") == "" and "tool" in obj:
        return ""
    if "arguments" in obj and "tool" in obj:
        args = obj["arguments"]
        if isinstance(args, dict):
            bits = " ".join(
                f"{key}={value}" for key, value in args.items() if key != "content"
            )
            return f"{obj['tool']} {bits}".rstrip()
    result = obj.get("result")
    if "tool" in obj and isinstance(result, dict) and isinstance(result.get("paths"), list):
        paths = [str(path) for path in result["paths"]]
        return f"{obj['tool']} {len(paths)} paths\n" + "\n".join(paths)
    return json.dumps(obj, indent=2)
