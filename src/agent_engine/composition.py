"""Wiring root for one run-task process.

This module opens the tenant database, the copy-on-write workspace, the jail,
the verifier, and the LLM. application/ does not import it. A live command
reads the tenant's llm_api_key_env and fails at startup when that variable,
the base URL, or the model name is missing. --llm canned uses a scripted
transport and does not read the secret. Model-backed leases last 180 seconds.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

from agent_engine.adapters.exec.jail import Jail
from agent_engine.adapters.mcp.client import McpClient
from agent_engine.adapters.llm.openai_chat import (
    CannedTransport,
    OpenAIChatLlm,
    UrllibTransport,
    completion_body,
)
from agent_engine.adapters.sqlite.db import open_tenant_db, resolve_tenant_dir
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.step_store import SqliteStepStore
from agent_engine.adapters.sqlite.tenant_store import get_task, read_tenant
from agent_engine.adapters.verifier.external import ExternalVerifier
from agent_engine.adapters.workspace.cow import CowWorkspace
from agent_engine.adapters.workspace.edits import OptimisticEditor
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.application.react import ReactLoop, RunResult
from agent_engine.application.verify_gate import VerifyGate
from agent_engine.domain.errors import FenceRejected
from agent_engine.domain.models import TenantId

_TERMINAL = ("completed", "rejected")
_TTL = 180.0


def format_report(report: RunResult) -> str:
    """Render the loop result for the CLI.

    Args:
        report: Status, cache hit, executions, and posts from one run.

    Returns:
        Status, and on a cache hit the execution and post counts.
    """
    lines = [report.status]
    if report.cache_hit:
        lines.append("cache hit")
        lines.append(f"executions={report.executions}")
        lines.append(f"posts={report.posts}")
    if report.detail:
        lines.append(report.detail)
    return "\n".join(lines)


def run_task(
    data_root: Path,
    tenant_id: TenantId,
    task_id: str,
    *,
    llm: str = "live",
    tool_choice: str | None = None,
) -> RunResult:
    """Acquire a lease, run the loop, and release the lease.

    Args:
        data_root: Directory that contains tenants/ and checks/.
        tenant_id: Tenant file to open. No other tenant file is attached.
        task_id: Task whose work root is tasks/{task_id}.
        llm: canned uses the scripted transport. Any other value is the live adapter.
        tool_choice: When llm is canned and this is finish, the scripted model
            calls finish immediately.

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
    if (task_dir / "base").is_dir() and not (workspace.work_root / "adder.py").exists():
        workspace.seed()
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
            model = _llm(data_root, tenant_id, work, llm, tool_choice, mcp.chat_tools())
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
            )
            return loop.run(now, clock=SystemClock())
        finally:
            try:
                service.release(lease, time.time())
            except FenceRejected:
                pass
    finally:
        connection.close()


def _llm(
    data_root: Path,
    tenant_id: TenantId,
    work: Path,
    llm: str,
    tool_choice: str | None,
    extra_tools: list[dict[str, object]] | None = None,
) -> OpenAIChatLlm:
    if llm == "canned":
        return OpenAIChatLlm(
            "http://canned.invalid",
            "canned-token",
            "canned",
            CannedTransport(_canned_bodies(work, tool_choice)),
            extra_tools=extra_tools,
        )
    tenant = read_tenant(data_root, tenant_id)
    api_key, base_url, model = _live_credentials(tenant.llm_api_key_env)
    return OpenAIChatLlm(base_url, api_key, model, UrllibTransport(), extra_tools=extra_tools)


def _live_credentials(env_name: str) -> tuple[str, str, str]:
    names = (env_name, "AGENT_LLM_BASE_URL", "AGENT_LLM_MODEL")
    for name in names:
        if not os.environ.get(name):
            raise ValueError(f"missing {name}")
    return os.environ[env_name], os.environ["AGENT_LLM_BASE_URL"], os.environ["AGENT_LLM_MODEL"]


def _canned_bodies(work: Path, tool_choice: str | None) -> list[dict[str, object]]:
    if tool_choice == "finish":
        return [completion_body([("finish", "{}")])]
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
