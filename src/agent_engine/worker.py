"""One worker process for a leased task.

The process opens its own tenant database. Heartbeats use a second connection
and renew every ttl/3. That connection is not the loop connection and it is
not inherited from the parent. When a renew matches zero rows, the HTTP socket
is closed and T2 does not start. AGENT_CRASH_AFTER skips later renews and
raises SIGKILL after that tool's side effect, before T2. AGENT_MUTATE_AFTER
appends one byte to the file that was just read, before the next tool.
completed is still only written by VerifyGate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass
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
from agent_engine.application.react import ReactLoop
from agent_engine.application.verify_gate import VerifyGate
from agent_engine.composition import _live_credentials
from agent_engine.domain.errors import FenceRejected
from agent_engine.domain.models import Lease, TenantId
from agent_engine.domain.ports import StepLog

_TERMINAL = ("completed", "rejected")


@dataclass
class WorkerConfig:
    """Arguments and hooks for one worker process."""

    data_root: Path
    tenant_id: str
    task_id: str
    llm: str
    ttl: float
    tool_choice: str | None = None
    canned: str = "repair"
    crash_after: str | None = None
    crash_ttl: float = 2.0
    mutate_after: str | None = None
    hold_sec: float | None = None
    ready_file: Path | None = None
    release_file: Path | None = None
    resume_only: bool = False
    force_finish_after_stale: bool = False
    heartbeat_interval: float | None = None


@dataclass
class WorkerReport:
    """What the worker prints when the loop returns."""

    status: str
    writes: int
    posts: int
    token: int
    cache_hit: bool
    executions: int
    fence_rejected: bool = False


class _WorkFiles:
    """Work-tree wrapper that counts writes and applies crash, hold, and mutate hooks."""

    def __init__(self, inner: CowWorkspace, worker: Worker) -> None:
        self._inner = inner
        self._worker = worker
        self._held = False

    def resolve(self, relative_path: str) -> Path:
        """Return a path inside work.

        Args:
            relative_path: Path relative to work.

        Returns:
            The resolved path.
        """
        return self._inner.resolve(relative_path)

    def read(self, relative_path: str) -> bytes:
        """Read one file, holding first when this worker is the suspend scenario.

        Args:
            relative_path: Path relative to work.

        Returns:
            The file bytes.
        """
        self._hold_once()
        return self._inner.read(relative_path)

    def write(self, relative_path: str, data: bytes) -> None:
        """Write a new inode, then honor AGENT_CRASH_AFTER.

        Args:
            relative_path: Path relative to work.
            data: Bytes for the new inode.

        Returns:
            None.
        """
        self._inner.write(relative_path, data)
        self._worker.writes += 1
        if self._worker.config.crash_after == "replace_text" and not self._worker.mutating:
            self._worker.crash()

    def cleanup_tmp(self) -> None:
        """Delete leftover temp files inside work.

        Returns:
            None.
        """
        self._inner.cleanup_tmp()

    def remove_file(self, relative_path: str) -> None:
        """Remove a work file.

        Args:
            relative_path: Path relative to work.

        Returns:
            None.
        """
        self._inner.remove_file(relative_path)

    def _hold_once(self) -> None:
        config = self._worker.config
        if self._held or config.hold_sec is None:
            return
        self._held = True
        if config.ready_file is not None:
            config.ready_file.write_text("ready", encoding="utf-8")
        if config.release_file is None:
            time.sleep(config.hold_sec)
            return
        deadline = time.time() + config.hold_sec
        while time.time() < deadline:
            if config.release_file.exists():
                return
            time.sleep(0.05)


class _GuardedStore:
    """Step log that refuses to open T2 after the heartbeat loses the fence."""

    def __init__(self, inner: StepLog, worker: Worker) -> None:
        self._inner = inner
        self._worker = worker

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def commit_step(
        self,
        tenant_id: TenantId,
        task_id: str,
        step_index: int,
        idempotency_key: str,
        result_json: str,
        worker_id: str,
        token: int,
        now: float,
        messages: tuple[tuple[str, str], ...],
    ) -> None:
        """Commit T2 unless the heartbeat already lost the lease.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.
            step_index: Intent row to commit.
            idempotency_key: Cache key.
            result_json: Tool result.
            worker_id: Lease owner.
            token: Fencing token.
            now: Instant compared with the lease expiry.
            messages: Transcript rows for this commit.

        Returns:
            None.

        Raises:
            FenceRejected: The heartbeat matched zero rows. T2 does not start.
        """
        if self._worker.lost.is_set():
            self._worker.close_http()
            raise FenceRejected(task_id)
        self._inner.commit_step(
            tenant_id,
            task_id,
            step_index,
            idempotency_key,
            result_json,
            worker_id,
            token,
            now,
            messages,
        )
        self._worker.after_commit(step_index, result_json)

    def set_terminal(
        self,
        tenant_id: TenantId,
        task_id: str,
        status: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Write a terminal status unless the heartbeat already lost the lease.

        Args:
            tenant_id: Tenant bound to this store.
            task_id: Task primary key.
            status: completed or rejected.
            worker_id: Lease owner.
            token: Fencing token.
            now: Instant compared with the lease expiry.

        Returns:
            None.

        Raises:
            FenceRejected: The heartbeat matched zero rows.
        """
        if self._worker.lost.is_set():
            self._worker.close_http()
            raise FenceRejected(task_id)
        self._inner.set_terminal(tenant_id, task_id, status, worker_id, token, now)


class _GuardedLlm:
    """Model port that does not POST after the lease is lost."""

    def __init__(self, inner: OpenAIChatLlm, worker: Worker) -> None:
        self._inner = inner
        self._worker = worker

    def post_count(self) -> int:
        """Return how many completions were requested.

        Returns:
            The post count on the inner client.
        """
        return self._inner.post_count()

    def complete(self, messages: list[dict[str, object]]) -> tuple[object, ...]:
        """POST one turn, or refuse when the heartbeat lost the fence.

        Args:
            messages: Transcript in Chat Completions shape.

        Returns:
            Tool calls from the inner client.

        Raises:
            FenceRejected: The lease was lost. The socket is closed first.
        """
        if self._worker.lost.is_set():
            self._worker.close_http()
            raise FenceRejected(self._worker.config.task_id)
        if self._worker.config.force_finish_after_stale and _saw_stale(messages):
            self._inner.tool_choice = "finish"
        return self._inner.complete(messages)

    def close(self) -> None:
        """Close the inner socket.

        Returns:
            None.
        """
        self._inner.close()


class Worker:
    """Lease, heartbeat, and ReAct loop for one subprocess."""

    def __init__(self, config: WorkerConfig) -> None:
        """Bind the hooks. The database is opened in serve.

        Args:
            config: Data root, task, ttl, and the crash and mutate hooks.
        """
        self.config = config
        self.writes = 0
        self.mutating = False
        self.lost = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._crashed = False
        self._lease: Lease | None = None
        self._service: LeaseService | None = None
        self._llm: _GuardedLlm | None = None
        self._workspace: _WorkFiles | None = None
        self._connection_for_steps: object | None = None

    def serve(self) -> WorkerReport:
        """Acquire, heartbeat, and run the loop.

        Returns:
            The status, write count, posts, and token. A lost fence sets
            fence_rejected and does not commit T2.

        Raises:
            ValueError: A live run is missing credentials.
            LeaseNotHeld: Another worker holds an unexpired lease.
            SystemExit: AGENT_CRASH_AFTER raised SIGKILL.
        """
        tenant_id = TenantId(self.config.tenant_id)
        task = get_task(self.config.data_root, tenant_id, self.config.task_id)
        task_dir = resolve_tenant_dir(self.config.data_root, tenant_id) / task.workspace_root
        cow = CowWorkspace(task_dir)
        if (task_dir / "base").is_dir() and not (cow.work_root / "adder.py").exists():
            cow.seed()
        files = _WorkFiles(cow, self)
        self._workspace = files
        if task.status in _TERMINAL:
            files.cleanup_tmp()
            return WorkerReport(task.status, 0, 0, task.fencing_token, True, 0)
        connection = open_tenant_db(self.config.data_root, tenant_id)
        self._connection_for_steps = connection
        beat: threading.Thread | None = None
        try:
            now = time.time()
            worker_id = str(uuid.uuid4())
            service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
            lease = service.acquire(self.config.task_id, worker_id, now, self.config.ttl)
            with self._lock:
                self._service = service
                self._lease = lease
            mcp = McpClient(self.config.data_root, Jail(), cow.work_root)
            model = self._model(cow.work_root, mcp.chat_tools())
            guarded_model = _GuardedLlm(model, self)
            self._llm = guarded_model
            beat = threading.Thread(target=self._heartbeat, name="lease-heartbeat")
            beat.start()
            store = _GuardedStore(SqliteStepStore(connection), self)
            checks = ExternalVerifier(self.config.data_root / "checks", Jail())
            loop = ReactLoop(
                tenant_id,
                self.config.task_id,
                worker_id,
                lease.fencing_token,
                cow.work_root,
                store,
                guarded_model,
                files,
                OptimisticEditor(),
                checks,
                VerifyGate(
                    tenant_id,
                    self.config.task_id,
                    worker_id,
                    lease.fencing_token,
                    cow.work_root,
                    checks,
                    store,
                ),
                mcp,
            )
            try:
                result = loop.run(now, clock=SystemClock(), resume_only=self.config.resume_only)
            except FenceRejected:
                return WorkerReport(
                    "running", self.writes, guarded_model.post_count(), lease.fencing_token, False, 0, True
                )
            return WorkerReport(
                result.status,
                self.writes,
                result.posts,
                lease.fencing_token,
                result.cache_hit,
                result.executions,
            )
        finally:
            self._stop.set()
            if beat is not None:
                beat.join(timeout=2)
            try:
                if (
                    self._lease is not None
                    and self._service is not None
                    and not self.lost.is_set()
                    and not self._crashed
                ):
                    self._service.release(self._lease, time.time())
            except FenceRejected:
                pass
            connection.close()

    def crash(self) -> None:
        """Stop the heartbeat, shorten the lease, and raise SIGKILL.

        Returns:
            None. The process does not continue into T2.

        Raises:
            SystemExit: A test replaced os.kill.
        """
        self._crashed = True
        with self._lock:
            self._stop.set()
            if self._lease is not None and self._service is not None:
                try:
                    self._lease = self._service.renew(
                        self._lease, time.time(), self.config.crash_ttl
                    )
                except FenceRejected:
                    self.lost.set()
        os.kill(os.getpid(), signal.SIGKILL)

    def close_http(self) -> None:
        """Close the model socket.

        Returns:
            None.
        """
        if self._llm is not None:
            self._llm.close()

    def after_commit(self, step_index: int, result_json: str) -> None:
        """Append one byte when AGENT_MUTATE_AFTER names the tool just committed.

        Args:
            step_index: Step whose tool name is read from the intent row.
            result_json: Stored result. Unused except to keep the hook next to T2.

        Returns:
            None.
        """
        del result_json
        if self.config.mutate_after != "read_file" or self._workspace is None:
            return
        connection = self._connection_for_steps
        if connection is None:
            return
        row = connection.execute(  # type: ignore[attr-defined]
            """
            SELECT tool_name, args_json FROM step_intents
            WHERE task_id = ? AND step_index = ?
            """,
            (self.config.task_id, step_index),
        ).fetchone()
        if row is None or row["tool_name"] != "read_file":
            return
        args = json.loads(row["args_json"])
        path = str(args["path"])
        self.mutating = True
        try:
            data = self._workspace.read(path) + b"x"
            self._workspace.write(path, data)
        finally:
            self.mutating = False

    def _heartbeat(self) -> None:
        tenant_id = TenantId(self.config.tenant_id)
        connection = open_tenant_db(self.config.data_root, tenant_id)
        try:
            service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
            interval = (
                self.config.ttl / 3
                if self.config.heartbeat_interval is None
                else self.config.heartbeat_interval
            )
            while not self._stop.wait(interval):
                with self._lock:
                    if self._stop.is_set() or self._lease is None:
                        break
                    try:
                        self._lease = service.renew(self._lease, time.time(), self.config.ttl)
                    except FenceRejected:
                        self.lost.set()
                        self.close_http()
                        break
        finally:
            connection.close()

    def _model(self, work: Path, extra_tools: list[dict[str, object]] | None = None) -> OpenAIChatLlm:
        if self.config.llm == "canned":
            return OpenAIChatLlm(
                "http://canned.invalid",
                "canned-token",
                "canned",
                CannedTransport(_canned_bodies(work, self.config.canned, self.config.tool_choice)),
                extra_tools=extra_tools,
            )
        tenant = read_tenant(self.config.data_root, TenantId(self.config.tenant_id))
        api_key, base_url, model = _live_credentials(tenant.llm_api_key_env)
        return OpenAIChatLlm(
            base_url, api_key, model, UrllibTransport(), extra_tools=extra_tools
        )


def serve(config: WorkerConfig) -> WorkerReport:
    """Run one worker from a config object.

    Args:
        config: Data root, task, ttl, and hooks.

    Returns:
        The worker report.
    """
    return Worker(config).serve()


def main(argv: list[str] | None = None) -> int:
    """Run the worker from the command line.

    Args:
        argv: Arguments excluding the program name. None reads sys.argv.

    Returns:
        0 when the loop returns. 1 when the fence rejects T2.
    """
    parser = argparse.ArgumentParser(prog="agent_engine.worker")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--llm", choices=("canned", "live"), default="canned")
    parser.add_argument("--ttl", type=float, default=180.0)
    parser.add_argument("--tool-choice")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    ready = os.environ.get("AGENT_READY_FILE")
    release = os.environ.get("AGENT_RELEASE_FILE")
    hold = os.environ.get("AGENT_HOLD_SEC")
    report = serve(
        WorkerConfig(
            data_root=Path(args.data_root),
            tenant_id=args.tenant,
            task_id=args.task,
            llm=args.llm,
            ttl=args.ttl,
            tool_choice=args.tool_choice,
            canned=os.environ.get("AGENT_CANNED", "repair"),
            crash_after=os.environ.get("AGENT_CRASH_AFTER") or None,
            crash_ttl=float(os.environ.get("AGENT_CRASH_TTL", "2")),
            mutate_after=os.environ.get("AGENT_MUTATE_AFTER") or None,
            hold_sec=float(hold) if hold else None,
            ready_file=Path(ready) if ready else None,
            release_file=Path(release) if release else None,
            resume_only=os.environ.get("AGENT_RESUME_ONLY") == "1",
            force_finish_after_stale=os.environ.get("AGENT_FORCE_FINISH_AFTER_STALE") == "1",
        )
    )
    _print_report(report)
    return 1 if report.fence_rejected else 0


def _print_report(report: WorkerReport) -> None:
    if report.fence_rejected:
        print("FenceRejected")
    print(f"token={report.token}")
    print(f"status={report.status}")
    print(f"writes={report.writes}")
    print(f"posts={report.posts}")
    print(f"executions={report.executions}")
    if report.cache_hit:
        print("cache hit")
        print("side_effects=0")


def _canned_bodies(
    work: Path, canned: str, tool_choice: str | None
) -> list[dict[str, object]]:
    if tool_choice == "finish" or canned == "finish":
        return [completion_body([("finish", "{}")])]
    if canned == "stop":
        return [completion_body([])]
    if canned == "read":
        return [completion_body([("read_file", json.dumps({"path": "adder.py"}))])]
    digest = hashlib.sha256((work / "adder.py").read_bytes()).hexdigest()
    replace = json.dumps(
        {
            "path": "adder.py",
            "expected_sha256": digest,
            "old": "return a - b",
            "new": "return a + b",
        }
    )
    if canned == "replace":
        return [completion_body([("replace_text", replace)])]
    if canned == "stale":
        return [
            completion_body([("read_file", json.dumps({"path": "adder.py"}))]),
            completion_body([("replace_text", replace)]),
            completion_body([("finish", "{}")]),
        ]
    return [
        completion_body([("read_file", json.dumps({"path": "adder.py"}))]),
        completion_body([("replace_text", replace)]),
        completion_body([("finish", "{}")]),
    ]


def _saw_stale(messages: list[dict[str, object]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and "StaleContent" in content:
            return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
