"""Fenced intent, cache, and transcript rows for one tenant connection.

This module does not open a second tenant file. insert_intent commits before
the caller runs a tool. commit_step updates that intent, inserts the cache
row, and appends messages in one transaction. The fence predicate is inside
the INSERT or UPDATE. Zero rows is TaskNotFound when the task is gone and
FenceRejected when the token does not match. A matching token with no
intent_logged row is not a lost lease.
"""

from __future__ import annotations

import hashlib
import sqlite3

from agent_engine.domain.errors import FenceRejected, TaskNotFound
from agent_engine.domain.models import StepView, TenantId

_TERMINAL = ("completed", "rejected")


class SqliteStepStore:
    """Step log for the connection passed to the constructor."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Bind the store to one connection.

        Args:
            connection: Open engine.db for a single tenant. The caller closes it.
        """
        self._connection = connection
        self._open = False

    def task_status(self, tenant_id: TenantId, task_id: str) -> str:
        """Return tasks.status.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.

        Returns:
            The status text.

        Raises:
            TaskNotFound: The task row does not exist.
        """
        row = self._task_row(tenant_id, task_id)
        if row is None:
            raise TaskNotFound(task_id)
        return str(row["status"])

    def open_intent(self, tenant_id: TenantId, task_id: str) -> StepView | None:
        """Return the newest intent_logged agent step.

        Args:
            tenant_id: Tenant id stored on the step row.
            task_id: Task primary key.

        Returns:
            The open step, or None. commit_marker rows are ignored.
        """
        row = self._connection.execute(
            """
            SELECT step_index, idempotency_key, tool_name, args_json, status, result_json
            FROM step_intents
            WHERE tenant_id = ? AND task_id = ? AND status = 'intent_logged'
              AND tool_name != 'commit_marker'
            ORDER BY step_index DESC
            LIMIT 1
            """,
            (tenant_id.value, task_id),
        ).fetchone()
        if row is None:
            return None
        return _view(row)

    def next_index(self, tenant_id: TenantId, task_id: str) -> int:
        """Return one past the highest step_index.

        Args:
            tenant_id: Tenant id stored on the step row.
            task_id: Task primary key.

        Returns:
            The next index. Marker rows count, so a later agent step does not collide.
        """
        row = self._connection.execute(
            """
            SELECT COALESCE(MAX(step_index) + 1, 0) AS next_index
            FROM step_intents
            WHERE tenant_id = ? AND task_id = ?
            """,
            (tenant_id.value, task_id),
        ).fetchone()
        return int(row["next_index"])

    def agent_count(self, tenant_id: TenantId, task_id: str) -> int:
        """Count agent steps.

        Args:
            tenant_id: Tenant id stored on the step row.
            task_id: Task primary key.

        Returns:
            The number of rows whose tool_name is not commit_marker.
        """
        row = self._connection.execute(
            """
            SELECT COUNT(*) AS n FROM step_intents
            WHERE tenant_id = ? AND task_id = ? AND tool_name != 'commit_marker'
            """,
            (tenant_id.value, task_id),
        ).fetchone()
        return int(row["n"])

    def list_messages(self, tenant_id: TenantId, task_id: str) -> list[tuple[int, str, str]]:
        """Return the transcript in seq order.

        Args:
            tenant_id: Tenant id stored on the message row.
            task_id: Task primary key.

        Returns:
            Tuples of seq, role, and body_json.
        """
        rows = self._connection.execute(
            """
            SELECT seq, role, body_json FROM messages
            WHERE tenant_id = ? AND task_id = ?
            ORDER BY seq
            """,
            (tenant_id.value, task_id),
        ).fetchall()
        return [(int(row["seq"]), str(row["role"]), str(row["body_json"])) for row in rows]

    def mark_running(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Set status to running when the task is still pending.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            worker_id: Current lease owner.
            token: fencing_token the caller holds.
            now: Compared with lease_expires_at using >.

        Returns:
            None. Already running with a matching fence is left unchanged.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
        """
        self._begin()
        try:
            changed = self._change(
                """
                UPDATE tasks SET status = 'running'
                WHERE tenant_id = ? AND id = ? AND status = 'pending'
                  AND fencing_token = ? AND lease_owner = ?
                  AND lease_expires_at > ?
                """,
                (tenant_id.value, task_id, token, worker_id, now),
            )
            if changed == 0:
                self._rollback()
                self._accept_running(tenant_id, task_id, worker_id, token, now)
                return
            self._commit()
        except Exception:
            self._rollback()
            raise

    def insert_intent(
        self,
        tenant_id: TenantId,
        task_id: str,
        step_index: int,
        idempotency_key: str,
        tool_name: str,
        args_json: str,
        worker_id: str,
        token: int,
        now: float,
        messages: tuple[tuple[str, str], ...],
    ) -> None:
        """Insert intent_logged and optional transcript rows, then commit.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            step_index: Step primary key.
            idempotency_key: Cache key for this step.
            tool_name: Tool that will run after this commit.
            args_json: Model arguments.
            worker_id: Current lease owner.
            token: fencing_token the caller holds.
            now: Compared with lease_expires_at using >.
            messages: Transcript rows written in this same transaction.

        Returns:
            None. A later tool crash can still see this row.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The fence does not match.
        """
        self._begin()
        try:
            changed = self._change(
                """
                INSERT INTO step_intents (
                    tenant_id, task_id, step_index, idempotency_key, tool_name,
                    args_json, status, result_json
                )
                SELECT ?, ?, ?, ?, ?, ?, 'intent_logged', NULL
                WHERE EXISTS (
                    SELECT 1 FROM tasks
                    WHERE tenant_id = ? AND id = ?
                      AND fencing_token = ? AND lease_owner = ?
                      AND lease_expires_at > ?
                )
                """,
                (
                    tenant_id.value,
                    task_id,
                    step_index,
                    idempotency_key,
                    tool_name,
                    args_json,
                    tenant_id.value,
                    task_id,
                    token,
                    worker_id,
                    now,
                ),
            )
            if changed == 0:
                self._rollback()
                self._raise_fence(tenant_id, task_id, worker_id, token, now)
            self._append_messages(tenant_id, task_id, messages)
            self._commit()
        except Exception:
            self._rollback()
            raise

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
        """Move the intent to committed and write the cache and transcript.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            step_index: Intent row that must still be intent_logged.
            idempotency_key: Cache primary key.
            result_json: Tool result stored on the step and in the cache.
            worker_id: Current lease owner.
            token: fencing_token the caller holds.
            now: Compared with lease_expires_at using >.
            messages: Transcript rows written in this same transaction.

        Returns:
            None.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
            ValueError: The fence matches and there is no intent_logged row.
        """
        digest = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
        self._begin()
        try:
            changed = self._change(
                """
                UPDATE step_intents
                SET status = 'committed', result_json = ?
                WHERE tenant_id = ? AND task_id = ? AND step_index = ?
                  AND status = 'intent_logged'
                  AND EXISTS (
                    SELECT 1 FROM tasks
                    WHERE tenant_id = ? AND id = ?
                      AND fencing_token = ? AND lease_owner = ?
                      AND lease_expires_at > ?
                  )
                """,
                (
                    result_json,
                    tenant_id.value,
                    task_id,
                    step_index,
                    tenant_id.value,
                    task_id,
                    token,
                    worker_id,
                    now,
                ),
            )
            if changed == 0:
                self._rollback()
                self._raise_commit_miss(tenant_id, task_id, step_index, worker_id, token, now)
                return
            self._connection.execute(
                """
                INSERT INTO idempotency_cache (
                    tenant_id, idempotency_key, result_json, result_sha256
                ) VALUES (?, ?, ?, ?)
                """,
                (tenant_id.value, idempotency_key, result_json, digest),
            )
            self._append_messages(tenant_id, task_id, messages)
            self._commit()
        except Exception:
            self._rollback()
            raise

    def cache_get(self, tenant_id: TenantId, idempotency_key: str) -> str | None:
        """Return one cached result from this file.

        Args:
            tenant_id: Tenant id stored on the cache row.
            idempotency_key: SHA256 key.

        Returns:
            result_json, or None when this file has no committed row.
            A rolled_back row is not a hit.
        """
        row = self._connection.execute(
            """
            SELECT result_json FROM idempotency_cache
            WHERE tenant_id = ? AND idempotency_key = ? AND status = 'committed'
            """,
            (tenant_id.value, idempotency_key),
        ).fetchone()
        if row is None:
            return None
        return str(row["result_json"])

    def set_terminal(
        self,
        tenant_id: TenantId,
        task_id: str,
        status: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Write completed or rejected under the fence.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            status: completed or rejected.
            worker_id: Current lease owner.
            token: fencing_token the caller holds.
            now: Compared with lease_expires_at using >.

        Returns:
            None. Writing the status the row already has is a no-op.

        Raises:
            ValueError: status is not a terminal value.
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
        """
        if status not in _TERMINAL:
            raise ValueError("status must be completed or rejected")
        self._begin()
        try:
            changed = self._change(
                """
                UPDATE tasks SET status = ?
                WHERE tenant_id = ? AND id = ?
                  AND status IN ('pending', 'running')
                  AND fencing_token = ? AND lease_owner = ?
                  AND lease_expires_at > ?
                """,
                (status, tenant_id.value, task_id, token, worker_id, now),
            )
            if changed == 0:
                self._rollback()
                self._accept_terminal(tenant_id, task_id, status, worker_id, token, now)
                return
            self._commit()
        except Exception:
            self._rollback()
            raise

    def _append_messages(
        self,
        tenant_id: TenantId,
        task_id: str,
        messages: tuple[tuple[str, str], ...],
    ) -> None:
        if not messages:
            return
        current = self._connection.execute(
            """
            SELECT COALESCE(MAX(seq) + 1, 0) AS seq FROM messages
            WHERE tenant_id = ? AND task_id = ?
            """,
            (tenant_id.value, task_id),
        ).fetchone()
        seq = int(current["seq"])
        for role, body in messages:
            self._connection.execute(
                """
                INSERT INTO messages (tenant_id, task_id, seq, role, body_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tenant_id.value, task_id, seq, role, body),
            )
            seq += 1

    def _accept_running(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        row = self._require_task(tenant_id, task_id)
        if not _fence_ok(row, worker_id, token, now):
            raise FenceRejected(task_id)
        if str(row["status"]) in {"running", *_TERMINAL}:
            return
        raise FenceRejected(task_id)

    def _accept_terminal(
        self,
        tenant_id: TenantId,
        task_id: str,
        status: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        row = self._require_task(tenant_id, task_id)
        if str(row["status"]) == status and _fence_ok(row, worker_id, token, now):
            return
        if not _fence_ok(row, worker_id, token, now):
            raise FenceRejected(task_id)
        raise FenceRejected(task_id)

    def _raise_fence(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        row = self._require_task(tenant_id, task_id)
        if not _fence_ok(row, worker_id, token, now):
            raise FenceRejected(task_id)
        raise FenceRejected(task_id)

    def _raise_commit_miss(
        self,
        tenant_id: TenantId,
        task_id: str,
        step_index: int,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        row = self._require_task(tenant_id, task_id)
        if not _fence_ok(row, worker_id, token, now):
            raise FenceRejected(task_id)
        step = self._connection.execute(
            """
            SELECT status FROM step_intents
            WHERE tenant_id = ? AND task_id = ? AND step_index = ?
            """,
            (tenant_id.value, task_id, step_index),
        ).fetchone()
        if step is not None and str(step["status"]) == "committed":
            return
        raise ValueError("no intent_logged row")

    def _require_task(self, tenant_id: TenantId, task_id: str) -> sqlite3.Row:
        row = self._task_row(tenant_id, task_id)
        if row is None:
            raise TaskNotFound(task_id)
        return row

    def _task_row(self, tenant_id: TenantId, task_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            """
            SELECT status, fencing_token, lease_owner, lease_expires_at
            FROM tasks
            WHERE tenant_id = ? AND id = ?
            """,
            (tenant_id.value, task_id),
        ).fetchone()

    def _change(self, sql: str, params: tuple[object, ...]) -> int:
        before = self._connection.total_changes
        self._connection.execute(sql, params)
        return self._connection.total_changes - before

    def _begin(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")
        self._open = True

    def _commit(self) -> None:
        self._connection.execute("COMMIT")
        self._open = False

    def _rollback(self) -> None:
        if not self._open:
            return
        self._connection.execute("ROLLBACK")
        self._open = False


def _view(row: sqlite3.Row) -> StepView:
    result = row["result_json"]
    return StepView(
        int(row["step_index"]),
        str(row["idempotency_key"]),
        str(row["tool_name"]),
        str(row["args_json"]),
        str(row["status"]),
        None if result is None else str(result),
    )


def _fence_ok(row: sqlite3.Row, worker_id: str, token: int, now: float) -> bool:
    expires = row["lease_expires_at"]
    return (
        int(row["fencing_token"]) == token
        and row["lease_owner"] == worker_id
        and expires is not None
        and float(expires) > now
    )
