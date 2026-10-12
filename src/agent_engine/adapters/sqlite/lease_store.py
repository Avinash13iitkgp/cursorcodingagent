"""SQL for fenced leases on one open tenant connection.

This module does not open a second tenant file. Each method runs inside
BEGIN IMMEDIATE. Steal matches lease_owner IS NULL OR lease_expires_at <= now
and raises LeaseNotHeld on zero rows when the task exists. Renew matches
lease_owner, fencing_token, and lease_expires_at > now, and raises
FenceRejected on zero rows. Release matches lease_owner and fencing_token,
and raises FenceRejected on zero rows. commit_marker inserts a row only while
fencing_token, lease_owner, and lease_expires_at > now still match, and raises
FenceRejected or TaskNotFound on zero rows.
"""

from __future__ import annotations

import sqlite3

from agent_engine.domain.errors import FenceRejected, LeaseNotHeld, TaskNotFound
from agent_engine.domain.models import TenantId


class SqliteLeaseStore:
    """Lease statements for the connection passed to the constructor."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Bind the store to one connection.

        Args:
            connection: Open engine.db for a single tenant. The caller closes it.
        """
        self._connection = connection
        self._open = False

    def steal(
        self, tenant_id: TenantId, task_id: str, worker_id: str, now: float, expiry: float
    ) -> int:
        """Bump fencing_token when the lease is free or expired.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            worker_id: New lease_owner.
            now: Compared with lease_expires_at using <=.
            expiry: Stored lease_expires_at.

        Returns:
            The fencing token after the bump.

        Raises:
            TaskNotFound: Zero rows because the task is absent.
            LeaseNotHeld: Zero rows because the owner is still unexpired.
        """
        self._begin()
        try:
            if self._task_row(tenant_id, task_id) is None:
                self._rollback()
                raise TaskNotFound(task_id)
            changed = self._change(
                """
                UPDATE tasks
                SET fencing_token = fencing_token + 1,
                    lease_owner = ?,
                    lease_expires_at = ?
                WHERE tenant_id = ? AND id = ?
                  AND (lease_owner IS NULL OR lease_expires_at <= ?)
                """,
                (worker_id, expiry, tenant_id.value, task_id, now),
            )
            if changed == 0:
                self._rollback()
                raise LeaseNotHeld(task_id)
            token = self._fencing_token(tenant_id, task_id)
            self._commit()
            return token
        except Exception:
            self._rollback()
            raise

    def renew(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
        expiry: float,
    ) -> None:
        """Extend lease_expires_at for the same fencing token.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            worker_id: Current lease_owner.
            token: fencing_token the caller holds.
            now: Compared with lease_expires_at using >.
            expiry: New lease_expires_at.

        Returns:
            None.

        Raises:
            TaskNotFound: Zero rows because the task is absent.
            FenceRejected: Zero rows because the token, owner, or expiry does not match.
        """
        self._begin()
        try:
            changed = self._change(
                """
                UPDATE tasks
                SET lease_expires_at = ?
                WHERE tenant_id = ? AND id = ?
                  AND lease_owner = ? AND fencing_token = ?
                  AND lease_expires_at > ?
                """,
                (expiry, tenant_id.value, task_id, worker_id, token, now),
            )
            if changed == 0:
                self._rollback()
                self._raise_unchanged(tenant_id, task_id)
            self._commit()
        except Exception:
            self._rollback()
            raise

    def release(
        self, tenant_id: TenantId, task_id: str, worker_id: str, token: int
    ) -> int:
        """Clear the owner and bump fencing_token.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            worker_id: Current lease_owner.
            token: fencing_token the caller holds.

        Returns:
            The fencing token after the bump.

        Raises:
            TaskNotFound: Zero rows because the task is absent.
            FenceRejected: Zero rows because the token or owner does not match.
        """
        self._begin()
        try:
            changed = self._change(
                """
                UPDATE tasks
                SET fencing_token = fencing_token + 1,
                    lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE tenant_id = ? AND id = ?
                  AND lease_owner = ? AND fencing_token = ?
                """,
                (tenant_id.value, task_id, worker_id, token),
            )
            if changed == 0:
                self._rollback()
                self._raise_unchanged(tenant_id, task_id)
            bumped = self._fencing_token(tenant_id, task_id)
            self._commit()
            return bumped
        except Exception:
            self._rollback()
            raise

    def commit_marker(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> None:
        """Insert a committed marker while the fence still matches.

        Args:
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            worker_id: Current lease_owner.
            token: fencing_token the caller holds.
            now: Compared with lease_expires_at using >.

        Returns:
            None.

        Raises:
            TaskNotFound: Zero rows because the task is absent.
            FenceRejected: Zero rows because the fence does not match.
        """
        marker = f"commit-marker-{worker_id}-{token}-{now}"
        self._begin()
        try:
            changed = self._change(
                """
                INSERT INTO step_intents (
                    tenant_id, task_id, step_index, idempotency_key, tool_name,
                    args_json, status, result_json
                )
                SELECT ?, ?,
                    COALESCE(
                        (SELECT MAX(step_index) + 1 FROM step_intents
                         WHERE tenant_id = ? AND task_id = ?),
                        0
                    ),
                    ?, 'commit_marker', '{}', 'committed', '{"marker":true}'
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
                    tenant_id.value,
                    task_id,
                    marker,
                    tenant_id.value,
                    task_id,
                    token,
                    worker_id,
                    now,
                ),
            )
            if changed == 0:
                self._rollback()
                self._raise_unchanged(tenant_id, task_id)
            self._commit()
        except Exception:
            self._rollback()
            raise

    def _task_row(self, tenant_id: TenantId, task_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            """
            SELECT fencing_token, lease_owner, lease_expires_at
            FROM tasks
            WHERE tenant_id = ? AND id = ?
            """,
            (tenant_id.value, task_id),
        ).fetchone()

    def _fencing_token(self, tenant_id: TenantId, task_id: str) -> int:
        row = self._task_row(tenant_id, task_id)
        if row is None:
            raise TaskNotFound(task_id)
        return int(row["fencing_token"])

    def _raise_unchanged(self, tenant_id: TenantId, task_id: str) -> None:
        if self._task_row(tenant_id, task_id) is None:
            raise TaskNotFound(task_id)
        raise FenceRejected(task_id)

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
