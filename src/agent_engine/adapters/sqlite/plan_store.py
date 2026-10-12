"""Stored plans for one tenant connection.

A task has at most one plan row. propose replaces a proposed row and refuses
to overwrite an accepted one. accept moves that row to accepted without a
model call. This module does not open a second tenant file and does not write
tasks.status. The stamp trigger on task_plans aborts an insert whose
tenant_id disagrees with tenant.id.
"""

from __future__ import annotations

import sqlite3

from agent_engine.domain.models import TenantId

PlanRow = tuple[str, str, str]


class SqlitePlanStore:
    """Plan rows for the connection passed to the constructor."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Bind the store to one connection.

        Args:
            connection: Open engine.db for a single tenant. The caller closes it.
        """
        self._connection = connection

    def read(self, tenant_id: TenantId, task_id: str) -> PlanRow | None:
        """Return the stored plan for one task.

        Args:
            tenant_id: Tenant id stored on the plan row.
            task_id: Task primary key.

        Returns:
            Prompt, body_json, and status, or None when the task has no plan.
        """
        row = self._connection.execute(
            """
            SELECT prompt, body_json, status FROM task_plans
            WHERE tenant_id = ? AND task_id = ?
            """,
            (tenant_id.value, task_id),
        ).fetchone()
        if row is None:
            return None
        return str(row["prompt"]), str(row["body_json"]), str(row["status"])

    def propose(
        self, tenant_id: TenantId, task_id: str, prompt: str, body_json: str
    ) -> None:
        """Store one proposed plan, replacing an earlier proposal.

        Args:
            tenant_id: Tenant that owns the row.
            task_id: Task primary key.
            prompt: User sentence this plan answers.
            body_json: Validated plan body.

        Returns:
            None.

        Raises:
            ValueError: The stored plan is already accepted.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            stored = self.read(tenant_id, task_id)
            if stored is not None and stored[2] == "accepted":
                self._connection.execute("ROLLBACK")
                raise ValueError(f"plan for {task_id} is already accepted")
            self._connection.execute(
                "DELETE FROM task_plans WHERE tenant_id = ? AND task_id = ?",
                (tenant_id.value, task_id),
            )
            self._connection.execute(
                """
                INSERT INTO task_plans (tenant_id, task_id, prompt, body_json, status)
                VALUES (?, ?, ?, ?, 'proposed')
                """,
                (tenant_id.value, task_id, prompt, body_json),
            )
            self._connection.execute("COMMIT")
        except ValueError:
            raise
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise

    def accept(self, tenant_id: TenantId, task_id: str) -> None:
        """Move the stored plan to accepted.

        Args:
            tenant_id: Tenant that owns the row.
            task_id: Task primary key.

        Returns:
            None. An accepted plan stays accepted.

        Raises:
            ValueError: The task has no plan row.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            changed = self._connection.execute(
                """
                UPDATE task_plans SET status = 'accepted'
                WHERE tenant_id = ? AND task_id = ?
                """,
                (tenant_id.value, task_id),
            ).rowcount
            if changed == 0:
                self._connection.execute("ROLLBACK")
                raise ValueError(f"{task_id} has no plan to accept")
            self._connection.execute("COMMIT")
        except ValueError:
            raise
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
