"""Fenced trajectory and knowledge log.

T2 appends inference, tool call, tool result, and knowledge rows in the same
transaction as the step result. A summary row is its own fenced insert and
does not change tasks.status. Stamp triggers on knowledge and trajectory abort
an insert whose tenant_id disagrees with tenant.id.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

from agent_engine.domain.errors import FenceRejected
from agent_engine.domain.models import TenantId

TrajectoryRow = tuple[int, str, int | None, str]
KnowledgeRow = tuple[int, str, str, str]
Note = tuple[str, str, str, int]


class TrajectoryStore:
    """Read and append the context log for one open tenant connection."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        worker_id: str,
        token: int,
        now: Callable[[], float],
    ) -> None:
        """Bind the log to the lease that may append a summary.

        Args:
            connection: Open tenant database. The caller closes it.
            worker_id: Lease owner for a summary insert.
            token: Fencing token for a summary insert.
            now: Instant compared with lease_expires_at.

        Returns:
            None.
        """
        self._connection = connection
        self._worker_id = worker_id
        self._token = token
        self._now = now

    def list_trajectory(self, tenant_id: TenantId, task_id: str) -> list[TrajectoryRow]:
        """Return trajectory rows in seq order.

        Args:
            tenant_id: Tenant bound to this file.
            task_id: Task primary key.

        Returns:
            Tuples of seq, kind, step_index, and body_json.
        """
        rows = self._connection.execute(
            """
            SELECT seq, kind, step_index, body_json FROM trajectory
            WHERE tenant_id = ? AND task_id = ?
            ORDER BY seq
            """,
            (tenant_id.value, task_id),
        ).fetchall()
        return [
            (int(row["seq"]), str(row["kind"]), _step(row["step_index"]), str(row["body_json"]))
            for row in rows
        ]

    def list_knowledge(self, tenant_id: TenantId, task_id: str) -> list[KnowledgeRow]:
        """Return knowledge notes newest first.

        Args:
            tenant_id: Tenant bound to this file.
            task_id: Task primary key.

        Returns:
            Tuples of note_id, path, quote, and body.
        """
        rows = self._connection.execute(
            """
            SELECT note_id, path, quote, body FROM knowledge
            WHERE tenant_id = ? AND task_id = ?
            ORDER BY note_id DESC
            """,
            (tenant_id.value, task_id),
        ).fetchall()
        return [
            (int(row["note_id"]), str(row["path"]), str(row["quote"]), str(row["body"]))
            for row in rows
        ]

    def write_summary(self, tenant_id: TenantId, task_id: str, summary: str) -> None:
        """Insert one context_summary row while the fence matches.

        Args:
            tenant_id: Tenant bound to this file.
            task_id: Task primary key.
            summary: Redacted summary text.

        Returns:
            None. tasks.status is left unchanged.

        Raises:
            FenceRejected: The token, owner, or expiry does not match.
        """
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            changed = self._connection.execute(
                """
                UPDATE tasks SET lease_owner = lease_owner
                WHERE tenant_id = ? AND id = ?
                  AND fencing_token = ? AND lease_owner = ?
                  AND lease_expires_at > ?
                """,
                (tenant_id.value, task_id, self._token, self._worker_id, self._now()),
            ).rowcount
            if changed == 0:
                self._connection.execute("ROLLBACK")
                raise FenceRejected(task_id)
            _insert_rows(
                self._connection,
                tenant_id,
                task_id,
                (("context_summary", None, json.dumps({"summary": summary})),),
            )
            self._connection.execute("COMMIT")
        except FenceRejected:
            raise
        except Exception:
            self._connection.execute("ROLLBACK")
            raise


def append_open(
    connection: sqlite3.Connection,
    tenant_id: TenantId,
    task_id: str,
    trajectory: tuple[tuple[str, int | None, str], ...],
    knowledge: Note | None,
) -> None:
    """Insert trajectory and one knowledge note inside the caller's transaction.

    Args:
        connection: Connection that already holds the T2 transaction.
        tenant_id: Tenant bound to this file.
        task_id: Task primary key.
        trajectory: Kind, step_index, and body_json rows.
        knowledge: Path, quote, body, and source step. None skips the note.

    Returns:
        None.
    """
    if trajectory:
        _insert_rows(connection, tenant_id, task_id, trajectory)
    if knowledge is None:
        return
    path, quote, body, source_step = knowledge
    note_id = _next_note(connection, tenant_id, task_id)
    connection.execute(
        """
        INSERT INTO knowledge (tenant_id, task_id, note_id, path, quote, body, source_step)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (tenant_id.value, task_id, note_id, path, quote, body, source_step),
    )


def _insert_rows(
    connection: sqlite3.Connection,
    tenant_id: TenantId,
    task_id: str,
    rows: tuple[tuple[str, int | None, str], ...],
) -> None:
    seq = _next_seq(connection, tenant_id, task_id)
    for kind, step_index, body in rows:
        connection.execute(
            """
            INSERT INTO trajectory (tenant_id, task_id, seq, step_index, kind, body_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (tenant_id.value, task_id, seq, step_index, kind, body),
        )
        seq += 1


def _next_seq(connection: sqlite3.Connection, tenant_id: TenantId, task_id: str) -> int:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(seq), -1) AS seq FROM trajectory
        WHERE tenant_id = ? AND task_id = ?
        """,
        (tenant_id.value, task_id),
    ).fetchone()
    return int(row["seq"]) + 1


def _next_note(connection: sqlite3.Connection, tenant_id: TenantId, task_id: str) -> int:
    row = connection.execute(
        """
        SELECT COALESCE(MAX(note_id), 0) AS note_id FROM knowledge
        WHERE tenant_id = ? AND task_id = ?
        """,
        (tenant_id.value, task_id),
    ).fetchone()
    return int(row["note_id"]) + 1


def _step(value: object) -> int | None:
    if value is None:
        return None
    return int(value)
