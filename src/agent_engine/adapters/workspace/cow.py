"""Copy-on-write work tree for one task.

Layout is tasks/{task_id}/{base,work,snapshots/{step}}. seed hardlinks base
into work. write never opens base and never writes an inode whose st_nlink is
greater than 1: os.replace puts a new inode in the work directory. snapshot
hardlinks the work files and writes tombstones.json. It may fsync. rollback
relinks work from that snapshot, restores tombstones, and marks later
step_intents, cache rows, and trajectory rows rolled_back under the fence.
The relink is the only work inside the timing measurement. os.link errors,
including EXDEV, propagate.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

from agent_engine.domain.errors import FenceRejected, TaskNotFound


class CowWorkspace:
    """Hardlink workspace. base stays unchanged when work is edited."""

    def __init__(self, task_dir: Path) -> None:
        """Bind the workspace to one task directory.

        Args:
            task_dir: Directory that contains base, work, and snapshots.
                Callers pass tenants/{tenant}/tasks/{task}.
        """
        self._task_dir = task_dir
        self._base = task_dir / "base"
        self._work = task_dir / "work"
        self._snapshots = task_dir / "snapshots"
        self._tombstones: set[str] = set()
        self.relink_ms = 0.0

    @property
    def work_root(self) -> Path:
        """Return the editable work directory.

        Returns:
            The work path. Checks and the editor use this directory.
        """
        return self._work

    def seed(self) -> None:
        """Hardlink every base file into work.

        Returns:
            None. An existing work path is left in place.

        Raises:
            FileNotFoundError: base does not exist.
            OSError: os.link fails, including when the directories are on
                different devices (EXDEV).
        """
        if not self._base.is_dir():
            raise FileNotFoundError("base is missing")
        self._work.mkdir(parents=True, exist_ok=True)
        for path in self._files(self._base):
            rel = path.relative_to(self._base)
            dest = self._work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                continue
            os.link(path, dest)

    def resolve(self, relative_path: str) -> Path:
        """Return relative_path inside work.

        Args:
            relative_path: Slash-separated path under work.

        Returns:
            The resolved path.

        Raises:
            ValueError: The path is absolute or leaves work.
        """
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("path must stay under the work root")
        root = self._work.resolve()
        candidate = (root / relative_path).resolve()
        candidate.relative_to(root)
        return candidate

    def read(self, relative_path: str) -> bytes:
        """Read one work file.

        Args:
            relative_path: Path relative to work.

        Returns:
            The file bytes.

        Raises:
            ValueError: The path escapes work.
            FileNotFoundError: The file does not exist.
        """
        return self.resolve(relative_path).read_bytes()

    def write(self, relative_path: str, data: bytes) -> None:
        """Replace a work directory entry with a new inode.

        Args:
            relative_path: Path relative to work.
            data: Bytes stored in the new inode.

        Returns:
            None. A hardlinked inode is not opened for write and is not chmod'd.

        Raises:
            ValueError: The path escapes work.
        """
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=target.parent)
        tmp = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        self._tombstones.discard(relative_path)

    def remove_file(self, relative_path: str) -> None:
        """Unlink a work entry and remember a tombstone. base is not unlinked.

        Args:
            relative_path: Path relative to work.

        Returns:
            None.

        Raises:
            ValueError: The path escapes work.
            FileNotFoundError: The work file does not exist.
        """
        target = self.resolve(relative_path)
        if not target.is_file():
            raise FileNotFoundError(relative_path)
        target.unlink()
        self._tombstones.add(relative_path)

    def cleanup_tmp(self) -> None:
        """Delete leftover ``*.tmp`` files that resolve inside work.

        Returns:
            None.
        """
        root = self._work.resolve()
        if not root.exists():
            return
        for path in root.rglob("*.tmp"):
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            path.unlink()

    def snapshot(self, step: int) -> None:
        """Hardlink work into snapshots/{step} and write tombstones.json.

        Args:
            step: Snapshot directory name. Later rollback names this step.

        Returns:
            None. File and directory fsync run here.

        Raises:
            OSError: os.link fails, including EXDEV.
        """
        dest_root = self._snapshots / str(step)
        dest_root.mkdir(parents=True, exist_ok=True)
        for path in self._files(self._work):
            rel = path.relative_to(self._work)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                os.link(path, dest)
            _fsync(dest)
        payload = json.dumps(sorted(self._tombstones)).encode("utf-8")
        tombstone_path = dest_root / "tombstones.json"
        tombstone_path.write_bytes(payload)
        _fsync(tombstone_path)
        _fsync(dest_root)

    def relink(self, step: int) -> None:
        """Point work at the snapshot inodes and restore tombstones.

        Args:
            step: Snapshot to restore. This method does not fsync and does
                not write the tenant database.

        Returns:
            None.

        Raises:
            FileNotFoundError: The snapshot directory is missing.
            OSError: os.link fails, including EXDEV.
        """
        snap = self._snapshots / str(step)
        if not snap.is_dir():
            raise FileNotFoundError(f"snapshot {step} is missing")
        tombstones = json.loads((snap / "tombstones.json").read_text(encoding="utf-8"))
        self._clear_work()
        for path in self._files(snap):
            if path.name == "tombstones.json":
                continue
            rel = path.relative_to(snap)
            dest = self._work / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)
        for rel in tombstones:
            target = self._work / rel
            if target.is_file():
                target.unlink()
        self._tombstones = set(tombstones)

    def rollback(
        self,
        step: int,
        connection: sqlite3.Connection,
        tenant_id: str,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
    ) -> float:
        """Relink work, then mark later step and cache rows rolled_back.

        Args:
            step: Snapshot to restore. Rows with a greater step_index rewind.
            connection: Open engine.db for this tenant. The caller closes it.
            tenant_id: Tenant id stored on the task row.
            task_id: Task primary key.
            worker_id: Current lease owner.
            token: Fencing token the caller holds.
            now: Instant compared with lease_expires_at.

        Returns:
            Milliseconds spent in relink. The database update is outside that
            measurement.

        Raises:
            FileNotFoundError: The snapshot directory is missing.
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
        """
        started = time.perf_counter()
        self.relink(step)
        elapsed = (time.perf_counter() - started) * 1000
        self.relink_ms = elapsed
        self._rewind(connection, tenant_id, task_id, worker_id, token, now, step)
        return elapsed

    def _rewind(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        task_id: str,
        worker_id: str,
        token: int,
        now: float,
        step: int,
    ) -> None:
        connection.execute("BEGIN IMMEDIATE")
        try:
            before = connection.total_changes
            connection.execute(
                """
                UPDATE tasks
                SET lease_owner = lease_owner
                WHERE tenant_id = ? AND id = ?
                  AND fencing_token = ? AND lease_owner = ?
                  AND lease_expires_at > ?
                """,
                (tenant_id, task_id, token, worker_id, now),
            )
            if connection.total_changes - before == 0:
                connection.execute("ROLLBACK")
                self._raise_fence(connection, tenant_id, task_id)
            connection.execute(
                """
                UPDATE idempotency_cache
                SET status = 'rolled_back'
                WHERE tenant_id = ?
                  AND idempotency_key IN (
                    SELECT idempotency_key FROM step_intents
                    WHERE tenant_id = ? AND task_id = ? AND step_index > ?
                  )
                """,
                (tenant_id, tenant_id, task_id, step),
            )
            connection.execute(
                """
                UPDATE step_intents
                SET status = 'rolled_back'
                WHERE tenant_id = ? AND task_id = ? AND step_index > ?
                """,
                (tenant_id, task_id, step),
            )
            self._rewind_trajectory(connection, tenant_id, task_id, step)
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

    def _raise_fence(
        self, connection: sqlite3.Connection, tenant_id: str, task_id: str
    ) -> None:
        row = connection.execute(
            "SELECT 1 FROM tasks WHERE tenant_id = ? AND id = ?",
            (tenant_id, task_id),
        ).fetchone()
        if row is None:
            raise TaskNotFound(task_id)
        raise FenceRejected(task_id)

    def _rewind_trajectory(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        task_id: str,
        step: int,
    ) -> None:
        found = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'trajectory'"
        ).fetchone()
        if found is None:
            return
        columns = [row[1] for row in connection.execute("PRAGMA table_info(trajectory)")]
        if "status" not in columns:
            return
        connection.execute(
            """
            UPDATE trajectory
            SET status = 'rolled_back'
            WHERE tenant_id = ? AND task_id = ? AND step_index > ?
            """,
            (tenant_id, task_id, step),
        )

    def _clear_work(self) -> None:
        root = self._work.resolve()
        root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_symlink() or path.is_file():
                try:
                    path.resolve().relative_to(root)
                except ValueError:
                    continue
                path.unlink()
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass

    def _files(self, root: Path) -> list[Path]:
        if not root.exists():
            return []
        found: list[Path] = []
        base = root.resolve()
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                path.resolve().relative_to(base)
            except ValueError:
                continue
            if path.suffix == ".tmp":
                continue
            found.append(path)
        return found


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
