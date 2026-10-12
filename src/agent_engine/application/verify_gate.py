"""The only writer of completed and rejected.

finish runs the operator's checks and then sets the terminal status under the
fence. It does not read the transcript. A failed finish returns only the
status, not the compiler or test output. FenceRejected leaves the task running
because the status update is inside the fenced statement.
"""

from __future__ import annotations

from pathlib import Path

from agent_engine.domain.models import TenantId
from agent_engine.domain.ports import CheckRunner, StepLog


class VerifyGate:
    """Run checks/commands and write the terminal task status."""

    def __init__(
        self,
        tenant_id: TenantId,
        task_id: str,
        worker_id: str,
        token: int,
        work_root: Path,
        checks: CheckRunner,
        store: StepLog,
    ) -> None:
        """Bind the gate to one leased task.

        Args:
            tenant_id: Tenant whose task row may become terminal.
            task_id: Task primary key.
            worker_id: Lease owner required by the fence.
            token: Fencing token required by the fence.
            work_root: Directory the checks compile and import.
            checks: Command list runner. The model does not choose its argv.
            store: Status writer. This gate does not ask it for messages.
        """
        self._tenant_id = tenant_id
        self._task_id = task_id
        self._worker_id = worker_id
        self._token = token
        self._work_root = work_root
        self._checks = checks
        self._store = store

    def finish(self, now: float) -> str:
        """Pass writes completed. Fail writes rejected.

        Args:
            now: Instant compared with lease_expires_at.

        Returns:
            The terminal status. Compiler and test output are not included.

        Raises:
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match. The
                task status stays running.
        """
        ok, _stdout, _stderr = self._checks.run(self._work_root)
        status = "completed" if ok else "rejected"
        self._store.set_terminal(
            self._tenant_id,
            self._task_id,
            status,
            self._worker_id,
            self._token,
            now,
        )
        return status
