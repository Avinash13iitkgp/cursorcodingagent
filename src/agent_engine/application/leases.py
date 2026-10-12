"""Fenced leases for one tenant.

Steal bumps fencing_token and replaces the owner. Renew only extends
lease_expires_at and keeps the token. Release bumps the token and clears the
owner so the next worker can acquire without waiting for expiry. This service
never opens a database and never touches a second tenant: the store is already
bound to one tenant connection.
"""

from __future__ import annotations

import time

from agent_engine.domain.errors import TenantMismatch
from agent_engine.domain.models import Lease, TenantId
from agent_engine.domain.ports import Clock, LeaseStore


class SystemClock:
    """Wall clock for production lease expiry."""

    def now(self) -> float:
        """Return time.time().

        Returns:
            Seconds since the Unix epoch.
        """
        return time.time()


class LeaseService:
    """Lease acquire, renew, release, and fenced commit for one tenant."""

    def __init__(self, tenant_id: TenantId, store: LeaseStore, clock: Clock) -> None:
        """Bind the service to one tenant.

        Args:
            tenant_id: The only tenant this service will name in store calls.
            store: Lease rows for the connection already opened for tenant_id.
            clock: Source of the current instant when a caller omits now.
        """
        self._tenant_id = tenant_id
        self._store = store
        self._clock = clock

    def acquire(self, task_id: str, worker_id: str, now: float, ttl: float) -> Lease:
        """Take a free or expired lease and return the new fencing token.

        Args:
            task_id: Task inside the bound tenant.
            worker_id: UUID of the worker that will own the lease.
            now: Instant compared with the current expiry.
            ttl: Seconds the new lease remains valid.

        Returns:
            The lease, including the bumped token.

        Raises:
            ValueError: ttl is not positive or worker_id is empty.
            TaskNotFound: The task row does not exist.
            LeaseNotHeld: An unexpired owner already holds the lease.
        """
        self._require_worker(worker_id)
        self._require_ttl(ttl)
        expiry = now + ttl
        token = self._store.steal(self._tenant_id, task_id, worker_id, now, expiry)
        return Lease(self._tenant_id, task_id, worker_id, token, expiry)

    def renew(self, lease: Lease, now: float, ttl: float) -> Lease:
        """Extend expiry and return the same fencing token.

        Args:
            lease: Lease previously returned by acquire.
            now: Instant compared with the current expiry.
            ttl: Seconds from now until the new expiry.

        Returns:
            The same token with the new expiry.

        Raises:
            TenantMismatch: lease.tenant_id is not this service's tenant.
            ValueError: ttl is not positive.
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
        """
        self._require_bound(lease)
        self._require_ttl(ttl)
        expiry = now + ttl
        self._store.renew(
            self._tenant_id,
            lease.task_id,
            lease.worker_id,
            lease.fencing_token,
            now,
            expiry,
        )
        return Lease(
            self._tenant_id,
            lease.task_id,
            lease.worker_id,
            lease.fencing_token,
            expiry,
        )

    def release(self, lease: Lease, now: float) -> None:
        """Drop the lease and invalidate the holder's token.

        Args:
            lease: Lease previously returned by acquire.
            now: Accepted for a uniform call shape. The SQL predicate does not
                compare it with expiry.

        Returns:
            None.

        Raises:
            TenantMismatch: lease.tenant_id is not this service's tenant.
            TaskNotFound: The task row does not exist.
            FenceRejected: The token or owner does not match.
        """
        del now
        self._require_bound(lease)
        self._store.release(
            self._tenant_id, lease.task_id, lease.worker_id, lease.fencing_token
        )

    def commit_marker(self, lease: Lease, now: float) -> None:
        """Write a committed marker if this lease still holds the fence.

        Args:
            lease: Lease whose token must still be current.
            now: Instant compared with lease_expires_at.

        Returns:
            None.

        Raises:
            TenantMismatch: lease.tenant_id is not this service's tenant.
            TaskNotFound: The task row does not exist.
            FenceRejected: The token, owner, or expiry does not match.
        """
        self._require_bound(lease)
        self._store.commit_marker(
            self._tenant_id,
            lease.task_id,
            lease.worker_id,
            lease.fencing_token,
            now,
        )

    def _require_bound(self, lease: Lease) -> None:
        if lease.tenant_id != self._tenant_id:
            raise TenantMismatch(
                f"lease tenant is {lease.tenant_id.value}, service tenant is {self._tenant_id.value}"
            )

    def _require_worker(self, worker_id: str) -> None:
        if not worker_id:
            raise ValueError("worker_id is required")

    def _require_ttl(self, ttl: float) -> None:
        if ttl <= 0:
            raise ValueError("ttl must be positive")
