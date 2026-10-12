"""Frozen domain values. This module performs no I/O.

Types: TenantId, Tenant, Task, Lease, TextEdit, and EditResult. Constructing
them does not touch the filesystem or SQLite. Invalid tenant ids raise ValueError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class TenantId:
    """Tenant boundary key. The value is also the directory name under tenants/."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _TENANT_ID.fullmatch(self.value) is None:
            raise ValueError(
                "tenant id must match ^[a-z0-9][a-z0-9_-]{0,63}$"
            )


@dataclass(frozen=True)
class Tenant:
    """One local tenant. llm_api_key_env is an environment variable name, not a secret."""

    id: TenantId
    display_name: str
    llm_api_key_env: str = "AGENT_LLM_API_KEY"


@dataclass(frozen=True)
class Task:
    """A unit of work stored in one tenant database.

    workspace_root is relative to the tenant directory. fencing_token starts at 0
    until a worker acquires the lease.
    """

    tenant_id: TenantId
    id: str
    status: str
    fencing_token: int
    lease_owner: str | None
    lease_expires_at: float | None
    workspace_root: str

    def __post_init__(self) -> None:
        if _TASK_ID.fullmatch(self.id) is None:
            raise ValueError("task id must match ^[a-z0-9][a-z0-9_-]{0,63}$")
        if self.workspace_root != f"tasks/{self.id}":
            raise ValueError("workspace_root must be tasks/{task id}")


@dataclass(frozen=True)
class Lease:
    """A worker's right to commit for one task.

    fencing_token is the capability to commit. expires_at is only a liveness
    hint: a stale token is rejected even when the clock still says the lease
    is current, and a clock expiry only lets another worker acquire a new token.
    """

    tenant_id: TenantId
    task_id: str
    worker_id: str
    fencing_token: int
    expires_at: float


@dataclass(frozen=True)
class TextEdit:
    """One optimistic replacement. expected_sha256 is the caller's preimage hash."""

    path: str
    expected_sha256: str
    old: str
    new: str


@dataclass(frozen=True)
class EditResult:
    """Outcome of an edit that changed nothing after the check, or wrote the new bytes."""

    status: Literal["applied", "already_applied"]
