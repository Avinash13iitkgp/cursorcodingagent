"""Frozen domain values. This module performs no I/O.

Types: TenantId, Tenant, and Task. Constructing them does not touch the filesystem
or SQLite. Invalid tenant ids raise ValueError.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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
