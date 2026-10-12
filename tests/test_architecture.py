"""Domain boundaries: frozen values, tenant id shape, and the import ban."""

import ast
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from agent_engine.domain.models import Task, Tenant, TenantId


def test_tenant_id_rejects_uppercase_dots_and_separators() -> None:
    """TenantId accepts a lowercase slug and rejects uppercase, dots, and slashes."""
    assert TenantId("acme").value == "acme"
    assert TenantId("a").value == "a"
    assert TenantId("acme-1_b").value == "acme-1_b"
    for raw in ("Acme", "acme.com", "acme/globex", "acme\\globex", "..", "-acme", ""):
        with pytest.raises(ValueError):
            TenantId(raw)


def test_models_are_frozen() -> None:
    """TenantId, Tenant, and Task reject attribute assignment."""
    tenant_id = TenantId("acme")
    tenant = Tenant(id=tenant_id, display_name="Acme")
    task = Task(
        tenant_id=tenant_id,
        id="fix-adder",
        status="pending",
        fencing_token=0,
        lease_owner=None,
        lease_expires_at=None,
        workspace_root="tasks/fix-adder",
    )
    with pytest.raises(FrozenInstanceError):
        tenant_id.value = "globex"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        tenant.display_name = "Other"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        task.status = "running"  # type: ignore[misc]
    with pytest.raises(ValueError):
        Task(
            tenant_id=tenant_id,
            id="Bad",
            status="pending",
            fencing_token=0,
            lease_owner=None,
            lease_expires_at=None,
            workspace_root="tasks/Bad",
        )
    with pytest.raises(ValueError):
        Task(
            tenant_id=tenant_id,
            id="fix-adder",
            status="pending",
            fencing_token=0,
            lease_owner=None,
            lease_expires_at=None,
            workspace_root="elsewhere/fix-adder",
        )


def test_domain_imports_neither_sqlite_nor_adapters() -> None:
    """domain/ does not import sqlite3, subprocess, os.killpg, or adapters."""
    root = Path(__file__).resolve().parents[1] / "src" / "agent_engine" / "domain"
    banned_roots = {"sqlite3", "subprocess"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    _reject_module(path, alias.name, banned_roots)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                _reject_module(path, module, banned_roots)
                if module == "os" and any(
                    alias.name in {"kill", "killpg", "setsid"} for alias in node.names
                ):
                    pytest.fail(f"{path} imports an os process-kill name")
                if any(alias.name == "adapters" for alias in node.names):
                    pytest.fail(f"{path} imports adapters")


def _reject_module(path: Path, module: str, banned_roots: set[str]) -> None:
    if module.split(".")[0] in banned_roots or module.startswith("agent_engine.adapters"):
        pytest.fail(f"{path} imports {module}")
