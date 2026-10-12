"""Command line for tenant provisioning and the step 1 self-test.

This module is the process edge. It opens SQLite only by calling the sqlite
adapter, and it does not import application code. Invalid ids and tenant
boundary errors exit non-zero. self-test prints PASS or FAIL.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_engine.adapters.sqlite.tenant_store import (
    get_task,
    init_task,
    provision,
    read_pragmas,
)
from agent_engine.domain.errors import TaskNotFound, TenantMismatch, UnknownTenant
from agent_engine.domain.models import TenantId


def main(argv: list[str] | None = None) -> int:
    """Run one agent_engine command.

    Args:
        argv: Arguments excluding the program name. None reads sys.argv.

    Returns:
        Process exit code. 0 is success. self-test returns 0 only when it
        prints PASS.
    """
    args_list = list(sys.argv[1:] if argv is None else argv)
    data_root, rest = _take_data_root(args_list)
    parser = argparse.ArgumentParser(prog="agent_engine")
    sub = parser.add_subparsers(dest="command", required=True)

    provision_cmd = sub.add_parser("provision")
    provision_cmd.add_argument("--tenant", required=True)
    provision_cmd.add_argument("--display-name", required=True)

    task_cmd = sub.add_parser("init-task")
    task_cmd.add_argument("--tenant", required=True)
    task_cmd.add_argument("--task", required=True)

    pragma_cmd = sub.add_parser("pragma")
    pragma_cmd.add_argument("--tenant", required=True)

    self_test = sub.add_parser("self-test")
    self_test.add_argument("--step", type=int)
    self_test.add_argument("--through", type=int)
    self_test.add_argument("--live", action="store_true")

    try:
        parsed = parser.parse_args(rest)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2
    try:
        if parsed.command == "self-test":
            live_status = _missing_token_if_live(parsed.live)
            if live_status != 0:
                return live_status
            return _self_test(data_root, parsed.step, parsed.through)
        tenant_id = TenantId(parsed.tenant)
        if parsed.command == "provision":
            path = provision(data_root, tenant_id, parsed.display_name)
            print(path)
            return 0
        if parsed.command == "init-task":
            task = init_task(data_root, tenant_id, parsed.task)
            print(task.id)
            return 0
        journal, synchronous, _busy = read_pragmas(data_root, tenant_id)
        print(journal)
        print(synchronous)
        return 0
    except (UnknownTenant, TenantMismatch, TaskNotFound, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


def _take_data_root(argv: list[str]) -> tuple[Path, list[str]]:
    root = Path("var")
    rest: list[str] = []
    index = 0
    while index < len(argv):
        if argv[index] == "--data-root" and index + 1 < len(argv):
            root = Path(argv[index + 1])
            index += 2
            continue
        rest.append(argv[index])
        index += 1
    return root, rest


def _missing_token_if_live(live: bool) -> int:
    if live and not os.environ.get("AGENT_LLM_API_KEY"):
        print("missing AGENT_LLM_API_KEY", file=sys.stderr)
        return 1
    return 0


def _self_test(data_root: Path, step: int | None, through: int | None) -> int:
    if step is None and through is None:
        print("FAIL", file=sys.stderr)
        print("self-test requires --step or --through", file=sys.stderr)
        return 1
    last = through if through is not None else step
    assert last is not None
    if last != 1 or (step is not None and step != 1):
        print("FAIL")
        print(f"step {last} is not implemented", file=sys.stderr)
        return 1
    try:
        _run_step1(data_root)
    except (UnknownTenant, TenantMismatch, TaskNotFound, ValueError, OSError) as exc:
        print("FAIL")
        print(exc, file=sys.stderr)
        return 1
    print("PASS")
    return 0


def _run_step1(data_root: Path) -> None:
    """Provision acme and globex, store fix-adder in each, and check WAL."""
    provision(data_root, TenantId("acme"), "Acme")
    provision(data_root, TenantId("globex"), "Globex")
    init_task(data_root, TenantId("acme"), "fix-adder")
    init_task(data_root, TenantId("globex"), "fix-adder")
    journal, synchronous, _busy = read_pragmas(data_root, TenantId("acme"))
    if journal != "wal" or synchronous != 2:
        raise OSError(f"expected wal and 2, got {journal} and {synchronous}")
    acme = get_task(data_root, TenantId("acme"), "fix-adder")
    globex = get_task(data_root, TenantId("globex"), "fix-adder")
    if acme.id != "fix-adder" or globex.id != "fix-adder":
        raise OSError("fix-adder missing from a tenant file")
    acme_db = data_root / "tenants" / "acme" / "engine.db"
    globex_db = data_root / "tenants" / "globex" / "engine.db"
    if not acme_db.is_file() or not globex_db.is_file():
        raise OSError("expected two engine.db files")
    if acme_db.resolve() == globex_db.resolve():
        raise OSError("tenant databases resolved to one file")


if __name__ == "__main__":
    raise SystemExit(main())
