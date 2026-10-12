"""Command line for tenants, leases, the jail, edits, plans, run-task, and demos.

This module is the process edge. It parses arguments, opens one tenant
database, and calls LeaseService for acquire, renew, and release. jail and
replace do not take a tenant. run-task wires the loop through composition.
plan stores a proposed tree for a prompt, accept marks it accepted, show-plan
prints it, and apply runs the loop only after that accept. demo starts the
scenarios. The scripted self-test procedure lives in agent_engine.self_test.
Invalid ids, tenant boundary errors, lease errors, a plan that is not
accepted, and edit errors exit non-zero.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_engine.adapters.exec.jail import Jail, tool_env
from agent_engine.adapters.sqlite.db import open_tenant_db
from agent_engine.adapters.sqlite.lease_store import SqliteLeaseStore
from agent_engine.adapters.sqlite.tenant_store import (
    get_task,
    init_task,
    provision,
    read_pragmas,
)
from agent_engine.adapters.workspace.edits import FlatWorkspace, OptimisticEditor
from agent_engine.application.leases import LeaseService, SystemClock
from agent_engine.composition import (
    accept_plan,
    apply_plan,
    format_report,
    propose_plan,
    run_task,
    show_plan,
)
from agent_engine.demo import run_scenarios
from agent_engine.domain.errors import (
    AmbiguousEdit,
    FenceRejected,
    LeaseNotHeld,
    StaleContent,
    TaskNotFound,
    TenantMismatch,
    ToolTimeout,
    UnknownTenant,
)
from agent_engine.domain.models import Lease, TenantId, TextEdit
from agent_engine.self_test import run_self_test


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

    _add_lease_command(sub, "acquire", ttl=True)
    _add_lease_command(sub, "renew", ttl=True, token=True)
    _add_lease_command(sub, "release", token=True)
    show_lease = sub.add_parser("show-lease")
    show_lease.add_argument("--tenant", required=True)
    show_lease.add_argument("--task", required=True)

    jail_cmd = sub.add_parser("jail")
    jail_cmd.add_argument("--timeout", type=float, required=True)
    jail_cmd.add_argument("argv", nargs=argparse.REMAINDER)

    replace_cmd = sub.add_parser("replace")
    replace_cmd.add_argument("--root", required=True)
    replace_cmd.add_argument("--path", required=True)
    replace_cmd.add_argument("--expected-sha256", required=True)
    replace_cmd.add_argument("--old", required=True)
    replace_cmd.add_argument("--new", required=True)

    run = sub.add_parser("run-task")
    run.add_argument("--tenant", required=True)
    run.add_argument("--task", required=True)
    run.add_argument("--llm", choices=("canned", "live"), default="live")
    run.add_argument("--tool-choice")

    plan_cmd = sub.add_parser("plan")
    plan_cmd.add_argument("--tenant", required=True)
    plan_cmd.add_argument("--task", required=True)
    plan_cmd.add_argument("--prompt", required=True)
    plan_cmd.add_argument("--llm", choices=("canned", "live"), default="live")

    accept_cmd = sub.add_parser("accept")
    accept_cmd.add_argument("--tenant", required=True)
    accept_cmd.add_argument("--task", required=True)

    show_plan_cmd = sub.add_parser("show-plan")
    show_plan_cmd.add_argument("--tenant", required=True)
    show_plan_cmd.add_argument("--task", required=True)

    apply_cmd = sub.add_parser("apply")
    apply_cmd.add_argument("--tenant", required=True)
    apply_cmd.add_argument("--task", required=True)
    apply_cmd.add_argument("--llm", choices=("canned", "live"), default="live")
    apply_cmd.add_argument("--max-steps", type=int, default=32)
    apply_cmd.add_argument("--checks-cmd", action="append", default=[])

    demo_cmd = sub.add_parser("demo")
    demo_cmd.add_argument("--tenant", required=True)
    demo_cmd.add_argument("--scenario", type=int)
    demo_cmd.add_argument("--all", action="store_true", dest="all_scenarios")

    try:
        parsed = parser.parse_args(rest)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2
    try:
        if parsed.command == "self-test":
            live_status = _missing_token_if_live(parsed.live)
            if live_status != 0:
                return live_status
            return run_self_test(
                data_root, parsed.step, parsed.through, live=parsed.live
            )
        if parsed.command == "jail":
            return _run_jail(parsed)
        if parsed.command == "replace":
            return _run_replace(parsed)
        tenant_id = TenantId(parsed.tenant)
        if parsed.command == "provision":
            path = provision(data_root, tenant_id, parsed.display_name)
            print(path)
            return 0
        if parsed.command == "init-task":
            task = init_task(data_root, tenant_id, parsed.task)
            print(task.id)
            return 0
        if parsed.command == "pragma":
            journal, synchronous, _busy = read_pragmas(data_root, tenant_id)
            print(journal)
            print(synchronous)
            return 0
        if parsed.command == "show-lease":
            _print_lease(get_task(data_root, tenant_id, parsed.task))
            return 0
        if parsed.command == "run-task":
            report = run_task(
                data_root,
                tenant_id,
                parsed.task,
                llm=parsed.llm,
                tool_choice=parsed.tool_choice,
            )
            print(format_report(report))
            return 0
        if parsed.command in {"plan", "accept", "show-plan", "apply"}:
            return _run_plan_command(data_root, tenant_id, parsed)
        if parsed.command == "demo":
            return _run_demo(data_root, parsed)
        if parsed.command in {"acquire", "renew", "release"}:
            return _run_lease_command(data_root, tenant_id, parsed)
        print(f"unknown command {parsed.command}", file=sys.stderr)
        return 2
    except (
        UnknownTenant,
        TenantMismatch,
        TaskNotFound,
        FenceRejected,
        LeaseNotHeld,
        StaleContent,
        AmbiguousEdit,
        ToolTimeout,
        ValueError,
    ) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _run_plan_command(
    data_root: Path, tenant_id: TenantId, parsed: argparse.Namespace
) -> int:
    if parsed.command == "plan":
        print(
            propose_plan(
                data_root, tenant_id, parsed.task, parsed.prompt, llm=parsed.llm
            )
        )
        return 0
    if parsed.command == "accept":
        accept_plan(data_root, tenant_id, parsed.task)
        print("accepted")
        return 0
    if parsed.command == "show-plan":
        print(show_plan(data_root, tenant_id, parsed.task))
        return 0
    report = apply_plan(
        data_root,
        tenant_id,
        parsed.task,
        llm=parsed.llm,
        max_steps=parsed.max_steps,
        checks_cmd=tuple(parsed.checks_cmd),
    )
    print(format_report(report))
    return 0


def _run_demo(data_root: Path, parsed: argparse.Namespace) -> int:
    if parsed.tenant != "acme":
        print("demos run as acme", file=sys.stderr)
        return 2
    if parsed.all_scenarios:
        numbers: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
        llm = "live"
    elif parsed.scenario is None:
        print("demo requires --scenario or --all", file=sys.stderr)
        return 2
    else:
        numbers = (parsed.scenario,)
        llm = "canned" if parsed.scenario in (4, 6) else "live"
    try:
        run_scenarios(data_root, numbers, llm=llm)
    except (OSError, ValueError) as exc:
        print("FAIL")
        print(exc, file=sys.stderr)
        return 1
    print("PASS")
    return 0


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


def _add_lease_command(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    ttl: bool = False,
    token: bool = False,
) -> None:
    command = sub.add_parser(name)
    command.add_argument("--tenant", required=True)
    command.add_argument("--task", required=True)
    command.add_argument("--worker-id", required=True)
    command.add_argument("--now", type=float)
    if ttl:
        command.add_argument("--ttl", type=float, required=True)
    if token:
        command.add_argument("--token", type=int, required=True)


def _run_jail(parsed: argparse.Namespace) -> int:
    argv = list(parsed.argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    result = Jail().run(argv, cwd=Path.cwd(), timeout_s=parsed.timeout, env=tool_env())
    if result.returncode != 0:
        sys.stderr.write(result.stderr or "")
        return int(result.returncode)
    sys.stdout.write(result.stdout or "")
    return 0


def _run_replace(parsed: argparse.Namespace) -> int:
    workspace = FlatWorkspace(Path(parsed.root))
    result = OptimisticEditor().apply(
        workspace,
        TextEdit(parsed.path, parsed.expected_sha256, parsed.old, parsed.new),
    )
    print(result.status)
    return 0


def _run_lease_command(data_root: Path, tenant_id: TenantId, parsed: argparse.Namespace) -> int:
    now = SystemClock().now() if parsed.now is None else parsed.now
    if parsed.command == "acquire":
        lease = _lease_call(
            data_root,
            tenant_id,
            lambda service: service.acquire(parsed.task, parsed.worker_id, now, parsed.ttl),
        )
        print(lease.fencing_token)
        return 0
    held = Lease(tenant_id, parsed.task, parsed.worker_id, parsed.token, now)
    if parsed.command == "renew":
        lease = _lease_call(
            data_root,
            tenant_id,
            lambda service: service.renew(held, now, parsed.ttl),
        )
        print(lease.fencing_token)
        return 0
    _lease_call(data_root, tenant_id, lambda service: service.release(held, now))
    print("released")
    return 0


def _lease_call(data_root: Path, tenant_id: TenantId, fn):
    connection = open_tenant_db(data_root, tenant_id)
    try:
        service = LeaseService(tenant_id, SqliteLeaseStore(connection), SystemClock())
        return fn(service)
    finally:
        connection.close()


def _print_lease(task) -> None:
    owner = "null" if task.lease_owner is None else task.lease_owner
    print(f"fencing_token={task.fencing_token}")
    print(f"lease_owner={owner}")
    print(f"lease_expires_at={_format_expiry(task.lease_expires_at)}")


def _format_expiry(value: float | None) -> str:
    if value is None:
        return "null"
    if float(value).is_integer():
        return str(int(value))
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
