"""Operator checks, run in the jail.

The command list is checks/commands, one argv per line, outside the writable
work tree. When that file is absent the Python fixture is the default list:
python -m compileall on the work tree, then python -m unittest on the check
file. That fixture is the default, not the only language. The child
environment is PATH plus PYTHONPATH pointed at the work tree. A missing
compiler is a failed result. This module does not read the transcript and
does not write tasks.status.
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from agent_engine.adapters.exec.jail import Jail, tool_env
from agent_engine.domain.errors import ToolTimeout


class ExternalVerifier:
    """Run checks/commands. The model cannot supply the argv."""

    def __init__(self, checks_dir: Path, jail: Jail, timeout_s: float = 30) -> None:
        """Bind the runner to one checks directory.

        Args:
            checks_dir: Directory that holds commands and the check file.
                It is not the task work root.
            jail: Process-group runner. The child env is built here.
            timeout_s: Seconds each command may run before ToolTimeout.
        """
        self._checks_dir = checks_dir
        self._jail = jail
        self._timeout_s = timeout_s

    def run(self, work_root: Path) -> tuple[bool, str, str]:
        """Compile and test the work tree.

        Args:
            work_root: Task directory. A test_adder.py here that differs from
                checks/test_adder.py fails the run before any command.

        Returns:
            ok, stdout, and stderr. Timeout and non-zero exits are ok False.
            This method does not change tasks.status.
        """
        if self._check_file_differs(work_root):
            return False, "", "check file differs"
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        env = tool_env()
        env["PYTHONPATH"] = str(work_root)
        for argv in self._commands(work_root):
            try:
                result = self._jail.run(
                    argv, cwd=work_root, timeout_s=self._timeout_s, env=env
                )
            except ToolTimeout as exc:
                return False, "".join(stdout_parts), str(exc)
            stdout_parts.append(result.stdout or "")
            stderr_parts.append(result.stderr or "")
            if result.returncode != 0:
                return False, "".join(stdout_parts), "".join(stderr_parts)
        return True, "".join(stdout_parts), "".join(stderr_parts)

    def _check_file_differs(self, work_root: Path) -> bool:
        work_test = work_root / "test_adder.py"
        canon = self._checks_dir / "test_adder.py"
        if not work_test.is_file() or not canon.is_file():
            return False
        return work_test.read_bytes() != canon.read_bytes()

    def _commands(self, work_root: Path) -> list[list[str]]:
        commands = self._checks_dir / "commands"
        if commands.is_file():
            parsed: list[list[str]] = []
            for line in commands.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    parsed.append(shlex.split(line))
            return parsed
        return [
            [sys.executable, "-m", "compileall", "-q", str(work_root)],
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(self._checks_dir),
                "-p",
                "test_adder.py",
            ],
        ]
