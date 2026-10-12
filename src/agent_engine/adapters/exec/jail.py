"""Process-group jail.

run starts the child with start_new_session so its process group is not the
caller's. On timeout it killpg's that group, waits, and raises ToolTimeout.
The child environment is exactly the mapping passed in. This module does not
copy os.environ and does not add keys the caller left out.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from agent_engine.domain.errors import ToolTimeout


def tool_env() -> dict[str, str]:
    """Return the only environment a jailed tool should receive.

    Returns:
        PATH and PYTHONPATH copied from the current process when those
        variables are set. AGENT_LLM_API_KEY is not included.
    """
    env: dict[str, str] = {}
    for key in ("PATH", "PYTHONPATH"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


class Jail:
    """Run one command in a new session and kill the whole group on timeout."""

    def run(
        self,
        argv: list[str],
        *,
        cwd: Path,
        timeout_s: float,
        env: dict[str, str],
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run argv in its own process group.

        Args:
            argv: Command and arguments. The jail does not use a shell.
            cwd: Working directory of the child.
            timeout_s: Seconds before the process group is killed.
            env: Exact environment of the child.
            stdin: Text written to the child, then closed. None leaves stdin
                inherited, which is the behavior of existing callers.

        Returns:
            The completed process when the command finishes in time.

        Raises:
            ValueError: argv is empty or timeout_s is not positive.
            ToolTimeout: The command exceeded timeout_s. The group was killed.
        """
        if not argv:
            raise ValueError("jail requires a command")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=None if stdin is None else subprocess.PIPE,
            text=True,
        )
        try:
            if stdin is None:
                stdout, stderr = process.communicate(timeout=timeout_s)
            else:
                stdout, stderr = process.communicate(stdin, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(process)
            raise ToolTimeout(f"command exceeded {timeout_s}s") from None
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _kill_group(process: subprocess.Popen[str]) -> None:
    if process.pid:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()
