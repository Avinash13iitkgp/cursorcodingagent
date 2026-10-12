"""Spawn a child and sleep in the jail's session.

This process must not call setsid. start_new_session on the jail puts this
process in its own group, and the forked child stays in that group so killpg
reaches both.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> None:
    """Write the leader pid and the child pid, then sleep.

    Args:
        None. The destination path is sys.argv[1].

    Returns:
        None. The process is expected to be killed by the jail.
    """
    destination = Path(sys.argv[1])
    child = os.fork()
    if child == 0:
        time.sleep(30)
        os._exit(0)
    destination.write_text(f"{os.getpid()}\n{child}\n", encoding="utf-8")
    time.sleep(30)


if __name__ == "__main__":
    main()
