"""Flat work directory for the ReAct loop.

The editor and the loop call this Workspace port and do not take a raw base
path. resolve keeps every path under the work root. write uses a same-directory
temp file and os.replace. cleanup_tmp deletes leftover *.tmp files that stay
inside the root and skips a symlink that resolves outside it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


class PlainWorkspace:
    """One task work root as a normal directory."""

    def __init__(self, root: Path) -> None:
        """Bind the workspace to one directory.

        Args:
            root: Directory that contains the editable files. A write creates
                missing parents. Callers pass the task work root.
        """
        self._root = root

    def resolve(self, relative_path: str) -> Path:
        """Return relative_path inside this root.

        Args:
            relative_path: Slash-separated path under the root.

        Returns:
            The resolved path.

        Raises:
            ValueError: The path is absolute or leaves the root.
        """
        if not relative_path or Path(relative_path).is_absolute():
            raise ValueError("path must stay under the work root")
        root = self._root.resolve()
        candidate = (root / relative_path).resolve()
        candidate.relative_to(root)
        return candidate

    def read(self, relative_path: str) -> bytes:
        """Read one file.

        Args:
            relative_path: Path relative to the work root.

        Returns:
            The file bytes.

        Raises:
            ValueError: The path escapes the work root.
            FileNotFoundError: The file does not exist.
        """
        return self.resolve(relative_path).read_bytes()

    def write(self, relative_path: str, data: bytes) -> None:
        """Write data via a same-directory temp file and os.replace.

        Args:
            relative_path: Path relative to the work root.
            data: Bytes that replace the file.

        Returns:
            None.

        Raises:
            ValueError: The path escapes the work root.
        """
        target = self.resolve(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".", suffix=".tmp", dir=target.parent)
        tmp = Path(name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, target)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def cleanup_tmp(self) -> None:
        """Delete leftover ``*.tmp`` files that resolve inside this root.

        Returns:
            None.
        """
        root = self._root.resolve()
        if not root.exists():
            return
        for path in root.rglob("*.tmp"):
            if not path.is_file():
                continue
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            path.unlink()
