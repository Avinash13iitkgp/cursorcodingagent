"""Optimistic text edits against a Workspace port.

apply hashes the bytes from Workspace.read and compares that digest to the
caller's expected_sha256. It does not substitute a hash it computed for the
one the caller sent. old must match once. Zero matches, or a different hash,
raise StaleContent and write nothing. Several matches raise AmbiguousEdit and
write nothing. When the file is already the postimage of that preimage, apply
returns already_applied and does not write. A real change writes a
same-directory temp file and os.replace's it into place.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from agent_engine.domain.errors import AmbiguousEdit, StaleContent
from agent_engine.domain.models import EditResult, TextEdit
from agent_engine.domain.ports import Workspace


class FlatWorkspace:
    """A work root that is one directory. Step 4's plain adapter can replace it."""

    def __init__(self, root: Path) -> None:
        """Bind the workspace to one directory.

        Args:
            root: Directory that contains the editable files. It is created
                when a write needs it. Callers pass the task work root.
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


class OptimisticEditor:
    """Apply one text replacement when the preimage hash still matches."""

    def apply(self, workspace: Workspace, edit: TextEdit) -> EditResult:
        """Replace edit.old once, or report that the postimage is already present.

        Args:
            workspace: Work root that owns the file.
            edit: Relative path, expected preimage hash, and the two snippets.

        Returns:
            applied when the file was replaced. already_applied when the bytes
            are already the postimage. Neither result rewrites expected_sha256.

        Raises:
            ValueError: The path escapes the work root, or old is empty.
            StaleContent: The hash does not match, or old matches zero times.
            AmbiguousEdit: old matches more than once.
        """
        workspace.resolve(edit.path)
        if edit.old == "":
            raise ValueError("old must be non-empty")
        data = workspace.read(edit.path)
        old = edit.old.encode("utf-8")
        new = edit.new.encode("utf-8")
        if _is_postimage(data, old, new, edit.expected_sha256):
            return EditResult("already_applied")
        digest = hashlib.sha256(data).hexdigest()
        if digest != edit.expected_sha256:
            raise StaleContent(edit.path)
        count = data.count(old)
        if count == 0:
            raise StaleContent(edit.path)
        if count != 1:
            raise AmbiguousEdit(edit.path)
        workspace.write(edit.path, data.replace(old, new, 1))
        return EditResult("applied")

    def resume(self, workspace: Workspace) -> None:
        """Delete leftover temp files in the work root.

        Args:
            workspace: Work root that may contain ``*.tmp`` files from a crash.

        Returns:
            None.
        """
        workspace.cleanup_tmp()


def _is_postimage(data: bytes, old: bytes, new: bytes, expected_sha256: str) -> bool:
    if new == b"":
        return False
    start = 0
    while True:
        index = data.find(new, start)
        if index < 0:
            return False
        restored = data[:index] + old + data[index + len(new) :]
        restored_hash = hashlib.sha256(restored).hexdigest()
        if restored_hash == expected_sha256 and restored.replace(old, new, 1) == data:
            return True
        start = index + 1
