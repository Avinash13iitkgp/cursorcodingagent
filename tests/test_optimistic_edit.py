"""Optimistic edits write only when the preimage hash matches a single span."""

import hashlib
from pathlib import Path

import pytest

from agent_engine.adapters.workspace.edits import FlatWorkspace, OptimisticEditor
from agent_engine.domain.errors import AmbiguousEdit, StaleContent
from agent_engine.domain.models import EditResult, TextEdit


class _Guard:
    """Workspace double whose write fails the test if the editor opens the file."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self.writes = 0

    def resolve(self, relative_path: str) -> Path:
        if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ValueError("path must stay under the work root")
        return Path(relative_path)

    def read(self, relative_path: str) -> bytes:
        self.resolve(relative_path)
        return self._data

    def write(self, relative_path: str, data: bytes) -> None:
        del relative_path, data
        self.writes += 1
        raise AssertionError("opened for write")

    def cleanup_tmp(self) -> None:
        return None


def test_replace_matching_hash_swaps_bytes(tmp_path: Path) -> None:
    """A matching preimage hash replaces the one occurrence of old."""
    root = tmp_path / "work"
    root.mkdir()
    original = b"return a - b\n"
    (root / "adder.py").write_bytes(original)
    result = OptimisticEditor().apply(*_edit(root, original, "return a - b", "return a + b"))
    assert result == EditResult("applied")
    assert (root / "adder.py").read_bytes() == b"return a + b\n"


def test_stale_hash_writes_nothing(tmp_path: Path) -> None:
    """A mismatched hash leaves the file bytes untouched."""
    root = tmp_path / "work"
    root.mkdir()
    original = b"return a - b\n"
    stale = original + b"x"
    (root / "adder.py").write_bytes(stale)
    digest = hashlib.sha256(original).hexdigest()
    with pytest.raises(StaleContent):
        OptimisticEditor().apply(
            FlatWorkspace(root),
            TextEdit("adder.py", digest, "return a - b", "return a + b"),
        )
    assert (root / "adder.py").read_bytes() == stale
    absent = b"nothing to replace\n"
    (root / "adder.py").write_bytes(absent)
    with pytest.raises(StaleContent):
        OptimisticEditor().apply(
            FlatWorkspace(root),
            TextEdit("adder.py", hashlib.sha256(absent).hexdigest(), "return a - b", "return a + b"),
        )
    assert (root / "adder.py").read_bytes() == absent


def test_already_applied_does_not_open_for_write() -> None:
    """A file that is already the postimage is not opened for write."""
    preimage = b"return a - b\n"
    postimage = b"return a + b\n"
    guard = _Guard(postimage)
    result = OptimisticEditor().apply(
        guard,
        TextEdit("adder.py", hashlib.sha256(preimage).hexdigest(), "return a - b", "return a + b"),
    )
    assert result == EditResult("already_applied")
    assert guard.writes == 0


def test_two_matches_raise_ambiguous_edit(tmp_path: Path) -> None:
    """Two matches raise AmbiguousEdit and leave both spans in place."""
    root = tmp_path / "work"
    root.mkdir()
    body = b"return a - b\nreturn a - b\n"
    (root / "adder.py").write_bytes(body)
    with pytest.raises(AmbiguousEdit):
        OptimisticEditor().apply(*_edit(root, body, "return a - b", "return a + b"))
    assert (root / "adder.py").read_bytes() == body
    with pytest.raises(ValueError):
        OptimisticEditor().apply(
            FlatWorkspace(root),
            TextEdit("adder.py", hashlib.sha256(body).hexdigest(), "", "x"),
        )


def test_dotdot_and_absolute_paths_write_nothing(tmp_path: Path) -> None:
    """Paths that leave the work root raise before any write."""
    root = tmp_path / "work"
    root.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_bytes(b"stay")
    editor = OptimisticEditor()
    workspace = FlatWorkspace(root)
    for path in ("../outside.py", str(outside)):
        with pytest.raises(ValueError):
            editor.apply(workspace, TextEdit(path, "unused", "stay", "go"))
    assert outside.read_bytes() == b"stay"


def test_acme_edit_leaves_globex_tree_unchanged(tmp_path: Path) -> None:
    """An edit under acme leaves the globex database and task tree byte-identical."""
    acme = tmp_path / "tenants" / "acme" / "tasks" / "fix-adder"
    globex_root = tmp_path / "tenants" / "globex"
    globex = globex_root / "tasks" / "fix-adder"
    acme.mkdir(parents=True)
    globex.mkdir(parents=True)
    original = b"return a - b\n"
    (acme / "adder.py").write_bytes(original)
    (globex / "adder.py").write_bytes(original)
    (globex_root / "engine.db").write_bytes(b"globex-db")
    before = _snapshot(globex_root)
    result = OptimisticEditor().apply(*_edit(acme, original, "return a - b", "return a + b"))
    assert result.status == "applied"
    assert (acme / "adder.py").read_bytes() == b"return a + b\n"
    assert _snapshot(globex_root) == before


def test_resume_removes_orphan_tmp_files(tmp_path: Path) -> None:
    """Resume deletes leftover tmp files inside the work root and keeps sources."""
    root = tmp_path / "work"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "keep.py").write_text("ok", encoding="utf-8")
    (root / "orphan.tmp").write_text("x", encoding="utf-8")
    (nested / "also.tmp").write_text("y", encoding="utf-8")
    outside = tmp_path / "escaped.tmp"
    outside.write_text("safe", encoding="utf-8")
    (root / "link.tmp").symlink_to(outside)
    OptimisticEditor().resume(FlatWorkspace(root))
    FlatWorkspace(tmp_path / "missing").cleanup_tmp()
    assert not (root / "orphan.tmp").exists()
    assert not (nested / "also.tmp").exists()
    assert (root / "keep.py").read_text(encoding="utf-8") == "ok"
    assert outside.read_text(encoding="utf-8") == "safe"


def _edit(root: Path, body: bytes, old: str, new: str) -> tuple[FlatWorkspace, TextEdit]:
    return (
        FlatWorkspace(root),
        TextEdit("adder.py", hashlib.sha256(body).hexdigest(), old, new),
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
