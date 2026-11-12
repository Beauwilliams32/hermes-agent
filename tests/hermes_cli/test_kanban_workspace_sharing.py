"""Cross-board workspace sharing must not be decided by a stray board DB.

A schema-less ``kanban.db`` (0 bytes, or any readable file without a
``tasks`` table) sitting next to the real boards used to make the whole
cross-board workspace scan fail closed: ``_workspace_in_use_by_other()``
returned ``"unknown"`` for EVERY task on EVERY board, so post-completion
workspace cleanup was deferred globally and ~7.5 GB of scratch accumulated.

Two layers are covered here:

* the sibling scan only considers genuine board directories (one holding
  ``board.json``), so a stray ``kanban.db`` in ``boards/default/`` or
  ``boards/_archived/`` is not treated as a board;
* a readable sibling DB without a ``tasks`` table references no workspace
  and must not defer cleanup, while a genuinely CORRUPT file still fails
  closed (``"unknown"``).

The second layer also covers the creation path: ``connect()`` refuses to
create a board database at a path that is not a board's own database, so a
stray file cannot be recreated by a stray ``sqlite3.connect()``.
"""
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_workspace as kbw
from hermes_cli import kanban_transfer


@pytest.fixture
def board(tmp_path, monkeypatch):
    """Isolate every Kanban path into a temp home (never ``~/.hermes``)."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    return tmp_path


def _board_dir(slug: str) -> Path:
    d = kb.board_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    (d / "board.json").write_text(f'{{"slug": "{slug}"}}\n', encoding="utf-8")
    return d


def test_sibling_scan_ignores_directories_that_are_not_boards(board):
    """``boards/default/`` and ``boards/_archived/`` are not boards: their
    stray ``kanban.db`` files must not enter the cross-board scan."""
    root = kb.boards_root()
    for stray_dir in ("default", "_archived"):
        d = root / stray_dir
        d.mkdir(parents=True, exist_ok=True)
        (d / "kanban.db").write_bytes(b"")
    real = _board_dir("real")
    (real / "kanban.db").write_bytes(b"")

    with kbc.connect_closing() as conn:
        found = {p.resolve() for p in kbw._sibling_board_db_files(conn)}

    assert (root / "default" / "kanban.db").resolve() not in found
    assert (root / "_archived" / "kanban.db").resolve() not in found
    assert (real / "kanban.db").resolve() in found


def test_schema_less_sibling_db_does_not_defer_cleanup(board):
    """A readable sibling DB without a ``tasks`` table references no
    workspace, so cleanup must not be deferred."""
    other = _board_dir("other")
    (other / "kanban.db").write_bytes(b"")  # 0-byte: readable, no tables

    with kbc.connect_closing() as conn:
        reason = kbw._workspace_in_use_by_other(conn, "t_missing", Path("/tmp/ws"))

    assert reason is None


def test_valid_sibling_db_without_tasks_table_does_not_defer_cleanup(board):
    """Same for a well-formed SQLite file that simply has no ``tasks`` table."""
    other = _board_dir("other")
    with sqlite3.connect(str(other / "kanban.db")) as oconn:
        oconn.execute("CREATE TABLE unrelated (id INTEGER)")

    with kbc.connect_closing() as conn:
        reason = kbw._workspace_in_use_by_other(conn, "t_missing", Path("/tmp/ws"))

    assert reason is None


def test_corrupt_sibling_db_still_fails_closed(board):
    """A file that is not a database at all stays unreadable: cleanup is
    deferred rather than guessed."""
    other = _board_dir("other")
    (other / "kanban.db").write_bytes(b"this is definitely not a sqlite database")

    with kbc.connect_closing() as conn:
        reason = kbw._workspace_in_use_by_other(conn, "t_missing", Path("/tmp/ws"))

    assert reason == "unknown"


def test_valid_sibling_db_is_still_consulted_for_sharing(board):
    """The fix must not blind the scan: a real sibling board that uses the
    same workspace still reports ``"shared"``."""
    ws = Path("/tmp/shared-ws")
    other = _board_dir("other")
    with kbc.connect_closing(board="other") as oconn:
        kb.create_task(oconn, title="live", workspace_path=str(ws))
    assert (other / "kanban.db").is_file()

    with kbc.connect_closing() as conn:
        reason = kbw._workspace_in_use_by_other(conn, "t_missing", ws)

    assert reason == "shared"


def test_connect_refuses_to_create_a_stray_board_db(board):
    """``sqlite3.connect()`` creates a missing file, so an open at a path
    that is not a board's database used to leave a 0-byte stub behind."""
    root = kb.boards_root()
    stray = root / "_archived" / "kanban.db"
    stray.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="not a board"):
        kbc.connect(db_path=stray)
    assert not stray.exists()


def test_connect_refuses_to_create_a_default_board_db_in_boards_root(board):
    """The default board's database lives at ``<home>/kanban.db``; a copy
    under ``boards/default/`` is a stray by construction."""
    root = kb.boards_root()
    stray = root / "default" / "kanban.db"
    stray.parent.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="default board"):
        kbc.connect(db_path=stray)
    assert not stray.exists()


def test_create_board_still_creates_its_own_database(board):
    """Legitimate board creation is untouched."""
    kb.create_board("fresh")
    assert (kb.board_dir("fresh") / "kanban.db").is_file()
    assert kb.board_exists("fresh")


def test_snapshot_db_refuses_a_missing_source(board, tmp_path):
    """Export must fail loudly instead of leaving a 0-byte ``kanban.db``."""
    missing = tmp_path / "gone.db"
    with pytest.raises(FileNotFoundError):
        kanban_transfer._snapshot_db(missing, tmp_path / "out.db")
    assert not missing.exists()
