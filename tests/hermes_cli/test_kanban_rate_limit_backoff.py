"""Regression coverage for rate-limit cooldown ordering and retry behavior."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_guard as kbg
from hermes_cli import kanban_ops


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_rate_limit_timestamp_ordering_and_retry_estimate(kanban_home, monkeypatch):
    """Mixed timestamp formats and repeated quota outcomes produce ordered backoff."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = int(time.time())
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="mixed timestamps", assignee="a")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'rate_limited', 'rate_limited', ?, ?)",
            (tid, now - 601, datetime.fromtimestamp(now - 600, UTC).isoformat()),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'rate_limited', 'rate_limited', ?, ?)", (tid, now - 20, now - 10),
        )
        conn.commit()
        monkeypatch.setattr(kb.time, "time", lambda: now)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        details = kbg.rate_limit_retry_details(conn, tid, now=now)
        assert details == {"retry_at": now + 590, "retry_after_seconds": 590,
                           "consecutive_rate_limits": 2}

        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'review_requested', 'review_requested', ?, ?)", (tid, now - 5, now - 1),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid, lane="review") is None
        assert kbg.rate_limit_retry_details(conn, tid, now=now) is None


@pytest.mark.parametrize("ended_at", [None, "invalid-timestamp", float("inf")])
def test_unreadable_rate_limit_timestamp_fails_closed(kanban_home, monkeypatch, ended_at):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="unreadable end", assignee="a")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'rate_limited', 'rate_limited', 1, ?)", (tid, ended_at),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) == "invalid_rate_limit_timestamp"


def test_older_unreadable_rate_limit_stays_guarded_after_progress(kanban_home, monkeypatch):
    """An unorderable historical quota run cannot safely be assumed superseded."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = int(time.time())
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="uncertain history", assignee="a")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'rate_limited', 'rate_limited', ?, NULL)", (tid, now - 600),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'review_requested', 'review_requested', ?, ?)", (tid, now - 60, now - 30),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid, lane="review") == "invalid_rate_limit_timestamp"
        assert kbg.rate_limit_retry_details(conn, tid, now=now)["retry_at"] is None


def test_zero_cooldown_disables_unreadable_timestamp_hold(kanban_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="operator override", assignee="a")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'rate_limited', 'rate_limited', 1, NULL)", (tid,),
        )
        conn.commit()
        assert kbd.check_respawn_guard(conn, tid) is None


def test_repeated_invalid_timestamp_guard_event_is_deduplicated(
    kanban_home, monkeypatch, all_assignees_spawnable,
):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="invalid event", assignee="a")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'a', 'rate_limited', 'rate_limited', 1, 'bad')", (tid,),
        )
        conn.commit()
        for _ in range(3):
            result = kbd.dispatch_once(
                conn, spawn_fn=lambda *_: pytest.fail("invalid timestamp must not spawn"),
            )
            assert (tid, "invalid_rate_limit_timestamp") in result.respawn_guarded
        count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='respawn_guarded'", (tid,),
        ).fetchone()[0]
        assert count == 1
        kb.add_comment(conn, tid, "operator", "Investigating timestamp")
        kbd.dispatch_once(conn, spawn_fn=lambda *_: pytest.fail("invalid timestamp must not spawn"))
        count = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='respawn_guarded'", (tid,),
        ).fetchone()[0]
        assert count == 2


def test_dispatch_json_reports_rate_limit_retry_estimate(
    kanban_home, monkeypatch, capsys, all_assignees_spawnable,
):
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    now = int(time.time())
    monkeypatch.setattr(kb.time, "time", lambda: now)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="report quota", assignee="a")
        for end in (now - 700, now - 400):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'a', 'rate_limited', 'rate_limited', ?, ?)", (tid, end - 1, end),
            )
        conn.commit()
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=True)
    assert kanban_ops._cmd_dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    item = next(item for item in payload["respawn_guarded"] if item["task_id"] == tid)
    assert item == {"task_id": tid, "reason": "rate_limit_cooldown", "retry_at": now + 200,
                    "retry_after_seconds": 200, "consecutive_rate_limits": 2}
