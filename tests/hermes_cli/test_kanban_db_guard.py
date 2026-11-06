"""Respawn guard: the reasons a card must not be re-spawned.

Split out of ``tests/hermes_cli/test_kanban_db.py`` when the guard moved to
``hermes_cli/kanban_db_guard.py``; that file was already over its line budget.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_guard as kbg


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_respawn_guard_ignores_stale_quota_stamp_after_progress_outcome(kanban_home):
    """A quota stamp from an EARLIER run must not park a card that has since
    made forward progress.

    The rate-limited requeue path stamps ``last_failure_error`` WITHOUT
    incrementing ``consecutive_failures`` — the stamp is informational. But
    the guard read it as a live blocker, and it did so BEFORE the review-lane
    early return. A card that was rate-limited 20 times and then successfully
    handed off for review (``review_requested``) was therefore parked forever:
    the reviewer could never spawn, and the card re-guarded itself on every
    dispatch tick (384 ``respawn_guarded`` events on one card).
    """
    now = int(time.time())

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="progress", assignee="a")

        # An earlier rate-limited run wrote the quota-flavoured stamp.
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at) VALUES (?, 'a', 'rate_limited', "
            "'rate_limited', ?, ?)",
            (tid, now - 360, now - 60),
        )
        conn.commit()

        # The guard sees the stale stamp with the rate-limited run as the
        # latest one -> deferred by the rate-limit cooldown (the pre-existing,
        # correct behaviour for a card that has NOT moved on).
        assert kbg.check_respawn_guard(conn, tid) == "rate_limit_cooldown"

        # Now the card progresses: a later run hands it off for review.
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at) VALUES (?, 'a', 'review_requested', "
            "'review_requested', ?, ?)",
            (tid, now - 50, now - 10),
        )
        conn.execute(
            "UPDATE tasks SET status='review', current_run_id=NULL, "
            "claim_lock=NULL, claim_expires=NULL, worker_pid=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()

        # The stale quota stamp must NOT park it: the reviewer has to be
        # able to spawn, and the ready lane must not trip on it either.
        assert kbg.check_respawn_guard(conn, tid, lane="review") is None
        assert kbg.check_respawn_guard(conn, tid, lane="ready") is None


def test_respawn_guard_still_parks_quota_stamp_without_progress(kanban_home):
    """The relaxation must not weaken the guard: a quota/auth stamp whose
    latest run did NOT make progress still parks the card."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="stuck", assignee="a")
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued", tid),
        )
        # Latest run made no progress -> the stamp is the live blocker.
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, "
            "started_at, ended_at) VALUES (?, 'a', 'blocked', 'blocked', 100, 200)",
            (tid,),
        )
        conn.commit()
        assert kbg.check_respawn_guard(conn, tid) == "blocker_auth"

        # No run at all -> still parked on the stamp.
        tid2 = kb.create_task(conn, title="stuck2", assignee="a")
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            ("401 auth failed", tid2),
        )
        conn.commit()
        assert kbg.check_respawn_guard(conn, tid2) == "blocker_auth"
