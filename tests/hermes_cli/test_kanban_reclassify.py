"""``kanban reclassify`` re-types an already-blocked card's wait, in place.

Regression cover for the stale ``block_kind`` leak: a card parked by the
breaker kept a ``dependency`` marker from an earlier lane, so it looked like
a parent wait while every parent was already done. The operation corrects the
KIND of the wait without releasing the card: status, failure counter,
recurrence counter and run history all survive, the card is not promoted, and
the audit event stays outside the notifier's TERMINAL_KINDS so nobody is woken.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_reclassify as kbr


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _parked(conn, kind: str = "dependency") -> str:
    """A breaker-parked card still carrying a stale ``dependency`` marker.

    Mirrors the real card: ``block_kind`` was set on an earlier lane and
    survived promotion, so the breaker parked it with a marker that no longer
    described the wait.
    """
    tid = kb.create_task(conn, title="parked card")
    kb.claim_task(conn, tid)
    assert kb.block_task(conn, tid, reason="waiting on a parent", kind="needs_input")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET block_kind = ?, consecutive_failures = 3, "
            "block_recurrences = 2 WHERE id = ?", (kind, tid))
    return tid


def _payload(event) -> dict:
    raw = event.payload
    return raw if isinstance(raw, dict) else json.loads(raw or "{}")


def test_reclassify_retypes_the_block_without_releasing_the_card(kanban_home):
    """The kind changes; the wait does not. Every counter and the run history
    survive, and the only new event is the non-terminal ``reclassified``."""
    with kbc.connect() as conn:
        tid = _parked(conn)
        before = kb.get_task(conn, tid)
        assert (before.status, before.block_kind) == ("blocked", "dependency")
        runs_before = len(kb.list_runs(conn, tid))
        blocked_events_before = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]

        assert kbr.reclassify_block(conn, tid, kind="needs_input", reason="parents are done") is True

        after = kb.get_task(conn, tid)
        assert after.status == "blocked"
        assert after.block_kind == "needs_input"
        assert after.consecutive_failures == 3
        assert after.block_recurrences == 2
        assert after.current_run_id is None
        assert len(kb.list_runs(conn, tid)) == runs_before

        events = kb.list_events(conn, tid)
        assert [e for e in events if e.kind == "blocked"] == blocked_events_before
        reclassified = [e for e in events if e.kind == "reclassified"]
        assert len(reclassified) == 1
        assert _payload(reclassified[0]) == {
            "kind": "needs_input", "previous_kind": "dependency", "reason": "parents are done",
        }


def test_reclassified_card_is_not_auto_promoted(kanban_home):
    """Re-classification is what makes the wait sticky: the same parked card
    that ``recompute_ready`` promotes before the correction stays blocked
    after it, so only an explicit ``kanban unblock`` releases it."""
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.claim_task(conn, parent)
        kb.complete_task(conn, parent, result="done")

        def park():
            # How the breaker parks a card: no sticky ``blocked`` event, a kind
            # marker left over from an earlier lane, and a failure counter still
            # under the limit — so nothing but a sticky event holds it.
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status = 'blocked', block_kind = 'dependency', "
                    "consecutive_failures = 1, current_run_id = NULL WHERE id = ?", (child,))

        # Control: an untyped wait with no sticky event is promoted.
        park()
        assert kb.recompute_ready(conn) == 1
        promoted = kb.get_task(conn, child)
        assert promoted is not None and promoted.status == "ready"

        # Park it again with the stale marker, then correct the marker.
        park()
        assert kbr.reclassify_block(conn, child, kind="needs_input", reason="parents are done") is True

        assert kb.recompute_ready(conn) == 0
        after = kb.get_task(conn, child)
        assert after is not None
        assert after.status == "blocked"
        assert after.block_kind == "needs_input"


def test_reclassify_refuses_cards_it_must_not_touch(kanban_home):
    """A card that is not blocked, one already carrying the target kind, and a
    bad kind are all refused — and a refusal writes no event."""
    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running card")
        kb.claim_task(conn, running)
        assert kbr.reclassify_block(conn, running, kind="needs_input") is False

        tid = _parked(conn)
        assert kbr.reclassify_block(conn, tid, kind="needs_input") is True
        assert kbr.reclassify_block(conn, tid, kind="needs_input") is False

        with pytest.raises(ValueError):
            kbr.reclassify_block(conn, tid, kind="not-a-kind")

        kinds = [e.kind for e in kb.list_events(conn, running)]
        assert "reclassified" not in kinds


def test_cli_reclassify_retypes_and_comments(kanban_home):
    """The supported CLI path applies the correction and records the reason."""
    with kbc.connect() as conn:
        tid = _parked(conn)

    out = kc.run_slash(f"reclassify {tid} --kind needs_input parents are done")
    assert "Reclassified" in out

    with kbc.connect() as conn:
        after = kb.get_task(conn, tid)
        assert (after.status, after.block_kind) == ("blocked", "needs_input")
        assert after.consecutive_failures == 3
        assert after.block_recurrences == 2
        comments = kb.list_comments(conn, tid)
        assert any("RECLASSIFIED" in c.body for c in comments)


def test_cli_reclassify_is_orchestrator_only(kanban_home, monkeypatch):
    """Re-typing somebody else's wait is a supervisor action, like unblock."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_somebody_else")
    out = kc.run_slash("reclassify t_whatever --kind needs_input")
    assert "orchestrator-only" in out


def test_reclassified_event_is_never_claimed_or_woken():
    """The audit event must stay outside the notifier's claim and wake sets —
    a re-classification may not deliver a notification or wake a subscriber."""
    from gateway.kanban_watchers_notifier import TERMINAL_KINDS, _WAKE_KINDS

    assert "reclassified" not in TERMINAL_KINDS
    assert "reclassified" not in _WAKE_KINDS
    assert kbr.RECLASSIFIED_EVENT == "reclassified"
