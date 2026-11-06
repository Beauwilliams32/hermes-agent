"""Re-typing the block of a parked card without releasing it.

Split out of ``hermes_cli.kanban_db``; origin-resident helpers are reached
late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import sqlite3
from typing import Optional

# The event kind a re-classification writes. It is deliberately NOT one of the
# notifier's TERMINAL_KINDS: nobody claims it, nobody is woken by it, so
# correcting a stale block kind can never release the card through a
# subscriber's wake path.
RECLASSIFIED_EVENT = "reclassified"


def _has_sticky_block(conn: sqlite3.Connection, task_id: str) -> bool:
    """True when the newest ``blocked``/``unblocked``/``gave_up`` event says the
    block must wait for an operator: an explicit ``kanban_block`` (#28712), or a
    breaker trip ``_record_task_failure`` stamped ``sticky`` — the clean-exit
    protocol-violation budget or a systemic same-error wave. Those trip on a
    policy independent of ``consecutive_failures``, so ``recompute_ready``'s
    counter check cannot see them — without this the trip is promoted back to
    ``ready`` in the same tick and the card respawns forever. A plain
    (unified-budget) ``gave_up`` carries no marker and is judged by the counter,
    so raising ``failure_limit`` or ``assign_task`` to a fresh profile still
    releases it; a task with no such event at all (direct DB edit) auto-recovers.
    A ``reclassified`` event is an operator's re-typed wait — it holds the card
    exactly like an explicit ``kanban_block``, and only an explicit
    ``kanban unblock`` releases it.
    """
    from hermes_cli import kanban_db as _kb

    row = conn.execute(
        "SELECT kind FROM task_events "
        "WHERE task_id = ? AND kind IN ('blocked', 'unblocked', 'reclassified') "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    if row and row["kind"] in ("blocked", "reclassified"):
        return True
    trip = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'gave_up' AND id > COALESCE("
        "  (SELECT MAX(id) FROM task_events WHERE task_id = ? AND kind = 'unblocked'), 0) "
        "ORDER BY id DESC LIMIT 1", (task_id, task_id),
    ).fetchone()
    return bool(trip) and bool(_kb._json_dict(trip["payload"]).get("sticky"))


def reclassify_block(
    conn: sqlite3.Connection, task_id: str, *, kind: str, reason: Optional[str] = None,
) -> bool:
    """Re-type the block of an already-``blocked`` card, in place.

    The card stays ``blocked`` and keeps its failure counter, recurrence
    counter, live-run guard and run history exactly as the breaker left them:
    a re-classification corrects the KIND of a wait, it never releases one.
    Nothing is promoted and no subscriber is woken -- the audit event kind is
    outside the notifier's TERMINAL_KINDS.
    """
    from hermes_cli import kanban_db as _kb

    if kind not in _kb.VALID_BLOCK_KINDS:
        raise ValueError(f"block kind must be one of {sorted(_kb.VALID_BLOCK_KINDS)}")
    with _kb.write_txn(conn):
        row = conn.execute(
            "SELECT status, block_kind, current_run_id FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        if row is None or row["status"] != "blocked" or row["current_run_id"] is not None:
            return False
        previous = row["block_kind"]
        if previous == kind:
            return False
        changed = conn.execute(
            "UPDATE tasks SET block_kind = ? WHERE id = ? AND status = 'blocked' "
            "AND current_run_id IS NULL", (kind, task_id),
        ).rowcount
        if changed != 1:
            return False
        _kb._append_event(conn, task_id, RECLASSIFIED_EVENT, {
            "kind": kind, "previous_kind": previous, "reason": reason,
        })
    return True
