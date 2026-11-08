"""Timestamp-safe retry estimates for the kanban dispatcher guard."""

from __future__ import annotations

import sqlite3
import time
from typing import Any
from typing import Optional

from hermes_cli import kanban_db as _kb


def _guard_epoch(value: Any) -> Optional[int]:
    """Convert a stored guard timestamp, treating nonfinite values as unknown."""
    try:
        return _kb._to_epoch(value)
    except (OverflowError, ValueError):
        return None


def _recent_finished_runs(conn: sqlite3.Connection, task_id: str) -> list:
    """Return the five newest runs by parsed timestamp, retaining unknown quota runs."""
    from heapq import nlargest

    rows = conn.execute(
        "SELECT id, outcome, ended_at, metadata FROM task_runs "
        "WHERE task_id = ? AND (ended_at IS NOT NULL OR outcome = 'rate_limited')",
        (task_id,),
    )

    def order(row):
        epoch = _guard_epoch(row["ended_at"])
        uncertain_limit = epoch is None and row["outcome"] == "rate_limited"
        return (uncertain_limit, epoch if epoch is not None else float("-inf"), row["id"])

    return nlargest(5, rows, key=order)


def retry_details(runs: list, base: int, now: Optional[int] = None) -> Optional[dict]:
    if not runs or runs[0]["outcome"] != "rate_limited":
        return None
    ended_at = _guard_epoch(runs[0]["ended_at"])
    if ended_at is None:
        return {"retry_at": None, "retry_after_seconds": None, "consecutive_rate_limits": None}
    streak = 0
    for run in runs:
        if run["outcome"] != "rate_limited":
            break
        streak += 1
    retry_at = ended_at + max(base, 0) * (2 ** max(streak - 1, 0))
    retry_after = max(0, retry_at - (int(time.time()) if now is None else now))
    return {
        "retry_at": retry_at,
        "retry_after_seconds": retry_after,
        "consecutive_rate_limits": streak,
    }


def rate_limit_retry_details(
    conn: sqlite3.Connection, task_id: str, *, now: Optional[int] = None,
) -> Optional[dict]:
    """Return the latest quota retry estimate; this does not probe a provider."""
    return retry_details(
        _recent_finished_runs(conn, task_id), _kb._resolve_rate_limit_cooldown_seconds(), now,
    )


def record_guarded_event(conn: sqlite3.Connection, task_id: str, reason: str) -> None:
    """Avoid writing the same unreadable-timestamp guard event on every tick."""
    if reason == "invalid_rate_limit_timestamp":
        last = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1", (task_id,),
        ).fetchone()
        payload = {"reason": reason}
        if last is not None and last["kind"] == "respawn_guarded" and last["payload"] == _kb._json_or_null(payload):
            return
    _kb._append_event(conn, task_id, "respawn_guarded", {"reason": reason})
