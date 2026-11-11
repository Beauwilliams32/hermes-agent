"""Respawn guard: the reasons a card must not be re-spawned.

Split out of ``hermes_cli.kanban_db_dispatch``; origin-resident helpers are
reached late-bound via ``_kb`` (import-cycle breaking) so monkeypatching
``kanban_db.<name>`` keeps working.
"""

from __future__ import annotations

import re
import sqlite3
import time
from typing import Any
from typing import Iterator
from typing import Optional

# Import the board facade only when a guard is called. Managed workers import
# dispatch -> guard before board has finished defining the dispatcher exports.

# ---------------------------------------------------------------------------
# Respawn guard constants
# ---------------------------------------------------------------------------

# Patterns in last_failure_error that indicate a quota / auth blocker.
# These errors won't resolve by retrying immediately — auto-block instead.
# The auth family is a curated list, not an open `auth\w*` stem: that stem
# also matched ordinary English words like "author"/"authored"/"authoring"/
# "authoritative" in worker progress prose, parking a healthy card forever
# (#117009).
_RESPAWN_BLOCKER_RE = re.compile(
    r"\b(quota|rate[\s_\-]?limit|429|403|"
    r"auth|authenticat(?:e|es|ed|ing|ion)|authoriz(?:e|es|ed|ing|ation)|"
    r"authoris(?:e|es|ed|ing|ation)|authz|"
    r"unauthorized|forbidden|billing|subscription|"
    r"access[\s_]denied|permission[\s_]denied|"
    r"invalid[\s_]api[\s_]key)\b",
    re.IGNORECASE,
)

# Outcomes that mean the worker made real forward progress: it finished the
# work and handed it off (``completed`` / ``review_requested`` /
# ``changes_requested``). A quota- or auth-flavoured ``last_failure_error``
# stamped by an EARLIER run is stale once a later run has advanced the card.
# The rate-limited requeue path writes that stamp WITHOUT incrementing
# ``consecutive_failures`` precisely because it is informational, so treating
# it as a live blocker parked a handed-off card forever: the guard read the
# stale stamp BEFORE the review-lane early return, so a card in ``review``
# could never be reviewed and re-guarded itself on every tick.
_RESPAWN_PROGRESS_OUTCOMES = frozenset({
    "completed", "review_requested", "changes_requested",
})

# Within this window a completed run counts as "recent proof"; don't re-spawn.
_RESPAWN_GUARD_SUCCESS_WINDOW = 3600  # 1 hour

# Within this window a GitHub PR URL in a comment blocks re-spawn.
_RESPAWN_GUARD_PR_WINDOW = 86400  # 24 hours

_RESPAWN_GUARD_PR_URL_RE = re.compile(
    r"https?://github\.com/[^/\s]+/[^/\s]+/pull/\d+",
    re.IGNORECASE,
)


def _guard_epoch(value: Any) -> Optional[int]:
    """Convert a stored guard timestamp, treating nonfinite values as unknown."""
    from hermes_cli import kanban_db as _kb

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
    """Return bounded exponential retry timing for a newest rate-limit streak."""
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
    from hermes_cli import kanban_db as _kb

    return retry_details(
        _recent_finished_runs(conn, task_id), _kb._resolve_rate_limit_cooldown_seconds(), now,
    )


def record_guarded_event(conn: sqlite3.Connection, task_id: str, reason: str) -> None:
    """Avoid writing the same unreadable-timestamp guard event on every tick."""
    from hermes_cli import kanban_db as _kb

    if reason == "invalid_rate_limit_timestamp":
        last = conn.execute(
            "SELECT kind, payload FROM task_events WHERE task_id = ? "
            "ORDER BY created_at DESC, id DESC LIMIT 1", (task_id,),
        ).fetchone()
        payload = {"reason": reason}
        if last is not None and last["kind"] == "respawn_guarded" and last["payload"] == _kb._json_or_null(payload):
            return
    _kb._append_event(conn, task_id, "respawn_guarded", {"reason": reason})


def check_respawn_guard(
    conn: sqlite3.Connection, task_id: str, *, lane: str = "ready",
) -> Optional[str]:
    """Return a guard reason if ``task_id`` should NOT be re-spawned, else None.

    Called per ready/review row before any claim attempt. Priority order:
    ``"infrastructure_cooldown"`` (latest run is a ``spawn_failed`` the host
    refused — no restart-safe scope — within the cooldown; never counted),
    ``"rate_limit_cooldown"`` (latest run ``rate_limited`` within the cooldown;
    checked BEFORE ``blocker_auth`` because the requeue stamps a quota-flavored
    ``last_failure_error`` that would otherwise park the task forever — that
    path never increments ``consecutive_failures``), ``"blocker_auth"``
    (quota/auth pattern; the breaker still trips eventually — but NOT when
    the latest run made forward progress, which means the stamp is stale
    context from an earlier run), then for the
    ready lane only ``"recent_success"`` (completed run within the window, unless
    a re-queue event arrived after it — a deliberate re-run) and ``"active_pr"``
    (PR URL in a recent comment; re-spawning risks a duplicate PR — unless a
    handoff event followed the comment: the named profile must work on that
    PR). The review lane skips the last two: they are the *inputs* to a review
    handoff. Stale / dead claim locks are NOT a guard reason — the reclaim
    passes own those.
    """
    from hermes_cli import kanban_db as _kb

    row = conn.execute(
        "SELECT last_failure_error FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return None

    now = int(time.time())

    # 1. Rate-limit cooldown — see docstring for why this precedes blocker_auth.
    #    LATEST run only: a newer crash/completion supersedes the rate-limit run.
    #    An infrastructure spawn refusal (#114720) shares the cooldown: the host
    #    condition is not the card's, so it retries forever, spaced, and never
    #    reaches the breaker.
    rl_cooldown = _kb._resolve_rate_limit_cooldown_seconds()
    runs = _recent_finished_runs(conn, task_id)
    latest_run = runs[0] if runs else None
    if latest_run is not None and latest_run["outcome"] == "spawn_failed":
        if rl_cooldown > 0 and _kb._json_dict(latest_run["metadata"]).get("infrastructure"):
            ended_at = _guard_epoch(latest_run["ended_at"])
            if ended_at is not None and (now - ended_at) < rl_cooldown:
                return "infrastructure_cooldown"
    if latest_run is not None and latest_run["outcome"] == "rate_limited":
        if rl_cooldown <= 0:
            # Cooldown disabled — respawn immediately, skipping blocker_auth so
            # the stamped rate-limit text doesn't re-trap the task.
            return None
        retry = retry_details(runs, rl_cooldown, now)
        if retry is None or retry["retry_at"] is None:
            return "invalid_rate_limit_timestamp"
        if retry["retry_after_seconds"] > 0:
            return "rate_limit_cooldown"
        # Cooldown elapsed — return early so blocker_auth doesn't catch the
        # stamped rate-limit text; this path intentionally retries forever
        # (spaced by the cooldown) until quota returns or a real run supersedes it.
        return None

    # 2. Quota / auth blocker: retrying immediately will not help.  A plain
    # crash is different: its persisted error includes the worker's last
    # captured output, which is context rather than a diagnosis and may contain
    # benign commands such as ``claude auth status`` (#117097).
    err = _kb._lossy_text(row["last_failure_error"])
    latest_outcome = latest_run["outcome"] if latest_run is not None else None
    # A forward-progress outcome means the card moved on after the run that
    # wrote the stamp, so the stamp is stale context, not a live blocker.
    if (
        err
        and latest_outcome not in _RESPAWN_PROGRESS_OUTCOMES
        and latest_outcome != "crashed"
        and _RESPAWN_BLOCKER_RE.search(err)
    ):
        return "blocker_auth"

    # Review-lane spawns stop here: a recent completed run and a fresh PR URL
    # are the canonical *inputs* to a review handoff, not duplicate-work signals.
    if lane == "review":
        return None

    # 3. Completed run within guard window. Exception: an explicit re-queue
    #    AFTER that success (done→ready drag, re-promotion, unblock, reclaim) is
    #    a deliberate "run it again" — otherwise a manual done→ready would sit
    #    silently held until the window elapses.
    success = max((t for t, _ in _rows_since(
        conn, "task_runs", task_id, now - _RESPAWN_GUARD_SUCCESS_WINDOW, "AND outcome = 'completed'")), default=None)
    if success is not None and not any(_rows_since(
            conn, "task_events", task_id, success, "AND kind IN ('status', 'promoted', 'unblocked', 'reclaimed')")):
        return "recent_success"

    # 4. GitHub PR URL in a recent comment — prior worker already opened a PR.
    #    Exception: a handoff AFTER the newest PR comment (operator reassign,
    #    reviewer changes_requested, review reopen) names the profile that must
    #    now work on THAT PR — a closer or the implementer finishing it, not a
    #    duplicate implementation (#111910). A crash/reclaim is not a handoff,
    #    so the worker that opened the PR is still not re-spawned against it.
    pr_times = [t for t, c in _rows_since(conn, "task_comments", task_id, now - _RESPAWN_GUARD_PR_WINDOW,
                                          cols=", body", unreadable=now)  # duplicate-PR risk: fail closed
                if _RESPAWN_GUARD_PR_URL_RE.search(_kb._lossy_text(c["body"]) or "")]
    if pr_times:
        # Strictly after: a same-second tie stays guarded (fail closed).
        if any(_is_handoff_event(e["kind"], e["payload"]) for _, e in _rows_since(
                conn, "task_events", task_id, max(pr_times),
                "AND kind IN ('assigned', 'changes_requested', 'review_reopened')", cols=", kind, payload", strict=True)):
            return None
        return "active_pr"

    return None


_GUARD_TIME_COLS = {"task_runs": "ended_at", "task_events": "created_at", "task_comments": "created_at"}


def _rows_since(conn: sqlite3.Connection, table: str, task_id: str, since: int, where: str = "", *,
                cols: str = "", strict: bool = False, unreadable: Optional[int] = None) -> Iterator[tuple[int, Any]]:
    """``(epoch, row)`` for ``task_id``'s ``table`` rows timed at/after ``since`` (after, if ``strict``).
    SQLite sorts TEXT above every INTEGER, so the indexed range also returns every text stamp (written
    outside the mutators, #97455); those are decided on their parsed epoch. An unreadable time is
    dropped (never guards or lifts) unless the caller substitutes ``unreadable``."""
    from hermes_cli import kanban_db as _kb

    col = _GUARD_TIME_COLS[table]
    sql = f"SELECT {col}{cols} FROM {table} WHERE task_id = ? {where} AND {col} {'>' if strict else '>='} ?"
    for row in conn.execute(sql, (task_id, since)):
        t = _kb._to_epoch(row[0])
        t = unreadable if t is None else t
        if t is not None and (t > since if strict else t >= since):
            yield t, row


def _is_handoff_event(kind: str, payload: Optional[str]) -> bool:
    """Only an ``assigned`` event that moves the card to a DIFFERENT profile is
    a handoff. A no-op re-assign (dev→dev via CLI/dashboard/``reassign
    --reclaim``), an unassign, or the dispatcher's own
    ``kanban.default_assignee`` write would otherwise lift ``active_pr`` for
    the very implementer that opened the PR. Events without ``from`` (written
    before it was recorded) are not trusted as handoffs — fail closed."""
    from hermes_cli import kanban_db as _kb

    if kind != "assigned":
        return True
    data = _kb._json_or(payload, {})
    if not isinstance(data, dict) or data.get("source") == "kanban.default_assignee":
        return False
    to = data.get("assignee")
    return bool(to) and "from" in data and data["from"] != to


