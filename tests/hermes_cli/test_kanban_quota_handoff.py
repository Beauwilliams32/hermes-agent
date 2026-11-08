"""A rate-limited task retries once and reports completion to its origin."""

import asyncio
import time
from pathlib import Path

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_notify as kbn


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append((chat_id, text))

    async def handle_message(self, event):
        self.handled.append(event)
        event._gateway_accepted = True


def test_quota_retry_completes_and_wakes_origin_once(tmp_path: Path, monkeypatch, all_assignees_spawnable):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kb, "_resolve_rate_limit_cooldown_seconds", lambda: 300)
    now = int(time.time())
    clock = [now]
    monkeypatch.setattr(kbd.time, "time", lambda: clock[0])
    kb.init_db()
    spawns = []

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="quota recovery", assignee="worker")
        kbn.add_notify_sub(conn, task_id=task_id, platform="telegram", chat_id="origin",
                           delivery_mode="notify+wake")
        conn.execute(
            "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
            "VALUES (?, 'worker', 'rate_limited', 'rate_limited', ?, ?)",
            (task_id, now - 1, now),
        )
        conn.commit()

        def complete_claim(task, workspace):
            spawns.append(task.id)
            assert kb.complete_task(conn, task.id, summary="Recovered result", expected_run_id=task.current_run_id)

        held = kbd.dispatch_once(conn, spawn_fn=complete_claim)
        assert (task_id, "rate_limit_cooldown") in held.respawn_guarded
        assert not spawns
        clock[0] += 301
        retried = kbd.dispatch_once(conn, spawn_fn=complete_claim)
        assert [row[0] for row in retried.spawned] == [task_id]
        kbd.dispatch_once(conn, spawn_fn=complete_claim)
        assert spawns == [task_id]

    adapter = RecordingAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_dispatcher_lock_handle = object()
    real_sleep = asyncio.sleep

    async def stop_after_tick(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", stop_after_tick)
    asyncio.run(runner._kanban_notifier_watcher(interval=1))
    assert len(adapter.sent) == 1
    assert adapter.sent[0][0] == "origin"
    assert "Recovered result" in adapter.sent[0][1]
    assert len(adapter.handled) == 1
    runner._running = True
    asyncio.run(runner._kanban_notifier_watcher(interval=1))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
