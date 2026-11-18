"""Reclaim must not release a live worker or signal a recycled process group."""
import os
import signal
import subprocess
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


def _running(conn):
    tid = kb.create_task(conn, title="reclaim probe", assignee="worker")
    task = kb.claim_task(conn, tid, claimer="beaus-MacBook-Pro-10.local:100")
    assert task is not None and task.current_run_id is not None and task.claim_lock is not None
    return tid, task.current_run_id


def test_changed_hostname_still_signals_fingerprinted_worker(board, monkeypatch):
    tid, _ = _running(board)
    fingerprint = dispatch._process_fingerprint(os.getpid())
    assert fingerprint
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?",
                      (os.getpid(), fingerprint, tid))
    monkeypatch.setattr(kb, "_host_prefix", lambda: "beaus-MBP-10.vsz-east.managedwifi.xfinity.com:")
    sent = []
    monkeypatch.setattr(dispatch, "_poll_worker_exit", lambda *args: True)
    info = dispatch._terminate_reclaimed_worker(
        os.getpid(), "beaus-MacBook-Pro-10.local:100", started_at=fingerprint,
        signal_fn=lambda pid, sig: sent.append((pid, sig)))
    assert info["host_local"] is True
    assert info["termination_attempted"] is True
    assert sent == [(os.getpid(), signal.SIGTERM)]


def test_manual_reclaim_defers_when_worker_survives(board, monkeypatch):
    tid, _ = _running(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?",
                      (os.getpid(), dispatch._process_fingerprint(os.getpid()), tid))
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **kw: {
        "host_local": True, "termination_attempted": True, "terminated": False})
    lock = kb.get_task(board, tid).claim_lock
    assert kb.reclaim_task(board, tid, reason="operator") is False
    task = kb.get_task(board, tid)
    assert task.status == "running" and task.claim_lock == lock
    assert task.worker_pid == os.getpid()
    assert "reclaim_deferred" in [e.kind for e in kb.list_events(board, tid)]


def test_adoption_replaces_launcher_with_bound_interpreter(board):
    tid, run_id = _running(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?",
                      (999999, "stale", tid))
        board.execute("UPDATE task_runs SET worker_pid=?, worker_started_at=? WHERE id=?",
                      (999999, "stale", run_id))
    lock = kb.get_task(board, tid)
    assert lock is not None and lock.claim_lock is not None
    assert dispatch.adopt_worker_pid(board, tid, run_id, os.getpid(), claim_lock=lock.claim_lock)
    row = board.execute("SELECT worker_pid, worker_started_at FROM tasks WHERE id=?", (tid,)).fetchone()
    run = board.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
    assert row["worker_pid"] == run["worker_pid"] == os.getpid()
    assert row["worker_started_at"] == run["worker_started_at"] == dispatch._process_fingerprint(os.getpid())


def test_submitter_watch_keeps_spawned_identity_after_interpreter_adoption(board):
    from gateway.session_kanban import _submitter_identity

    tid, run_id = _running(board)
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET worker_pid=?, worker_started_at=? WHERE id=?",
                      (987654, "submitter-birth", tid))
        board.execute("UPDATE task_runs SET worker_pid=?, worker_started_at=? WHERE id=?",
                      (987654, "submitter-birth", run_id))
        kb._append_event(board, tid, "spawned", {"pid": 987654, "started_at": "submitter-birth"}, run_id=run_id)
    lock = kb.get_task(board, tid)
    assert lock is not None and lock.claim_lock is not None
    assert dispatch.adopt_worker_pid(board, tid, run_id, os.getpid(), claim_lock=lock.claim_lock)
    assert _submitter_identity(board, run_id) == (987654, "submitter-birth")


def test_adoption_refuses_claim_replaced_before_binding(board):
    tid, run_id = _running(board)
    task = kb.get_task(board, tid)
    assert task is not None and task.claim_lock is not None
    lock = task.claim_lock
    with kb.write_txn(board):
        board.execute("UPDATE tasks SET claim_lock=?, worker_pid=? WHERE id=?", ("new-owner:200", 987654, tid))
    assert not dispatch.adopt_worker_pid(board, tid, run_id, os.getpid(), claim_lock=lock)
    assert kb.get_task(board, tid).worker_pid == 987654


@pytest.mark.platforms("posix")
def test_group_signal_uses_group_identity_and_refuses_recycled_leader(monkeypatch):
    fingerprint = dispatch._process_fingerprint(os.getpid())
    sent = []
    monkeypatch.setattr(dispatch, "_poll_worker_exit", lambda *args: True)
    monkeypatch.setattr(dispatch.os, "killpg", lambda pgid, sig: sent.append((pgid, sig)))
    lock = kb._claimer_id()
    # The worker is group leader in the spawn topology; mock group membership, not platform.
    monkeypatch.setattr(dispatch.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(dispatch.os, "getpgrp", lambda: -1)
    info = dispatch._terminate_reclaimed_worker(os.getpid(), lock, started_at=fingerprint)
    assert info["termination_attempted"] and sent == [(os.getpid(), signal.SIGTERM)]
    sent.clear()
    info = dispatch._terminate_reclaimed_worker(os.getpid(), lock, started_at="other-boot|0")
    assert sent == []
    assert info["pid_recycled"] is True


@pytest.mark.platforms("posix")
def test_reclaim_worker_death_reaps_detached_tool_group(tmp_path):
    """The existing owner-death supervisor sweeps a terminal command's own setsid group."""
    import psutil

    marker = tmp_path / "tool-pid"
    code = (
        "import os, pathlib, time\n"
        "from tools.environments.local import LocalEnvironment\n"
        "env = LocalEnvironment(cwd=os.getcwd())\n"
        "tool = env._run_bash('sleep 90')\n"
        f"pathlib.Path({str(marker)!r}).write_text(str(tool.pid))\n"
        "time.sleep(90)\n"
    )
    env = {**os.environ, "HERMES_HOME": str(tmp_path / "home")}
    worker = subprocess.Popen([sys.executable, "-c", code], cwd=os.getcwd(), env=env,
                              start_new_session=True, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    child_pid = None
    child = None
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and time.monotonic() < deadline and worker.poll() is None:
            time.sleep(0.05)
        assert marker.exists(), "tool did not spawn"
        child_pid = int(marker.read_text())
        child = psutil.Process(child_pid)
        assert os.getpgid(child_pid) != worker.pid
        fingerprint = dispatch._process_fingerprint(worker.pid)
        assert fingerprint
        termination = dispatch._terminate_reclaimed_worker(worker.pid, kb._claimer_id(), started_at=fingerprint)
        assert termination["termination_attempted"] and termination["terminated"]
        worker.wait(timeout=5)
        deadline = time.monotonic() + 15
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=5)
        if child is not None and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()

