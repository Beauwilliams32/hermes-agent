"""A recurring job that spent its repeat budget must still be re-armable.

The shape this pins:

``CLAUDE recovery dispatch - bounded`` is an ``interval`` job that runs every 10 minutes
with ``repeat: {times: 144, completed: 144}``. 144 x 10m is 24 hours of budget. When the
budget ran out the scheduler set ``state=completed`` and stopped firing it - correct.

But the job still had a schedule with future occurrences. Treating a repeat-capped
RECURRING job as terminal meant every recovery path refused it::

    hermes cron resume <id>                 -> "Cannot activate terminal cron job"
    hermes cron resume <id> --run-now       -> same
    hermes cron resume <id> --at <ts>       -> same
    hermes cron edit <id> --repeat 288      -> the edit lands, the resume still refuses
    hermes cron run <id>                    -> "Job is paused/disabled; resume it first"

All of them route through ``update_job``, and ``update_job`` raised before saving. The only
way out was to hand-edit ``~/.hermes/cron/jobs.json``.

The fix exempts a recurring job whose repeat budget has been RAISED above what it has
spent (``completed < times``), because raising the budget is the explicit statement that
the job should keep running. The exemption is deliberately narrow - see the control tests
at the bottom, which pin that a one-shot and an exactly-spent recurring job stay blocked.
"""

from datetime import datetime, timedelta, timezone, UTC
import copy

import pytest

from cron.jobs import (
    create_job,
    get_job,
    load_jobs,
    mark_job_run,
    resume_job,
    save_jobs,
    trigger_job,
    update_job,
)


@pytest.fixture()
def tmp_cron_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")
    return tmp_path


def _exhaust_recurring_budget(job_id: str, times: int) -> None:
    """Drive a recurring job to ``state=completed`` by spending its whole repeat budget."""
    record = get_job(job_id)
    record["repeat"] = {"times": times, "completed": times}
    record["state"] = "completed"
    record["enabled"] = False
    record["next_run_at"] = None
    save_jobs([record])


class TestRepeatCappedRecurringJobStaysRearmable:
    def test_exhausted_interval_job_can_be_resumed_once_its_budget_is_raised(self, tmp_cron_dir):
        job = create_job("recovery", "every 10m", repeat=144, name="recovery dispatch")
        _exhaust_recurring_budget(job["id"], 144)

        exhausted = get_job(job["id"])
        assert exhausted["state"] == "completed"
        assert exhausted["enabled"] is False
        assert exhausted["repeat"]["completed"] == 144

        # The budget is raised - the explicit statement that this job should keep running.
        update_job(job["id"], {"repeat": {"times": 288, "completed": 144}})

        resumed = resume_job(job["id"])

        assert resumed is not None, "resume must not refuse a repeat-capped recurring job"
        assert resumed["state"] == "scheduled"
        assert resumed["enabled"] is True
        assert resumed["next_run_at"] is not None, (
            "a resumed recurring job must get a next occurrence"
        )
        assert resumed["repeat"]["completed"] == 144, (
            "the spent count must not be rewritten by the re-arm"
        )
        assert resumed["repeat"]["times"] == 288

    def test_budget_raise_alone_does_not_activate_the_job(self, tmp_cron_dir):
        """Raising the cap must not silently start a job that was retired."""
        job = create_job("recovery", "every 10m", repeat=144)
        _exhaust_recurring_budget(job["id"], 144)

        updated = update_job(job["id"], {"repeat": {"times": 288, "completed": 144}})

        assert updated["enabled"] is False
        assert updated["state"] == "completed"
        assert updated["next_run_at"] is None

    def test_resume_preserves_the_stored_past_slot_for_late_or_catch_up(self, tmp_cron_dir):
        """A recurring job paused across a slot must not lose that slot silently.

        The stored ``next_run_at`` survives the resume as the due instant, and the ordinary
        late/catch-up policy decides whether it fires or is skipped - never a silent
        re-anchor past it.
        """
        job = create_job("recovery", "every 10m", repeat=144)
        past_slot = (datetime.now(UTC) - timedelta(minutes=3)).isoformat()
        record = get_job(job["id"])
        record.update({
            "repeat": {"times": 288, "completed": 144},
            "state": "completed",
            "enabled": False,
            "next_run_at": past_slot,
        })
        save_jobs([record])

        resumed = resume_job(job["id"])

        assert resumed["enabled"] is True
        assert resumed["state"] == "scheduled"
        assert resumed["next_run_at"] == past_slot, (
            "the elapsed slot must be preserved as the due instant"
        )

    def test_the_whole_recovery_sequence_works_end_to_end(self, tmp_cron_dir):
        """edit --repeat, then resume, then the job is live again.

        This is the exact sequence an operator would run, and every step used to fail.
        """
        job = create_job("recovery", "every 10m", repeat=144, name="recovery dispatch")
        _exhaust_recurring_budget(job["id"], 144)

        with pytest.raises(ValueError, match="terminal"):
            resume_job(job["id"])

        update_job(job["id"], {"repeat": {"times": 288, "completed": 144}})
        resumed = resume_job(job["id"])

        assert resumed["enabled"] is True
        assert resumed["state"] == "scheduled"
        assert resumed["next_run_at"] is not None


class TestTheExemptionStaysNarrow:
    """Controls: the fix must not weaken protection where the job really is done."""

    def test_a_oneshot_is_still_blocked_on_every_path(self, tmp_cron_dir):
        """A one-shot has no future occurrence. It stays terminal."""
        job = create_job("done", "in 30m", repeat=1)
        mark_job_run(job["id"], success=True)
        assert get_job(job["id"])["state"] == "completed"

        with pytest.raises(ValueError, match="terminal"):
            resume_job(job["id"])
        with pytest.raises(ValueError, match="terminal"):
            trigger_job(job["id"])

    def test_an_exactly_spent_recurring_job_is_still_blocked(self, tmp_cron_dir):
        """A budget that is spent but NOT raised stays exactly as blocked as before.

        ``test_exhausted_recurring_job_trigger_is_refused`` in the sibling module pins the
        trigger path; this pins resume and update.
        """
        job = create_job("done", "every 1h", repeat=1)
        _exhaust_recurring_budget(job["id"], 1)

        with pytest.raises(ValueError, match="terminal"):
            resume_job(job["id"])
        with pytest.raises(ValueError, match="terminal"):
            trigger_job(job["id"])
        with pytest.raises(ValueError, match="terminal"):
            update_job(job["id"], {"enabled": True})

    def test_a_recurring_job_with_no_budget_is_unaffected(self, tmp_cron_dir):
        """``repeat.times = None`` means run forever - not a capped job."""
        job = create_job("forever", "every 1h")
        record = get_job(job["id"])
        record.update({"state": "completed", "enabled": False, "next_run_at": None})
        save_jobs([record])

        with pytest.raises(ValueError, match="terminal"):
            resume_job(job["id"])

    def test_the_spent_count_is_not_silently_reset(self, tmp_cron_dir):
        """Re-arming must not erase the history of how many times the job already ran."""
        job = create_job("recovery", "every 10m", repeat=144)
        _exhaust_recurring_budget(job["id"], 144)
        before = copy.deepcopy(get_job(job["id"]))

        update_job(job["id"], {"repeat": {"times": 288, "completed": 144}})
        resumed = resume_job(job["id"])

        assert resumed["repeat"]["completed"] == before["repeat"]["completed"] == 144
