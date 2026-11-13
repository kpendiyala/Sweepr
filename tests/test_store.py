"""Tests for the run history database. No processes: we save made-up results by hand."""

import sqlite3
from pathlib import Path

import pytest

from sweepr import store
from sweepr.classify import Classification
from sweepr.supervisor import AttemptResult


def attempt(number, failure=None):
    return AttemptResult(
        attempt=number,
        classification=Classification(failure, failure or "exited 0"),
        exit_code=1 if failure else 0,
        killed_by=None,
        last_step=10,
        last_checkpoint=None,
        duration_s=1.5,
        log_path=Path(f"runs/x/attempt-{number}.log"),
    )


@pytest.fixture
def db(tmp_path):
    connection = store.connect(tmp_path / "sweepr.db")
    yield connection
    connection.close()


def test_connect_twice_keeps_the_data(tmp_path):
    path = tmp_path / "sweepr.db"
    first = store.connect(path)
    store.start_sweep(first, "a", tmp_path, total_runs=1)
    first.close()
    second = store.connect(path)  # CREATE TABLE IF NOT EXISTS doesn't wipe anything
    assert len(store.list_sweeps(second)) == 1
    second.close()


def test_a_sweep_with_runs_and_attempts(db, tmp_path):
    sweep_id = store.start_sweep(db, "lr", tmp_path, total_runs=3)

    ok = store.start_run(db, sweep_id, "run-001", {"lr": 0.1})
    store.save_attempt(db, ok, ["python", "train.py"], attempt(1))
    store.finish_run(db, ok, "succeeded", {"lr": 0.1})

    healed = store.start_run(db, sweep_id, "run-002", {"lr": 0.2, "batch_size": 64})
    store.save_attempt(db, healed, ["python", "train.py"], attempt(1, "OOM"))
    store.save_attempt(db, healed, ["python", "train.py"], attempt(2))
    store.finish_run(db, healed, "succeeded", {"lr": 0.2, "batch_size": 32})

    broken = store.start_run(db, sweep_id, "run-003", {"lr": 0.3})
    store.save_attempt(db, broken, ["python", "train.py"], attempt(1, "CODE_BUG"))
    store.finish_run(db, broken, "failed", {"lr": 0.3}, "CODE_BUG", "retrying won't help")
    store.finish_sweep(db, sweep_id)

    [sweep] = store.list_sweeps(db)
    assert (sweep["runs_started"], sweep["succeeded"], sweep["failed"]) == (3, 2, 1)
    assert sweep["finished_at"] is not None

    runs = store.list_runs(db, sweep_id)
    assert [(r["name"], r["status"], r["attempts"]) for r in runs] == [
        ("run-001", "succeeded", 1),
        ("run-002", "succeeded", 2),
        ("run-003", "failed", 1),
    ]
    assert runs[2]["failure"] == "CODE_BUG"


def test_unfinished_work_is_visible(db, tmp_path):
    # What a crash halfway through leaves behind: a running run, an unfinished sweep.
    sweep_id = store.start_sweep(db, "crashed", tmp_path, total_runs=5)
    store.start_run(db, sweep_id, "run-001", {})
    [sweep] = store.list_sweeps(db)
    assert sweep["finished_at"] is None
    assert store.list_runs(db, sweep_id)[0]["status"] == "running"


def test_a_sweep_with_no_runs_is_still_listed(db, tmp_path):
    store.start_sweep(db, "empty", tmp_path, total_runs=0)
    [sweep] = store.list_sweeps(db)
    assert (sweep["runs_started"], sweep["succeeded"], sweep["failed"]) == (0, 0, 0)


def test_foreign_keys_are_enforced(db):
    with pytest.raises(sqlite3.IntegrityError):
        store.start_run(db, sweep_id=999, name="orphan", params={})  # no such sweep


def test_values_are_never_treated_as_sql(db, tmp_path):
    name = "x'); DROP TABLE runs; --"
    sweep_id = store.start_sweep(db, name, tmp_path, total_runs=1)
    store.start_run(db, sweep_id, name, {})
    assert store.list_runs(db, sweep_id)[0]["name"] == name  # stored as plain text
