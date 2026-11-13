"""Run history: saves every sweep, run and attempt in a SQLite database file.

Three tables, each row pointing at its parent:

    sweeps   one row per `supervise sweep`
    runs     one row per grid combination      (runs.sweep_id  -> sweeps.id)
    attempts one row per try of a run          (attempts.run_id -> runs.id)

Runs and attempts are separate because one run can take several attempts: "run-003 hit
OOM on attempt 1, then succeeded on attempt 2 with batch_size=32".

Results are saved as the sweep goes, one small transaction at a time. If the supervisor
crashes halfway, the database still holds everything that finished, and a run left as
"running" shows where it stopped.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    folder      TEXT NOT NULL,      -- where the logs and checkpoints are
    total_runs  INTEGER NOT NULL,
    started_at  TEXT NOT NULL,      -- SQLite has no date type; ISO text sorts correctly
    finished_at TEXT                -- NULL while running, or if it was interrupted
);

CREATE TABLE IF NOT EXISTS runs (
    id       INTEGER PRIMARY KEY,
    sweep_id INTEGER NOT NULL REFERENCES sweeps(id),
    name     TEXT NOT NULL,         -- e.g. run-003
    params   TEXT NOT NULL,         -- JSON, because every sweep has different grid keys
    status   TEXT NOT NULL,         -- running, succeeded, failed or interrupted
    failure  TEXT,                  -- the failure label, if it failed
    reason   TEXT,                  -- why it ended, in a few words
    UNIQUE (sweep_id, name)
);

CREATE TABLE IF NOT EXISTS attempts (
    id              INTEGER PRIMARY KEY,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    number          INTEGER NOT NULL,  -- 1, 2, 3...
    command         TEXT NOT NULL,     -- JSON list: the exact command this attempt ran
    exit_code       INTEGER,
    failure         TEXT,              -- NULL if this attempt succeeded
    reason          TEXT,
    last_step       INTEGER,
    last_checkpoint TEXT,
    duration_s      REAL NOT NULL,
    log_path        TEXT NOT NULL,
    finished_at     TEXT NOT NULL,
    UNIQUE (run_id, number)
);

-- Looking up the runs of a sweep, and the attempts of a run, are the common queries.
CREATE INDEX IF NOT EXISTS runs_by_sweep ON runs (sweep_id);
CREATE INDEX IF NOT EXISTS attempts_by_run ON attempts (run_id);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open (or create) the database file and make sure the tables exist."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row  # rows can be read by column name: row["status"]
    # SQLite ignores REFERENCES unless this is turned on, for every connection.
    db.execute("PRAGMA foreign_keys = ON")
    # Write-ahead logging: other programs (the dashboard, later) can read while we write.
    db.execute("PRAGMA journal_mode = WAL")
    db.executescript(SCHEMA)
    return db


def now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


# Every write below runs inside `with db:`, which commits the transaction when the block
# ends, or rolls it back if an error happens. Values always go in through ? placeholders,
# never by formatting them into the SQL string, which is what prevents SQL injection.


def start_sweep(db: sqlite3.Connection, name: str, folder: Path, total_runs: int) -> int:
    with db:
        cursor = db.execute(
            "INSERT INTO sweeps (name, folder, total_runs, started_at) VALUES (?, ?, ?, ?)",
            (name, str(folder), total_runs, now()),
        )
    return cursor.lastrowid


def finish_sweep(db: sqlite3.Connection, sweep_id: int) -> None:
    with db:
        db.execute("UPDATE sweeps SET finished_at = ? WHERE id = ?", (now(), sweep_id))


def start_run(db: sqlite3.Connection, sweep_id: int, name: str, params: dict) -> int:
    with db:
        cursor = db.execute(
            "INSERT INTO runs (sweep_id, name, params, status) VALUES (?, ?, ?, 'running')",
            (sweep_id, name, json.dumps(params)),
        )
    return cursor.lastrowid


def save_attempt(db: sqlite3.Connection, run_id: int, command: list[str], result) -> None:
    """Save one finished attempt (a supervisor.AttemptResult)."""
    with db:
        db.execute(
            """
            INSERT INTO attempts (run_id, number, command, exit_code, failure, reason,
                                  last_step, last_checkpoint, duration_s, log_path, finished_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                result.attempt,
                json.dumps(command),
                result.exit_code,
                result.classification.failure,
                result.classification.reason,
                result.last_step,
                result.last_checkpoint,
                result.duration_s,
                str(result.log_path),
                now(),
            ),
        )


def finish_run(
    db: sqlite3.Connection,
    run_id: int,
    status: str,
    params: dict,
    failure: str | None = None,
    reason: str | None = None,
) -> None:
    with db:
        db.execute(
            "UPDATE runs SET status = ?, params = ?, failure = ?, reason = ? WHERE id = ?",
            (status, json.dumps(params), failure, reason, run_id),
        )


# --- Reading the history back ---


def list_sweeps(db: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    """The newest sweeps, each with how many of its runs succeeded and failed."""
    return db.execute(
        """
        SELECT sweeps.id, sweeps.name, sweeps.started_at, sweeps.finished_at,
               sweeps.total_runs,
               COUNT(runs.id) AS runs_started,
               COALESCE(SUM(runs.status = 'succeeded'), 0) AS succeeded,
               COALESCE(SUM(runs.status = 'failed'), 0) AS failed
        FROM sweeps
        LEFT JOIN runs ON runs.sweep_id = sweeps.id   -- LEFT: keep sweeps with no runs yet
        GROUP BY sweeps.id
        ORDER BY sweeps.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def list_runs(db: sqlite3.Connection, sweep_id: int) -> list[sqlite3.Row]:
    """The runs of one sweep, each with its number of attempts."""
    return db.execute(
        """
        SELECT runs.name, runs.status, runs.failure, runs.params, runs.reason,
               COUNT(attempts.id) AS attempts
        FROM runs
        LEFT JOIN attempts ON attempts.run_id = runs.id
        WHERE runs.sweep_id = ?
        GROUP BY runs.id
        ORDER BY runs.name
        """,
        (sweep_id,),
    ).fetchall()
