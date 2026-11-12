"""Training-side SDK: the few functions a training script calls.

Every call appends one JSON record (one line) to the file named by SWEEPR_METRICS_FILE.
The supervisor reads that file while the script runs: each `log` call shows the run is still
making progress, and each `checkpoint` call tells it where a retry can resume from.

When the variable is not set (the script was started without the supervisor), every call
does nothing, so the same script still runs on its own.
"""

import atexit
import json
import os
import time

METRICS_FILE_ENV = "SWEEPR_METRICS_FILE"
RESUME_ENV = "SWEEPR_RESUME_FROM"

_file = None  # the open metrics file, or None before the first call / without the supervisor
_opened = False  # True once we've looked at SWEEPR_METRICS_FILE


def _write(record: dict) -> None:
    global _file, _opened
    if not _opened:
        _opened = True
        path = os.environ.get(METRICS_FILE_ENV)
        if path:
            _file = open(path, "a", encoding="utf-8")
            atexit.register(_close)  # close the file even if the script never calls finish()
    if _file is None:
        return
    record["t"] = time.time()
    # NaN and inf are written as the bare words NaN / Infinity. That is not strict JSON, but
    # Python's json module reads it back, and the supervisor needs to see NaN losses.
    _file.write(json.dumps(record) + "\n")
    # Flush every time, so the supervisor sees the record right away and nothing is stuck in
    # a buffer if the process is killed.
    _file.flush()


def _close() -> None:
    global _file
    if _file is not None:
        _file.close()
        _file = None


def log(metrics: dict, step: int | None = None) -> None:
    """Record scalar metrics, for example `log({"loss": 0.31}, step=100)`."""
    _write({"type": "metric", "step": step, "data": metrics})


def checkpoint(path: str, step: int | None = None) -> None:
    """Record that a checkpoint was saved at `path`. Call it after the file is fully written."""
    _write({"type": "checkpoint", "step": step, "path": str(path)})


def finish() -> None:
    """Mark the run as finished cleanly."""
    _write({"type": "finish"})
    _close()


def resume_path() -> str | None:
    """The checkpoint to resume from when the supervisor is retrying this run, else None."""
    return os.environ.get(RESUME_ENV) or None


def _reset_for_tests() -> None:
    """Close and forget the metrics file, so each test starts fresh."""
    global _opened
    _close()
    _opened = False
