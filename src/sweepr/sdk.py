"""Training-side SDK.

A training script calls `log`, `checkpoint`, and `finish`. Every call appends one JSON
record to the file named by SWEEPR_METRICS_FILE, which the supervisor tails. When that
variable is unset (the script was started without the supervisor), every call is a no-op,
so the same script runs fine on its own.

A background thread also writes a heartbeat every few seconds. The heartbeat carries the
time of the last `log` call, which lets the supervisor tell two cases apart:
  - no heartbeats at all: the whole process is frozen or gone
  - heartbeats, but `last_log_t` is stale: the process is alive but training is stuck
"""

from __future__ import annotations

import atexit
import json
import os
import threading
import time
from typing import Any

METRICS_FILE_ENV = "SWEEPR_METRICS_FILE"
RESUME_ENV = "SWEEPR_RESUME_FROM"
HEARTBEAT_ENV = "SWEEPR_HEARTBEAT_INTERVAL"
DEFAULT_HEARTBEAT_INTERVAL = 5.0


class _Run:
    """Writes records for the current process. Created lazily on the first SDK call."""

    def __init__(self, path: str | None, heartbeat_interval: float) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._file = open(path, "a", encoding="utf-8") if path else None  # noqa: SIM115
        self._last_log_t: float | None = None
        self._stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        if self._file is not None:
            self._heartbeat = threading.Thread(
                target=self._heartbeat_loop, args=(heartbeat_interval,), daemon=True
            )
            self._heartbeat.start()

    @property
    def enabled(self) -> bool:
        return self._file is not None

    def write(self, record: dict[str, Any]) -> None:
        if self._file is None:
            return
        record = {"t": time.time(), **record}
        # NaN and inf are written as the bare tokens NaN/Infinity. That is not strict JSON,
        # but Python's json module reads it back, and the supervisor needs to see NaN losses.
        line = json.dumps(record, allow_nan=True)
        with self._lock:
            self._file.write(line + "\n")
            # Flush on every write so the supervisor sees records immediately, and so
            # nothing is lost in a buffer if the process is killed.
            self._file.flush()

    def log(self, metrics: dict[str, float], step: int | None) -> None:
        self._last_log_t = time.time()
        self.write({"type": "metric", "step": step, "data": metrics})

    def _heartbeat_loop(self, interval: float) -> None:
        while not self._stop.wait(interval):
            self.write({"type": "heartbeat", "last_log_t": self._last_log_t})

    def close(self) -> None:
        self._stop.set()
        if self._heartbeat is not None:
            self._heartbeat.join(timeout=1)
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


_run: _Run | None = None
_run_lock = threading.Lock()


def _get_run() -> _Run:
    global _run
    with _run_lock:
        if _run is None:
            interval = float(os.environ.get(HEARTBEAT_ENV, DEFAULT_HEARTBEAT_INTERVAL))
            _run = _Run(os.environ.get(METRICS_FILE_ENV), interval)
            atexit.register(_run.close)
        return _run


def log(metrics: dict[str, float], step: int | None = None) -> None:
    """Record scalar metrics, for example `log({"loss": 0.31}, step=100)`."""
    _get_run().log(metrics, step)


def checkpoint(path: str, step: int | None = None) -> None:
    """Record that a checkpoint was saved at `path`. Call it after the file is fully written."""
    _get_run().write({"type": "checkpoint", "step": step, "path": str(path)})


def finish() -> None:
    """Mark the run as finished cleanly and stop the heartbeat thread."""
    run = _get_run()
    run.write({"type": "finish"})
    run.close()


def resume_path() -> str | None:
    """Checkpoint to resume from, if the supervisor is retrying this run. None on a fresh start."""
    return os.environ.get(RESUME_ENV) or None


def _reset_for_tests() -> None:
    """Close and forget the current run so a test can start a fresh one."""
    global _run
    with _run_lock:
        if _run is not None:
            _run.close()
        _run = None
