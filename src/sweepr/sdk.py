"""Training-side SDK: the few functions a training script calls.

Every call appends one JSON record (one line) to the file named by SWEEPR_METRICS_FILE.
The supervisor reads that file while the script runs. When the variable is not set (the
script was started without the supervisor), every call does nothing, so the same script
still runs on its own.

A background thread also writes a heartbeat record every few seconds. Each heartbeat carries
the time of the last `log` call, which lets the supervisor tell two problems apart:
  - no records at all: the whole process is frozen or gone
  - heartbeats keep coming, but no new metrics: the process is alive but training is stuck
"""

import atexit
import json
import os
import threading
import time

METRICS_FILE_ENV = "SWEEPR_METRICS_FILE"
RESUME_ENV = "SWEEPR_RESUME_FROM"
HEARTBEAT_ENV = "SWEEPR_HEARTBEAT_INTERVAL"
DEFAULT_HEARTBEAT_INTERVAL = 5.0


class _Run:
    """Owns the metrics file and the heartbeat thread for the current process."""

    def __init__(self, path: str | None, heartbeat_interval: float):
        # The heartbeat thread and the training code both write to the file,
        # so every write holds this lock.
        self.lock = threading.Lock()
        self.file = None
        self.last_log_t = None
        self.stop_event = threading.Event()
        self.heartbeat_thread = None

        if path:
            self.file = open(path, "a", encoding="utf-8")  # closed in close()
            # daemon=True: this thread never keeps the process alive on its own.
            self.heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, args=(heartbeat_interval,), daemon=True
            )
            self.heartbeat_thread.start()

    def write(self, record: dict) -> None:
        with self.lock:
            if self.file is None:
                return
            record["t"] = time.time()
            # NaN and inf are written as the bare words NaN / Infinity. That is not strict
            # JSON, but Python's json module reads it back, and the supervisor needs to see
            # NaN losses.
            self.file.write(json.dumps(record) + "\n")
            # Flush every time, so the supervisor sees the record right away and nothing is
            # stuck in a buffer if the process is killed.
            self.file.flush()

    def _heartbeat_loop(self, interval: float) -> None:
        while True:
            # wait() sleeps for `interval` seconds, but returns True early if close() was called.
            stopped = self.stop_event.wait(interval)
            if stopped:
                return
            self.write({"type": "heartbeat", "last_log_t": self.last_log_t})

    def close(self) -> None:
        self.stop_event.set()
        if self.heartbeat_thread is not None:
            self.heartbeat_thread.join(timeout=1)
        with self.lock:
            if self.file is not None:
                self.file.close()
                self.file = None


# One run per process, created on the first SDK call.
_run = None
_run_lock = threading.Lock()


def _get_run() -> _Run:
    global _run
    with _run_lock:
        if _run is None:
            interval = float(os.environ.get(HEARTBEAT_ENV, DEFAULT_HEARTBEAT_INTERVAL))
            _run = _Run(os.environ.get(METRICS_FILE_ENV), interval)
            atexit.register(_run.close)  # close the file even if the script never calls finish()
        return _run


def log(metrics: dict, step: int | None = None) -> None:
    """Record scalar metrics, for example `log({"loss": 0.31}, step=100)`."""
    run = _get_run()
    run.last_log_t = time.time()
    run.write({"type": "metric", "step": step, "data": metrics})


def checkpoint(path: str, step: int | None = None) -> None:
    """Record that a checkpoint was saved at `path`. Call it after the file is fully written."""
    _get_run().write({"type": "checkpoint", "step": step, "path": str(path)})


def finish() -> None:
    """Mark the run as finished cleanly and stop the heartbeat thread."""
    run = _get_run()
    run.write({"type": "finish"})
    run.close()


def resume_path() -> str | None:
    """The checkpoint to resume from when the supervisor is retrying this run, else None."""
    return os.environ.get(RESUME_ENV) or None


def _reset_for_tests() -> None:
    """Close and forget the current run, so each test starts fresh."""
    global _run
    with _run_lock:
        if _run is not None:
            _run.close()
        _run = None
