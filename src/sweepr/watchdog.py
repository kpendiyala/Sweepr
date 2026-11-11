"""Health monitoring: decides when a running attempt should be stopped.

It catches three problems: the process froze (no records at all), training is stuck
(heartbeats but no new metrics), or the loss went NaN.

This file has no clocks, files, or processes. The supervisor calls `observe` for each record
it reads from the metrics file and `check` on every poll, passing in the current time. That
means tests can pass made-up times and never need to sleep.

The times are when the supervisor *saw* each record (its own time.monotonic() clock), never
the `t` the script wrote into the record. The script's wall clock can jump (for example
when NTP corrects it), but a monotonic clock only moves forward.
"""

import math
from dataclasses import dataclass

from sweepr.classify import HEARTBEAT_TIMEOUT, NAN_LOSS, PROGRESS_TIMEOUT


@dataclass
class WatchdogConfig:
    # No record at all (heartbeat or metric) for this many seconds: the process is frozen.
    # Keep it several times the SDK heartbeat interval (5 s) to avoid false alarms.
    heartbeat_timeout: float = 60.0
    # Records keep arriving, but no new metric for this long: training is stuck.
    progress_timeout: float = 600.0
    # No record at all this long after launch. None turns it off, which is the default,
    # because imports and data loading can be slow and scripts without the SDK never
    # write a record.
    startup_timeout: float | None = None
    # Stop early when this metric becomes NaN or inf. None turns the check off.
    nan_key: str | None = "loss"


class Watchdog:
    def __init__(self, config: WatchdogConfig, started_at: float):
        self.config = config
        self.started_at = started_at
        self.first_record_at = None  # when we saw the first record of any type
        self.last_record_at = None  # when we saw the latest record of any type
        self.last_metric_at = None  # when we saw the latest metric record
        self.last_step = None
        self.last_checkpoint = None
        self.saw_nonfinite_loss = False
        self.finished = False

    def observe(self, record: dict, now: float) -> None:
        """Update our state from one record of the metrics file."""
        if self.first_record_at is None:
            self.first_record_at = now
        self.last_record_at = now

        record_type = record.get("type")
        if record_type == "metric":
            self.last_metric_at = now
            if record.get("step") is not None:
                self.last_step = record["step"]
            metrics = record.get("data") or {}
            value = metrics.get(self.config.nan_key)
            if isinstance(value, (int, float)) and not math.isfinite(value):
                self.saw_nonfinite_loss = True
        elif record_type == "checkpoint":
            self.last_checkpoint = record.get("path")
        elif record_type == "finish":
            self.finished = True

    def check(self, now: float) -> str | None:
        """Return a kill reason if the attempt should be stopped now, or None to let it run."""
        config = self.config

        if self.saw_nonfinite_loss:
            return NAN_LOSS

        if self.finished:
            # Training is done and heartbeats have stopped; let the process exit on its own.
            return None

        if self.last_record_at is None:
            # We have never heard from the script, so the normal timeouts don't apply yet.
            if config.startup_timeout is not None:
                if now - self.started_at > config.startup_timeout:
                    return HEARTBEAT_TIMEOUT
            return None

        if now - self.last_record_at > config.heartbeat_timeout:
            return HEARTBEAT_TIMEOUT

        # Before the first metric arrives, count progress from the first record we saw.
        if self.last_metric_at is not None:
            progress_since = self.last_metric_at
        else:
            progress_since = self.first_record_at
        if now - progress_since > config.progress_timeout:
            return PROGRESS_TIMEOUT

        return None
