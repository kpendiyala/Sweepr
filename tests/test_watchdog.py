"""Watchdog tests. Time is just a number we pass in, so nothing here sleeps."""

import math

from sweepr.classify import HEARTBEAT_TIMEOUT, NAN_LOSS, PROGRESS_TIMEOUT
from sweepr.watchdog import Watchdog, WatchdogConfig

CONFIG = WatchdogConfig(heartbeat_timeout=10, progress_timeout=60)
HEARTBEAT = {"type": "heartbeat", "last_log_t": 0}


def metric(step, loss=0.5):
    return {"type": "metric", "step": step, "data": {"loss": loss}}


def test_healthy_run_is_left_alone():
    watchdog = Watchdog(CONFIG, started_at=0)
    for t in range(0, 300, 5):
        watchdog.observe(metric(t), now=t)
        assert watchdog.check(now=t + 1) is None


def test_not_armed_until_the_first_record():
    watchdog = Watchdog(CONFIG, started_at=0)
    assert watchdog.check(now=10_000) is None


def test_startup_timeout_when_configured():
    watchdog = Watchdog(WatchdogConfig(startup_timeout=30), started_at=0)
    assert watchdog.check(now=29) is None
    assert watchdog.check(now=31) == HEARTBEAT_TIMEOUT


def test_no_records_at_all_means_frozen():
    watchdog = Watchdog(CONFIG, started_at=0)
    watchdog.observe(metric(1), now=5)
    assert watchdog.check(now=15) is None
    assert watchdog.check(now=16) == HEARTBEAT_TIMEOUT


def test_heartbeats_without_metrics_means_stuck():
    watchdog = Watchdog(CONFIG, started_at=0)
    watchdog.observe(metric(1), now=0)
    for t in range(5, 61, 5):
        watchdog.observe(HEARTBEAT, now=t)
        assert watchdog.check(now=t) is None
    watchdog.observe(HEARTBEAT, now=65)
    assert watchdog.check(now=65) == PROGRESS_TIMEOUT


def test_nan_loss_stops_immediately():
    watchdog = Watchdog(CONFIG, started_at=0)
    watchdog.observe(metric(1), now=0)
    watchdog.observe(metric(2, loss=math.nan), now=1)
    assert watchdog.check(now=1) == NAN_LOSS


def test_nan_check_can_be_turned_off():
    watchdog = Watchdog(WatchdogConfig(nan_key=None), started_at=0)
    watchdog.observe(metric(1, loss=math.nan), now=0)
    assert watchdog.check(now=0) is None


def test_no_timeouts_after_finish():
    watchdog = Watchdog(CONFIG, started_at=0)
    watchdog.observe(metric(1), now=0)
    watchdog.observe({"type": "finish"}, now=1)
    assert watchdog.check(now=1_000) is None


def test_tracks_last_step_and_checkpoint():
    watchdog = Watchdog(CONFIG, started_at=0)
    watchdog.observe(metric(10), now=0)
    watchdog.observe({"type": "checkpoint", "step": 10, "path": "ckpt/step_10.json"}, now=0)
    watchdog.observe(metric(11), now=1)
    assert watchdog.last_step == 11
    assert watchdog.last_checkpoint == "ckpt/step_10.json"
