"""Checks that each failure mode of examples/flaky_train.py fails the way the supervisor expects.

These run the script as a real subprocess, the same way the supervisor will.
"""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

# A killed process reports -9 (killed by signal 9) on Linux and macOS, and exit code 9 on Windows.
KILLED = 9 if sys.platform == "win32" else -9

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "flaky_train.py"


def run(tmp_path, *args, attempt=1, resume=None, timeout=30):
    metrics = tmp_path / f"metrics_{attempt}.jsonl"
    env = {
        **os.environ,
        "SWEEPR_METRICS_FILE": str(metrics),
        "SWEEPR_HEARTBEAT_INTERVAL": "0.05",
        "SWEEPR_ATTEMPT": str(attempt),
    }
    if resume:
        env["SWEEPR_RESUME_FROM"] = str(resume)
    cmd = [sys.executable, str(SCRIPT), "--steps", "100", "--step-time", "0", "--ckpt-dir"]
    cmd += [str(tmp_path / "ckpt"), *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return proc, read_records(metrics)


def read_records(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def metrics_of(records):
    return [r for r in records if r["type"] == "metric"]


def test_clean_run_converges(tmp_path):
    proc, records = run(tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert records[-1]["type"] == "finish"
    final = metrics_of(records)[-1]["data"]
    assert final["w"] == pytest.approx(3.0, abs=0.2)
    assert final["b"] == pytest.approx(2.0, abs=0.2)


def test_oom_fails_with_cuda_message_above_threshold(tmp_path):
    proc, _ = run(tmp_path, "--fail-mode", "oom", "--batch-size", "64", "--oom-above", "32")
    assert proc.returncode == 1
    assert "CUDA out of memory" in proc.stderr


def test_oom_succeeds_once_batch_size_is_halved(tmp_path):
    proc, _ = run(tmp_path, "--fail-mode", "oom", "--batch-size", "32", "--oom-above", "32")
    assert proc.returncode == 0, proc.stderr


def test_nan_mode_logs_nan_loss_from_fail_step(tmp_path):
    proc, records = run(tmp_path, "--fail-mode", "nan", "--fail-step", "20")
    losses = {r["step"]: r["data"]["loss"] for r in metrics_of(records)}
    assert not math.isnan(losses[19])
    assert math.isnan(losses[20])
    # The script itself keeps going; stopping it is the supervisor's job.
    assert proc.returncode == 0


def test_bug_mode_fails_on_every_attempt(tmp_path):
    for attempt in (1, 2, 3):
        proc, records = run(tmp_path, "--fail-mode", "bug", attempt=attempt)
        assert proc.returncode == 1
        assert "TypeError" in proc.stderr
        assert metrics_of(records) == []


def test_crash_mode_is_killed_then_recovers_on_retry(tmp_path):
    proc, records = run(tmp_path, "--fail-mode", "crash", "--fail-step", "30")
    assert proc.returncode == KILLED
    assert metrics_of(records)[-1]["step"] == 29

    proc, _ = run(tmp_path, "--fail-mode", "crash", "--fail-step", "30", attempt=2)
    assert proc.returncode == 0, proc.stderr


def test_hang_mode_keeps_heartbeating_but_stops_logging(tmp_path):
    metrics = tmp_path / "metrics.jsonl"
    env = {
        **os.environ,
        "SWEEPR_METRICS_FILE": str(metrics),
        "SWEEPR_HEARTBEAT_INTERVAL": "0.05",
    }
    cmd = [sys.executable, str(SCRIPT), "--step-time", "0", "--ckpt-dir", str(tmp_path / "ckpt")]
    cmd += ["--fail-mode", "hang", "--fail-step", "10"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(1.0)
        assert proc.poll() is None, "a hung run should still be alive"
    finally:
        proc.kill()
        proc.wait()

    records = read_records(metrics)
    assert metrics_of(records)[-1]["step"] == 9
    last_log_t = metrics_of(records)[-1]["t"]
    late_heartbeats = [r for r in records if r["type"] == "heartbeat" and r["t"] > last_log_t + 0.5]
    assert late_heartbeats, "heartbeats should continue while training is stuck"
    assert all(r["last_log_t"] == pytest.approx(last_log_t, abs=0.01) for r in late_heartbeats)


def test_resume_continues_from_checkpoint(tmp_path):
    proc, _ = run(tmp_path, "--fail-mode", "crash", "--fail-step", "60", "--ckpt-every", "25")
    assert proc.returncode == KILLED
    latest = tmp_path / "ckpt" / "step_50.json"
    assert latest.exists()

    proc, records = run(tmp_path, attempt=2, resume=latest)
    assert proc.returncode == 0, proc.stderr
    assert "resumed from" in proc.stdout
    assert metrics_of(records)[0]["step"] == 51
