"""Tests for the training-side SDK: what it writes to the metrics file."""

import json
import math
import time

import pytest

import sweepr
from sweepr import sdk


@pytest.fixture
def metrics_file(tmp_path, monkeypatch):
    """Point the SDK at a temp file with fast heartbeats, and reset it around each test."""
    path = tmp_path / "metrics.jsonl"
    monkeypatch.setenv(sdk.METRICS_FILE_ENV, str(path))
    monkeypatch.setenv(sdk.HEARTBEAT_ENV, "0.05")
    sdk._reset_for_tests()
    yield path
    sdk._reset_for_tests()


def read_records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_log_checkpoint_and_finish_are_written_in_order(metrics_file):
    sweepr.log({"loss": 0.5}, step=1)
    sweepr.checkpoint("ckpt/step_1.json", step=1)
    sweepr.finish()

    records = [r for r in read_records(metrics_file) if r["type"] != "heartbeat"]
    assert [r["type"] for r in records] == ["metric", "checkpoint", "finish"]
    assert records[0]["data"] == {"loss": 0.5}
    assert records[1]["path"] == "ckpt/step_1.json"


def test_nan_loss_survives_the_round_trip(metrics_file):
    sweepr.log({"loss": math.nan}, step=3)
    sweepr.finish()

    metric = read_records(metrics_file)[0]
    assert math.isnan(metric["data"]["loss"])


def test_heartbeats_report_time_of_last_log(metrics_file):
    sweepr.log({"loss": 1.0}, step=1)
    time.sleep(0.2)  # long enough for a few heartbeats
    sweepr.finish()

    records = read_records(metrics_file)
    logged_at = records[0]["t"]
    heartbeats = [r for r in records if r["type"] == "heartbeat"]
    assert len(heartbeats) >= 2
    assert heartbeats[-1]["last_log_t"] == pytest.approx(logged_at, abs=0.01)


def test_sdk_does_nothing_without_the_supervisor(tmp_path, monkeypatch):
    monkeypatch.delenv(sdk.METRICS_FILE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)
    sdk._reset_for_tests()

    sweepr.log({"loss": 0.1}, step=1)
    sweepr.finish()

    assert list(tmp_path.iterdir()) == []  # no file was created


def test_resume_path(monkeypatch):
    monkeypatch.delenv(sdk.RESUME_ENV, raising=False)
    assert sweepr.resume_path() is None
    monkeypatch.setenv(sdk.RESUME_ENV, "ckpt/step_50.json")
    assert sweepr.resume_path() == "ckpt/step_50.json"
