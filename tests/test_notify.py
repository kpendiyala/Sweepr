"""Tests for failure notifications, against a tiny local HTTP server instead of Slack."""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sweepr import notify
from sweepr.classify import Classification
from sweepr.cli import main
from sweepr.supervisor import AttemptResult
from sweepr.sweep import RunResult


class FakeWebhook:
    """Runs an HTTP server on a free local port and records every POST body it receives."""

    def __init__(self, status=200):
        self.received = []
        webhook = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                webhook.received.append(json.loads(self.rfile.read(length)))
                self.send_response(status)
                self.end_headers()

            def log_message(self, *args):
                pass  # keep test output quiet

        self.server = HTTPServer(("127.0.0.1", 0), Handler)  # port 0: pick any free port
        self.url = f"http://127.0.0.1:{self.server.server_port}/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def webhook():
    server = FakeWebhook()
    yield server
    server.close()


def failed_run(name, failure="CODE_BUG"):
    attempt = AttemptResult(
        attempt=1,
        classification=Classification(failure, "TypeError"),
        exit_code=1,
        killed_by=None,
        last_step=None,
        last_checkpoint=None,
        duration_s=0.1,
        log_path=Path(f"runs/s/{name}/attempt-1.log"),
    )
    return RunResult(
        name, {"lr": 0.1}, [attempt], False, failure, f"{failure}: retrying won't help"
    )


SUMMARY = "3/3 runs finished: 1 succeeded, 2 failed (CODE_BUG 2)"


def test_sends_one_message_listing_the_failed_runs(webhook, monkeypatch):
    monkeypatch.setenv(notify.WEBHOOK_ENV, webhook.url)
    notify.notify_if_needed("lr", 7, SUMMARY, [failed_run("run-002"), failed_run("run-003")])

    assert len(webhook.received) == 1
    text = webhook.received[0]["text"]
    assert "sweep 'lr' needs attention" in text
    assert SUMMARY in text
    assert "- run-002 CODE_BUG: CODE_BUG: retrying won't help" in text
    assert str(Path("runs/s/run-003/attempt-1.log")) in text
    assert "supervise history 7" in text


def test_nothing_is_sent_without_failures_or_without_a_url(webhook, monkeypatch):
    monkeypatch.setenv(notify.WEBHOOK_ENV, webhook.url)
    notify.notify_if_needed("lr", 1, "2/2 runs finished: 2 succeeded", [])
    monkeypatch.delenv(notify.WEBHOOK_ENV)
    notify.notify_if_needed("lr", 1, SUMMARY, [failed_run("run-001")])
    assert webhook.received == []


def test_long_lists_are_cut_short():
    runs = [failed_run(f"run-{i:03d}") for i in range(25)]
    text = notify.failure_message("big", 1, SUMMARY, runs)
    assert text.count("\n- ") == notify.MAX_LISTED_RUNS
    assert "... and 15 more" in text


def test_a_broken_webhook_only_warns(capsys):
    server = FakeWebhook(status=500)
    try:
        assert notify.send(server.url, "hello") is False  # server error
    finally:
        server.close()
    assert notify.send("http://127.0.0.1:9/nothing-listens-here", "hello", timeout=2) is False
    assert notify.send("not a url", "hello") is False
    assert capsys.readouterr().err.count("could not send the notification") == 3


def test_a_real_sweep_sends_the_alert(webhook, monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples" / "flaky_train.py"
    sweep_file = tmp_path / "sweep.yaml"
    sweep_file.write_text(
        f"command: [{sys.executable!r}, {str(script)!r},"
        " --steps, '5', --step-time, '0', --fail-mode, '{mode}', --ckpt-dir, '{run_dir}/ckpt']\n"
        "grid: {mode: [none, bug]}\n"
    )
    monkeypatch.setenv(notify.WEBHOOK_ENV, webhook.url)
    assert main(["sweep", str(sweep_file), "--runs-dir", str(tmp_path / "runs")]) == 1
    assert len(webhook.received) == 1
    assert "run-002 CODE_BUG" in webhook.received[0]["text"]
