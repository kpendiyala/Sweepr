"""Failure notifications: one message to a Slack-style webhook when a sweep needs a person.

Set SWEEPR_WEBHOOK_URL to a Slack incoming-webhook URL (or anything that accepts the same
JSON, such as a Discord webhook URL ending in /slack). It is read from the environment and
not from the sweep file, because anyone with the URL can post to your channel, and sweep
files get committed to git.

Only sweeps with failed runs send a message, and only one per sweep: a code bug can fail
every run within seconds, and one alert per run would bury the channel.
"""

import json
import os
import sys
import urllib.error
import urllib.request

WEBHOOK_ENV = "SWEEPR_WEBHOOK_URL"
MAX_LISTED_RUNS = 10  # keep the message short; the rest is in `supervise history`


def failure_message(sweep_name: str, sweep_id: int, summary: str, failed_runs: list) -> str:
    """The text of the alert: the summary line, then one line per failed run.

    `failed_runs` are sweep.RunResult objects for the runs that gave up.
    """
    lines = [f"Sweepr: sweep '{sweep_name}' needs attention.", summary]
    for run in failed_runs[:MAX_LISTED_RUNS]:
        log = run.attempts[-1].log_path
        lines.append(f"- {run.name} {run.failure}: {run.reason} (log: {log})")
    if len(failed_runs) > MAX_LISTED_RUNS:
        lines.append(f"... and {len(failed_runs) - MAX_LISTED_RUNS} more")
    lines.append(f"Details: supervise history {sweep_id}")
    return "\n".join(lines)


def send(url: str, text: str, timeout: float = 10.0) -> bool:
    """POST {"text": ...} to the webhook. Returns False, with a warning, if it fails.

    A failed notification must never fail the sweep: the results are already saved in the
    database, so we only warn.
    """
    body = json.dumps({"text": text}).encode("utf-8")
    try:
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=timeout):
            return True  # urlopen raises an HTTPError for any 4xx or 5xx status
    # URLError: no connection or an error status. ValueError: not a valid URL.
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(f"[sweepr] warning: could not send the notification: {error}", file=sys.stderr)
        return False


def notify_if_needed(sweep_name: str, sweep_id: int, summary: str, failed_runs: list) -> None:
    """Send one alert if any run failed and SWEEPR_WEBHOOK_URL is set. Otherwise do nothing."""
    url = os.environ.get(WEBHOOK_ENV)
    if not url or not failed_runs:
        return
    if send(url, failure_message(sweep_name, sweep_id, summary, failed_runs)):
        print("[sweepr] sent a failure notification", file=sys.stderr)
