"""End-to-end tests: supervise the real fault-injection script and check each outcome.

Timeouts are scaled down (heartbeats every 0.05 s, kills after about 1 s) so the tests stay
fast. Only the numbers are smaller; the logic is the same as with the real defaults.
"""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from sweepr.classify import (
    CODE_BUG,
    HEARTBEAT_TIMEOUT,
    HUNG,
    NAN,
    NAN_LOSS,
    NODE_FAILURE,
    OOM,
    PROGRESS_TIMEOUT,
)
from sweepr.cli import main
from sweepr.supervisor import MetricsReader, run_attempt
from sweepr.watchdog import WatchdogConfig

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "flaky_train.py"
FAST = WatchdogConfig(heartbeat_timeout=1.0, progress_timeout=1.0)
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="uses POSIX signals")


@pytest.fixture(autouse=True)
def fast_heartbeats(monkeypatch):
    monkeypatch.setenv("SWEEPR_HEARTBEAT_INTERVAL", "0.05")


def flaky_command(tmp_path, *extra_args):
    """The command line for flaky_train.py: 60 quick steps, a checkpoint every 20."""
    command = [sys.executable, str(SCRIPT), "--steps", "60", "--step-time", "0.01"]
    command += ["--ckpt-dir", str(tmp_path / "ckpt"), "--ckpt-every", "20"]
    return command + list(extra_args)


def supervise(tmp_path, command, attempt=1, resume_from=None):
    return run_attempt(
        command,
        run_dir=tmp_path / "run",
        attempt=attempt,
        resume_from=resume_from,
        config=FAST,
        poll_interval=0.05,
        stop_grace=1.0,
    )


# --- Each failure mode of flaky_train.py ends up with the right label ---


def test_clean_run_succeeds(tmp_path):
    result = supervise(tmp_path, flaky_command(tmp_path))
    assert result.classification.failure is None, result.classification.reason
    assert result.last_step == 60
    assert result.last_checkpoint.endswith("step_60.json")
    assert "done:" in result.log_path.read_text()


def test_oom_and_halving_the_batch_size_fixes_it(tmp_path):
    command = flaky_command(tmp_path, "--fail-mode", "oom", "--batch-size", "64")
    assert supervise(tmp_path, command).classification.failure == OOM

    # flaky_train.py only "runs out of memory" above --oom-above (default 32).
    command = flaky_command(tmp_path, "--fail-mode", "oom", "--batch-size", "32")
    result = supervise(tmp_path, command)
    assert result.classification.failure is None


def test_code_bug(tmp_path):
    result = supervise(tmp_path, flaky_command(tmp_path, "--fail-mode", "bug"))
    assert result.classification.failure == CODE_BUG


def test_node_failure_then_retry_resumes_from_checkpoint(tmp_path):
    command = flaky_command(tmp_path, "--fail-mode", "crash", "--fail-step", "30")
    first = supervise(tmp_path, command, attempt=1)
    assert first.classification.failure == NODE_FAILURE
    assert first.killed_by is None  # killed from outside, not by the supervisor
    assert first.last_checkpoint.endswith("step_20.json")

    # Attempt 2: flaky_train.py only crashes on attempt 1, like a flaky node.
    second = supervise(tmp_path, command, attempt=2, resume_from=first.last_checkpoint)
    assert second.classification.failure is None
    assert "resumed from" in second.log_path.read_text()


def test_nan_loss_is_stopped_early(tmp_path):
    # 100,000 steps: it would take about 17 minutes if we let it run.
    command = flaky_command(tmp_path, "--fail-mode", "nan", "--fail-step", "10")
    command += ["--steps", "100000"]
    result = supervise(tmp_path, command)
    assert result.killed_by == NAN_LOSS
    assert result.classification.failure == NAN
    assert result.duration_s < 10


def test_stuck_training_is_detected_by_progress_timeout(tmp_path):
    # In hang mode the training loop freezes but the heartbeat thread keeps running.
    command = flaky_command(tmp_path, "--fail-mode", "hang", "--fail-step", "10")
    result = supervise(tmp_path, command)
    assert result.killed_by == PROGRESS_TIMEOUT
    assert result.classification.failure == HUNG
    assert result.last_step == 9


# --- Safety: frozen processes and leftover child processes ---


@posix_only
def test_frozen_process_is_detected_by_heartbeat_timeout(tmp_path):
    # SIGSTOP freezes every thread, heartbeat included, like a wedged node.
    script = tmp_path / "freeze.py"
    script.write_text(
        textwrap.dedent("""
            import os, signal, sweepr
            sweepr.log({"loss": 1.0}, step=1)
            os.kill(os.getpid(), signal.SIGSTOP)
        """)
    )
    result = supervise(tmp_path, [sys.executable, str(script)])
    assert result.killed_by == HEARTBEAT_TIMEOUT
    assert result.classification.failure == HUNG


@posix_only
def test_stopping_kills_the_whole_process_group(tmp_path):
    # The script starts a child process (think DataLoader worker), then hangs. Killing only
    # the parent would leave the child running and holding resources.
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "spawn.py"
    script.write_text(
        textwrap.dedent(f"""
            import subprocess, sys, time, sweepr
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            open({str(pid_file)!r}, "w").write(str(child.pid))
            sweepr.log({{"loss": 1.0}}, step=1)
            while True:
                time.sleep(1)
        """)
    )
    result = supervise(tmp_path, [sys.executable, str(script)])
    assert result.classification.failure == HUNG

    child_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while is_running(child_pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not is_running(child_pid), "the child process survived"


def is_running(pid):
    """True if the process is running. A zombie (dead, but not cleaned up yet) counts as gone."""
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    state = state.stdout.strip()
    return state != "" and not state.startswith("Z")


# --- Reading the metrics file while it is being written ---


def test_reader_returns_only_new_records_and_waits_for_whole_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    reader = MetricsReader(path)
    assert reader.read_new() == []  # the file doesn't exist yet

    path.write_text('{"step": 1}\n{"step": 2, "lo')  # the second line is half-written
    assert reader.read_new() == [{"step": 1}]

    with open(path, "a") as f:
        f.write('ss": 0.5}\n')
    assert reader.read_new() == [{"step": 2, "loss": 0.5}]
    assert reader.read_new() == []


def test_reader_skips_corrupt_lines(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_bytes(b'{"step": 1}\r\nnot json\r\n{"step": 2}\r\n')  # Windows line endings
    assert MetricsReader(path).read_new() == [{"step": 1}, {"step": 2}]


# --- The command line ---


def test_cli_prints_the_result_and_sets_the_exit_code(tmp_path, capsys):
    options = ["run", "--runs-dir", str(tmp_path / "runs"), "--"]

    assert main(options + flaky_command(tmp_path)) == 0
    assert "succeeded" in capsys.readouterr().out

    assert main(options + flaky_command(tmp_path, "--fail-mode", "bug")) == 1
    assert "failed: CODE_BUG" in capsys.readouterr().out
