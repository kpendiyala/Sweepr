"""Table-driven tests for the classifier, plus integration tests against flaky_train.py."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sweepr.classify import (
    Evidence,
    FailureClass,
    KillReason,
    classify,
    last_exception_type,
)

OOM, NAN, HUNG = FailureClass.OOM, FailureClass.NAN, FailureClass.HUNG
NODE, BUG, UNKNOWN = FailureClass.NODE_FAILURE, FailureClass.CODE_BUG, FailureClass.UNKNOWN


def tb(exc_line: str) -> str:
    """A minimal Python traceback ending in `exc_line`."""
    return f'Traceback (most recent call last):\n  File "train.py", line 3, in <module>\n{exc_line}'


# (id, evidence, expected class). None means success.
CASES = [
    # Success
    ("clean exit", Evidence(0), None),
    ("exit 0 with warnings in log", Evidence(0, "UserWarning: something"), None),
    # Supervisor's own kill decision wins over everything else
    ("killed: no heartbeat", Evidence(-9, killed_by=KillReason.HEARTBEAT_TIMEOUT), HUNG),
    ("killed: no progress", Evidence(-15, killed_by=KillReason.PROGRESS_TIMEOUT), HUNG),
    ("killed: nan loss", Evidence(-15, killed_by=KillReason.NAN_LOSS), NAN),
    (
        "hang kill beats OOM text in log",
        Evidence(-9, tb("RuntimeError: CUDA out of memory"), killed_by=KillReason.PROGRESS_TIMEOUT),
        HUNG,
    ),
    # OOM
    ("torch CUDA OOM", Evidence(1, tb("torch.OutOfMemoryError: CUDA out of memory.")), OOM),
    ("old torch OOM", Evidence(1, tb("RuntimeError: CUDA out of memory. Tried 2 GiB")), OOM),
    ("cublas alloc", Evidence(1, tb("RuntimeError: CUBLAS_STATUS_ALLOC_FAILED")), OOM),
    ("tf/jax OOM", Evidence(1, "RESOURCE_EXHAUSTED: Out of memory while trying"), OOM),
    ("host MemoryError", Evidence(1, tb("MemoryError")), OOM),
    ("OOM beats NaN", Evidence(1, tb("CUDA out of memory"), saw_nonfinite_loss=True), OOM),
    ("OOM text on success is ignored", Evidence(0, "retrying after CUDA out of memory"), None),
    # NaN
    ("nan, script kept going", Evidence(0, saw_nonfinite_loss=True), NAN),
    ("nan, then crashed", Evidence(1, tb("ValueError: nan"), saw_nonfinite_loss=True), NAN),
    # Killed from outside
    ("SIGKILL", Evidence(-9), NODE),
    ("SIGTERM (preempted)", Evidence(-15), NODE),
    ("SIGKILL via shell (137)", Evidence(137), NODE),
    ("SIGTERM via shell (143)", Evidence(143), NODE),
    ("SIGSEGV is not an outside kill", Evidence(-11), UNKNOWN),
    # Transient infra errors
    ("NCCL error", Evidence(1, tb("RuntimeError: NCCL error in: socket timeout")), NODE),
    ("connection reset", Evidence(1, tb("ConnectionResetError: [Errno 104]")), NODE),
    ("stale NFS handle", Evidence(1, tb("OSError: [Errno 116] Stale file handle")), NODE),
    # Deterministic code bugs
    ("TypeError", Evidence(1, tb("TypeError: can only concatenate str")), BUG),
    ("module missing", Evidence(1, tb("ModuleNotFoundError: No module named 'x'")), BUG),
    ("SyntaxError", Evidence(1, "  File 'train.py'\nSyntaxError: invalid syntax"), BUG),
    ("KeyError", Evidence(1, tb("KeyError: 'lr'")), BUG),
    ("chained: last one wins", Evidence(1, tb("KeyError: 'a'") + "\n" + tb("TypeError: x")), BUG),
    # Not confident: UNKNOWN, never CODE_BUG
    ("bare RuntimeError", Evidence(1, tb("RuntimeError: something odd")), UNKNOWN),
    ("custom exception", Evidence(1, tb("mylib.DataLoaderException: worker died")), UNKNOWN),
    ("exit 1, empty log", Evidence(1), UNKNOWN),
    ("exit 2 from argparse", Evidence(2, "error: unrecognized arguments"), UNKNOWN),
    ("exit code unknown", Evidence(None), UNKNOWN),
]


@pytest.mark.parametrize(
    ("evidence", "expected"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_classify(evidence, expected):
    result = classify(evidence, windows=False)
    assert result.failure == expected, result.reason
    assert result.ok == (expected is None)
    assert result.reason


def test_exit_code_9_means_killed_only_on_windows():
    assert classify(Evidence(9), windows=True).failure == NODE
    assert classify(Evidence(9), windows=False).failure == UNKNOWN


@pytest.mark.parametrize(
    ("log", "expected"),
    [
        (tb("TypeError: bad"), "TypeError"),
        (tb("torch.OutOfMemoryError: CUDA"), "OutOfMemoryError"),
        (tb("__main__.OutOfMemoryError: CUDA"), "OutOfMemoryError"),
        (tb("KeyboardInterrupt"), "KeyboardInterrupt"),
        ("step 10: loss=0.3\nall good", None),
        ("", None),
    ],
)
def test_last_exception_type(log, expected):
    assert last_exception_type(log) == expected


# Integration: run the real fault-injection script and classify what comes out. This is the
# same evidence the supervisor will collect, minus the hang case, which needs the supervisor
# to do the killing (piece 5).

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "flaky_train.py"


def run_and_classify(tmp_path, *args):
    metrics = tmp_path / "metrics.jsonl"
    env = {**os.environ, "SWEEPR_METRICS_FILE": str(metrics), "SWEEPR_HEARTBEAT_INTERVAL": "0.05"}
    cmd = [sys.executable, str(SCRIPT), "--steps", "60", "--step-time", "0"]
    cmd += ["--ckpt-dir", str(tmp_path / "ckpt"), *args]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
    lines = metrics.read_text().splitlines() if metrics.exists() else []
    records = [json.loads(line) for line in lines]
    losses = [r["data"]["loss"] for r in records if r["type"] == "metric"]
    evidence = Evidence(
        exit_code=proc.returncode,
        log_tail=proc.stdout + proc.stderr,
        saw_nonfinite_loss=any(loss != loss or abs(loss) == float("inf") for loss in losses),
    )
    return classify(evidence).failure


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((), None),
        (("--fail-mode", "oom", "--batch-size", "64", "--oom-above", "32"), OOM),
        (("--fail-mode", "nan", "--fail-step", "20"), NAN),
        (("--fail-mode", "crash", "--fail-step", "20"), NODE),
        (("--fail-mode", "bug"), BUG),
    ],
    ids=["clean", "oom", "nan", "crash", "bug"],
)
def test_flaky_train_modes_are_classified(tmp_path, args, expected):
    assert run_and_classify(tmp_path, *args) == expected
