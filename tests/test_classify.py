"""Tests for the failure classifier. No processes here: we build Evidence by hand."""

import pytest

from sweepr import classify as classify_module
from sweepr.classify import (
    CODE_BUG,
    HANG_TIMEOUT,
    HUNG,
    NAN,
    NAN_LOSS,
    NODE_FAILURE,
    OOM,
    UNKNOWN,
    Evidence,
    classify,
    last_exception_type,
)


def traceback_ending_in(last_line):
    """A minimal Python traceback whose last line is `last_line`."""
    return f'Traceback (most recent call last):\n  File "train.py", line 3\n{last_line}'


# Each case: (evidence, expected label). None means the attempt succeeded.
CASES = [
    pytest.param(Evidence(0), None, id="clean exit"),
    # The supervisor's own kill decision wins over everything else.
    pytest.param(Evidence(-9, killed_by=HANG_TIMEOUT), HUNG, id="killed: hung"),
    pytest.param(Evidence(-15, killed_by=NAN_LOSS), NAN, id="killed: nan loss"),
    pytest.param(
        Evidence(-9, traceback_ending_in("CUDA out of memory"), killed_by=HANG_TIMEOUT),
        HUNG,
        id="hang kill beats OOM text",
    ),
    # Out of memory
    pytest.param(
        Evidence(1, traceback_ending_in("torch.OutOfMemoryError: CUDA out of memory.")),
        OOM,
        id="torch OOM",
    ),
    pytest.param(Evidence(1, traceback_ending_in("MemoryError")), OOM, id="host OOM"),
    pytest.param(
        Evidence(1, traceback_ending_in("CUDA out of memory"), saw_nonfinite_loss=True),
        OOM,
        id="OOM beats NaN",
    ),
    pytest.param(Evidence(0, "retrying after CUDA out of memory"), None, id="OOM text, exit 0"),
    # NaN loss
    pytest.param(Evidence(0, saw_nonfinite_loss=True), NAN, id="nan, script kept going"),
    # Killed from outside
    pytest.param(Evidence(-9), NODE_FAILURE, id="SIGKILL"),
    pytest.param(Evidence(137), NODE_FAILURE, id="SIGKILL via shell (128 + 9)"),
    pytest.param(Evidence(-11), UNKNOWN, id="SIGSEGV is not an outside kill"),
    # Deterministic code bugs
    pytest.param(
        Evidence(1, traceback_ending_in("TypeError: can only concatenate str")),
        CODE_BUG,
        id="TypeError",
    ),
    pytest.param(
        Evidence(1, traceback_ending_in("ModuleNotFoundError: No module named 'x'")),
        CODE_BUG,
        id="missing module",
    ),
    # Not confident: UNKNOWN, never CODE_BUG
    pytest.param(
        Evidence(1, traceback_ending_in("RuntimeError: something odd")),
        UNKNOWN,
        id="bare RuntimeError",
    ),
    pytest.param(Evidence(1), UNKNOWN, id="exit 1, empty log"),
    pytest.param(Evidence(None), UNKNOWN, id="exit code unknown"),
]


@pytest.mark.parametrize(("evidence", "expected"), CASES)
def test_classify(evidence, expected):
    result = classify(evidence)
    assert result.failure == expected, result.reason
    assert result.reason  # there is always a human-readable reason


def test_exit_code_9_means_killed_only_on_windows(monkeypatch):
    monkeypatch.setattr(classify_module, "IS_WINDOWS", True)
    assert classify(Evidence(9)).failure == NODE_FAILURE
    monkeypatch.setattr(classify_module, "IS_WINDOWS", False)
    assert classify(Evidence(9)).failure == UNKNOWN


def test_last_exception_type():
    assert last_exception_type(traceback_ending_in("TypeError: bad")) == "TypeError"
    assert last_exception_type(traceback_ending_in("torch.OutOfMemoryError: x")) == (
        "OutOfMemoryError"
    )
    # With chained tracebacks, the last exception is the one that ended the program.
    chained = traceback_ending_in("KeyError: 'a'") + "\n" + traceback_ending_in("TypeError: x")
    assert last_exception_type(chained) == "TypeError"
    assert last_exception_type("step 10: loss=0.3") is None
