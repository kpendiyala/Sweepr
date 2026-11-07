"""Failure classifier: turns what we saw about a finished attempt into one failure label.

This module is a pure function with no I/O. The supervisor collects the evidence (exit code,
log tail, metrics, and whether it killed the process itself) and calls `classify`. Keeping the
logic pure means every case is a one-line table-driven test.

Rules are checked in a fixed order and the first match wins, because one failure often
matches several signals. An OOM is also a Python traceback, for example, so the OOM rule must
run before the traceback rule.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import Enum


class FailureClass(str, Enum):
    OOM = "OOM"  # out of GPU or host memory; retrying with a smaller batch can fix it
    NAN = "NAN"  # loss became NaN or inf; retrying the same config won't help
    HUNG = "HUNG"  # stopped making progress; the supervisor killed it
    NODE_FAILURE = "NODE_FAILURE"  # killed from outside or a transient infra error; retry
    CODE_BUG = "CODE_BUG"  # a deterministic error; it fails the same way every time
    UNKNOWN = "UNKNOWN"  # failed, but no rule is confident about why


class KillReason(str, Enum):
    """Why the supervisor itself stopped the process, if it did."""

    HEARTBEAT_TIMEOUT = "heartbeat_timeout"  # no heartbeat at all: process frozen or gone
    PROGRESS_TIMEOUT = "progress_timeout"  # heartbeats, but no new metrics: training stuck
    NAN_LOSS = "nan_loss"  # stopped early because the loss went NaN


@dataclass(frozen=True)
class Evidence:
    """Everything the supervisor observed about one finished attempt."""

    exit_code: int | None  # None if the exit code is unknown
    log_tail: str = ""  # the last lines of stdout and stderr
    saw_nonfinite_loss: bool = False  # any logged loss was NaN or inf
    killed_by: KillReason | None = None


@dataclass(frozen=True)
class Classification:
    failure: FailureClass | None  # None means the attempt succeeded
    reason: str  # short, human-readable, shown in summaries and webhooks

    @property
    def ok(self) -> bool:
        return self.failure is None


# Messages that mean "ran out of memory", from PyTorch, TensorFlow/JAX, and plain Python.
OOM_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"CUDA out of memory",
        r"OutOfMemoryError",
        r"CUBLAS_STATUS_ALLOC_FAILED",
        r"RESOURCE_EXHAUSTED",
        r"^MemoryError\b",
    )
]

# Errors that come from the environment rather than the code, so a retry can succeed.
TRANSIENT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"NCCL (error|WARN.*timeout)",
        r"ECC error",
        r"\bConnection(Error|ResetError|RefusedError|AbortedError)\b",
        r"\bTimeoutError\b",
        r"\bBrokenPipeError\b",
        r"Stale file handle",
    )
]

# Exception types that fail the same way on every attempt. Anything not listed is UNKNOWN,
# not CODE_BUG: wrongly labelling a run a code bug means we never retry it.
DETERMINISTIC_EXCEPTIONS = frozenset(
    {
        "SyntaxError",
        "IndentationError",
        "NameError",
        "TypeError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "KeyError",
        "IndexError",
        "ValueError",
        "AssertionError",
        "ZeroDivisionError",
        "NotImplementedError",
        "FileNotFoundError",
        "UnboundLocalError",
        "RecursionError",
    }
)

# Signals that mean "something outside killed us": SIGHUP, SIGKILL, SIGTERM.
EXTERNAL_KILL_SIGNALS = frozenset({1, 9, 15})

# The last line of a Python traceback looks like "pkg.module.SomeError: message".
_EXCEPTION_LINE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))\b(?::|$)")


def classify(evidence: Evidence, *, windows: bool | None = None) -> Classification:
    """Label one finished attempt. The rules run in order, and the first match wins."""
    if windows is None:
        windows = sys.platform == "win32"
    code, log = evidence.exit_code, evidence.log_tail

    # 1. The supervisor's own decision is the strongest evidence we have.
    if evidence.killed_by is KillReason.NAN_LOSS:
        return Classification(FailureClass.NAN, "stopped early: loss became NaN/inf")
    if evidence.killed_by in (KillReason.HEARTBEAT_TIMEOUT, KillReason.PROGRESS_TIMEOUT):
        return Classification(
            FailureClass.HUNG, f"killed by supervisor: {evidence.killed_by.value}"
        )

    # 2. OOM shows up as a traceback too, so it has to come before the traceback rule.
    if code != 0 and (match := _first_match(OOM_PATTERNS, log)):
        return Classification(FailureClass.OOM, f"out of memory: {match!r}")

    # 3. NaN can end in exit 0 (the script kept going) or in a crash. Either way, the result
    # is useless, and the NaN is the root cause.
    if evidence.saw_nonfinite_loss:
        return Classification(FailureClass.NAN, "loss became NaN/inf")

    if code == 0:
        return Classification(None, "exited 0")

    # 4. Killed from outside: a lost node, preemption, or (ambiguously) the Linux OOM killer.
    if (sig := _killed_by_signal(code, windows)) is not None:
        return Classification(FailureClass.NODE_FAILURE, f"killed by signal {sig}")

    # 5. Transient infra errors, which may also print a traceback.
    if match := _first_match(TRANSIENT_PATTERNS, log):
        return Classification(FailureClass.NODE_FAILURE, f"transient error: {match!r}")

    # 6. A traceback whose exception type is known to be deterministic.
    exc = last_exception_type(log)
    if exc in DETERMINISTIC_EXCEPTIONS:
        return Classification(FailureClass.CODE_BUG, f"{exc} (fails the same way on retry)")

    detail = f"last exception {exc}" if exc else "no recognizable error"
    return Classification(FailureClass.UNKNOWN, f"exit code {code}, {detail}")


def last_exception_type(log: str) -> str | None:
    """Return the bare exception name from the last traceback line, e.g. 'TypeError'."""
    found = None
    for line in log.splitlines():
        if m := _EXCEPTION_LINE.match(line.strip()):
            found = m.group(1).rsplit(".", 1)[-1]  # "torch.OutOfMemoryError" -> "OutOfMemoryError"
    return found


def _first_match(patterns: list[re.Pattern[str]], log: str) -> str | None:
    for line in log.splitlines():
        for p in patterns:
            if m := p.search(line.strip()):
                return m.group(0)
    return None


def _killed_by_signal(code: int | None, windows: bool) -> int | None:
    """Signal number if the exit code means 'killed from outside', else None."""
    if code is None:
        return None
    if code < 0:  # Python's subprocess reports death by signal N as -N on POSIX
        return -code if -code in EXTERNAL_KILL_SIGNALS else None
    if code > 128 and code - 128 in EXTERNAL_KILL_SIGNALS:  # shells report it as 128 + N
        return code - 128
    if windows and code == 9:  # os.kill(pid, 9) on Windows exits with code 9
        return 9
    return None
