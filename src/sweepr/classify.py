"""Failure classifier: turns what we saw about a finished attempt into one failure label.

This module has no I/O. The supervisor collects the evidence (exit code, end of the log,
whether a loss went NaN, and whether it killed the process itself) and calls `classify`.
Because it is a plain function of its input, every case is easy to test.

The rules run in a fixed order and the first match wins, because one failure often matches
several signals. An OOM also prints a Python traceback, for example, so the OOM rule has to
run before the traceback rule.
"""

import sys
from dataclasses import dataclass

IS_WINDOWS = sys.platform == "win32"

# Failure labels. Plain strings, so they print, compare, and store (JSON, database) as is.
OOM = "OOM"  # out of GPU or host memory; retrying with a smaller batch can fix it
NAN = "NAN"  # loss became NaN or inf; retrying the same config won't help
HUNG = "HUNG"  # stopped making progress; the supervisor killed it
NODE_FAILURE = "NODE_FAILURE"  # killed from outside (lost node, preemption); retry
CODE_BUG = "CODE_BUG"  # a deterministic error; it fails the same way every time
UNKNOWN = "UNKNOWN"  # failed, but no rule is confident about why; retry

# Why the supervisor itself stopped the process.
HANG_TIMEOUT = "hang_timeout"  # no new metrics for too long: frozen or stuck
NAN_LOSS = "nan_loss"  # stopped early because the loss went NaN


@dataclass
class Evidence:
    """Everything the supervisor observed about one finished attempt."""

    exit_code: int | None  # None if the exit code is unknown
    log_tail: str = ""  # the last part of the combined stdout and stderr
    saw_nonfinite_loss: bool = False  # any logged loss was NaN or inf
    killed_by: str | None = None  # one of the kill reasons above, or None


@dataclass
class Classification:
    failure: str | None  # one of the labels above, or None if the attempt succeeded
    reason: str  # short and human-readable, for summaries and notifications


# Text that means "ran out of memory" (PyTorch, TensorFlow/JAX, plain Python).
# All matching is case-insensitive.
OOM_PATTERNS = [
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "RESOURCE_EXHAUSTED",
    "MemoryError",
]

# Exception types that fail the same way on every attempt. Anything not listed is UNKNOWN,
# not CODE_BUG: wrongly calling something a code bug means we would never retry it.
DETERMINISTIC_EXCEPTIONS = {
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

# Signals that mean "something outside killed us": SIGHUP (1), SIGKILL (9), SIGTERM (15).
EXTERNAL_KILL_SIGNALS = {1, 9, 15}


def classify(evidence: Evidence) -> Classification:
    """Label one finished attempt. The rules run in order, and the first match wins."""
    code = evidence.exit_code
    log = evidence.log_tail

    # 1. The supervisor's own decision is the strongest evidence we have.
    if evidence.killed_by == NAN_LOSS:
        return Classification(NAN, "stopped early: loss became NaN/inf")
    if evidence.killed_by == HANG_TIMEOUT:
        return Classification(HUNG, "killed by supervisor: no new metrics")

    # 2. An OOM also prints a traceback, so this must come before the traceback rule (5).
    if code != 0:
        oom_text = find_first_pattern(OOM_PATTERNS, log)
        if oom_text is not None:
            return Classification(OOM, f"out of memory: {oom_text}")

    # 3. NaN can end in exit 0 (the script kept going) or in a crash. Either way the result
    #    is useless, and the NaN is the root cause.
    if evidence.saw_nonfinite_loss:
        return Classification(NAN, "loss became NaN/inf")

    if code == 0:
        return Classification(None, "exited 0")

    # 4. Killed from outside: a lost node, a preemption, or (ambiguously) the Linux OOM killer.
    signal_number = killed_by_signal(code)
    if signal_number is not None:
        return Classification(NODE_FAILURE, f"killed by signal {signal_number}")

    # 5. A traceback whose exception type is known to be deterministic.
    exception = last_exception_type(log)
    if exception in DETERMINISTIC_EXCEPTIONS:
        return Classification(CODE_BUG, f"{exception} (fails the same way on retry)")

    if exception is not None:
        return Classification(UNKNOWN, f"exit code {code}, last exception {exception}")
    return Classification(UNKNOWN, f"exit code {code}, no recognizable error")


def find_first_pattern(patterns: list[str], log: str) -> str | None:
    """Return the first pattern that appears anywhere in the log (ignoring case), or None."""
    log_lower = log.lower()
    for pattern in patterns:
        if pattern.lower() in log_lower:
            return pattern
    return None


def last_exception_type(log: str) -> str | None:
    """Return the exception name from the last traceback line, e.g. "TypeError".

    The last line of a Python traceback looks like "TypeError: message" or
    "torch.OutOfMemoryError: message", or just "KeyboardInterrupt" with no message.
    """
    found = None
    for line in log.splitlines():
        name = line.strip().split(":")[0]  # the text before the first colon
        if " " in name or not name:
            continue  # e.g. '  File "train.py", line 3' is not an exception line
        if name.endswith(("Error", "Exception", "Interrupt", "Exit")):
            found = name.split(".")[-1]  # "torch.OutOfMemoryError" -> "OutOfMemoryError"
    return found


def killed_by_signal(code: int | None) -> int | None:
    """The signal number if the exit code means "killed from outside", else None."""
    if code is None:
        return None
    # Python reports "killed by signal N" as exit code -N on Linux and macOS.
    if code < 0 and -code in EXTERNAL_KILL_SIGNALS:
        return -code
    # Shells (and wrappers like srun) report it as 128 + N.
    if code > 128 and code - 128 in EXTERNAL_KILL_SIGNALS:
        return code - 128
    # Windows has no signals: os.kill(pid, 9) ends the process with exit code 9.
    if IS_WINDOWS and code == 9:
        return 9
    return None
