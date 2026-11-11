"""Runs one attempt of a training command under supervision.

`run_attempt` starts the command and then loops every `poll_interval` seconds:
  1. read the new records from the metrics file and give them to the watchdog
  2. stop if the process has exited
  3. ask the watchdog whether the process should be killed (hung, frozen, NaN loss)
When the process is gone, it collects the evidence and asks the classifier what happened.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from sweepr.classify import Classification, Evidence, classify
from sweepr.process import kill_leftovers, start_process, stop_process
from sweepr.sdk import METRICS_FILE_ENV, RESUME_ENV
from sweepr.watchdog import Watchdog, WatchdogConfig

ATTEMPT_ENV = "SWEEPR_ATTEMPT"
LOG_TAIL_BYTES = 64 * 1024  # how much of the end of the log the classifier looks at


@dataclass
class AttemptResult:
    attempt: int
    classification: Classification
    exit_code: int | None
    killed_by: str | None  # set if the supervisor stopped the process itself
    last_step: int | None
    last_checkpoint: str | None  # where a retry can resume from
    duration_s: float
    log_path: Path
    metrics_path: Path


class MetricsReader:
    """Reads the metrics file a little at a time while the SDK keeps appending to it.

    Each call to `read_new` returns only the records added since the previous call. A line
    the SDK hasn't finished writing yet (no newline at the end) is kept back until the rest
    of it arrives, so we never try to parse half a record.
    """

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0  # how many bytes of the file we have already read
        self.leftover = b""  # an unfinished last line, kept for the next read

    def read_new(self) -> list[dict]:
        try:
            # Reopening the file on every read costs almost nothing, and it works on Windows
            # while the training script still has the file open.
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read()
        except FileNotFoundError:
            return []  # the script hasn't written anything yet
        self.offset += len(chunk)

        lines = (self.leftover + chunk).split(b"\n")
        # The last piece has no newline after it yet: either b"" or a half-written line.
        self.leftover = lines.pop()

        records = []
        for line in lines:
            line = line.strip()  # also removes the \r of Windows line endings
            if not line:
                continue
            try:
                record = json.loads(line)  # also accepts the NaN / Infinity the SDK writes
            except ValueError:
                continue  # one corrupt line shouldn't take the supervisor down
            if isinstance(record, dict):
                records.append(record)
        return records


def run_attempt(
    cmd: list[str],
    run_dir: Path,
    attempt: int = 1,
    resume_from: str | None = None,
    config: WatchdogConfig | None = None,
    poll_interval: float = 0.5,
    stop_grace: float = 10.0,
) -> AttemptResult:
    """Run `cmd` once, stop it if it hangs or its loss goes NaN, and classify the result."""
    if config is None:
        config = WatchdogConfig()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"attempt-{attempt}.log"
    metrics_path = run_dir / f"attempt-{attempt}.metrics.jsonl"
    metrics_path.unlink(missing_ok=True)  # don't read records from an older attempt

    # The child gets our environment plus the variables the SDK reads.
    env = dict(os.environ)
    env[METRICS_FILE_ENV] = str(metrics_path)
    env[ATTEMPT_ENV] = str(attempt)
    env["PYTHONUNBUFFERED"] = "1"  # so the log is complete even if we have to kill it
    env.pop(RESUME_ENV, None)
    if resume_from:
        env[RESUME_ENV] = resume_from

    started = time.monotonic()
    proc = start_process(cmd, env, log_path)
    print(f"[sweepr] attempt {attempt} started (pid {proc.pid}), log: {log_path}", file=sys.stderr)

    reader = MetricsReader(metrics_path)
    watchdog = Watchdog(config, started_at=started)
    killed_by = None
    try:
        while True:
            # Check for exit *before* reading, so the records written right before the
            # exit are still read in this same pass.
            has_exited = proc.poll() is not None
            now = time.monotonic()
            for record in reader.read_new():
                watchdog.observe(record, now)
            if has_exited:
                break

            if killed_by is None:
                reason = watchdog.check(now)
                if reason is not None:
                    killed_by = reason
                    print(f"[sweepr] attempt {attempt}: {reason}, stopping it", file=sys.stderr)
                    stop_process(proc, stop_grace)
                    continue  # the process is gone now; the next pass reads its last records

            time.sleep(poll_interval)
    finally:
        # This also runs on Ctrl-C or any error in the loop above, so we never leave a
        # training run behind. The child is in its own process group, so Ctrl-C doesn't
        # reach it directly; we have to stop it ourselves.
        if proc.poll() is None:
            stop_process(proc, stop_grace)
        kill_leftovers(proc)

    evidence = Evidence(
        exit_code=proc.returncode,
        log_tail=read_end_of_file(log_path, LOG_TAIL_BYTES),
        saw_nonfinite_loss=watchdog.saw_nonfinite_loss,
        killed_by=killed_by,
    )
    return AttemptResult(
        attempt=attempt,
        classification=classify(evidence),
        exit_code=proc.returncode,
        killed_by=killed_by,
        last_step=watchdog.last_step,
        last_checkpoint=watchdog.last_checkpoint,
        duration_s=time.monotonic() - started,
        log_path=log_path,
        metrics_path=metrics_path,
    )


def read_end_of_file(path: Path, max_bytes: int) -> str:
    """Return the last `max_bytes` of a file as text ("" if the file doesn't exist)."""
    try:
        with open(path, "rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""
