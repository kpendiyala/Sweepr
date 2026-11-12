"""Sweeps: run a training command once per combination of a parameter grid, with retries.

A sweep file looks like this:

    command: python train.py --lr {lr} --batch-size {batch_size} --ckpt-dir {run_dir}/ckpt
    grid:
      lr: [0.01, 0.05]
      batch_size: [64]
    max_attempts: 3      # optional, default 3
    hang_timeout: 300    # optional, seconds without a new metric before a run counts as hung

Runs go one after another. Each run is retried until it succeeds, fails in a way a retry
can't fix, or reaches max_attempts. Every retry resumes from the run's latest checkpoint.
"""

import itertools
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from sweepr.classify import CODE_BUG, NAN, OOM
from sweepr.supervisor import AttemptResult, run_attempt

# Retrying won't help these: the same config diverges again, the same code fails again.
NEVER_RETRY = {NAN, CODE_BUG}


class SweepFileError(ValueError):
    """The sweep file is invalid. The message says what to fix."""


@dataclass
class Sweep:
    name: str
    command: list[str]  # the command template, already split into arguments
    grid: dict[str, list]
    max_attempts: int = 3
    hang_timeout: float = 300.0


@dataclass
class RunResult:
    name: str  # e.g. "run-003"; also the run's folder name
    params: dict  # the params of the last attempt (batch_size may have been halved)
    attempts: list[AttemptResult]
    succeeded: bool
    failure: str | None  # the label of the last attempt, if the run failed
    reason: str  # why the run ended, in a few words


# --- Reading the sweep file ---


def load_sweep(path: str | Path) -> Sweep:
    """Read and check a sweep file. Any mistake is reported before a single run starts."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SweepFileError(f"{path}: not valid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise SweepFileError(f"{path}: expected keys such as command: and grid:")

    allowed = {"command", "grid", "max_attempts", "hang_timeout"}
    unknown = set(raw) - allowed
    if unknown:
        raise SweepFileError(f"unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")

    # A string is split into arguments the way a shell would. On Windows, write paths with
    # forward slashes (examples/train.py), because shell splitting treats \ as an escape.
    command = raw.get("command")
    if isinstance(command, str):
        command = shlex.split(command)
    if not isinstance(command, list) or not command:
        raise SweepFileError("command: give the training command as a string or a list")
    command = [str(part) for part in command]

    grid = raw.get("grid") or {}
    if not isinstance(grid, dict):
        raise SweepFileError("grid: expected parameter names, each with a list of values")
    for name, values in grid.items():
        if not isinstance(values, list) or not values:
            raise SweepFileError(f"grid.{name}: expected a list with at least one value")

    max_attempts = raw.get("max_attempts", 3)
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise SweepFileError("max_attempts: must be a whole number, at least 1")
    hang_timeout = raw.get("hang_timeout", 300.0)
    if not isinstance(hang_timeout, (int, float)) or hang_timeout <= 0:
        raise SweepFileError("hang_timeout: must be a number of seconds above 0")

    sweep = Sweep(path.stem, command, grid, max_attempts, float(hang_timeout))
    # Fill in the first combination now, so a placeholder typo like {learning_rate} fails
    # here and not after the first runs have already finished.
    try:
        render_command(sweep.command, grid_combinations(grid)[0], run_dir="check")
    except KeyError as error:
        name = error.args[0]
        message = f"command uses {{{name}}}, which is not in the grid {sorted(grid)}"
        raise SweepFileError(message) from error
    except (IndexError, ValueError, TypeError) as error:
        raise SweepFileError(f"command: bad placeholder: {error}") from error
    return sweep


def grid_combinations(grid: dict[str, list]) -> list[dict]:
    """Every combination of grid values, e.g. {lr: [1, 2], bs: [8]} -> [{lr: 1, bs: 8}, ...]."""
    names = list(grid)
    combinations = itertools.product(*(grid[name] for name in names))
    return [dict(zip(names, values, strict=True)) for values in combinations]


def render_command(template: list[str], params: dict, run_dir: str) -> list[str]:
    """Fill in {placeholders}, e.g. "--lr={lr}" -> "--lr=0.01". {run_dir} is the run's folder.

    The template was split into arguments before filling in values, so a value with a
    space in it stays one argument.
    """
    return [part.format(**params, run_dir=run_dir) for part in template]


# --- Deciding whether to retry ---


def plan_retry(failure: str, attempts_so_far: int, max_attempts: int, params: dict):
    """After a failed attempt, return (params for the next attempt, reason).

    The params are None when the run should give up.
    """
    if failure in NEVER_RETRY:
        return None, f"{failure}: retrying won't help"
    if attempts_so_far >= max_attempts:
        return None, f"{failure}: gave up after {max_attempts} attempts"
    if failure == OOM and "batch_size" in params:
        smaller = params["batch_size"] // 2
        if smaller < 1:
            return None, "OOM: batch_size can't go any lower"
        return {**params, "batch_size": smaller}, f"OOM: retry with batch_size={smaller}"
    return params, f"{failure}: retry"


# --- Running ---


def run_with_retries(
    sweep: Sweep,
    params: dict,
    run_dir: Path,
    poll_interval: float = 0.5,
    stop_grace: float = 10.0,
) -> RunResult:
    """Run one grid combination until it succeeds or plan_retry says to give up."""
    attempts = []
    latest_checkpoint = None
    attempt = 0
    while True:
        attempt += 1
        command = render_command(sweep.command, params, run_dir=str(run_dir))
        result = run_attempt(
            command,
            run_dir=run_dir,
            attempt=attempt,
            resume_from=latest_checkpoint,  # None on the first attempt
            hang_timeout=sweep.hang_timeout,
            poll_interval=poll_interval,
            stop_grace=stop_grace,
        )
        attempts.append(result)
        # Keep the newest checkpoint across attempts: an attempt that dies before saving
        # one should still resume from the one before it.
        if result.last_checkpoint:
            latest_checkpoint = result.last_checkpoint

        failure = result.classification.failure
        if failure is None:
            return RunResult(run_dir.name, params, attempts, True, None, "succeeded")

        next_params, reason = plan_retry(failure, attempt, sweep.max_attempts, params)
        print(
            f"[sweepr] {run_dir.name} attempt {attempt}: {result.classification.reason}",
            file=sys.stderr,
        )
        print(f"[sweepr] {run_dir.name} -> {reason}", file=sys.stderr)
        if next_params is None:
            return RunResult(run_dir.name, params, attempts, False, failure, reason)
        params = next_params


def run_sweep(
    sweep: Sweep, sweep_dir: Path, poll_interval: float = 0.5, stop_grace: float = 10.0
) -> list[RunResult]:
    """Run every grid combination, one after another, and return how each run ended."""
    combinations = grid_combinations(sweep.grid)
    print(f"[sweepr] sweep {sweep.name}: {len(combinations)} runs in {sweep_dir}", file=sys.stderr)
    results = []
    for number, params in enumerate(combinations, start=1):
        run_dir = sweep_dir / f"run-{number:03d}"
        results.append(run_with_retries(sweep, params, run_dir, poll_interval, stop_grace))
        print(f"[sweepr] {summary_line(results, len(combinations))}", file=sys.stderr)
    return results


def summary_line(results: list[RunResult], total: int) -> str:
    """For example "12/12 runs finished: 8 succeeded, 4 failed (NAN 2, CODE_BUG 2)"."""
    succeeded = sum(1 for result in results if result.succeeded)
    line = f"{len(results)}/{total} runs finished: {succeeded} succeeded"
    failures = {}  # failure label -> how many runs ended with it
    for result in results:
        if not result.succeeded:
            failures[result.failure] = failures.get(result.failure, 0) + 1
    if failures:
        counts = ", ".join(f"{label} {count}" for label, count in failures.items())
        line += f", {len(results) - succeeded} failed ({counts})"
    return line


def describe_run(result: RunResult) -> str:
    """One line per run for the final report."""
    status = "succeeded" if result.succeeded else f"FAILED {result.failure}"
    params = " ".join(f"{name}={value}" for name, value in result.params.items())
    return f"{result.name}  {status}  after {len(result.attempts)} attempt(s)  {params}"
