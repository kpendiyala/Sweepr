"""Command line: `supervise run [options] -- <training command>`."""

import argparse
import secrets
import sys
import time
from pathlib import Path

from sweepr.supervisor import AttemptResult, run_attempt
from sweepr.watchdog import WatchdogConfig

EXAMPLE = "example: supervise run -- python examples/flaky_train.py --fail-mode crash"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supervise",
        description="Sweepr: supervise ML training runs and classify how they fail.",
        epilog=EXAMPLE,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="run one attempt of a training command",
        description="Run one attempt of a training command (put it after --), stop it if "
        "it hangs or its loss goes NaN, and classify the result.",
        epilog=EXAMPLE,
    )
    run.add_argument("--runs-dir", default="runs", help="where logs and metrics files go")
    run.add_argument(
        "--heartbeat-timeout", type=float, default=60.0, help="kill if no record for N seconds"
    )
    run.add_argument(
        "--progress-timeout", type=float, default=600.0, help="kill if no new metric for N seconds"
    )
    run.add_argument(
        "--startup-timeout", type=float, default=None, help="kill if silent N seconds after start"
    )
    run.add_argument(
        "--nan-key", default="loss", help="stop early if this metric is NaN; '' turns it off"
    )
    run.add_argument("--resume-from", default=None, help="checkpoint passed as SWEEPR_RESUME_FROM")
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Everything after "--" is the training command, passed through untouched.
    if "--" in argv:
        split_at = argv.index("--")
        options = argv[:split_at]
        command = argv[split_at + 1 :]
    else:
        options = argv
        command = []

    parser = build_parser()
    args = parser.parse_args(options)
    if not command:
        parser.error("give the training command after --\n" + EXAMPLE)

    config = WatchdogConfig(
        heartbeat_timeout=args.heartbeat_timeout,
        progress_timeout=args.progress_timeout,
        startup_timeout=args.startup_timeout,
        nan_key=args.nan_key or None,  # '' means "off"
    )
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    try:
        result = run_attempt(
            command,
            run_dir=Path(args.runs_dir) / run_id,
            resume_from=args.resume_from,
            config=config,
        )
    except KeyboardInterrupt:
        # run_attempt has already stopped the training process on its way out.
        print("[sweepr] interrupted; the training process was stopped", file=sys.stderr)
        return 130  # the usual exit code for "stopped by Ctrl-C"
    print(summarize(result))
    if result.classification.failure is None:
        return 0
    return 1


def summarize(result: AttemptResult) -> str:
    """One line such as "attempt 1 failed: OOM (out of memory: ...) after 0.4s, log ..."."""
    classification = result.classification
    if classification.failure is None:
        status = "succeeded"
    else:
        status = f"failed: {classification.failure}"

    first = f"attempt {result.attempt} {status} ({classification.reason})"
    parts = [f"{first} after {result.duration_s:.1f}s"]
    if result.last_step is not None:
        parts.append(f"last step {result.last_step}")
    if result.last_checkpoint:
        parts.append(f"last checkpoint {result.last_checkpoint}")
    parts.append(f"log {result.log_path}")
    return ", ".join(parts)
