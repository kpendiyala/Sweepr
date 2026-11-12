"""Command line: `supervise run [options] -- <command>` and `supervise sweep <file>`."""

import argparse
import secrets
import sys
import time
from pathlib import Path

from sweepr.supervisor import AttemptResult, run_attempt
from sweepr.sweep import SweepFileError, describe_run, load_sweep, run_sweep

EXAMPLES = """examples:
  supervise run -- python examples/flaky_train.py --fail-mode crash
  supervise sweep examples/sweep.yaml"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="supervise",
        description="Sweepr: supervise ML training runs, classify failures, and retry them.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser(
        "run",
        help="run one attempt of a training command and classify the result",
        description="Run one attempt of a training command (put it after --), stop it if "
        "it hangs or its loss goes NaN, and say why it failed.",
    )
    run.add_argument("--runs-dir", default="runs", help="where logs and metrics files go")
    run.add_argument(
        "--hang-timeout", type=float, default=300.0, help="kill if no new metric for N seconds"
    )
    run.add_argument("--resume-from", default=None, help="checkpoint passed as SWEEPR_RESUME_FROM")

    sweep = commands.add_parser(
        "sweep",
        help="run every combination in a sweep file, retrying failures",
        description="Run the command once per grid combination, one after another. Failed "
        "runs are retried (from their latest checkpoint) unless a retry can't help.",
    )
    sweep.add_argument("sweep_file", help="path to a sweep .yaml file")
    sweep.add_argument("--runs-dir", default="runs", help="where each sweep's folder goes")
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
    stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)  # a unique folder name

    try:
        if args.command == "sweep":
            return sweep_command(args, stamp)
        if not command:
            parser.error("give the training command after --")
        result = run_attempt(
            command,
            run_dir=Path(args.runs_dir) / stamp,
            resume_from=args.resume_from,
            hang_timeout=args.hang_timeout,
        )
    except KeyboardInterrupt:
        # run_attempt has already stopped the training process on its way out.
        print("[sweepr] interrupted; the training process was stopped", file=sys.stderr)
        return 130  # the usual exit code for "stopped by Ctrl-C"
    except OSError as error:
        print(f"supervise: could not start the command: {error}", file=sys.stderr)
        return 2

    print(summarize(result))
    if result.classification.failure is None:
        return 0
    return 1


def sweep_command(args, stamp: str) -> int:
    try:
        sweep = load_sweep(args.sweep_file)
    except (SweepFileError, OSError) as error:
        print(f"supervise sweep: {error}", file=sys.stderr)
        return 2
    results = run_sweep(sweep, Path(args.runs_dir) / f"{sweep.name}-{stamp}")
    print()
    for result in results:
        print(describe_run(result))
    if all(result.succeeded for result in results):
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
