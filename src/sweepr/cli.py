"""Command line: the `supervise run`, `supervise sweep` and `supervise history` commands."""

import argparse
import json
import secrets
import sys
import time
from pathlib import Path

from sweepr import store
from sweepr.supervisor import AttemptResult, run_attempt
from sweepr.sweep import SweepFileError, describe_run, load_sweep, run_sweep

EXAMPLES = """examples:
  supervise run -- python examples/flaky_train.py --fail-mode crash
  supervise sweep examples/sweep.yaml
  supervise history"""


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

    history = commands.add_parser(
        "history",
        help="list past sweeps, or the runs of one sweep",
        description="Without an id, list recent sweeps. With an id, list that sweep's runs.",
    )
    history.add_argument("sweep_id", nargs="?", type=int, help="a sweep id from the list")
    history.add_argument("--runs-dir", default="runs", help="the folder holding sweepr.db")
    return parser


def database_path(runs_dir: str) -> Path:
    """The run history lives next to the run folders: runs/sweepr.db."""
    return Path(runs_dir) / "sweepr.db"


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
        if args.command == "history":
            return history_command(args)
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
    db = store.connect(database_path(args.runs_dir))
    try:
        results = run_sweep(sweep, Path(args.runs_dir) / f"{sweep.name}-{stamp}", db)
    finally:
        db.close()
    print()
    for result in results:
        print(describe_run(result))
    if all(result.succeeded for result in results):
        return 0
    return 1


def history_command(args) -> int:
    path = database_path(args.runs_dir)
    if not path.exists():
        print(f"supervise history: no history yet ({path} doesn't exist)", file=sys.stderr)
        return 1
    db = store.connect(path)
    try:
        if args.sweep_id is None:
            print("id  started              runs   succeeded  failed  name")
            rows = store.list_sweeps(db)
            for row in rows:
                runs = f"{row['runs_started']}/{row['total_runs']}"
                if row["finished_at"] is None:
                    runs += "*"  # stopped early (Ctrl-C or a crash)
                print(
                    f"{row['id']:<3} {row['started_at']}  {runs:<6} "
                    f"{row['succeeded']:<10} {row['failed']:<7} {row['name']}"
                )
            if any(row["finished_at"] is None for row in rows):
                print("* = did not finish")
        else:
            rows = store.list_runs(db, args.sweep_id)
            if not rows:
                print(f"supervise history: no sweep with id {args.sweep_id}", file=sys.stderr)
                return 1
            for row in rows:
                status = row["status"]
                if row["failure"]:
                    status += f" {row['failure']}"
                params = " ".join(f"{k}={v}" for k, v in json.loads(row["params"]).items())
                print(f"{row['name']}  {status:<16} {row['attempts']} attempt(s)  {params}")
    finally:
        db.close()
    return 0


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
