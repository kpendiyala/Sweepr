"""A tiny training script that fails on purpose, for testing and demoing the supervisor.

It fits y = 3x + 2 with minibatch SGD in plain Python (no GPU or PyTorch needed, so tests
and CI stay fast) and can be told to fail in each way the supervisor must handle:

  --fail-mode oom    raise a CUDA-style out-of-memory error when --batch-size > --oom-above,
                     so retrying with a smaller batch size actually fixes it
  --fail-mode nan    the loss diverges to NaN at --fail-step and stays there
  --fail-mode hang   the training loop freezes at --fail-step (no more metrics)
  --fail-mode crash  the process is killed (SIGKILL) at --fail-step, like a node going away
  --fail-mode bug    a TypeError on startup, which no retry will fix

`hang` and `crash` only fire while SWEEPR_ATTEMPT <= --fail-attempts (default 1), so a retry
succeeds, the way a flaky node does. Checkpoints are written every --ckpt-every steps, and
the script resumes from SWEEPR_RESUME_FROM when the supervisor sets it.

Example:
  python examples/flaky_train.py --fail-mode oom --batch-size 64 --oom-above 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import sweepr

TRUE_W, TRUE_B = 3.0, 2.0


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.OutOfMemoryError, so the error text matches real PyTorch logs."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--step-time", type=float, default=0.01, help="seconds to sleep per step")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument(
        "--fail-mode", choices=["none", "oom", "nan", "hang", "crash", "bug"], default="none"
    )
    p.add_argument("--fail-step", type=int, default=50)
    p.add_argument("--fail-attempts", type=int, default=1)
    p.add_argument("--oom-above", type=int, default=32)
    return p.parse_args(argv)


def save_checkpoint(ckpt_dir: Path, step: int, w: float, b: float) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"step": step, "w": w, "b": b}))
    # Write then rename: a crash mid-write never leaves a half-written checkpoint behind.
    os.replace(tmp, path)
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    attempt = int(os.environ.get("SWEEPR_ATTEMPT", "1"))
    flaky_now = attempt <= args.fail_attempts

    if args.fail_mode == "bug":
        # A real code bug: wrong argument type. It fails the same way on every attempt.
        _ = "batch size: " + args.batch_size  # type: ignore[operator]

    if args.fail_mode == "oom" and args.batch_size > args.oom_above:
        raise OutOfMemoryError(
            f"CUDA out of memory. Tried to allocate {args.batch_size * 32} MiB "
            f"(batch_size={args.batch_size})"
        )

    rng = random.Random(args.seed)
    w, b, start = 0.0, 0.0, 0
    resume = sweepr.resume_path()
    if resume:
        state = json.loads(Path(resume).read_text())
        w, b, start = state["w"], state["b"], state["step"]
        print(f"resumed from {resume} at step {start}", flush=True)

    for step in range(start + 1, args.steps + 1):
        if step == args.fail_step and flaky_now:
            if args.fail_mode == "hang":
                print(f"step {step}: hanging", flush=True)
                while True:
                    time.sleep(3600)
            if args.fail_mode == "crash":
                print(f"step {step}: simulating node failure", flush=True)
                # 9 is SIGKILL on Linux and macOS. On Windows, os.kill ends the process
                # with exit code 9 instead, since Windows has no signals.
                os.kill(os.getpid(), 9)

        xs = [rng.uniform(-1, 1) for _ in range(args.batch_size)]
        ys = [TRUE_W * x + TRUE_B + rng.gauss(0, 0.1) for x in xs]
        errs = [(w * x + b) - y for x, y in zip(xs, ys, strict=True)]
        loss = sum(e * e for e in errs) / len(errs)
        grad_w = 2 * sum(e * x for e, x in zip(errs, xs, strict=True)) / len(errs)
        grad_b = 2 * sum(errs) / len(errs)
        w -= args.lr * grad_w
        b -= args.lr * grad_b

        if args.fail_mode == "nan" and step >= args.fail_step:
            w = b = loss = math.nan

        sweepr.log({"loss": loss, "w": w, "b": b}, step=step)
        if step % args.ckpt_every == 0:
            path = save_checkpoint(Path(args.ckpt_dir), step, w, b)
            sweepr.checkpoint(str(path), step=step)
        if step % 50 == 0:
            print(f"step {step}: loss={loss:.4f}", flush=True)
        time.sleep(args.step_time)

    print(f"done: w={w:.3f} b={b:.3f}", flush=True)
    sweepr.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
