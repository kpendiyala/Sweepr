# Sweepr

A self-healing supervisor for ML training sweeps. It watches every run in a sweep, works out
why a run failed (out of memory, NaN loss, hung, node failure, or a code bug), and applies the
recovery you configured, so you stop babysitting runs.

> Work in progress. Single runs and self-healing sweeps work; a run history database is next.

## Development

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest
```

## Try the fault-injection script

`examples/flaky_train.py` is a tiny training script that fails on purpose in each way the
supervisor handles:

```bash
python examples/flaky_train.py                                   # trains cleanly
python examples/flaky_train.py --fail-mode oom --batch-size 64   # CUDA-style OOM
python examples/flaky_train.py --fail-mode nan --fail-step 20    # loss goes NaN
python examples/flaky_train.py --fail-mode bug                   # TypeError on startup
```

## Supervise a run

```bash
supervise run -- python examples/flaky_train.py --fail-mode crash
# attempt 1 failed: NODE_FAILURE (killed by signal 9) after 0.5s, last step 49, ...

supervise run --hang-timeout 5 -- python examples/flaky_train.py --fail-mode hang
# attempt 1 failed: HUNG (killed by supervisor: no new metrics) after 5.5s, ...
```

Each attempt writes its output to `runs/<run id>/attempt-N.log` and its metrics to
`attempt-N.metrics.jsonl`. If `supervise` isn't on your PATH, use `python -m sweepr run -- ...`.

## Run a sweep that heals itself

```bash
supervise sweep examples/sweep.yaml
# [sweepr] run-003 attempt 1: out of memory: CUDA out of memory
# [sweepr] run-003 -> OOM: retry with batch_size=32
# [sweepr] run-005 attempt 1: killed by signal 9
# [sweepr] run-005 -> NODE_FAILURE: retry
# ...
# [sweepr] 12/12 runs finished: 8 succeeded, 4 failed (NAN 2, CODE_BUG 2)
```

A sweep file is a command template and a parameter grid:

```yaml
command: python train.py --lr {lr} --batch-size {batch_size} --ckpt-dir {run_dir}/ckpt
grid:
  lr: [0.01, 0.05]
  batch_size: [64]
max_attempts: 3     # optional
hang_timeout: 300   # optional: seconds without a new metric before a run counts as hung
```

Runs go one at a time. When an attempt fails, the retry rules are:

| Failure | What happens |
| --- | --- |
| OOM | retry with `batch_size` halved |
| NODE_FAILURE, HUNG, UNKNOWN | retry |
| NAN, CODE_BUG | give up: a retry would fail the same way |

Every retry resumes from the run's latest checkpoint, and a run stops after `max_attempts`.
`{run_dir}` is each run's own folder, so runs never overwrite each other's checkpoints.

## How the code is organized

| File | What it does |
| --- | --- |
| `src/sweepr/sdk.py` | Training side: `log`, `checkpoint`, `finish` append JSON lines to a file |
| `src/sweepr/process.py` | Starts the process in its own group and stops the whole group (all OS-specific code) |
| `src/sweepr/supervisor.py` | `run_attempt`: starts the command, reads the metrics file, stops it if it hangs or goes NaN, classifies the result |
| `src/sweepr/classify.py` | Failure classifier with no I/O: OOM, NAN, HUNG, NODE_FAILURE, CODE_BUG, UNKNOWN |
| `src/sweepr/sweep.py` | Sweep files, the retry rules, and running each grid combination with retries |
| `src/sweepr/cli.py` | `supervise run [options] -- <command>` and `supervise sweep <file>` (argparse) |

## Instrumenting your own script

```python
import sweepr

for step in range(num_steps):
    loss = train_step()
    sweepr.log({"loss": loss}, step=step)

sweepr.finish()
```

Outside the supervisor these calls do nothing, so your script still runs on its own.
