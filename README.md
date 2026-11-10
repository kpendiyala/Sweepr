# Sweepr

A self-healing supervisor for ML training sweeps. It watches every run in a sweep, works out
why a run failed (out of memory, NaN loss, hung, node failure, or a code bug), and applies the
recovery you configured, so you stop babysitting runs.

> Work in progress. `supervise run` works for single runs; sweeps and retries are next.

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

supervise run --progress-timeout 5 -- python examples/flaky_train.py --fail-mode hang
# attempt 1 failed: HUNG (killed by supervisor: progress_timeout) after 5.5s, ...
```

Each attempt writes its output to `runs/<run id>/attempt-N.log` and its metrics to
`attempt-N.metrics.jsonl`. If `supervise` isn't on your PATH, use `python -m sweepr run -- ...`.
Run `supervise run --help` for all options. Sweepr has no runtime dependencies beyond Python.

## How the code is organized

| File | What it does |
| --- | --- |
| `src/sweepr/sdk.py` | Training side: `log`, `checkpoint`, `finish` append JSON lines to a file; a thread writes heartbeats |
| `src/sweepr/supervisor.py` | `run_attempt`: starts the command, polls the metrics file, stops it if needed, classifies the result |
| `src/sweepr/watchdog.py` | Health checks with no I/O: frozen (no records), stuck (no new metrics), NaN loss |
| `src/sweepr/classify.py` | Failure classifier with no I/O: OOM, NAN, HUNG, NODE_FAILURE, CODE_BUG, UNKNOWN |
| `src/sweepr/process.py` | Starts the process in its own group and stops the whole group (all OS-specific code) |
| `src/sweepr/cli.py` | `supervise run [options] -- <command>` (argparse) |

## Instrumenting your own script

```python
import sweepr

for step in range(num_steps):
    loss = train_step()
    sweepr.log({"loss": loss}, step=step)

sweepr.finish()
```

Outside the supervisor these calls do nothing, so your script still runs on its own.
