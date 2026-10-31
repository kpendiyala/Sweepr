# Sweepr

A self-healing supervisor for ML training sweeps. It watches every run in a sweep, works out
why a run failed (out of memory, NaN loss, hung, node failure, or a code bug), and applies the
recovery you configured, so you stop babysitting runs.

> Work in progress. The supervisor CLI is being built now.

## Development

```bash
python -m venv .venv && source .venv/bin/activate
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

## Instrumenting your own script

```python
import sweepr

for step in range(num_steps):
    loss = train_step()
    sweepr.log({"loss": loss}, step=step)

sweepr.finish()
```

Outside the supervisor these calls do nothing, so your script still runs on its own.
