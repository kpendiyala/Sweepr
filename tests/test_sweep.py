"""Tests for sweeps: reading sweep files, the retry rules, and a real end-to-end sweep."""

import sys
from pathlib import Path

import pytest

from sweepr import store
from sweepr.classify import CODE_BUG, HUNG, NAN, NODE_FAILURE, OOM, UNKNOWN
from sweepr.cli import main
from sweepr.sweep import (
    SweepFileError,
    grid_combinations,
    load_sweep,
    plan_retry,
    render_command,
    run_sweep,
)

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "flaky_train.py"


def write_sweep(tmp_path, text):
    path = tmp_path / "sweep.yaml"
    path.write_text(text)
    return path


# --- Reading sweep files ---


def test_load_sweep(tmp_path):
    path = write_sweep(
        tmp_path,
        "command: python train.py --lr {lr} --out {run_dir}/ckpt\n"
        "grid:\n  lr: [0.1, 0.2]\n"
        "hang_timeout: 60\n",
    )
    sweep = load_sweep(path)
    assert sweep.name == "sweep"  # from the file name
    assert sweep.command == ["python", "train.py", "--lr", "{lr}", "--out", "{run_dir}/ckpt"]
    assert sweep.max_attempts == 3  # the default
    assert sweep.hang_timeout == 60


def test_grid_combinations():
    combinations = grid_combinations({"lr": [0.1, 0.2], "bs": [32, 64]})
    assert combinations == [
        {"lr": 0.1, "bs": 32},
        {"lr": 0.1, "bs": 64},
        {"lr": 0.2, "bs": 32},
        {"lr": 0.2, "bs": 64},
    ]
    assert grid_combinations({}) == [{}]  # no grid: a single run


def test_render_keeps_a_value_with_spaces_as_one_argument():
    command = render_command(
        ["train.py", "--name", "{name}", "--out={run_dir}"], {"name": "a b"}, "r1"
    )
    assert command == ["train.py", "--name", "a b", "--out=r1"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("command: python train.py --lr {learning_rate}\ngrid: {lr: [1]}\n", "learning_rate"),
        ("grid: {lr: [1]}\n", "command"),
        ("command: python train.py\ngrid: {lr: []}\n", "grid.lr"),
        ("command: python train.py\ngrid: {lr: 0.1}\n", "grid.lr"),
        ("command: python train.py\nmax_attempts: 0\n", "max_attempts"),
        ("command: python train.py\nhang_timeout: -1\n", "hang_timeout"),
        ("command: python train.py\nretries: 5\n", "unknown keys"),
        ("command: [unclosed\n", "not valid YAML"),
        ("- just a list\n", "expected keys"),
    ],
)
def test_bad_sweep_files_fail_before_anything_runs(tmp_path, text, message):
    with pytest.raises(SweepFileError) as error:
        load_sweep(write_sweep(tmp_path, text))
    assert message in str(error.value)


# --- The retry rules ---

PARAMS = {"lr": 0.1, "batch_size": 64}


@pytest.mark.parametrize(
    ("failure", "attempts_so_far", "retries"),
    [
        (OOM, 1, True),
        (NODE_FAILURE, 1, True),
        (HUNG, 2, True),
        (UNKNOWN, 1, True),
        (NAN, 1, False),  # the same config would diverge again
        (CODE_BUG, 1, False),  # the same code would fail again
        (NODE_FAILURE, 3, False),  # max_attempts (3) reached
    ],
)
def test_plan_retry(failure, attempts_so_far, retries):
    next_params, reason = plan_retry(failure, attempts_so_far, 3, PARAMS)
    assert (next_params is not None) == retries
    assert failure in reason


def test_oom_halves_the_batch_size_and_keeps_the_other_params():
    next_params, _ = plan_retry(OOM, 1, 3, PARAMS)
    assert next_params == {"lr": 0.1, "batch_size": 32}
    assert PARAMS["batch_size"] == 64  # the original dict is unchanged

    next_params, reason = plan_retry(OOM, 1, 3, {"batch_size": 1})
    assert next_params is None
    assert "can't go any lower" in reason

    next_params, _ = plan_retry(OOM, 1, 3, {"lr": 0.1})  # no batch_size: plain retry
    assert next_params == {"lr": 0.1}


# --- A real sweep ---


def test_sweep_recovers_what_it_can_and_gives_up_on_the_rest(tmp_path):
    path = write_sweep(
        tmp_path,
        f"command: [{sys.executable!r}, {str(SCRIPT)!r}, --fail-mode, '{{mode}}',"
        " --batch-size, '{batch_size}', --steps, '60', --step-time, '0.01',"
        " --fail-step, '30', --ckpt-every, '10', --ckpt-dir, '{run_dir}/ckpt']\n"
        "grid:\n  mode: [none, oom, crash, hang, nan, bug]\n  batch_size: [64]\n"
        "hang_timeout: 1\n",
    )
    db = store.connect(tmp_path / "sweepr.db")
    results = run_sweep(load_sweep(path), tmp_path / "runs", db, poll_interval=0.05, stop_grace=1)
    by_mode = {result.params["mode"]: result for result in results}

    for mode in ("none", "oom", "crash", "hang"):
        assert by_mode[mode].succeeded, (mode, by_mode[mode].reason)
    assert len(by_mode["none"].attempts) == 1
    assert by_mode["oom"].params["batch_size"] == 32  # the retry used half the batch size

    for mode in ("crash", "hang"):
        assert len(by_mode[mode].attempts) == 2
        second_log = by_mode[mode].attempts[1].log_path.read_text()
        assert "resumed from" in second_log  # from this run's own checkpoint folder
        assert by_mode[mode].name in second_log

    for mode, label in (("nan", NAN), ("bug", CODE_BUG)):
        assert not by_mode[mode].succeeded
        assert by_mode[mode].failure == label
        assert len(by_mode[mode].attempts) == 1  # never retried

    # Everything above was also saved to the database, attempt by attempt.
    runs = {row["name"]: row for row in store.list_runs(db, 1)}
    assert runs[by_mode["oom"].name]["status"] == "succeeded"
    assert runs[by_mode["oom"].name]["attempts"] == 2
    assert '"batch_size": 32' in runs[by_mode["oom"].name]["params"]  # the final params
    assert runs[by_mode["bug"].name]["failure"] == CODE_BUG
    failures = db.execute(
        "SELECT attempts.failure FROM attempts JOIN runs ON runs.id = attempts.run_id"
        " WHERE runs.name = ? ORDER BY number",
        (by_mode["oom"].name,),
    ).fetchall()
    assert [row["failure"] for row in failures] == [OOM, None]


def test_cli_sweep(tmp_path, capsys):
    path = write_sweep(
        tmp_path,
        f"command: [{sys.executable!r}, {str(SCRIPT)!r}, --steps, '5', --step-time, '0',"
        " --fail-mode, '{mode}', --ckpt-dir, '{run_dir}/ckpt']\n"
        "grid: {mode: [none, bug]}\n",
    )
    assert main(["sweep", str(path), "--runs-dir", str(tmp_path / "runs")]) == 1
    output = capsys.readouterr()
    assert "2/2 runs finished: 1 succeeded, 1 failed (CODE_BUG 1)" in output.err
    assert "run-002  FAILED CODE_BUG" in output.out

    # The sweep is in the history, and so are its runs.
    assert main(["history", "--runs-dir", str(tmp_path / "runs")]) == 0
    listing = capsys.readouterr().out
    assert "2/2" in listing and "sweep" in listing
    assert main(["history", "1", "--runs-dir", str(tmp_path / "runs")]) == 0
    assert "run-002  failed CODE_BUG" in capsys.readouterr().out


def test_cli_sweep_reports_a_bad_file(tmp_path, capsys):
    path = write_sweep(tmp_path, "command: python train.py --lr {lr}\n")
    assert main(["sweep", str(path)]) == 2
    assert "command uses {lr}" in capsys.readouterr().err
