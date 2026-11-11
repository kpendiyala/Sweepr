"""Starting and stopping the training process. All the OS-specific code lives here.

Each attempt runs in its own process group (a new session on Linux and macOS, a new process
group on Windows). Stopping the attempt then stops everything it started too: DataLoader
workers, torchrun children, shell pipelines. If those survived, they would keep holding GPU
memory and the retry would fail with OOM.
"""

import os
import signal
import subprocess
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


def start_process(cmd: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    """Start `cmd` in its own process group, with stdout and stderr appended to `log_path`."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Output goes straight to a file, with stderr merged into stdout so the order is kept.
    # A pipe would need someone reading it all the time, or the child blocks once it fills.
    with open(log_path, "ab") as log_file:
        if IS_WINDOWS:
            return subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # new session, so the process group id equals proc.pid
        )
    # The child has its own copy of the file handle, so closing ours here is fine.


def stop_process(proc: subprocess.Popen, grace_seconds: float) -> None:
    """Ask the whole process tree to exit, and force it if it hasn't after `grace_seconds`."""
    if proc.poll() is not None:
        return  # already exited

    if IS_WINDOWS:
        # Console programs on Windows have no SIGTERM, so kill the whole tree (/T) right away.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
        return

    # SIGTERM first, so the script gets a chance to save state and clean up.
    signal_group(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        signal_group(proc.pid, signal.SIGKILL)
        proc.wait()


def kill_leftovers(proc: subprocess.Popen) -> None:
    """After the main process has exited, kill anything it left running in its group."""
    if not IS_WINDOWS:
        signal_group(proc.pid, signal.SIGKILL)


def signal_group(group_id: int, sig: int) -> None:
    """Send `sig` to every process in the group. Does nothing if the group is already gone."""
    try:
        os.killpg(group_id, sig)
    except (ProcessLookupError, PermissionError):
        pass
