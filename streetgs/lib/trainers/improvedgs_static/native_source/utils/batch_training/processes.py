"""
Subprocess management for batch training.

Training and post-processing are launched as child processes. This module keeps
track of them so Ctrl-C or SIGTERM can stop the full process group instead of
leaving long-running GPU jobs alive.
"""
from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


child_processes: list[subprocess.Popen] = []
should_exit = False


def signal_handler(sig: int, frame: object) -> None:
    """
        Stop all known child processes and exit with the standard interrupt code.
    """
    del sig, frame
    global should_exit
    should_exit = True
    print("\nReceived stop signal, terminating all child processes...")

    for proc in list(child_processes):
        if proc.poll() is not None:
            continue
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass

    for proc in list(child_processes):
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    print("All child processes stopped.")
    raise SystemExit(130)


def install_signal_handlers() -> None:
    """
        Register cleanup handlers for SIGINT and SIGTERM.
    """
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def run_command(command: list[str], gpu_id: Any, dry_run: bool) -> int:
    """
        Run one command with an optional CUDA_VISIBLE_DEVICES override.
    """
    command_text = " ".join(command)
    print(command_text)
    if dry_run:
        return 0

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    child_processes.append(process)
    try:
        return process.wait()
    finally:
        if process in child_processes:
            child_processes.remove(process)
