from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
import shlex
import subprocess
import sys
from pathlib import Path


TIMED_RUNNER = r"""
from __future__ import annotations

import os
import signal
import subprocess
import sys


def main() -> None:
    timeout_seconds = float(sys.argv[1])
    kill_after_seconds = float(sys.argv[2])
    cwd = sys.argv[3]
    command = sys.argv[4:]
    if not command:
        raise SystemExit(2)

    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=os.environ.copy(),
    )

    try:
        raise SystemExit(proc.wait(timeout=timeout_seconds))
    except subprocess.TimeoutExpired:
        print(
            f"[spawn_detached] hard timeout reached after {timeout_seconds:.3f}s; terminating pid={proc.pid}",
            flush=True,
        )
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

        try:
            rc = proc.wait(timeout=kill_after_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"[spawn_detached] force killing pid={proc.pid} after {kill_after_seconds:.3f}s grace",
                flush=True,
            )
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            rc = proc.wait()

        raise SystemExit(124 if rc == 0 else rc)


if __name__ == "__main__":
    main()
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--kill-after-seconds", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("missing command")

    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.log_file.parent.mkdir(parents=True, exist_ok=True)

    command_display = " ".join(shlex.quote(part) for part in command)
    launch_lines = [
        f"[spawn_detached] utc_start={datetime.now(timezone.utc).isoformat()}",
        f"[spawn_detached] cwd={args.cwd}",
        f"[spawn_detached] command={command_display}",
    ]

    launch_command = command
    if args.timeout_seconds > 0:
        launch_lines.append(
            f"[spawn_detached] hard_timeout_seconds={args.timeout_seconds:.3f} kill_after_seconds={args.kill_after_seconds:.3f}"
        )
        launch_command = [
            sys.executable,
            "-u",
            "-c",
            TIMED_RUNNER,
            str(args.timeout_seconds),
            str(args.kill_after_seconds),
            str(args.cwd),
            *command,
        ]

    launch_header = ("\n".join(launch_lines) + "\n").encode("utf-8", errors="replace")

    with args.log_file.open("ab") as log_f:
        log_f.write(launch_header)
        log_f.flush()
        proc = subprocess.Popen(
            launch_command,
            cwd=str(args.cwd),
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=os.environ.copy(),
        )

    args.pid_file.write_text(str(proc.pid), encoding="utf-8")
    print(f"PID={proc.pid}")
    print(f"LOG_FILE={args.log_file}")
    print(f"CWD={args.cwd}")
    print(f"COMMAND={command_display}")
    if args.timeout_seconds > 0:
        print(f"HARD_TIMEOUT_SECONDS={args.timeout_seconds}")


if __name__ == "__main__":
    main()
