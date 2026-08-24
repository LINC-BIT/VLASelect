from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROLLOUT_RE = re.compile(r"\[rollout\]\s+update=(?P<update>\d+)\/(?P<total_updates>\d+)\s+step=(?P<step>\d+)\/(?P<total_steps>\d+)")
TRAIN_UPDATE_RE = re.compile(r"\[train\]\s+update=(?P<update>\d+)\/(?P<total_updates>\d+)")
TRAIN_STEP_RE = re.compile(r"(?:^|\s)step=(?P<global_step>\d+)(?:\s|$)")
ITER_RE = re.compile(r"\biter=(?P<iter>\d+)\b")
EVAL_ONCE_RE = re.compile(r"(?:\beval_success_once|\bsuccess_once)=(?P<value>[-+]?\d*\.?\d+|nan)")
EVAL_END_RE = re.compile(r"(?:\beval_success_(?:at_)?end|\bsuccess_(?:at_)?end)=(?P<value>[-+]?\d*\.?\d+|nan)")
ENV_PATTERNS = [
    re.compile(r"initial_env_id=(?P<env>\S+)"),
    re.compile(r"current_env_id=(?P<env>\S+)"),
    re.compile(r"current_env=(?P<env>\S+)"),
    re.compile(r"ENV_ID=(?P<env>\S+)"),
    re.compile(r"switching env from \S+ to (?P<env>\S+)"),
    re.compile(r"switched to env\[\d+\]=(?P<env>\S+)"),
    re.compile(r"\benv=(?P<env>\S+)")
]

PYTHON_EXCEPTION_RE = re.compile(r"^(?:[A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Exception):")

RUNNING_STATUSES = {"running", "launched"}
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "finished", "inherited"}


@dataclass
class MethodState:
    name: str
    display_name: str
    gpu: int | None
    manifest_status: str = "unknown"
    pid: int | None = None
    exit_code: int | None = None
    env_id: str | None = None
    phase: str | None = None
    update: int | None = None
    total_updates: int | None = None
    step: int | None = None
    total_steps: int | None = None
    global_step: int | None = None
    eval_success_once: float | None = None
    eval_success_end: float | None = None
    last_emit: str = ""
    last_emit_at: float = 0.0

    def parse_line(self, raw_line: str) -> None:
        line = raw_line.strip()
        if not line:
            return

        for env_re in ENV_PATTERNS:
            match = env_re.search(line)
            if match:
                self.env_id = match.group("env").rstrip(",")
                break

        if "[setup]" in line and self.phase is None:
            self.phase = "setup"

        rollout_match = ROLLOUT_RE.search(line)
        if rollout_match:
            self.phase = "rollout"
            self.update = int(rollout_match.group("update"))
            self.total_updates = int(rollout_match.group("total_updates"))
            self.step = int(rollout_match.group("step"))
            self.total_steps = int(rollout_match.group("total_steps"))

        train_match = TRAIN_UPDATE_RE.search(line)
        if train_match:
            self.phase = "train"
            self.update = int(train_match.group("update"))
            self.total_updates = int(train_match.group("total_updates"))
            step_match = TRAIN_STEP_RE.search(line)
            if step_match:
                self.global_step = int(step_match.group("global_step"))

        iter_match = ITER_RE.search(line)
        if iter_match and self.update is None:
            self.phase = self.phase or "train"
            self.update = int(iter_match.group("iter"))

        eval_once_match = EVAL_ONCE_RE.search(line)
        if eval_once_match:
            self.eval_success_once = parse_metric(eval_once_match.group("value"))

        eval_end_match = EVAL_END_RE.search(line)
        if eval_end_match:
            self.eval_success_end = parse_metric(eval_end_match.group("value"))

        lowered = line.lower()
        if "reached time limit" in lowered:
            self.phase = "time-limit"
        elif "zero-success early stop" in lowered:
            self.phase = "early-stop"
        elif "aborting during rollout" in lowered:
            self.phase = "aborting"
        elif line.startswith("Traceback ") or PYTHON_EXCEPTION_RE.match(line):
            self.phase = "error"

    def effective_status(self) -> str:
        if self.manifest_status in TERMINAL_STATUSES:
            return self.manifest_status
        if self.manifest_status in RUNNING_STATUSES:
            if is_pid_alive(self.pid):
                return "running"
            if self.exit_code is not None:
                return "failed" if self.exit_code != 0 else "completed"
            return "finished"
        return self.manifest_status

    def snapshot(self, label: str) -> str | None:
        status = self.effective_status()
        parts = [f"[{label}]", f"method={self.display_name}", f"status={status}"]
        if self.phase and status in {"running", "finished", "completed", "failed"}:
            parts.append(f"phase={self.phase}")
        if self.env_id:
            parts.append(f"env={self.env_id}")
        if self.update is not None and self.total_updates is not None:
            parts.append(f"update={self.update}/{self.total_updates}")
        elif self.update is not None:
            parts.append(f"update={self.update}")
        if self.step is not None and self.total_steps is not None:
            remaining = max(self.total_steps - self.step, 0)
            parts.append(f"step={self.step}/{self.total_steps}")
            parts.append(f"remaining_step={remaining}")
        elif self.global_step is not None:
            parts.append(f"global_step={self.global_step}")
        if self.eval_success_once is not None:
            parts.append(f"eval_once={format_metric(self.eval_success_once)}")
        if self.eval_success_end is not None:
            parts.append(f"eval_end={format_metric(self.eval_success_end)}")
        if self.gpu is not None:
            parts.append(f"gpu={self.gpu}")
        if self.exit_code is not None and status in {"failed", "completed"}:
            parts.append(f"exit={self.exit_code}")
        return " ".join(parts)


def parse_metric(raw: str) -> float | None:
    if raw.lower() == "nan":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def format_metric(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def is_pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.exists():
        try:
            fields = proc_stat.read_text(encoding="utf-8").split()
        except OSError:
            fields = []
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def tail_initial_lines(path: Path, limit: int = 200) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    method_states: dict[str, MethodState] = {}
    file_positions: dict[Path, int] = {}
    initialized_logs: set[Path] = set()

    while True:
        manifest = load_manifest(args.manifest)
        if manifest is not None:
            break
        time.sleep(0.5)

    while True:
        manifest = load_manifest(args.manifest)
        if manifest is None:
            time.sleep(args.interval_seconds)
            continue

        all_done = True
        for method_entry in manifest.get("methods", []):
            name = method_entry.get("name", "unknown")
            state = method_states.get(name)
            if state is None:
                state = MethodState(
                    name=name,
                    display_name=method_entry.get("display_name", name),
                    gpu=method_entry.get("gpu"),
                )
                method_states[name] = state

            state.display_name = method_entry.get("display_name", state.display_name)
            state.gpu = method_entry.get("gpu", state.gpu)
            state.manifest_status = str(method_entry.get("status", state.manifest_status))
            state.pid = method_entry.get("pid")
            state.exit_code = method_entry.get("exit_code")

            log_file_value = method_entry.get("log_file")
            log_path = Path(log_file_value) if log_file_value else None
            if log_path is not None and not log_path.is_absolute():
                log_path = Path.cwd() / log_path

            if log_path is not None and log_path.exists():
                if log_path not in initialized_logs:
                    for line in tail_initial_lines(log_path):
                        state.parse_line(line)
                    try:
                        file_positions[log_path] = log_path.stat().st_size
                    except OSError:
                        file_positions[log_path] = 0
                    initialized_logs.add(log_path)
                else:
                    last_pos = file_positions.get(log_path, 0)
                    try:
                        current_size = log_path.stat().st_size
                    except OSError:
                        current_size = last_pos
                    if current_size < last_pos:
                        last_pos = 0
                    try:
                        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
                            handle.seek(last_pos)
                            for line in handle:
                                state.parse_line(line)
                            file_positions[log_path] = handle.tell()
                    except OSError:
                        pass

            snapshot = state.snapshot(args.label)
            if snapshot:
                now = time.time()
                should_emit = snapshot != state.last_emit
                if not should_emit and state.effective_status() == "running":
                    should_emit = (now - state.last_emit_at) >= args.heartbeat_seconds
                if should_emit:
                    print(snapshot, flush=True)
                    state.last_emit = snapshot
                    state.last_emit_at = now

            if state.effective_status() == "running":
                all_done = False
            elif state.manifest_status == "queued":
                all_done = False

        suite_state = manifest.get("suite_state")
        if all_done and suite_state == "finished":
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
