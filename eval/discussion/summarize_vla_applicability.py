from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_history(path: Path) -> List[Dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def process_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.4f}"


def _maybe_float(value: Any) -> Optional[float]:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    return _maybe_float(metrics.get(key))


def final_metric(metrics: Optional[Dict[str, Any]], primary_key: str, fallback_key: str) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    return _maybe_float(metrics.get(primary_key, metrics.get(fallback_key)))


def summarize_octo_suite(entry: Dict[str, Any]) -> Dict[str, Any]:
    suite_dir = Path(entry["run_dir"])
    summary_rows = load_csv_rows(suite_dir / "summary.csv")
    row = summary_rows[0] if summary_rows else {}
    latest_eval_once = latest_eval_end = final_eval_once = final_eval_end = None
    history_len = 0
    status = "launching" if process_alive(entry.get("pid")) else "failed"
    if row:
        latest_eval_once = _maybe_float(row.get("final_success_once"))
        latest_eval_end = _maybe_float(row.get("final_success_at_end"))
        final_eval_once = latest_eval_once
        final_eval_end = latest_eval_end
        history_len = int(float(row.get("num_eval_points", "0") or "0"))
        status = row.get("status") or status
        if status == "completed":
            status = "completed"
        elif process_alive(entry.get("pid")):
            status = "running"
        else:
            status = "partial"
    return {
        "family": entry["family"],
        "status": status,
        "pid": entry.get("pid"),
        "log_file": entry.get("log_file"),
        "run_dir": str(suite_dir),
        "history_len": history_len,
        "latest_train_once": None,
        "latest_train_end": None,
        "latest_eval_once": latest_eval_once,
        "latest_eval_end": latest_eval_end,
        "final_eval_once": final_eval_once,
        "final_eval_end": final_eval_end,
    }


def resolve_status(entry: Dict[str, Any], run_dir: Path) -> str:
    pid = entry.get("pid")
    final_metrics = load_json(run_dir / "final_eval_metrics.json")
    latest_metrics = load_json(run_dir / "latest_metrics.json")
    if final_metrics is not None:
        return "completed"
    if latest_metrics is not None and process_alive(pid):
        return "running"
    if latest_metrics is not None:
        return "partial"
    if process_alive(pid):
        return "launching"
    return "failed"


def summarize_json_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(entry["run_dir"])
    latest_metrics = load_json(run_dir / "latest_metrics.json")
    final_metrics = load_json(run_dir / "final_eval_metrics.json")
    history = load_history(run_dir / "metrics_history.json")
    return {
        "family": entry["family"],
        "status": resolve_status(entry, run_dir),
        "pid": entry.get("pid"),
        "log_file": entry.get("log_file"),
        "run_dir": str(run_dir),
        "history_len": len(history),
        "latest_train_once": latest_metric(latest_metrics, "train_success_once"),
        "latest_train_end": latest_metric(latest_metrics, "train_success_at_end"),
        "latest_eval_once": latest_metric(latest_metrics, "eval_success_once"),
        "latest_eval_end": latest_metric(latest_metrics, "eval_success_at_end"),
        "final_eval_once": final_metric(final_metrics, "success_once", "success"),
        "final_eval_end": final_metric(final_metrics, "success_at_end", "success"),
    }


def summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    if entry.get("kind") == "octo_suite":
        return summarize_octo_suite(entry)
    return summarize_json_entry(entry)


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("")
    print("VLA applicability summary")
    print("family         status      latest_eval_once  latest_eval_end  final_eval_once  final_eval_end  updates")
    for row in rows:
        print(
            f"{row['family']:<14}"
            f"{row['status']:<12}"
            f"{format_metric(row['latest_eval_once']):<18}"
            f"{format_metric(row['latest_eval_end']):<17}"
            f"{format_metric(row['final_eval_once']):<17}"
            f"{format_metric(row['final_eval_end']):<16}"
            f"{row['history_len']}"
        )
    print("")
    for row in rows:
        print(f"[{row['family']}] run_dir={row['run_dir']}")
        print(f"[{row['family']}] log_file={row['log_file']}")


def all_finished(rows: List[Dict[str, Any]]) -> bool:
    return all(row["status"] in {"completed", "partial", "failed"} for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Invalid manifest: {manifest_path}")
    entries = manifest.get("runs")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(f"No runs found in manifest: {manifest_path}")

    try:
        while True:
            rows = [summarize_entry(entry) for entry in entries]
            print_summary(rows)
            if not args.wait or all_finished(rows):
                break
            time.sleep(max(1.0, args.poll_seconds))
    except KeyboardInterrupt:
        print("Interrupted while waiting for runs to finish.")


if __name__ == "__main__":
    main()
