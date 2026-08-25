from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from train.common.time_breakdown import write_time_breakdown_from_metrics_history


def _coerce_scalar(value: Any) -> float | int | bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except Exception:
            scalar = value
        if isinstance(scalar, (bool, int, float)):
            return scalar
        value = scalar
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _scalar_dict(payload: dict[str, Any] | None) -> dict[str, float | int | bool]:
    if not isinstance(payload, dict):
        return {}
    result: dict[str, float | int | bool] = {}
    for key, value in payload.items():
        scalar = _coerce_scalar(value)
        if scalar is not None:
            result[key] = scalar
    return result


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def normalize_eval_metrics(metrics: dict[str, Any] | None) -> dict[str, float | int | bool]:
    scalar_metrics = _scalar_dict(metrics)
    normalized: dict[str, float | int | bool] = {}
    for key, value in scalar_metrics.items():
        normalized_key = "success_at_end" if key == "success_end" else key
        normalized[f"eval_{normalized_key}"] = value
    return normalized


def build_metric_entry(
    *,
    update: int,
    global_step: int,
    current_env_id: str,
    current_env_index: int,
    elapsed_minutes: float,
    eval_metrics: dict[str, Any] | None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "update": int(update),
        "global_step": int(global_step),
        "current_env_id": current_env_id,
        "current_env_index": int(current_env_index),
        "elapsed_hours": float(elapsed_minutes) / 60.0,
    }
    entry.update(_scalar_dict(extras))
    entry.update(normalize_eval_metrics(eval_metrics))
    return entry


class JsonMetricsLogger:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.history_path = self.output_dir / "metrics_history.json"
        self.latest_path = self.output_dir / "latest_metrics.json"
        self.final_eval_path = self.output_dir / "final_eval_metrics.json"
        self.history: list[dict[str, Any]] = []
        self._load_history()

    def _load_history(self) -> None:
        if not self.history_path.exists():
            return
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        history = payload.get("history")
        if isinstance(history, list):
            self.history = [entry for entry in history if isinstance(entry, dict)]

    def append(self, metric: dict[str, Any]) -> None:
        self.history.append(metric)
        save_json(self.latest_path, metric)
        save_json(self.history_path, {"history": self.history})
        write_time_breakdown_from_metrics_history(self.output_dir, self.history)

    def save_final_eval(self, eval_metrics: dict[str, Any] | None) -> None:
        save_json(self.final_eval_path, _scalar_dict(eval_metrics))
