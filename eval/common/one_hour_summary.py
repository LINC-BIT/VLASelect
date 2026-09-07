from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any

EVAL_ROOT = Path(__file__).resolve().parents[1]
SUCCESS_KEYS = (
    'eval_success_once',
    'train_success_once',
    'success_once',
    'eval/success_once',
)
VLASELECT_NAMES = {'ours', 'ours_single_agent'}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def resolve_eval_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (EVAL_ROOT / path).resolve()


def parse_envs(raw_value: Any) -> list[str]:
    if isinstance(raw_value, list):
        return [str(value) for value in raw_value]
    try:
        parsed = ast.literal_eval(str(raw_value))
    except (SyntaxError, ValueError):
        parsed = [part.strip() for part in str(raw_value).split(',') if part.strip()]
    if isinstance(parsed, (list, tuple)):
        return [str(value) for value in parsed]
    return []


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def success_value(metric: dict[str, Any]) -> float | None:
    for key in SUCCESS_KEYS:
        value = finite_float(metric.get(key))
        if value is not None:
            return value
    return None


def env_index_value(metric: dict[str, Any]) -> int | None:
    for key in ('env_index', 'current_env_index'):
        value = finite_float(metric.get(key))
        if value is not None:
            return int(value)
    return None


def load_env_averages(run_dir: Path) -> dict[int, float]:
    history_path = run_dir / 'metrics_history.json'
    if not history_path.exists():
        return {}
    try:
        payload = load_json(history_path)
    except Exception:
        return {}
    history = payload.get('history', []) if isinstance(payload, dict) else payload
    if not isinstance(history, list):
        return {}
    values_by_env: dict[int, list[float]] = {}
    for item in history:
        if not isinstance(item, dict):
            continue
        env_index = env_index_value(item)
        value = success_value(item)
        if env_index is None or value is None:
            continue
        values_by_env.setdefault(env_index, []).append(value)
    return {
        env_index: sum(values) / len(values)
        for env_index, values in values_by_env.items()
        if values
    }


def summarize_panel(panel: dict[str, Any], env_count: int) -> list[str]:
    suite_manifest_raw = str(panel.get('suite_manifest', '')).strip()
    if not suite_manifest_raw:
        return [f"[1h-summary] {panel.get('family', 'unknown')}: no suite_manifest"]
    suite_manifest_path = resolve_eval_path(suite_manifest_raw)
    if not suite_manifest_path.exists():
        return [f"[1h-summary] {panel.get('family', 'unknown')}: missing suite_manifest={suite_manifest_path}"]
    suite_manifest = load_json(suite_manifest_path)
    methods = [method for method in suite_manifest.get('methods', []) if isinstance(method, dict)]
    env_names = parse_envs(panel.get('envs_id', suite_manifest.get('envs_id', [])))

    ours_by_env: dict[int, list[float]] = {}
    baseline_by_env: dict[int, list[float]] = {}
    for method in methods:
        name = str(method.get('name', '')).strip()
        run_dir_raw = str(method.get('run_dir', '')).strip()
        if not name or not run_dir_raw:
            continue
        env_avgs = load_env_averages(resolve_eval_path(run_dir_raw))
        target = ours_by_env if name in VLASELECT_NAMES else baseline_by_env
        for env_index, avg in env_avgs.items():
            if 0 <= env_index < env_count:
                target.setdefault(env_index, []).append(avg)

    lines = []
    family = str(panel.get('family', 'unknown'))
    workload = str(panel.get('workload_name', family))
    for env_index in range(env_count):
        ours_values = ours_by_env.get(env_index, [])
        baseline_values = baseline_by_env.get(env_index, [])
        env_name = env_names[env_index] if env_index < len(env_names) else f'env{env_index + 1}'
        if not ours_values or not baseline_values:
            lines.append(f"[1h-summary] {family} {workload} env{env_index + 1} {env_name}: improvement=NaN (missing data)")
            continue
        ours_avg = sum(ours_values) / len(ours_values)
        baseline_avg = sum(baseline_values) / len(baseline_values)
        improvement_pp = (ours_avg - baseline_avg) * 100.0
        lines.append(
            f"[1h-summary] {family} {workload} env{env_index + 1} {env_name}: "
            f"improvement={improvement_pp:+.2f}% "
            f"(vlaselect={ours_avg * 100.0:.2f}%, baselines_avg={baseline_avg * 100.0:.2f}%)"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description='Print per-env VLASelect improvement for one-hour runs.')
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--env-count', type=int, default=2)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    panels = [panel for panel in manifest.get('panels', manifest.get('families', [])) if isinstance(panel, dict)]
    if not panels:
        print('[1h-summary] no panels found')
        return
    for panel in panels:
        for line in summarize_panel(panel, args.env_count):
            print(line)


if __name__ == '__main__':
    main()
