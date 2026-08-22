from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def maybe_float(value: Any) -> Optional[float]:
    if value in (None, '', 'None'):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_metric(value: Optional[float]) -> str:
    return '-' if value is None else f'{value:.4f}'


def summarize_run(entry: Dict[str, Any]) -> Dict[str, Any]:
    run_dir = Path(entry['run_dir'])
    result_json = entry.get('result_json', '')
    result_path = Path(result_json) if result_json else None
    metrics = load_json(result_path) if result_path and result_path.is_file() else load_json(run_dir / 'metrics.json')

    if isinstance(metrics, list):
        rows = metrics
        latest = rows[-1] if rows else {}
        best = max(rows, key=lambda row: maybe_float(row.get('score')) or float('-inf')) if rows else {}
        latest_success_once = maybe_float(latest.get('eval_success_once'))
        latest_success_at_end = maybe_float(latest.get('eval_success_at_end'))
        best_success_once = maybe_float(best.get('eval_success_once'))
        best_success_at_end = maybe_float(best.get('eval_success_at_end'))
        updates = len(rows)
    elif isinstance(metrics, dict):
        latest_success_once = maybe_float(metrics.get('success_once'))
        latest_success_at_end = maybe_float(metrics.get('success_at_end'))
        best_success_once = latest_success_once
        best_success_at_end = latest_success_at_end
        updates = 0
    else:
        latest_success_once = None
        latest_success_at_end = None
        best_success_once = None
        best_success_at_end = None
        updates = 0

    return {
        'method': entry.get('method', entry.get('family', 'unknown')),
        'status': entry.get('status', 'unknown'),
        'run_dir': str(run_dir),
        'log_file': entry.get('log_file', ''),
        'updates': updates,
        'latest_success_once': latest_success_once,
        'latest_success_at_end': latest_success_at_end,
        'best_success_once': best_success_once,
        'best_success_at_end': best_success_at_end,
    }


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print('')
    print('Multi-agent discussion summary')
    print('method                      status      latest_once  latest_end   best_once    best_end     updates')
    for row in rows:
        print(
            f"{row['method']:<28}"
            f"{row['status']:<12}"
            f"{format_metric(row['latest_success_once']):<13}"
            f"{format_metric(row['latest_success_at_end']):<13}"
            f"{format_metric(row['best_success_once']):<13}"
            f"{format_metric(row['best_success_at_end']):<13}"
            f"{row['updates']}"
        )
    print('')
    for row in rows:
        print(f"[{row['method']}] run_dir={row['run_dir']}")
        print(f"[{row['method']}] log_file={row['log_file']}")
    mappo = next((row for row in rows if row['method'] == 'mappo'), None)
    ours = next((row for row in rows if row['method'] == 'ours'), None)
    if mappo and ours:
        once_gain = None
        end_gain = None
        if mappo['latest_success_once'] is not None and ours['latest_success_once'] is not None:
            once_gain = ours['latest_success_once'] - mappo['latest_success_once']
        if mappo['latest_success_at_end'] is not None and ours['latest_success_at_end'] is not None:
            end_gain = ours['latest_success_at_end'] - mappo['latest_success_at_end']
        print('')
        print('[comparison] ours vs mappo')
        print(f"[comparison] latest_success_once_gain={format_metric(once_gain)}")
        print(f"[comparison] latest_success_at_end_gain={format_metric(end_gain)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    manifest = load_json(Path(args.manifest))
    if not isinstance(manifest, dict):
        raise SystemExit(f'Invalid manifest: {args.manifest}')
    runs = manifest.get('runs')
    if not isinstance(runs, list) or not runs:
        raise SystemExit(f'No runs found in manifest: {args.manifest}')
    rows = [summarize_run(entry) for entry in runs]
    print_summary(rows)


if __name__ == '__main__':
    main()
