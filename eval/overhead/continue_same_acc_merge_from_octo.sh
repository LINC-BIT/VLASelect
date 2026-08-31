#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EVAL_ROOT"

OLD_SUITE_STAMP="${OLD_SUITE_STAMP:-}"
NEW_SUITE_STAMP="${NEW_SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
METHODS="${METHODS:-self_improv,vla_rft,world_env,vlaselect}"
FAMILY_SELECTION="${FAMILY_SELECTION:-vla_adapter_new,tinyvla,edgevla}"
MWE="${MWE:-1}"
RENDER_PLOTS="${RENDER_PLOTS:-0}"
SAME_ACC_ACCURACY_COMPAT="${SAME_ACC_ACCURACY_COMPAT:-1}"
SAME_ACC_BREAKDOWN_COMPAT="${SAME_ACC_BREAKDOWN_COMPAT:-1}"

if [[ -z "$OLD_SUITE_STAMP" ]]; then
    echo "OLD_SUITE_STAMP is required." >&2
    echo "Example: OLD_SUITE_STAMP=20260825-061524 bash continue_same_acc_merge_from_octo.sh" >&2
    exit 1
fi

TOP_TABLE_ROOT="overhead/overhead_same_acc_table"
ACC_TABLE_ROOT="acc_comparison/acc_task_env_from_same_acc_table"

OLD_TOP_MANIFEST="${TOP_TABLE_ROOT}/${OLD_SUITE_STAMP}/manifest.json"
NEW_TOP_ROOT="${TOP_TABLE_ROOT}/${NEW_SUITE_STAMP}"
NEW_TOP_MANIFEST="${NEW_TOP_ROOT}/manifest.json"
NEW_TOP_JSONL="${NEW_TOP_ROOT}/panels.jsonl"

NEW_ACC_ROOT="${ACC_TABLE_ROOT}/${NEW_SUITE_STAMP}"
NEW_ACC_MANIFEST="${NEW_ACC_ROOT}/manifest.json"
NEW_ACC_JSONL="${NEW_ACC_ROOT}/panels.jsonl"

if [[ ! -f "$OLD_TOP_MANIFEST" ]]; then
    echo "Old manifest not found: $OLD_TOP_MANIFEST" >&2
    exit 1
fi

echo "[continue-merge] old octo suite: $OLD_SUITE_STAMP"
echo "[continue-merge] new merged suite: $NEW_SUITE_STAMP"

env \
    SUITE_STAMP="$NEW_SUITE_STAMP" \
    MWE="$MWE" \
    FAMILY_SELECTION="$FAMILY_SELECTION" \
    METHODS="$METHODS" \
    SAME_ACC_ACCURACY_COMPAT="$SAME_ACC_ACCURACY_COMPAT" \
    SAME_ACC_BREAKDOWN_COMPAT="$SAME_ACC_BREAKDOWN_COMPAT" \
    bash overhead/overhead_same_acc.sh

python - <<'PY' \
    "$OLD_TOP_MANIFEST" \
    "$NEW_TOP_MANIFEST" \
    "$NEW_TOP_JSONL" \
    "$NEW_ACC_MANIFEST" \
    "$NEW_ACC_JSONL" \
    "$NEW_SUITE_STAMP"
from __future__ import annotations

import json
import sys
from pathlib import Path

old_top_manifest = Path(sys.argv[1])
new_top_manifest = Path(sys.argv[2])
new_top_jsonl = Path(sys.argv[3])
new_acc_manifest = Path(sys.argv[4])
new_acc_jsonl = Path(sys.argv[5])
new_suite_stamp = sys.argv[6]

family_order = ["octo", "vla_adapter_new", "tinyvla", "edgevla"]


def load_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid manifest payload: {path}")
    return payload


def index_panels(payload: dict) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in payload.get("panels", payload.get("families", [])):
        if isinstance(item, dict) and item.get("family"):
            result[str(item["family"])] = item
    return result


old_top = load_payload(old_top_manifest)
new_top = load_payload(new_top_manifest)
old_panels = index_panels(old_top)
new_panels = index_panels(new_top)

merged_panels: list[dict] = []
for family in family_order:
    if family == "octo":
        panel = old_panels.get(family)
    else:
        panel = new_panels.get(family)
    if panel is None:
        raise SystemExit(f"Missing panel for family={family}")
    merged_panels.append(dict(panel))

new_top["suite_stamp"] = new_suite_stamp
new_top["panels"] = merged_panels
new_top["families"] = merged_panels
new_top_manifest.write_text(json.dumps(new_top, indent=2), encoding="utf-8")

new_top_jsonl.parent.mkdir(parents=True, exist_ok=True)
with new_top_jsonl.open("w", encoding="utf-8") as handle:
    for panel in merged_panels:
        handle.write(json.dumps(panel, ensure_ascii=True) + "\n")

acc_payload = {
    "suite_stamp": new_suite_stamp,
    "table_root": "acc_comparison/acc_task_env_from_same_acc_table",
    "figure_output": "acc_comparison/FIG_ACC_TASK_ENV_FROM_SAME_ACC.pdf",
    "panels": merged_panels,
    "families": merged_panels,
}
new_acc_manifest.parent.mkdir(parents=True, exist_ok=True)
new_acc_manifest.write_text(json.dumps(acc_payload, indent=2), encoding="utf-8")
with new_acc_jsonl.open("w", encoding="utf-8") as handle:
    for panel in merged_panels:
        handle.write(json.dumps(panel, ensure_ascii=True) + "\n")
PY

printf '%s\n' "$NEW_SUITE_STAMP" > "${TOP_TABLE_ROOT}/latest.txt"
printf '%s\n' "$NEW_SUITE_STAMP" > "${ACC_TABLE_ROOT}/latest.txt"

python overhead/plot_breakdown_impl.py --manifest "$NEW_TOP_MANIFEST" --prepare-only

echo "[continue-merge] merged top manifest: ${EVAL_ROOT}/${NEW_TOP_MANIFEST}"
echo "[continue-merge] merged accuracy manifest: ${EVAL_ROOT}/${NEW_ACC_MANIFEST}"
echo "[continue-merge] latest top pointer: ${EVAL_ROOT}/${TOP_TABLE_ROOT}/latest.txt"
echo "[continue-merge] latest accuracy pointer: ${EVAL_ROOT}/${ACC_TABLE_ROOT}/latest.txt"
echo "[continue-merge] merged breakdown csv: ${EVAL_ROOT}/${NEW_TOP_ROOT}/BREAKDOWN_ALL_METHODS.csv"
echo "[continue-merge] merged breakdown csv: ${EVAL_ROOT}/${NEW_TOP_ROOT}/BREAKDOWN_MODULES.csv"

if [[ "$RENDER_PLOTS" == "1" ]]; then
    python overhead/plot_overhead.py
    env \
        PLOT_ACC_TABLE_ROOT="$ACC_TABLE_ROOT" \
        PLOT_ACC_FIGURE_STEM="FIG_ACC_TASK_ENV_FROM_SAME_ACC" \
        PLOT_ACC_SUMMARY_STEM="acc_task_env_from_same_acc_summary" \
        PLOT_ACC_PANEL_DIR="FIG_ACC_TASK_ENV_FROM_SAME_ACC_panels" \
        PLOT_ACC_VIS_PAYLOAD_DIR="vis_payload_task_env_from_same_acc" \
        python acc_comparison/plot_acc_task_env.py
    echo "[continue-merge] rendered figure: ${EVAL_ROOT}/overhead/FIG_MEMORY_FOOTPOINT.pdf"
    echo "[continue-merge] rendered figure: ${EVAL_ROOT}/acc_comparison/FIG_ACC_TASK_ENV_FROM_SAME_ACC.pdf"
fi
