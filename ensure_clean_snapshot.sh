#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash ensure_clean_snapshot.sh [--force]

Remove existing experiment run results while preserving source files,
pretrained checkpoints, and datasets. The default is a dry run.

The script removes only run directories identified by result markers
(metrics_history.json, final_eval_metrics.json, latest_metrics.json, or
time_breakdown.json), plus generated logs, W&B/TensorBoard files, plots, and
Python bytecode caches. It never deletes a checkpoint directory merely because
it is named ckpt/.

Options:
  --force     Delete the listed generated run results and caches.
  -h, --help  Show this help text.

Examples:
  bash ensure_clean_snapshot.sh
  bash ensure_clean_snapshot.sh --force
EOF
}

force=0
for argument in "$@"; do
    case "$argument" in
        --force) force=1 ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null)" || {
    echo "This script must be run from inside a Git worktree." >&2
    exit 2
}

cd "$repo_root"
declare -a candidates=()
declare -A seen=()

add_candidate() {
    local path="$1"
    [[ -e "$path" || -L "$path" ]] || return 0
    [[ -z "${seen[$path]+x}" ]] || return 0
    seen["$path"]=1
    candidates+=("$path")
}

add_result_run_dir() {
    local marker_path="$1"
    local marker_dir
    local run_dir

    marker_dir="$(dirname "$marker_path")"
    if [[ "$(basename "$marker_dir")" == "[agent]" ]]; then
        run_dir="$(dirname "$marker_dir")"
    else
        run_dir="$marker_dir"
    fi
    case "/$run_dir/" in
        */pretrain_*/|*/pretrained_*/|*/foundation_*/)
            echo "[clean-snapshot] preserving pretrained asset: ${run_dir#"$repo_root"/}" >&2
            return 0
            ;;
    esac
    add_candidate "$run_dir"
}

# A run directory must contain at least one of these metrics artifacts. This
# deliberately excludes pretrained checkpoint directories without run metrics.
while IFS= read -r -d '' marker_path; do
    add_result_run_dir "$marker_path"
done < <(
    find "$repo_root/eval/ckpt" "$repo_root/api" \
        -type f \( \
            -name metrics_history.json -o \
            -name final_eval_metrics.json -o \
            -name latest_metrics.json -o \
            -name time_breakdown.json \
        \) -print0 2>/dev/null
)

# Generated artifacts outside the per-run checkpoint directories.
while IFS= read -r -d '' path; do
    add_candidate "$path"
done < <(
    find "$repo_root/eval" "$repo_root/api" \
        -type d \( \
            -name nohup_out -o \
            -name launch_logs -o \
            -name __pycache__ -o \
            -name '*_table' -o \
            -path '*/discussion/results' \
        \) -prune -print0 2>/dev/null
)

while IFS= read -r -d '' path; do
    add_candidate "$path"
done < <(
    find "$repo_root/eval" "$repo_root/api" \
        -type f \( \
            -name '*.pyc' -o \
            -name '*.pyo' -o \
            -name 'events.out.tfevents.*' \
        \) -print0 2>/dev/null
)


if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "No generated experiment results or caches found."
    exit 0
fi

printf '%s\n' "Generated experiment results and caches:"
for path in "${candidates[@]}"; do
    printf '  %s\n' "${path#"$repo_root"/}"
done

if [[ "$force" != "1" ]]; then
    echo "Dry run only. Re-run with --force to delete these paths."
    exit 0
fi

for path in "${candidates[@]}"; do
    rm -rf -- "$path"
done

# Remove only now-empty parents created by deleted run directories.
find "$repo_root/eval/ckpt" -depth -type d -empty -delete 2>/dev/null || true
echo "Removed ${#candidates[@]} generated experiment result/cache path(s)."
