#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash ensure_clean_snapshot.sh [--force] [--check]

Remove ignored experiment outputs and caches while preserving source files,
pretrained assets, and datasets. The default is a dry run.

Options:
  --force  Delete the listed ignored paths.
  --check  Fail unless the workspace has no Git changes after cleanup.
  -h, --help  Show this help text.

Examples:
  bash ensure_clean_snapshot.sh
  bash ensure_clean_snapshot.sh --force
  bash ensure_clean_snapshot.sh --force --check
EOF
}

force=0
check_clean=0
for argument in "$@"; do
    case "$argument" in
        --force) force=1 ;;
        --check) check_clean=1 ;;
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

add_candidate() {
    local absolute_path="$1"
    local git_root
    local git_relative_path
    local relative_path="${absolute_path#"$repo_root"/}"

    [[ -e "$absolute_path" || -L "$absolute_path" ]] || return 0
    git_root="$(git -C "$(dirname "$absolute_path")" rev-parse --show-toplevel 2>/dev/null)" || return 0
    git_relative_path="${absolute_path#"$git_root"/}"
    if [[ -d "$absolute_path" ]]; then
        git_relative_path="${git_relative_path%/}/"
    fi
    git -C "$git_root" check-ignore -q -- "$git_relative_path" || return 0
    candidates+=("$relative_path")
}

# Fixed experiment-output roots. Do not include datasets or root-level ckpt/:
# they can contain downloaded inputs and pretrained checkpoints.
for relative_path in \
    "eval/ckpt" \
    "eval/wandb" \
    "eval/discussion/results" \
    "api/results" \
    "wandb"; do
    add_candidate "$repo_root/$relative_path"
done

# Remove nested experiment outputs without touching unignored source directories.
while IFS= read -r -d '' absolute_path; do
    add_candidate "$absolute_path"
done < <(
    find "$repo_root/eval" "$repo_root/api" \
        -type d \( \
            -name '__pycache__' -o \
            -name 'nohup_out' -o \
            -name 'launch_logs' -o \
            -name 'cl_suite' -o \
            -name 'outputs' -o \
            -name 'results' -o \
            -name '*_table' \
        \) -prune -print0
)

while IFS= read -r -d '' absolute_path; do
    add_candidate "$absolute_path"
done < <(
    find "$repo_root/eval" "$repo_root/api" \
        -type d \( \
            -name '__pycache__' -o \
            -name 'nohup_out' -o \
            -name 'launch_logs' -o \
            -name 'cl_suite' -o \
            -name 'outputs' -o \
            -name 'results' -o \
            -name '*_table' \
        \) -prune -o \
        -type f \( \
            -name '*.pyc' -o \
            -name '*.pyo' -o \
            -name '*.log' -o \
            -name 'events.out.tfevents.*' \
        \) -print0
)

if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "No ignored run outputs or caches found."
else
    printf '%s\n' "Ignored run outputs and caches:"
    printf '  %s\n' "${candidates[@]}"
    if [[ "$force" != "1" ]]; then
        echo "Dry run only. Re-run with --force to delete these paths."
    else
        for relative_path in "${candidates[@]}"; do
            rm -rf -- "$repo_root/$relative_path"
        done
        echo "Removed ${#candidates[@]} ignored run-output/cache path(s)."
    fi
fi

if [[ "$check_clean" == "1" ]]; then
    if [[ "$force" != "1" ]]; then
        echo "--check requires --force so the check is performed after cleanup." >&2
        exit 2
    fi
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Workspace is not a clean source snapshot: source changes or untracked files remain." >&2
        echo "These are intentionally preserved; inspect them with: git status --short" >&2
        exit 1
    fi
    echo "Workspace is a clean source snapshot."
fi
