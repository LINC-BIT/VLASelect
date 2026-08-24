#!/usr/bin/env bash

vlaselect_apply_env_id_order() {
    local envs_var_name="$1"
    local time_points_var_name="$2"
    local primary_env_var_name="${3:-}"
    local order_spec="${4:-${ENV_ID_ORDER:-}}"

    if [[ -z "$order_spec" ]]; then
        return 0
    fi

    local envs_value="${!envs_var_name:-}"
    local time_points_value="${!time_points_var_name:-}"
    if [[ -z "$envs_value" ]]; then
        echo "[env-order] missing env sequence in ${envs_var_name}" >&2
        return 1
    fi
    if [[ -z "$time_points_value" ]]; then
        echo "[env-order] missing env change time points in ${time_points_var_name}" >&2
        return 1
    fi

    local helper_output
    if ! helper_output=$(python3 - "$envs_value" "$time_points_value" "$order_spec" <<'PY_HELPER'
import ast
import sys

try:
    envs = list(ast.literal_eval(sys.argv[1]))
    time_points = list(ast.literal_eval(sys.argv[2]))
except (SyntaxError, ValueError) as exc:
    raise SystemExit(f"failed to parse env sequence literals: {exc}")

order_chunks = [chunk.strip() for chunk in sys.argv[3].split(',') if chunk.strip()]
if not order_chunks:
    raise SystemExit('ENV_ID_ORDER is empty after parsing')
try:
    order = [int(chunk) for chunk in order_chunks]
except ValueError as exc:
    raise SystemExit(f"ENV_ID_ORDER must be a comma-separated list of integers: {exc}")

if len(envs) != len(time_points):
    raise SystemExit(
        f"env sequence length {len(envs)} does not match env change time points length {len(time_points)}"
    )
expected = list(range(len(envs)))
if sorted(order) != expected:
    raise SystemExit(
        f"ENV_ID_ORDER must be a permutation of 0..{len(envs) - 1}, got {order}"
    )

reordered_envs = [envs[index] for index in order]
print(repr(reordered_envs))
print(repr(time_points))
print(reordered_envs[0] if reordered_envs else '')
PY_HELPER
); then
        return 1
    fi

    local -a helper_lines=()
    mapfile -t helper_lines <<< "$helper_output"
    if [[ "${#helper_lines[@]}" -lt 2 ]]; then
        echo "[env-order] failed to reorder env sequence for ${envs_var_name}" >&2
        return 1
    fi

    printf -v "$envs_var_name" '%s' "${helper_lines[0]}"
    printf -v "$time_points_var_name" '%s' "${helper_lines[1]}"
    if [[ -n "$primary_env_var_name" && "${#helper_lines[@]}" -ge 3 ]]; then
        printf -v "$primary_env_var_name" '%s' "${helper_lines[2]}"
    fi
}
