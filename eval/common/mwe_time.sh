#!/usr/bin/env bash

vlaselect_convert_mwe_schedule_seconds_to_minutes() {
    local raw_time_points="${1:-}"

    if [[ -z "$raw_time_points" ]]; then
        echo "[mwe-time] missing time points input" >&2
        return 1
    fi

    python3 - "$raw_time_points" <<'PY_HELPER'
import ast
import math
import sys

try:
    time_points = list(ast.literal_eval(sys.argv[1]))
except (SyntaxError, ValueError) as exc:
    raise SystemExit(f"failed to parse time point literal: {exc}")

converted = []
for value in time_points:
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise SystemExit(f"time point must be finite, got {value!r}")
    converted.append(numeric_value / 60.0)

print(repr(converted))
PY_HELPER
}
