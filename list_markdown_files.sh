#!/usr/bin/env bash

set -euo pipefail

# Resolve the project root from the script location so the script is cwd-independent.
script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
temp_file="$(mktemp "${TMPDIR:-/tmp}/vlaselect-markdown-files.XXXXXX")"

# A file directly below the project root has find-depth 2; deeper files are
# included as well, while markdown files in the root itself are excluded.
(
    cd -- "$script_dir"
    find . -mindepth 2 -type f -name '*.md' -print
) | LC_ALL=C sort > "$temp_file"

printf '%s\n' "$temp_file"
