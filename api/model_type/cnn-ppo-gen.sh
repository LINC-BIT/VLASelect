#!/usr/bin/env bash
set -euo pipefail

# PPO-Gen is called ConRFT in the comparison labels. Keep the user-facing
# PPO-Gen script name while sharing the single implementation.
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/cnn-conrft.sh" "$@"
