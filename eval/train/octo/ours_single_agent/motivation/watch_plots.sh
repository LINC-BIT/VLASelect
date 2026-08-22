#!/usr/bin/env bash

set -euo pipefail

python -u -m train.octo.ours_single_agent.motivation.plot_accuracy --watch --interval-seconds 60 &
ACC_PID=$!

python -u -m train.octo.ours_single_agent.motivation.visualize_neuron_importance --watch --interval-seconds 60 &
IMP_PID=$!

trap 'kill $ACC_PID $IMP_PID 2>/dev/null || true' EXIT

wait $ACC_PID $IMP_PID
