#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$EVAL_ROOT"
source "${EVAL_ROOT}/common/interrupt_cleanup.sh"

SUITE_STAMP="${SUITE_STAMP:-$(date -u +"%Y%m%d-%H%M%S")}"
TABLE_ROOT="ablation/ablation_table"
RUN_ROOT="${TABLE_ROOT}/${SUITE_STAMP}"
MANIFEST_JSON="${RUN_ROOT}/manifest.json"
LATEST_POINTER="${TABLE_ROOT}/latest.txt"
LAUNCH_LOG_DIR="${RUN_ROOT}/launch_logs"

RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-0}"
SMOKE="${SMOKE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TAIL_LOG="${TAIL_LOG:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
ABLATION_SELECTION="${ABLATION_SELECTION:-}"
LIST_ONLY="${LIST_ONLY:-0}"
MODEL_SELECTION="${MODEL_SELECTION:-}"
MODEL_CKPT_PATH="${MODEL_CKPT_PATH:-${CHECKPOINT_PATH:-}}"
MWE="${MWE:-0}"

: "${MWE_RUNTIME_LIMIT_SECONDS:=300}"
export MWE_RUNTIME_LIMIT_SECONDS
if [[ "$MWE" == "1" && "${MWE_TIMEOUT_APPLIED:-0}" != "1" ]]; then
    if command -v timeout >/dev/null 2>&1; then
        export MWE_TIMEOUT_APPLIED=1
        exec timeout --preserve-status -k 10s "${MWE_RUNTIME_LIMIT_SECONDS}s" bash "$SCRIPT_PATH" "$@"
    fi
    echo "[warn] timeout command not found; MWE runtime is not hard-capped" >&2
fi
if [[ "$MWE" == "1" ]]; then
    RUN_EXPERIMENTS="${RUN_EXPERIMENTS:-1}"
    SMOKE="${SMOKE:-1}"
    ABLATION_SELECTION="${ABLATION_SELECTION:-scaling_law_function:with_scaling_law,scaling_law_function:without_scaling_law}"
fi
vlaselect_install_cleanup_trap

BASE_ENV_ID="${BASE_ENV_ID:-PickCubeObjectScaleUp1p2-v1}"
BASE_ENVS_ID="${BASE_ENVS_ID:-['PickCubeObjectScaleUp1p2-v1','PickCubeLightStronger50-v1','PickCubeObjectScaleUp1p4-v1','PickCubeLightWeaker50-v1','PushCubeLightWeaker50-v1','PushCubeLightStronger50-v1','PushCubeColorTempHigher50-v1','PushCubeColorTempLower50-v1','PickCubeColorTempHigher50-v1','PickCubeObjectScaleDown1p2-v1']}"
BASE_ENV_CHANGE_TIME_POINTS="${BASE_ENV_CHANGE_TIME_POINTS:-[31,62,96,131,151,163,207,247,271,300]}"

DEFAULT_ENV_CONFIG_PATH="${DEFAULT_ENV_CONFIG_PATH:-datasets/PickCube-v1/motionplanning/trajectory.rgb+depth+state_dict.pd_ee_delta_pos.physx_cpu.json}"
DEFAULT_STATE_NORM_STATS_PATH="${DEFAULT_STATE_NORM_STATS_PATH:-ckpt/PickCube-v1/ours/octo/PickCube-v1-state-max-min.pth}"
DEFAULT_CHECKPOINT_PATH="${DEFAULT_CHECKPOINT_PATH:-ckpt/PickCube-v1/ours/octo/pretrain_large_model_ppo/20260201-183518-lr3e-4/checkpoints/best_success_once-copy.pt}"
FALLBACK_ENV_CONFIG_PATH="${RUN_ROOT}/fallback_env_config.json"
FALLBACK_STATE_NORM_STATS_PATH="${RUN_ROOT}/fallback_state_norm_stats.pth"
MISSING_CHECKPOINT_PATH="${RUN_ROOT}/missing_pretrained_checkpoint.pt"

mkdir -p "$RUN_ROOT" "$LAUNCH_LOG_DIR"
printf "%s\n" "$SUITE_STAMP" > "$LATEST_POINTER"

ensure_env_config() {
    if [[ -f "$DEFAULT_ENV_CONFIG_PATH" ]]; then
        printf "%s" "$DEFAULT_ENV_CONFIG_PATH"
        return
    fi
    mkdir -p "$(dirname "$FALLBACK_ENV_CONFIG_PATH")"
    cat > "$FALLBACK_ENV_CONFIG_PATH" <<'JSON'
{
  "env_info": {
    "env_kwargs": {
      "obs_mode": "rgb+depth+state_dict",
      "control_mode": "pd_joint_delta_pos",
      "reward_mode": "normalized_dense",
      "render_mode": "all",
      "num_envs": 1
    }
  }
}
JSON
    printf "%s" "$FALLBACK_ENV_CONFIG_PATH"
}

ensure_state_norm_stats() {
    if [[ -f "$DEFAULT_STATE_NORM_STATS_PATH" ]]; then
        printf "%s" "$DEFAULT_STATE_NORM_STATS_PATH"
        return
    fi
    "$PYTHON_BIN" - <<'PY' "$FALLBACK_STATE_NORM_STATS_PATH"
import sys
from pathlib import Path

import torch

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
state_max = torch.ones(42, dtype=torch.float32)
state_min = torch.zeros(42, dtype=torch.float32)
torch.save((state_max, state_min), path)
PY
    printf "%s" "$FALLBACK_STATE_NORM_STATS_PATH"
}

resolve_checkpoint_path() {
    if [[ -n "$MODEL_CKPT_PATH" ]]; then
        if [[ -f "$MODEL_CKPT_PATH" ]]; then
            printf "%s" "$MODEL_CKPT_PATH"
            return
        fi
        printf "%s" "$MISSING_CHECKPOINT_PATH"
        return
    fi
    if [[ -f "$DEFAULT_CHECKPOINT_PATH" ]]; then
        printf "%s" "$DEFAULT_CHECKPOINT_PATH"
        return
    fi
    printf "%s" "$MISSING_CHECKPOINT_PATH"
}

ENV_CONFIG_PATH="$(ensure_env_config)"
STATE_NORM_STATS_PATH="$(ensure_state_norm_stats)"
CHECKPOINT_PATH="$(resolve_checkpoint_path)"

list_curve_keys() {
    cat <<'KEYS'
scaling_law_function:with_scaling_law
scaling_law_function:without_scaling_law
neuron_grained_scaling_up:random
neuron_grained_scaling_up:inverse
neuron_grained_scaling_up:neuron_grained
scaling_down_freezing_vs_pruning:pruning
scaling_down_freezing_vs_pruning:freezing
neuron_swapping:with_swapping
neuron_swapping:without_swapping
knowledge_accumulation:selective_accumulation
knowledge_accumulation:no_accumulation
knowledge_accumulation:accumulate_every_rollout
KEYS
}

should_run_curve() {
    local key="$1"
    if [[ -z "$ABLATION_SELECTION" ]]; then
        return 0
    fi

    while IFS= read -r item; do
        item="${item#"${item%%[![:space:]]*}"}"
        item="${item%"${item##*[![:space:]]}"}"
        [[ -z "$item" ]] && continue
        if [[ "$item" == "$key" ]]; then
            return 0
        fi
    done < <(printf "%s" "$ABLATION_SELECTION" | tr ',' '\n')

    return 1
}

write_manifest() {
    "$PYTHON_BIN" - <<'PY' "$MANIFEST_JSON" "$SUITE_STAMP" "$CHECKPOINT_PATH"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
suite_stamp = sys.argv[2]
checkpoint_path = sys.argv[3]

panels = [
    {
        "panel_label": "a",
        "panel_id": "scaling_law_function",
        "title": "(a) Scaling law function",
        "workload_name": "Single-arm robot",
        "curves": [
            {
                "curve_id": "with_scaling_law",
                "label": "With scaling law",
                "color": "#C44E52",
                "linestyle": "-",
                "run_dir": f"ckpt/ablation/{suite_stamp}/scaling_law_function/with_scaling_law/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Default VLASelect continual setting.",
                "changed_options": [],
            },
            {
                "curve_id": "without_scaling_law",
                "label": "Without scaling law",
                "color": "#4D4D4D",
                "linestyle": "--",
                "run_dir": f"ckpt/ablation/{suite_stamp}/scaling_law_function/without_scaling_law/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Target-batch generation without the target-trajectory scaling-law path.",
                "changed_options": ["small_model_generation_strategy"],
            },
        ],
    },
    {
        "panel_label": "b",
        "panel_id": "neuron_grained_scaling_up",
        "title": "(b) Neuron-grained scaling up",
        "workload_name": "Single-arm robot",
        "curves": [
            {
                "curve_id": "random",
                "label": "Random",
                "color": "#4D4D4D",
                "linestyle": "--",
                "run_dir": f"ckpt/ablation/{suite_stamp}/neuron_grained_scaling_up/random/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Random neuron selection.",
                "changed_options": ["small_model_ab_strategy"],
            },
            {
                "curve_id": "inverse",
                "label": "Most accuracy-unrelated",
                "color": "#4C78A8",
                "linestyle": ":",
                "run_dir": f"ckpt/ablation/{suite_stamp}/neuron_grained_scaling_up/inverse/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Keep the least important neurons.",
                "changed_options": ["small_model_ab_strategy"],
            },
            {
                "curve_id": "neuron_grained",
                "label": "Most accuracy-related",
                "color": "#C44E52",
                "linestyle": "-",
                "run_dir": f"ckpt/ablation/{suite_stamp}/neuron_grained_scaling_up/neuron_grained/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Default score-based neuron selection.",
                "changed_options": [],
            },
        ],
    },
    {
        "panel_label": "c",
        "panel_id": "scaling_down_freezing_vs_pruning",
        "title": "(c) Scaling down by freezing vs pruning",
        "workload_name": "Single-arm robot",
        "curves": [
            {
                "curve_id": "pruning",
                "label": "Pruning",
                "color": "#C44E52",
                "linestyle": "-",
                "run_dir": f"ckpt/ablation/{suite_stamp}/scaling_down_freezing_vs_pruning/pruning/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Compressed small model with max_sparsity=0.8.",
                "changed_options": [],
            },
            {
                "curve_id": "freezing",
                "label": "Freezing",
                "color": "#4D4D4D",
                "linestyle": "--",
                "run_dir": f"ckpt/ablation/{suite_stamp}/scaling_down_freezing_vs_pruning/freezing/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "No pruning path, using max_sparsity=0.0 as the large-model frozen-size baseline.",
                "changed_options": ["max_sparsity"],
            },
        ],
    },
    {
        "panel_label": "d",
        "panel_id": "neuron_swapping",
        "title": "(d) Neuron swapping",
        "workload_name": "Single-arm robot",
        "curves": [
            {
                "curve_id": "with_swapping",
                "label": "With swapping",
                "color": "#C44E52",
                "linestyle": "-",
                "run_dir": f"ckpt/ablation/{suite_stamp}/neuron_swapping/with_swapping/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Repeated regeneration with partial channel replacement.",
                "changed_options": [],
            },
            {
                "curve_id": "without_swapping",
                "label": "Without swapping",
                "color": "#4D4D4D",
                "linestyle": "--",
                "run_dir": f"ckpt/ablation/{suite_stamp}/neuron_swapping/without_swapping/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "One-shot small model without iterative swapping.",
                "changed_options": ["small_model_regeneration_schedule"],
            },
        ],
    },
    {
        "panel_label": "e",
        "panel_id": "knowledge_accumulation",
        "title": "(e) Knowledge accumulation",
        "workload_name": "Single-arm robot",
        "curves": [
            {
                "curve_id": "no_accumulation",
                "label": "No accumulation",
                "color": "#4D4D4D",
                "linestyle": "--",
                "run_dir": f"ckpt/ablation/{suite_stamp}/knowledge_accumulation/no_accumulation/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Disable knowledge accumulation by using feedback_alpha=0.",
                "changed_options": ["small_model_feedback_alpha"],
            },
            {
                "curve_id": "accumulate_every_rollout",
                "label": "Every-rollout accumulation",
                "color": "#4C78A8",
                "linestyle": ":",
                "run_dir": f"ckpt/ablation/{suite_stamp}/knowledge_accumulation/accumulate_every_rollout/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Accumulate knowledge before every rollout.",
                "changed_options": ["small_model_feedback_schedule"],
            },
            {
                "curve_id": "selective_accumulation",
                "label": "Selective accumulation",
                "color": "#C44E52",
                "linestyle": "-",
                "run_dir": f"ckpt/ablation/{suite_stamp}/knowledge_accumulation/selective_accumulation/[agent]",
                "metric_source": "tensorboard",
                "metric_key": "eval/success_end",
                "notes": "Feedback only after significant success improvement.",
                "changed_options": [],
            },
        ],
    },
]

payload = {
    "suite_stamp": suite_stamp,
    "table_root": "ablation/ablation_table",
    "figure_output": "ablation/FIG_ABLATION.pdf",
    "summary_csv": "ablation/ablation_summary.csv",
    "checkpoint_path": checkpoint_path,
    "notes": [
        "The underlying small-model generation, channel inheritance, optimizer remapping, and feedback logic come from eval/ours.",
        "Each ablation curve records the intended changed_options so the comparison is easier to audit.",
        "If a run directory has no metrics yet, plot_ablation.py will emit 0 rows in ablation_summary.csv and keep a placeholder bar in the vis-style figure.",
    ],
    "panels": panels,
}

manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

launch_curve() {
    local panel_id="$1"
    local curve_id="$2"
    local exp_name="ablation/${SUITE_STAMP}/${panel_id}/${curve_id}"
    local log_file="${LAUNCH_LOG_DIR}/${panel_id}__${curve_id}.log"
    local run_dir="ckpt/ablation/${SUITE_STAMP}/${panel_id}/${curve_id}/[agent]"
    local run_manifest="${run_dir}/manifest.json"

    vlaselect_register_cleanup_manifest "$run_manifest"

    local total_timesteps="100000000"
    local learning_rate="3e-5"
    local num_envs="256"
    local num_eval_envs="32"
    local num_minibatches="16"
    local update_epochs="2"
    local max_time="301"
    local num_steps="50"
    local num_eval_steps="50"

    if [[ "$SMOKE" == "1" ]]; then
        total_timesteps="4096"
        num_envs="8"
        num_eval_envs="2"
        num_minibatches="2"
        update_epochs="1"
        max_time="1"
        num_steps="8"
        num_eval_steps="8"
    fi

    local -a cmd=(
        "$PYTHON_BIN" -u -m train.octo.ours_single_agent.online_rl_cl
        --exp-name "$exp_name"
        --env-id "$BASE_ENV_ID"
        --envs-id "$BASE_ENVS_ID"
        --env-change-time-points "$BASE_ENV_CHANGE_TIME_POINTS"
        --env_config_path "$ENV_CONFIG_PATH"
        --state-norm-stats-path "$STATE_NORM_STATS_PATH"
        --checkpoint "$CHECKPOINT_PATH"
        --total_timesteps "$total_timesteps"
        --learning_rate "$learning_rate"
        --eval_freq 1
        --num_envs "$num_envs"
        --num_eval_envs "$num_eval_envs"
        --num_steps "$num_steps"
        --num_eval_steps "$num_eval_steps"
        --num_minibatches "$num_minibatches"
        --update_epochs "$update_epochs"
        --max_time "$max_time"
        --small_model_generation_policy small
        --tag "${panel_id}-${curve_id}"
    )

    case "${panel_id}:${curve_id}" in
        scaling_law_function:with_scaling_law)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        scaling_law_function:without_scaling_law)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-batch
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        neuron_grained_scaling_up:random)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_ab_strategy random
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        neuron_grained_scaling_up:inverse)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_ab_strategy inverse
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        neuron_grained_scaling_up:neuron_grained)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        scaling_down_freezing_vs_pruning:pruning)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        scaling_down_freezing_vs_pruning:freezing)
            cmd+=(
                --max-sparsity 0.0
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        neuron_swapping:with_swapping)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        neuron_swapping:without_swapping)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule once
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        knowledge_accumulation:selective_accumulation)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout_if_success_improv_is_larger_than_0.2
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        knowledge_accumulation:no_accumulation)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule once
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.0
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        knowledge_accumulation:accumulate_every_rollout)
            cmd+=(
                --max-sparsity 0.8
                --small_model_generation_strategy target-single-traj
                --small_model_feedback_schedule before_per_rollout
                --small_model_regeneration_schedule before_per_rollout_if_success_improv_less_than_0.1_for_4_iters
                --small_model_feedback_alpha 0.1
                --small_model_regeneration_increment_ratio 0.05
                --reset_optimizer_after_regeneration
            )
            ;;
        *)
            echo "Unknown ablation curve: ${panel_id}:${curve_id}" >&2
            return 1
            ;;
    esac

    echo "[ablation] launching ${panel_id}/${curve_id}"
    if [[ "$TAIL_LOG" == "1" ]]; then
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" 2>&1 | tee "$log_file"
    else
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${cmd[@]}" > "$log_file" 2>&1
    fi
}

maybe_launch_curve() {
    local panel_id="$1"
    local curve_id="$2"
    local key="${panel_id}:${curve_id}"
    if should_run_curve "$key"; then
        launch_curve "$panel_id" "$curve_id"
    fi
}

write_manifest

if [[ "$LIST_ONLY" == "1" ]]; then
    list_curve_keys
    exit 0
fi

if [[ "$RUN_EXPERIMENTS" == "1" ]]; then
    maybe_launch_curve scaling_law_function with_scaling_law
    maybe_launch_curve scaling_law_function without_scaling_law
    maybe_launch_curve neuron_grained_scaling_up random
    maybe_launch_curve neuron_grained_scaling_up inverse
    maybe_launch_curve neuron_grained_scaling_up neuron_grained
    maybe_launch_curve scaling_down_freezing_vs_pruning pruning
    maybe_launch_curve scaling_down_freezing_vs_pruning freezing
    maybe_launch_curve neuron_swapping with_swapping
    maybe_launch_curve neuron_swapping without_swapping
    maybe_launch_curve knowledge_accumulation no_accumulation
    maybe_launch_curve knowledge_accumulation accumulate_every_rollout
    maybe_launch_curve knowledge_accumulation selective_accumulation
fi

echo "Manifest written to ${MANIFEST_JSON}"
echo "Results root: ${RUN_ROOT}"
echo "Run plot via: cd ${SCRIPT_DIR} && ${PYTHON_BIN} plot_ablation.py"
