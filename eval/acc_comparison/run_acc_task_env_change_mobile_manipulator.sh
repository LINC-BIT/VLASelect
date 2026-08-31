#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_SELECTION="${MODEL_SELECTION:-tinyvla}"
ENV_ID_ORDER="${ENV_ID_ORDER:-}"
BASELINE_PRETRAIN_CKPT_NOISE_SCALE="${BASELINE_PRETRAIN_CKPT_NOISE_SCALE:-}"
BASELINE_PRETRAIN_CKPT_NOISE_SEED="${BASELINE_PRETRAIN_CKPT_NOISE_SEED:-}"

if [[ -n "$BASELINE_PRETRAIN_CKPT_NOISE_SCALE" && -z "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE:-}" ]]; then
    export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SCALE="$BASELINE_PRETRAIN_CKPT_NOISE_SCALE"
fi
if [[ -n "$BASELINE_PRETRAIN_CKPT_NOISE_SEED" && -z "${VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED:-}" ]]; then
    export VLASELECT_BASELINE_PRETRAIN_CKPT_NOISE_SEED="$BASELINE_PRETRAIN_CKPT_NOISE_SEED"
fi

ACC_SINGLE_WORKLOAD_SERIAL_GPU="${ACC_SINGLE_WORKLOAD_SERIAL_GPU:-0}"
if [[ -z "${GPU_BY_METHOD_OVERRIDE:-}" ]]; then
    build_serial_gpu_override() {
        local family="$1"
        local raw_methods="${METHODS:-${RUN_METHODS:-}}"
        local -a resolved_methods=()
        local method=""

        if [[ -n "$raw_methods" ]]; then
            while IFS= read -r method; do
                method="${method#${method%%[![:space:]]*}}"
                method="${method%${method##*[![:space:]]}}"
                [[ -z "$method" ]] && continue
                case "$method" in
                    vlaselect)
                        if [[ "$family" == "octo" ]]; then
                            resolved_methods+=("ours_single_agent")
                        else
                            resolved_methods+=("ours")
                        fi
                        ;;
                    ours)
                        if [[ "$family" == "octo" ]]; then
                            resolved_methods+=("ours_single_agent")
                        else
                            resolved_methods+=("ours")
                        fi
                        ;;
                    ours_single_agent)
                        if [[ "$family" == "octo" ]]; then
                            resolved_methods+=("ours_single_agent")
                        else
                            resolved_methods+=("ours")
                        fi
                        ;;
                    *)
                        resolved_methods+=("$method")
                        ;;
                esac
            done < <(printf '%s' "$raw_methods" | tr ',' '
')
        fi

        if [[ "${#resolved_methods[@]}" -eq 0 ]]; then
            case "$family" in
                octo)
                    resolved_methods=(conrft flare improv_vla edgeta convertnet ours_single_agent ppo_gen self_improv vla_rft world_env)
                    ;;
                edgevla|tinyvla)
                    resolved_methods=(ppo_gen ours conrft flare edgeta convertnet improv_vla self_improv vla_rft world_env)
                    ;;
                vla_adapter_new)
                    resolved_methods=(conrft flare improv_vla edgeta convertnet ours ppo_gen self_improv vla_rft world_env)
                    ;;
                *)
                    resolved_methods=(ours)
                    ;;
            esac
        fi

        local out=""
        for method in "${resolved_methods[@]}"; do
            if [[ -n "$out" ]]; then
                out+="," 
            fi
            out+="${method}=${ACC_SINGLE_WORKLOAD_SERIAL_GPU}"
        done
        printf '%s' "$out"
    }

    GPU_BY_METHOD_OVERRIDE="$(build_serial_gpu_override "$MODEL_SELECTION")"
fi

exec env MODEL_SELECTION="$MODEL_SELECTION" ENV_ID_ORDER="$ENV_ID_ORDER" GPU_BY_METHOD_OVERRIDE="$GPU_BY_METHOD_OVERRIDE" bash "$SCRIPT_DIR/run_acc_task_env_change.sh" "$@"
