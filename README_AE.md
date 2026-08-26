# VLASelect: Minimal Reproduction for Artifact Evaluation
This artifact accompanies the paper **"VLASelect: Selective Large-small Model Co-learning for Self-evolving VLA Agents."** This document is for artifact evaluators who want to check the paper-reproduction with the minimum working examples (MWE).

## Requirements

The MWE scripts typically require:

- Linux with Docker;
- an NVIDIA GPU with more than 20 GB of VRAM;

- at least 80 GB of free disk space.

The sim-to-real check is optional: it requires a DOFBOT-SE arm and an AmazingHand dexterous hand.## 



## Scope

This evaluation covers only the primary experiments that reproduce the paper results:

- accuracy under task/environment and resource changes;
- runtime, memory, and energy overheads;
- time breakdowns;
- ablation studies; and
- discussion experiments.

The model-integration and extensibility examples from Section 3 of the original README are out of scope. All commands below set `MWE=1`; full or regular validation runs are intentionally omitted.

## Get the source code



``` bash
git clone https://github.com/LINC-BIT/VLASelect.git
cd VLASelect
```

## Setup

Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) first!!! then run the following commands:

```bash
bash dep.sh # duration depends on your network condition
bash start_docker.sh # start docker containor
```

> Beformentioned commands need to download ~80GB of docker images, checkpoints, and datasets.

GPU avilability check Inside the container:
```bash
python -c "import torch; print(torch.cuda.is_available())"

# expect outputs
# torch.cuda.is_available() with True
```

If Docker cannot be used, we also provides `dep-non-docker.sh`allow you to run the code on your host devices.

## Run the Minimal Evaluation

### One click test

Run six MWE groups and generate their figures/tables:

```bash
cd eval

MWE=1 bash run.sh
```


For a lighter comparison using three representative baselines:

```bash
cd eval

MWE=1 \
METHODS=self_improv,vla_rft,world_env,vlaselect \
bash run.sh
```

The one-click script covers Figures 7-12 and Tables 2-3. This step takes considerable time (about one whole day) even when minimized, so don’t run it if you’re short on time.
## Run Individual Evaluation



### Accuracy

```bash
cd eval/acc_comparison

MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash run_acc_task_env_change.sh
python3 plot_acc_task_env.py

MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash run_acc_res_change.sh
python3 plot_acc_res_change.py
```

| Figure No. & Experiment | Path | Experiment Results |
| --- | --- | --- |
| Figure 7: Accuracy under task change and new environment | `acc_comparison/FIG_ACC_TASK_ENV.pdf` | [Task & Env Accuracy](https://www.google.com/search?q=) |
| Figure 8: Accuracy under resource change | `acc_comparison/FIG_ACC_RESOURCE.pdf` | [Resource Accuracy](https://www.google.com/search?q=) |

### Overhead and Time Breakdown

```bash
cd ../overhead

MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash overhead_same_acc.sh
python3 plot_overhead.py

MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash overhead_breakdown_all_methods.sh
python3 plot_breakdown_all_methods.py

MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash overhead_breakdown_modules.sh
python3 plot_breakdown_modules.py
```

| Figure No. & Experiment | Path | Experiment Results |
| --- | --- | --- |
| Figure 9: Memory footprint | `overhead/FIG_MEMORY_FOOTPOINT.pdf` | [Memory Footprint](https://www.google.com/search?q=) |
| Tables 2-3: Overhead and energy breakdown | `overhead/overhead_breakdown_table/TAB_OVERHEAD.csv`, `overhead/overhead_breakdown_table/TAB_ENERGY.csv` | [Overhead & Energy](https://www.google.com/search?q=) |
| Figure 10: Overhead breakdown across all methods | `overhead/FIG_BREAKDOWN_ALL_METHODS.pdf` | [Methods Overhead](https://www.google.com/search?q=) |
| Figure 11: Overhead breakdown across VLASelect | `overhead/FIG_BREAKDOWN_MODULES.pdf` | [VLASelect Breakdown](https://www.google.com/search?q=) |

### Ablation

```bash
cd ../ablation
MWE=1 bash run_ablation.sh
python3 plot_ablation.py
```

| Figure No. & Experiment | Path | Experiment Results |
| --- | --- | --- |
| Figure 12: Ablation study | `ablation/FIG_ABLATION.pdf` | [Ablation Results](./ablation_results.md) |

### Discussion
> This part are not included in one click run
```bash
cd ../discussion

# ICL comparison
MWE=1 bash compare_icl.sh

# Applicability to one VLA model
MWE=1 MODEL_SELECTION=octo bash run_vla_models.sh

# Maximum supported model size
MWE=1 MODEL_SIZE_LIMIT_FAMILY=tinyvla bash sweep_model_size.sh

# Multi-agent scenario
MWE=1 bash run_multi_agent.sh
```

Generated artifacts:

| Figure No. & Experiment | Path | Experiment Results |
| --- | --- | --- |
| Figure: VLA applicability check | `discussion/FIG_VLA_APPLICABILITY.pdf` | [VLA Applicability](https://www.google.com/search?q=) |
| Discussion: Other discussion results | Terminal outputs | [Discussion Results](https://www.google.com/search?q=) |

> Run  sim-to-real MWE only if the required robot hardware is connected
```bash
cd discussion
MWE=1 bash run_sim_to_real.sh
```


## Expected Cost

Approximate upper bounds reported for an individual MWE group are:

| Experiment | Time | GPU memory |
| --- | ---: | ---: |
| Accuracy, overhead, or time breakdown | 20 minutes | 8-20 GB |
| Ablation | 25 minutes | 8-20 GB |
| Discussion experiment | 5 minutes | 8-20 GB |

Runtime depends on the selected methods, model, GPU, cached assets, and container startup time. Successful completion should produce the files listed above without an uncaught error.

## Full-Run Data and Expected Variance

MWE is only a short pipeline check. If full-run data is already available, evaluators can plot it directly without rerunning the experiments. The plotting scripts use the newest `manifest.json` under these directories:

| Result | Full-run data directory | Plot command |
| --- | --- | --- |
| Accuracy (Figures 7-8) | `eval/acc_comparison/*_table/<run-id>/` | `cd eval/acc_comparison && python3 plot_acc_task_env.py` or `python3 plot_acc_res_change.py` |
| Overhead (Figure 9, Tables 2-3) | `eval/overhead/overhead_same_acc_table/<run-id>/` | `cd eval/overhead && python3 plot_overhead.py` |
| Time breakdown (Figures 10-11) | `eval/overhead/overhead_breakdown_*_table/<run-id>/` | `cd eval/overhead && python3 plot_breakdown_all_methods.py` or `python3 plot_breakdown_modules.py` |
| Ablation (Figure 12) | `eval/ablation/ablation_table/<run-id>/` | `cd eval/ablation && python3 plot_ablation.py` |

The manifest points to the underlying JSON/CSV logs, such as `metrics_history.json`, `gpu_metrics.csv`, and `memory_accounting.json`. Copy the manifest directory together with its referenced run directories when sharing full-run data.

### Supporting Various VLA Models, Scaling Strategies, and Knowledge Exchange Granularities

```bash
## run in the root path of VLASelect
MWE=1 bash <script_path>
```
> example of VLA-Adapter Model Support Verification:  \
> MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh

| Model & Evaluation Metric | Script Path | Experiment Results |
| --- | --- | --- |
| **VLA-Adapter**: Model Support Verification | `api/vla_model_interface_examples/vla_adapter_impl_verify.sh` | [VLA-Adapter Support](./model_support.md) |
| **VLA-Adapter**: Scaling Strategies | `api/vla_model_interface_examples/vla_adapter_impl_verify-all_scaling_methods-only4.sh` | [VLA-Adapter Scaling](./model_support.md) |
| **VLA-Adapter**: Knowledge Exchange Granularities | `api/vla_model_interface_examples/vla_adapter_impl_verify-all_granularities.sh` | [VLA-Adapter Granularity](./model_support.md) |
| **TinyVLA**: Model Support Verification | `api/vla_model_interface_examples/tinyvla_impl_verify.sh` | [TinyVLA Support](./model_support.md) |
| **TinyVLA**: Scaling Strategies | `api/vla_model_interface_examples/tinyvla_impl_verify-all_scaling_methods-only4.sh` | [TinyVLA Scaling](./model_support.md) |
| **TinyVLA**: Knowledge Exchange Granularities | `api/vla_model_interface_examples/tinyvla_impl_verify-all_granularities.sh` | [TinyVLA Granularity](./model_support.md) |
| **EdgeVLA**: Model Support Verification | `api/vla_model_interface_examples/edgevla_impl_verify.sh` | [EdgeVLA Support](./model_support.md) |
| **EdgeVLA**: Scaling Strategies | `api/vla_model_interface_examples/edgevla_impl_verify-all_scaling_methods-only4.sh` | [EdgeVLA Scaling](./model_support.md) |
| **EdgeVLA**: Knowledge Exchange Granularities | `api/vla_model_interface_examples/edgevla_impl_verify-all_granularities.sh` | [EdgeVLA Granularity](./model_support.md) |



## Expected Results and Variance

Exact values may vary with GPU/CPU hardware, CUDA and driver versions, PyTorch versions, and random seeds. For artifact screening, allow approximately:

| Metric | Normal variation |
| --- | ---: |
| Accuracy or success rate | +/-5 percentage points; up to +/-10 points for a short MWE tail |
| Runtime | +/-20% on similar hardware; up to +/-35% on different GPUs |
| GPU memory | +/-10% |
| Energy | +/-25% |

The main checks are successful completion, valid output files, populated plots, and the same qualitative trend or method ordering as the paper. If results differ more, first check the selected manifest, model/checkpoint, method list, and seed.
