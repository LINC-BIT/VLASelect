# VLASelect: Minimal Working Examples for Artifact Evaluation
This artifact accompanies the paper **"VLASelect: Selective Large-small Model Co-learning for Self-evolving VLA Agents."** This document is for artifact evaluators who want to check the paper-reproduction with the minimal working examples (MWE).

---

> This guide covers only Minimal Working Examples (MWE). For comprehensive documentation or troubleshooting, please check the main [README](README.md).

## Requirements

### Hardware Requirements

Minimum hardware requirements for running minimal working example:

| Device Type | RAM | CPU (Examples) | Disk | GPU |
| :--- | :--- | :--- | :--- | :--- |
| **Server (CPU-only)** | 16–32 GB | 8-core server CPU (e.g., Intel Xeon E-2388G) | ≥ 80 GB free | — |
| **GPU Server** | 32–64 GB | 12-core server CPU (e.g., Intel Xeon Silver 4310) | ≥ 80 GB free | NVIDIA GPU w/ 8–12 GB VRAM (e.g., RTX 3060) |
| **Desktop / Laptop (CPU-only)** | 16–32 GB | 12–16 core CPU (e.g., Intel Ultra 7 155U / i7-13700) | ≥ 80 GB free | Integrated graphics |
| **Desktop / Laptop (GPU-accelerated)** | 16–32 GB | 16–20 core CPU (e.g., Intel i7-14650HX / i7-14700) | ≥ 80 GB free | NVIDIA GPU w/ ≥ 8 GB VRAM (e.g., RTX 4060) |

### Software Requirements

Minimum software requirements for running minimal working examples:

| Operating System | CUDA | Others |
|---|---|---|
| Ubuntu LTS 20.04+ | CUDA 12.x+ (when using a GPU) | Kernel 5.4+, Docker 29.0+ |
| Windows 10+ | CUDA 12.x+ (when using a GPU) | Docker 29.0+ |
| macOS 14+ | - | - |
| Debian 11+ | CUDA 12.x+ (when using a GPU) | Kernel 5.4+, Docker 29.0+ |
| RHEL 8+ | CUDA 12.x+ (when using a GPU) | Kernel 5.4+, Docker 29.0+ |

---

## Get source code

>You can obtain the source code for artifacts evaluation by the following command. **The code does not perform any malicious or destructive operations**.

``` bash
git clone https://github.com/LINC-BIT/VLASelect.git
cd VLASelect
```

---


## Install Dependencies

**Step 1:** Install [Docker Engine](https://docs.docker.com/engine/install/ubuntu/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
  # Add Docker's official GPG key:
  sudo apt update
  sudo apt install ca-certificates curl
  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  # Add the repository to Apt sources:
  sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
  Types: deb
  URIs: https://download.docker.com/linux/ubuntu
  Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
  Components: stable
  Architectures: $(dpkg --print-architecture)
  Signed-By: /etc/apt/keyrings/docker.asc
  EOF

  sudo apt update

  sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

  sudo systemctl status docker --no-pager
  sudo docker run hello-world
```
[Example running screenshots](install-step-1-example.md)

**Step 2:** Install Docker plugin for using CUDA

  ```bash
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
[Example running screenshots](install-step-2-example.md)

**Step 3:** Install the required dependencies of this artifact

  ```bash
  cd <VLASelect directory>

  # Option 1: 
  # pull the full Docker image (33GB)
  # without installing other dependencies
  bash dep.sh

  # Option 2: 
  # pull the minimum Docker image (100MB)
  # and install other dependencies in the Docker container
  TYPE=100M bash dep.sh
  ```
[Example running screenshots](install-step-3-example.md)


**Step 4:** Check the installation

  ```bash
  bash start_docker.sh
  python -c "import torch; print(torch.__version__)"
  python -c "import torch; print(torch.cuda.is_available())"
  python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA not available')"
  ```
  [Example running screenshots](imgs/4.1.png)

---

> If the docker is not available on your host machine, you can follow the instruction for non-docker installation in section  [1.2.5 Install Dependencies (if Docker cannot be installed)](./README.md#125-install-dependencies-if-docker-cannot-be-installed)

---

## Evaluation Reproduction

### One-click Reproduction

**Minimal working example (completed within 1 day and 20GB memory)**

```bash
cd <VLASelect directory>
bash start_docker.sh
cd <VLASelect directory in the container>/eval
MWE=1 bash run.sh
```

### Step-by-Step Reproduction

Run the following command at the beginning:

```bash
cd <VLASelect directory>
bash start_docker.sh
cd <VLASelect directory in the container>/eval
```

#### 1. (Figure 7) Accuracy Under Tasks/Environment Changes

```bash
cd acc_comparison

# You can run vlaselect and three baseline methods by one command:
MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect bash run_acc_task_env_change.sh
python3 plot_acc_task_env.py

# Or can run all baseline methods by one command:
MWE=1 bash run_acc_task_env_change.sh
python3 plot_acc_task_env.py
```
resource requirements and outputs:

| | Resource Requirements | Example Running Outputs |
| --- | --- | --- |
| Minimum working example on three methods | 1 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_acc.md#mwe-run) |
| Minimum working example on all methods | 3 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_acc.md#mwe-run) |

>**Note:** In minimum working examples, we have limited the training time for each method to 120s. However, the total runtime remains several hours due to:
>  - loading large model checkpoints (>1GB) for each method
>  - initializing RL environments based on the physical simulation engine
>  - evaluating each method's accuracy periodically

#### 2. (Figure 8) Accuracy Under Available Resource Changes

```bash
cd acc_comparison

# You can run vlaselect and three baseline methods by one command:
MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect bash run_acc_res_change.sh
python3 plot_acc_res_change.py

# Or can run all baseline methods by one command:
MWE=1 bash run_acc_res_change.sh
python3 plot_acc_res_change.py
```

resource requirements and outputs:

| | Resource Requirements | Example Running Outputs |
| --- | --- | --- |
| Minimum working example on three methods | 1 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_acc.md#mwe-run-1) |
| Minimum working example on all methods | 2 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_acc.md#mwe-run-1) |


#### 3. (Figure 9 and Tables 2/3) Overheads Under The Same Accuracy

```bash
cd overhead

# You can run vlaselect and three baseline methods by one command:
bash overhead_same_acc.sh
python3 plot_overhead.py

# Or can run all baseline methods by one command:
bash overhead_same_acc.sh
python3 plot_overhead.py
```
resource requirements and outputs:

  | | Resource Requirements | Example Running Outputs |
  | --- | --- | --- |
  | Minimum working example on three methods | 1 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_overhead.md#mwe-run) |
  | Minimum working example on all methods | 2 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_overhead.md#mwe-run) |

#### 4. (Figure 10) Time Breakdown of VLASelect's Modules

```bash
cd overhead

# You can run vlaselect and three baseline methods by one command:
  MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect bash overhead_breakdown_all_methods.sh
  python3 plot_breakdown_all_methods.py

# Or can run all baseline methods by one command:
  MWE=1 bash overhead_breakdown_all_methods.sh
  python3 plot_breakdown_all_methods.py
```

resource requirements and outputs:

  | | Resource Requirements | Example Running Outputs |
  | --- | --- | --- |
  | Minimum working example on three methods | 1 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_overhead.md#mwe-run-2) |
  | Minimum working example on all methods | 2 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_overhead.md#mwe-run-2) |


#### 5. (Figure 11) Training Time Breakdown in Each Workload

```bash
cd overhead

# You can run vlaselect and three baseline methods by one command:
  MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect bash overhead_breakdown_all_methods.sh
  python3 plot_breakdown_all_methods.py

# Or can run all baseline methods by one command:
  MWE=1 bash overhead_breakdown_all_methods.sh
  python3 plot_breakdown_all_methods.py
```

resource requirements and outputs:

  | | Resource Requirements | Example Running Outputs |
  | --- | --- | --- |
  | Minimum working example on three methods | 1 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_overhead.md#mwe-run-2) |
  | Minimum working example on all methods | 2 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_overhead.md#mwe-run-2) |


#### 6. (Figure 12) Design Choice Validation by Ablation

  ```bash
  cd ablation
  MWE=1 bash run_ablation.sh
  python3 plot_ablation.py
  ```

resource requirements and outputs:

| | Resource Requirements | Example Running Outputs |
  | --- | --- | --- |
  | Minimum working example | 1 hours<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/ablation_results.md#2-mwe-run) |


#### 7. Discussion

```bash
cd discussion

# Discussion 1: Sim-to-real transfer (Run only if the required robot hardware is connected)
# bash run_sim_to_real.sh

# Discussion 2: ICL (In-Context Learning)
# MWE: MWE=1 bash compare_icl.sh | Full run: bash compare_icl.sh
MWE=1 bash compare_icl.sh

# Discussion 3: Maximum Supported Model Size
MODEL_SIZE_LIMIT_FAMILY=tinyvla bash sweep_model_size.sh

# Discussion 4: Applicability to multi-agent scenarios
# MWE: MWE=1 bash run_multi_agent.sh | Full run: bash run_multi_agent.sh
MWE=1 bash run_multi_agent.sh
```

| Experiment | Resource Requirements | Example Running Outputs |
| :--- | :--- | :--- |
| **Sim-to-real transfer (DOFBOT-SE)** | a DOFBOT-SE single-arm robot | [Link](https://github.com/user-attachments/assets/9f7905c2-c88b-4d91-8b25-9d59a78566af) |
| **Sim-to-real transfer (AmazingHand)** | an AmazingHand dexterous hand | [Link](https://github.com/user-attachments/assets/c56c4114-c24c-4c35-b272-e9cf03848504) |
| **In-Context Learning (ICL)** | 10 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_discussion.md#icl) |
| **Maximum Supported Model Size (TinyVLA)** | 1 hour<br>32GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_discussion.md#maximum-supported-model-size) |
| **Multi-agent scenarios** | 20 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/results_discussion.md#applicability-to-multi-agent-scenarios) |

---

### Supporting Various VLA Models, Scaling Strategies, and Knowledge Exchange Granularities

```bash
## run in the root path of VLASelect
MWE=1 bash <script_path>

# An example of VLA-Adapter Model Support Verification: 
MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh
```

| Model & Evaluation Metric | Command to Run | Resource Requirements | Experiment Results |
| :--- | :--- | :--- | :--- |
| **VLA-Adapter**: Model Support Verification | `MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify.sh` | 3 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#311-supporting-the-vla-adapter) |
| **VLA-Adapter**: Scaling Strategies (Representative 3 methods) | `MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify-all_scaling_methods-only4.sh` | 20 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#312-supporting-different-scaling-strategies) |
| **VLA-Adapter**: Knowledge Exchange Granularities | `MWE=1 bash api/vla_model_interface_examples/vla_adapter_impl_verify-all_granularities.sh` | 30 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#313-supporting-different-knowledge-exchange-granularities) |
| **TinyVLA**: Model Support Verification | `MWE=1 bash api/vla_model_interface_examples/tinyvla_impl_verify.sh` | 3 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#321-supporting-the-tinyvla) |
| **TinyVLA**: Scaling Strategies (Representative 3 methods) | `MWE=1 bash api/vla_model_interface_examples/tinyvla_impl_verify-all_scaling_methods-only4.sh` | 20 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#322-supporting-different-scaling-strategies) |
| **TinyVLA**: Knowledge Exchange Granularities | `MWE=1 bash api/vla_model_interface_examples/tinyvla_impl_verify-all_granularities.sh` | 30 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#323-supporting-different-knowledge-exchange-granularities) |
| **EdgeVLA**: Model Support Verification | `MWE=1 bash api/vla_model_interface_examples/edgevla_impl_verify.sh` | 3 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#331-supporting-for-the-edgevla) |
| **EdgeVLA**: Scaling Strategies (Representative 3 methods) | `MWE=1 bash api/vla_model_interface_examples/edgevla_impl_verify-all_scaling_methods-only4.sh` | 20 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#332-baseline-comparison-on-edgevla) |
| **EdgeVLA**: Knowledge Exchange Granularities | `MWE=1 bash api/vla_model_interface_examples/edgevla_impl_verify-all_granularities.sh` | 30 minutes<br>20GB memory | [Link](https://github.com/LINC-BIT/VLASelect/blob/main/model_support.md#333-swapping-granularity-ablation-on-edgevla) |

---

## Notes on Results Variation

>Exact values may vary slightly across different testbeds and hardware environments due to performance fluctuations. However, these minor variations do not affect the overall trends or conclusions.
