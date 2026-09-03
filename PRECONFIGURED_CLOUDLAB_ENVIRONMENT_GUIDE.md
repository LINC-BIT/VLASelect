# Instructions for Open Access to Our Preconfigured CloudLab Environment

We provide a **preconfigured CloudLab environment** for artifact evaluation. 

Reviewers can access this environment and run the **minimum working examples** directly by following the instructions below.

## 1. Hardware and Software Specifications

<p align="center">
  <table align="center">
    <tr>
      <td><b>Operating System</b></td>
      <td>Ubuntu 22.04.2 LTS (Kernel 5.15)</td>
    </tr>
    <tr>
      <td><b>CPU</b></td>
      <td>AMD EPYC 7542 (32C)</td>
    </tr>
    <tr>
      <td><b>Memory</b></td>
      <td>512 GB DDR4</td>
    </tr>
    <tr>
      <td><b>GPU</b></td>
      <td>NVIDIA Tesla V100 (32 GB)</td>
    </tr>
    <tr>
      <td><b>CUDA</b></td>
      <td>Driver 580.173.02, CUDA 13.0</td>
    </tr>
  </table>
</p>

## 2. Access the Environment

Open the URL [http://clgpu015.clemson.cloudlab.us:8080](http://clgpu015.clemson.cloudlab.us:8080) in a web browser (e.g. Chrome or Microsoft Edge). The page of **preconfigured environment** will appear as below:

<p align="center">
  <img src="./imgs/step1-cloud.png" alt="Login page" width="90%" />
</p>

## 3. Launch a Terminal

Click the **"New Terminal" button** in the menu to launch a terminal.
<p align="center">
  <img src="./imgs/step2.1-cloud.png" alt="Platform home page" width="90%" />
</p>

<p align="center">
  <img src="./imgs/step2.2-cloud.png" alt="Terminal" width="90%" />
</p>

## 4. Start the Docker Container

Run the following command in the terminal:

```bash
bash start_docker.sh
```

The expected output is shown as below:

<p align="center">
  <img src="./imgs/step3-cloud.png" alt="VLASelect Docker container started" width="90%" />
</p>

<!-- After the command attaches to the terminal, run all subsequent evaluation commands in the container shell. -->

## 5. Run Minimum Working Examples

Follow [Section 2.2: Step-by-Step Reproduction](README.md#22-step-by-step-reproduction) in the README.md to run the minimum working examples.

The expected terminal output is shown below:

<p align="center">
  <img src="./imgs/step4-cloud.png" alt="Input commands in the terminal" width="90%" />
</p>

## 6. Check Results

Use the file manager on the left panel to check the results.

<p align="center">
  <img src="./imgs/step5.1-cloud.png" alt="Output path in the terminal" width="90%" />
</p>

## 7. Run Example Experiment 1 (Figure 7): Accuracy Under Tasks/Environment Changes

<!-- To run **Experiment 1**, first complete the procedures in [Sections 2 to 4](#2-access-the-environment) above, and perform the steps below. -->

### Step 1: Enter the environment<img src="./heading-divider.svg" alt="" width="100%" height="1">

Complete the procedures in [Sections 2 to 4](#2-access-the-environment) above:
- [Section 2: Accessing the Environment](#2-access-the-environment)
- [Section 3: Launching a Terminal](#3-launch-a-terminal)
- [Section 4: Starting the Docker container](#4-start-the-docker-container)

### Step 2: Find the evaluation script<img src="./heading-divider.svg" alt="" width="100%" height="1">

The **evaluation script** for Experiment 1 is provided in [Section 2.2.1 Experiment 1: (Figure 7 in Section 5.2.1) Accuracy Under Tasks/Environment Changes](README.md#221-experiment-1-figure-7-in-section-521-accuracy-under-tasksenvironment-changes) in the README.md, i.e.:

```bash
cd eval/acc_comparison

MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash run_acc_task_env_change.sh

python3 plot_acc_task_env.py
```

The **resource requirements** and **expected output** of this experiment are also listed in that section, i.e.:

|  | Expected runtime | Resource requirements | Output |
| --- | --- | --- | --- |
| Minimum working example | 1.5 hours | 20 GB memory<br>32 GB disk space | `eval/acc_comparison/FIG_ACC_TASK_ENV.pdf` |

### Step 3: Run the evaluation script<img src="./heading-divider.svg" alt="" width="100%" height="1">

Run the script (found in Step 2) in the terminal, and wait for completion:

<p align="center">
  <img src="./imgs/step7.2-cloud.png" alt="Input commands in the terminal" width="90%" />
</p>

### Step 4: Check the results<img src="./heading-divider.svg" alt="" width="100%" height="1">

1. **Obtain the output file's path**. After Step 3 completes, the terminal will print the path of the output file:

  <p align="center">
    <img src="./imgs/step7.3-cloud.png" alt="Output path in the terminal" width="90%" />
  </p>

2. **View the output file**. Use the file manager to locate the file, and double-click the file. The file will be displayed in the right panel:

  <p align="center">
    <img src="./imgs/step5.2-cloud.png" alt="Example output in the file manager" width="90%" />
  </p>

3. **Download the output file**. Right-click the file in the file manager, and click the "Download" button:

<p align="center">
  <img src="./imgs/download.png" alt="Download output file from the file manager" width="90%" />
</p>

## 8. Solve Possible Unusual Behaviors

### 8.1 Unusual behavior 1: File display is prohibited by the browser<img src="./heading-divider.svg" alt="" width="100%" height="1">


This is due to the browser's default safety restriction strategy. 

**Solution: Change the browser settings as follows:** 

1. For Microsoft Edge browser, enter `edge://flags/#unsafely-treat-insecure-origin-as-secure` in the address bar. For Chrome browser, enter `chrome://flags/#unsafely-treat-insecure-origin-as-secure` in the address bar.

  <p align="center">
    <img src="./imgs/warn1.png" alt="Example output in the file manager" width="90%" />
  </p>

2. Set **Insecure origins treated as secure** to **Enabled**, and enter `http://clgpu015.clemson.cloudlab.us:8080` in the textarea below.

  <p align="center">
    <img src="./imgs/warn2.png" alt="Example output in the file manager" width="90%" />
  </p>

3. Click the **Restart** button.

  <p align="center">
    <img src="./imgs/warn3.png" alt="Example output in the file manager" width="90%" />
  </p>

4. Re-open [http://clgpu015.clemson.cloudlab.us:8080](http://clgpu015.clemson.cloudlab.us:8080) in the browser to view the files.
