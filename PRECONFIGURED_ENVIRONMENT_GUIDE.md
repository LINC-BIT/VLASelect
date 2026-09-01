# Instructions for Open Access to Our Preconfigured Small Machine

We provide a preconfigured small machine for artifact evaluation. Reviewers can access this machine and run the **minimum working examples** directly by following the instructions below.

## 1. Hardware and Software Specifications

<p align="center">
  <table>
    <tr>
      <th></th>
      <th></th>
    </tr>
    <tr>
      <td><b>Operating System</b></td>
      <td>Ubuntu 22.04.4 LTS (Kernel 6.8)</td>
    </tr>
    <tr>
      <td><b>CPU Architecture</b></td>
      <td>Intel Xeon E5-2698 v4 (16C)</td>
    </tr>
    <tr>
      <td><b>System Memory</b></td>
      <td>32 GB DDR4</td>
    </tr>
    <tr>
      <td><b>GPU & VRAM</b></td>
      <td>NVIDIA Tesla V100 (32 GB)</td>
    </tr>
    <tr>
      <td><b>CUDA Toolchain</b></td>
      <td>Driver 550.127.05, CUDA 12.4</td>
    </tr>
  </table>
</p>

## 2. Access the Environment

Open the following URL in a web browser:

- **URL:** [http://js4.blockelite.cn:24158](http://js4.blockelite.cn:24158)
- **Password:** `Jupyterserver`

Enter the password on the login page.

<p align="center">
  <img src="./imgs/server_web_login.png" alt="Login page" width="90%" />
</p>

After login, the platform opens directly in the preconfigured VLASelect project environment:

<p align="center">
  <img src="./imgs/web_wellcom_page_1.png" alt="Platform home page" width="90%" />
</p>

## 3. Open a Terminal

Select **Terminal** under **Other**. A terminal will open in the project environment.

<p align="center">
  <img src="./imgs/web_wellcom_page.png" alt="Platform home page" width="90%" />
</p>

<p align="center">
  <img src="./imgs/web_terminal_1.png" alt="Platform home page" width="90%" />
</p>


## 4. Start the Docker Container

Run the following command to start the Docker container:

```bash
bash start_docker.sh
```

<br>


<p align="center">
  <img src="./imgs/start_docker.png" alt="VLASelect Docker container started" width="90%" />
</p>

After the command attaches to the container, run all subsequent evaluation commands in the container shell.

## 5. Run the Evaluation

After starting the container, follow [Section 2.2: Step-by-Step Reproduction](README.md#22-step-by-step-reproduction) in the README to run the evaluation experiments.

**Example:**

The following example reproduces **Experiment 1 (Figure 7): Accuracy Under Tasks/Environment Changes** for the minimum working examples.

```bash
cd eval/acc_comparison
MWE=1 METHODS=self_improv,vla_rft,world_env,vlaselect \
  bash run_acc_task_env_change.sh
python3 plot_acc_task_env.py
```

**The expected terminal output is shown below:**

<p align="center">
  <img src="./imgs/exmaple_1.png" alt="Example output for the task and environment change experiment" width="90%" />
</p>

**The resource requirements and output are listed below:**

<p align="center">
  <table>
    <tr>
      <th>Configuration</th>
      <th>Expected runtime</th>
      <th>Resource requirements</th>
      <th>Output</th>
    </tr>
    <tr>
      <td>Minimum working example</td>
      <td>1.5 hours</td>
      <td>20GB memory <br> 32GB disk space</td>
      <td><code>eval/acc_comparison/FIG_ACC_TASK_ENV.pdf</code></td>
    </tr>
  </table>
</p>

## 6. Check Results

Use the file manager on the left to inspect outputs.

<p align="center">
  <img src="./imgs/file_manager.png" alt=" file manager" width="90%" />
</p>