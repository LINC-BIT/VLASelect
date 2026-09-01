# Pre-configured JupyterLab Environment Guide

We provide a browser-based JupyterLab environment where reviewers can access the VLASelect repository and run the **Minimal Working Example** directly.

## 1. Access the JupyterLab Environment

Open the following URL in a web browser:

- **URL:** [http://js4.blockelite.cn:24158](http://js4.blockelite.cn:24158)
- **Password:** `Jupyterserver`

Enter the password on the Jupyter login page.

<p align="center">
  <img src="./imgs/server_web_login.png" alt="Jupyter login page" width="90%" />
</p>

After login, JupyterLab opens directly in the pre-configured VLASelect project environment.

## 2. Open a Terminal

In the JupyterLab Launcher, select **Terminal** under **Other**. A web terminal will open in the project environment.

<p align="center">
  <img src="./imgs/web_wellcom_page.png" alt="JupyterLab Launcher in the VLASelect project environment" width="90%" />
</p>

## 3. Start the Docker Container

In the web terminal, start the pre-configured Docker container:

```bash
bash start_docker.sh
```

After the command attaches to the container, run all subsequent evaluation commands in the container shell.

<p align="center">
  <img src="./imgs/start_docker.png" alt="VLASelect Docker container started" width="90%" />
</p>

## 4. Run the Evaluation

After starting the container, follow [Section 2.2: Step-by-Step Reproduction](README.md#22-step-by-step-reproduction) in the README to run the scheduled evaluation experiments.

**Example:**

The following example reproduces **Experiment 1 (Figure 7): Accuracy Under Tasks/Environment Changes** in Minimal Working Example mode.

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
      <td>Minimal Working Example</td>
      <td>1.5 hours</td>
      <td>20GB memory <br> 32GB disk space</td>
      <td><code>eval/acc_comparison/FIG_ACC_TASK_ENV.pdf</code></td>
    </tr>
  </table>
</p>


