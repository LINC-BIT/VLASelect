# Pre-configured CloudLab Environment Guide

We provide a browser-based CloudLab environment where reviewers can access the VLASelect repository and run the **Minimal Working Example** directly.

## 1. Access the CloudLab Environment

Open the following URL in a web browser:

- **URL:** [http://clgpu015.clemson.cloudlab.us:8080](http://clgpu015.clemson.cloudlab.us:8080)

<p align="center">
  <img src="./imgs/step1-cloud.png" alt="CloudLab login page" width="90%" />
</p>

## 2. Open a Terminal

In the CloudLab JupyterLab launcher, open a terminal.

<p align="center">
  <img src="./imgs/step2.1-cloud.png" alt="CloudLab launcher" width="90%" />
</p>

<p align="center">
  <img src="./imgs/step2.2-cloud.png" alt="CloudLab terminal" width="90%" />
</p>

## 3. Start the Docker Container

In the terminal, start the pre-configured Docker container:

```bash
bash start_docker.sh
```

<p align="center">
  <img src="./imgs/step3-cloud.png" alt="CloudLab Docker container started" width="90%" />
</p>

## 4. Run the Evaluation

After starting the container, run the evaluation commands in the container shell.

<p align="center">
  <img src="./imgs/step4-cloud.png" alt="Running evaluation commands" width="90%" />
</p>

## 5. Check Results

Use the file manager on the left to inspect outputs.

<p align="center">
  <img src="./imgs/step5-cloud.png" alt="CloudLab file manager" width="90%" />
</p>
