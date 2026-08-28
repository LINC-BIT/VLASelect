# Expected running outputs

## Discussion 2: ICL (In-Context Learning)

### Minimal working example

> **Key observation:** RICL fails entirely because the brief runtime is insufficient for its policy to update, whereas VLASelect rapidly adapts and **maintains an average accuracy of 37.2%**.

<img src="./imgs/2.5.2.png" alt="alt text" style="zoom:33%;" />


## Discussion 3: Maximum Supported Model Size

| Source | Supported Model Size | Hardware / Environment |
| --- | --- | --- |
| **Full run** | Up to 11.3 GB (Xavier), Up to 24.0 GB (Orin) | Jetson Xavier / Jetson Orin |
| **Minimal working example** | ~2.5–24.0 GB | Host with 32 GB VRAM |

## Discussion 4: Applicability to multi-agent scenarios

### Minimal working example

> **Key observation:** MAPPO completely fails due to the short runtime being insufficient for policy updates, whereas **VLASelect reaches up to 60.0% accuracy**.

<img src="./imgs/2.2.4.5.png" alt="alt text" style="zoom:33%;" />
