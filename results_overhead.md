# Example Running Outputs<img src="./heading-divider.svg" alt="" width="100%" height="1">

## (Figure 9 Tables 2/3 in section 5.3.1) Overheads Under The Same Accuracy<img src="./heading-divider.svg" alt="" width="100%" height="1">

> **Key observation:** VLASelect consistently achieves the target accuracy with **the shortest execution** time and **the most reduced resource consumption** compared to all baseline methods.

### Full run

#### Figure 9: Full-scale Example for Memory Footprint Comparison in a new task

<img src="imgs/2.2.2.1.png" alt="Overhead Comparison - Full Run" style="zoom:150%;" />

<br><br>

#### Table 2: Full-scale Example for Average Energy Consumption (kJ) in each new task

| Method | Time (h) - Single-arm | Time (h) - Dexterous | Time (h) - Mobile | Time (h) - Humanoid | Memory (GB) - Single-arm | Memory (GB) - Dexterous | Memory (GB) - Mobile | Memory (GB) - Humanoid | Energy (kJ) - Single-arm | Energy (kJ) - Dexterous | Energy (kJ) - Mobile | Energy (kJ) - Humanoid |
| :--: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ConRFT** | 3.88 | 3.14 | 2.83 | 3.78 | 19.12 | 20.88 | 27.92 | 35.71 | 360.39 | 294.89 | 334.87 | 600.23 |
| **FlaRe** | 2.56 | 3.60 | 2.92 | 2.91 | 23.14 | 31.39 | 27.49 | 33.70 | 295.13 | 384.55 | 384.75 | 492.30 |
| **iRe-VLA** | 2.71 | 3.38 | 2.78 | 3.31 | 21.93 | 24.55 | 27.30 | 42.19 | 253.73 | 374.00 | 382.41 | 630.36 |
| **Self-Improvement** | 4.52 | 3.48 | 2.86 | 3.95 | 20.93 | 23.52 | 27.53 | 42.11 | 547.71 | 309.29 | 337.90 | 763.35 |
| **RLVLA** | 2.74 | 3.32 | 3.52 | 3.20 | 15.26 | 17.31 | 18.78 | 32.11 | 275.99 | 353.41 | 546.13 | 608.56 |
| **VLA-RFT** | 3.20 | 3.39 | 2.39 | 4.07 | 28.54 | 30.99 | 29.04 | 60.02 | 319.56 | 412.78 | 341.89 | 704.55 |
| **World-Env** | 3.10 | 3.63 | 3.87 | 4.16 | 25.66 | 30.05 | 29.27 | 60.23 | 279.42 | 429.66 | 505.86 | 738.14 |
| **EdgeTA** | 4.59 | 3.17 | 2.84 | 3.12 | 17.54 | 19.23 | 21.12 | 37.87 | 523.80 | 382.06 | 440.83 | 467.90 |
| **ConvertNet** | 4.82 | 3.67 | 2.69 | 2.73 | 15.26 | 17.31 | 18.78 | 32.11 | 478.14 | 434.77 | 430.10 | 510.83 |
| **VLASelect** | 0.39 | 0.29 | 0.21 | 0.32 | 15.26 | 17.31 | 18.78 | 32.11 | 45.53 | 34.53 | 28.76 | 47.56 |

<br><br>

#### Table 3: Full-scale Example for Overhead comparison under the same learning accuracy

| Method | Trial / Task | Single-arm robot | Dexterous hand | Mobile manipulator | Humanoid robot |
| :--: | :---: | :---: | :---: | :---: | :---: |
| **ConRFT** | Task 1<br>Task 2 | 342.37<br>403.51 | 346.31<br>274.05 | 499.84<br>291.68 | 787.19<br>570.22 |
| **FlaRe** | Task 1<br>Task 2 | 318.06<br>254.18 | 477.07<br>317.51 | 461.29<br>379.37 | 671.17<br>452.06 |
| **iRe-VLA** | Task 1<br>Task 2 | 383.33<br>213.81 | 530.53<br>204.04 | 505.40<br>273.55 | 661.88<br>564.21 |
| **Self-Improvement** | Task 1<br>Task 2 | 551.76<br>473.14 | 416.37<br>293.83 | 497.49<br>297.81 | 801.52<br>673.49 |
| **RLVLA** | Task 1<br>Task 2 | 326.26<br>206.67 | 360.71<br>314.32 | 573.44<br>466.58 | 622.44<br>458.46 |
| **VLA-RFT** | Task 1<br>Task 2 | 323.69<br>277.80 | 433.42<br>280.17 | 374.09<br>257.39 | 795.60<br>632.21 |
| **World-Env** | Task 1<br>Task 2 | 329.26<br>265.45 | 339.87<br>451.14 | 666.90<br>480.57 | 775.05<br>639.55 |
| **EdgeTA** | Task 1<br>Task 2 | 349.40<br>549.99 | 383.21<br>282.31 | 464.28<br>355.50 | 629.97<br>444.50 |
| **ConvertNet** | Task 1<br>Task 2 | 608.37<br>454.23 | 456.51<br>327.35 | 451.61<br>378.12 | 536.37<br>451.60 |
| **VLASelect** | Task 1<br>Task 2 | 46.06<br>44.24 | 35.41<br>25.63 | 39.46<br>25.57 | 64.71<br>45.18 |

<br><br>

### Minimal working example

#### Figure 9: Minimal Working Example for Memory Footprint Comparison in a new task

![Overhead Comparison - MWE](imgs/2.2.2.1-mwe.jpg)

#### Table 2: Minimal Working Example for Average Energy Consumption (kJ) in each new task

| Method | Time (h) - Single-arm | Time (h) - Dexterous | Time (h) - Mobile | Time (h) - Humanoid | Memory (GB) - Single-arm | Memory (GB) - Dexterous | Memory (GB) - Mobile | Memory (GB) - Humanoid | Energy (kJ) - Single-arm | Energy (kJ) - Dexterous | Energy (kJ) - Mobile | Energy (kJ) - Humanoid |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Self-Improvement** | 0.04 | 0.17 | 0.09 | 0.08 | 7.32 | 13.41 | 12.47 | 12.94 | 15.82 | 62.99 | 37.17 | 25.65 |
| **VLA-RFT** | 0.03 | 0.08 | 0.07 | 0.07 | 8.48 | 13.09 | 11.07 | 11.21 | 13.83 | 27.52 | 26.83 | 25.24 |
| **World-Env** | 0.03 | 0.07 | 0.08 | 0.08 | 9.77 | 13.03 | 10.25 | 11.12 | 12.98 | 23.96 | 33.92 | 33.02 |
| **VLASelect** | 0.01 | 0.01 | 0.03 | 0.03 | 5.23 | 11.97 | 10.40 | 10.40 | 0.58 | 2.96 | 12.28 | 9.91 |

<br><br>

#### Table 3: Minimal Working Example for Overhead comparison under the same learning accuracy

| Method | Single-arm robot | Dexterous hand | Mobile manipulator | Humanoid robot |
| :--- | :---: | :---: | :---: | :---: |
| **Self-Improvement** | 15.82 | 62.99 | 37.17 | 26.22 |
| **VLA-RFT** | 13.83 | 27.52 | 26.83 | 24.62 |
| **World-Env** | 12.98 | 23.96 | 33.92 | 33.03 |
| **VLASelect** | 0.68 | 11.37 | 12.28 | 2.58 |

<br><br>

## (Figure 10 in section 5.3.2) Time Breakdown of VLASelect's Modules<img src="./heading-divider.svg" alt="" width="100%" height="1">

> **Key observation:** The execution time of VLASelect core modules are obviously less than training iteration time.

| | Results |
| :--- | :--- |
| **Full run** | ![Time Breakdown - Full Run](imgs/2.2.2.2.png) |
| **Minimal working example** | <img src="imgs/2.2.2.2-mwe.png" alt="Time Breakdown - MWE" style="zoom: 25%;" /> |

---

## (Figure 11 in section 5.3.2) Training Time Breakdown in Each Workload<img src="./heading-divider.svg" alt="" width="100%" height="1">

> **Key observation:** Under identical workloads, VLASelect achieves **shorter runtime** while maintaining high accuracy compared with other baselines.

| | Results |
| :--- | :--- |
| **Full run** | ![Training Time Breakdown - Full Run](imgs/2.2.2.3.png) |
| **Minimal working example** | ![Training Time Breakdown - MWE](imgs/2.2.2.3-mwe.jpg) |