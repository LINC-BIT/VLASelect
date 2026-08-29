# EuroSys'27 Artifact Checklist

The official checklist for EuroSys'27 artifact evaluation is listed in [https://sysartifacts.github.io/eurosys2027/badges](https://sysartifacts.github.io/eurosys2027/badges). 

We have conducted a self-check and listed the results below.

## 1. Available

| Checklist Item | Our Artifact |
|---|---|
| The artifact is available on a **public archive with irrevocable versioning and long-term storage**, such as Zenodo but not GitHub | ✅<br>[GitHub](https://github.com/LINC-BIT/VLASelect)<br>[Zenodo](https://zenodo.org/records/22119671) |
| The artifact has a **license that allows comparison and extension**, such as the [CC-BY](https://creativecommons.org/licenses/by/4.0/) or [MIT](https://opensource.org/license/mit/) licenses | ✅<br>[Apache license](./LICENSE) |
| The artifact has a **"read me" file referencing the paper** | ✅<br>1. [README.md for full experiment](./READMD.md)<br>2. [README.md for minimum working example](./READMD.md) |

## 2. Functional

- The artifact has a “read me” file with:
    | Checklist Item | Our Artifact |
    |---|---|
    | A description of each artifact component and how it relates to the paper | ✅<br>[Section 2 in README.md](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) |
    | A description of the exact environment the authors used, such as OS version and hardware | ✅<br>[Sections 1.2.1 and 1.2.2 in README.md](https://github.com/LINC-BIT/VLASelect#121-hardware-requirements) |
    | If the artifact includes code that deliberately performs malicious or destructive operations, appropriate warnings and context | ✅<br>Our code has no malicious or destructive operations, as stated in [Sections 1.2.3 in README.md](https://github.com/LINC-BIT/VLASelect#123-get-source-code). |

- The artifact includes **all code and data relevant to the paper**, and only those:
    | Checklist Item | Our Artifact |
    |---|---|
    | The artifact must not include obsolete or unrelated code or data | ✅ |
    | If existing code or data has been modified, the artifact should clearly separate the modifications from the original | ✅<br>We do not modify existing code or data. |
    | If the paper makes soundness claims, such as proofs, there should be simple scripts to verify these, such as listing proof assumptions | ✅<br>No proof |
    | If the paper makes quantifiable claims, such as code size per module, there should be simple scripts to output these | ✅<br>We provide 10 scripts to output all quantifiable claims in the paper, as listed in [Section 2 in README.md](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction). |

- For data, **modifications made to the raw data are documented**:
    | Checklist Item | Our Artifact |
    |---|---|
    | For instance, whether parts of the raw data were anonymized or discarded | ✅<br>We do not anonymize or discard any raw data. |

- For executable artifacts, the “read me” file also contains **documentation** to:
    | Checklist Item | Our Artifact |
    |---|---|
    | Run and extend a “minimal working example” | ✅<br>We provide 19 minimal working example in [Sections 2](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) [and 3 in README.md](https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities). Running them can reproduce evaluation and verify VLASelect's applicability. |
    | Compile and execute the artifact, including pre-installation steps | ✅<br>[Sections 1.2.3 to 1.2.6 in README.md](https://github.com/LINC-BIT/VLASelect#123-get-source-code) |
    | Configure the artifact, such as selecting IP addresses or disks | ✅<br>[Sections 1.2.3 to 1.2.6 in README.md](https://github.com/LINC-BIT/VLASelect#123-get-source-code) |
    | Know the expected resource use per kind of experiment, such as “5 minutes, 10 GB of disk space” | ✅<br>1. We provide the expected resource use per experiment in [Sections 2](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) [and 3 in README.md](https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities).<br>2. We provide the overall expected resource use in [Section 1.2.1 in README.md](https://github.com/LINC-BIT/VLASelect#121-hardware-requirements). |
    | Know what unusual behavior to expect, such as warning messages emitted by another system used as a baseline for experiments | ✅<br>[Section 1.3 in README.md](https://github.com/LINC-BIT/VLASelect#13-treatment-measure-for-unusual-behaviors) |

- For executable artifacts, the artifact includes a **precise list of dependencies**:
    | Checklist Item | Our Artifact |
    |---|---|
    | Whenever possible, it should be usable by a package manager | ✅<br>We list the dependencies in `eval/requirements.txt` which can be usable by Python package manager. |
    | Exotic dependencies must have associated automation to download and build them | ✅<br>We provide `dep.sh` to automatically download all dependencies. |
    | OS-level dependencies must involve a VM/container, accompanied by a script to generate the VM/container | ✅<br>We do not have any OS-level dependencies. |
    | Proprietary dependencies must have associated instructions to obtain them, along with “mock” versions to demonstrate their use | ✅<br>We do not have any proprietary dependencies. |

- The artifact includes an **example input and configuration for each kind of experiment** in the paper:
    | Checklist Item | Our Artifact |
    |---|---|
    | Authors are encouraged, but not required, to provide inputs, configurations, and outputs for all experiments described in the paper. | ✅<br>1. We pack the inputs and configurations into the scripts.<br>2. We provide example running outputs of each script in [Sections 2](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) [and 3 in README.md](https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities). |

- Others:
    | Checklist Item | Our Artifact |
    |---|---|
    | Artifacts must be usable on other environments than the authors’, though software may require specific hardware such as one model of network card. | ✅<br>1. We provide several options of hardware/software in [Sections 1.2.1 and 1.2.2 in README.md](https://github.com/LINC-BIT/VLASelect#121-hardware-requirements).<br>2. We have tested the artifact evaluation in a smaller environment than the paper, and provided the [evaluation report](https://github.com/LINC-BIT/VLASelect/blob/main/Artifact%20Evaluation%20Report%20for%20VLASelect.pdf). |
    | Manual work such as writing configuration files must be minimized. There must be no redundant manual steps such as writing the same configuration values in multiple places, as this inevitably leads to human error. | ✅<br>We provide 19 automatic scripts for evaluation reproduction, which do not include any manual work of writing additional files. |

## 3. Reproduced

- The artifact includes a single script to run each experiment and output results, given the necessary input and configuration:
    | Checklist Item | Our Artifact |
    |---|---|
    | The scripts must be documented, allowing researchers to ensure they correspond to the claims, merely producing the right output is not enough. | ✅<br>We provide each script's documentation, example running outputs, and key observations in [Sections 2](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) [and 3 in README.md](https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities). |
    | The scripts must handle common edge cases in a reasonable fashion, such as forgetting arguments or running the same script twice. | ✅<br>Our scripts can handle common edge cases. For example, the script does not require any arguments. Running the same scripts multiple times just outputs multiple results. |

- The artifact includes a **script to convert each experiment’s results into human-readable ones** as close to the paper presentation as possible:
    | Checklist Item | Our Artifact |
    |---|---|
    | For simple results presentation such as tables, this and the previous script can be merged into one. | ✅<br>The result presentation for one experiment only requires one script, as listed in [Sections 2](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) [and 3 in README.md](https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities). |
    | The artifact may contain separate installation steps for the dependencies of plotting scripts, subject to the same criteria. | ✅<br>[Section 1.2.6 in README.md](https://github.com/LINC-BIT/VLASelect#126-install-dependencies-for-plotting-scripts) |

- Others:
    | Checklist Item | Our Artifact |
    |---|---|
    | The expected workflow for an evaluator or a researcher looking to reuse the artifact is to install the artifact using a handful of commands, run experiments with one command each, and plot data as necessary. | ✅<br>[Outline of README.md](https://github.com/LINC-BIT/VLASelect#outline) |
    | In the absence of problems requiring debugging, active time must not exceed a few minutes. | ✅<br>The minimum working example on each method takes 6-10 minutes to complete, as listed in [Sections 2](https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction) [and 3 in README.md](https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities). |