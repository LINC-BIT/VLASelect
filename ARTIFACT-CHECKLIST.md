# EuroSys'27 Artifact Checklist

The official checklist for EuroSys'27 artifact evaluation is listed in [https://sysartifacts.github.io/eurosys2027/badges](https://sysartifacts.github.io/eurosys2027/badges). 

We have conducted a self-check and listed the results below.

## 1. Available

<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>The artifact is available on a <strong>public archive with irrevocable versioning and long-term storage</strong>, such as Zenodo but not GitHub</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect">GitHub</a><br><a href="https://zenodo.org/records/22119671">Zenodo</a></td></tr>
<tr><td>The artifact has a <strong>license that allows comparison and extension</strong>, such as the <a href="https://creativecommons.org/licenses/by/4.0/">CC-BY</a> or <a href="https://opensource.org/license/mit/">MIT</a> licenses</td><td>✅<br><a href="./LICENSE">Apache license</a></td></tr>
<tr><td>The artifact has a <strong>"read me" file referencing the paper</strong></td><td>✅<br>1. <a href="./READMD.md">README.md for full experiment</a><br>2. <a href="./READMD.md">README.md for minimum working example</a></td></tr>
</tbody>
</table>

## 2. Functional

(1) The artifact has a “read me” file with:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>A description of each artifact component and how it relates to the paper</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Section 2 in README.md</a></td></tr>
<tr><td>A description of the exact environment the authors used, such as OS version and hardware</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#121-hardware-requirements">Sections 1.2.1 and 1.2.2 in README.md</a></td></tr>
<tr><td>If the artifact includes code that deliberately performs malicious or destructive operations, appropriate warnings and context</td><td>✅<br>Our code has no malicious or destructive operations, as stated in <a href="https://github.com/LINC-BIT/VLASelect#123-get-source-code">Sections 1.2.3 in README.md</a>.</td></tr>
</tbody>
</table>
<br>

(2) The artifact includes **all code and data relevant to the paper**, and only those:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>The artifact must not include obsolete or unrelated code or data</td><td>✅</td></tr>
<tr><td>If existing code or data has been modified, the artifact should clearly separate the modifications from the original</td><td>✅<br>We do not modify existing code or data.</td></tr>
<tr><td>If the paper makes soundness claims, such as proofs, there should be simple scripts to verify these, such as listing proof assumptions</td><td>✅<br>No proof</td></tr>
<tr><td>If the paper makes quantifiable claims, such as code size per module, there should be simple scripts to output these</td><td>✅<br>We provide 10 scripts to output all quantifiable claims in the paper, as listed in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Section 2 in README.md</a>.</td></tr>
</tbody>
</table>
<br>

(3) For data, **modifications made to the raw data are documented**:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody><tr><td>For instance, whether parts of the raw data were anonymized or discarded</td><td>✅<br>We do not anonymize or discard any raw data.</td></tr></tbody>
</table>
<br>

(4) For executable artifacts, the “read me” file also contains **documentation** to:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>Run and extend a “minimal working example”</td><td>✅<br>We provide 19 minimal working example in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Sections 2</a> <a href="https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities">and 3 in README.md</a>. Running them can reproduce evaluation and verify VLASelect's applicability.</td></tr>
<tr><td>Compile and execute the artifact, including pre-installation steps</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#123-get-source-code">Sections 1.2.3 to 1.2.6 in README.md</a></td></tr>
<tr><td>Configure the artifact, such as selecting IP addresses or disks</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#123-get-source-code">Sections 1.2.3 to 1.2.6 in README.md</a></td></tr>
<tr><td>Know the expected resource use per kind of experiment, such as “5 minutes, 10 GB of disk space”</td><td>✅<br>1. We provide the expected resource use per experiment in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Sections 2</a> <a href="https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities">and 3 in README.md</a>.<br>2. We provide the overall expected resource use in <a href="https://github.com/LINC-BIT/VLASelect#121-hardware-requirements">Section 1.2.1 in README.md</a>.</td></tr>
<tr><td>Know what unusual behavior to expect, such as warning messages emitted by another system used as a baseline for experiments</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#13-treatment-measure-for-unusual-behaviors">Section 1.3 in README.md</a></td></tr>
</tbody>
</table>
<br>

(5) For executable artifacts, the artifact includes a **precise list of dependencies**:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>Whenever possible, it should be usable by a package manager</td><td>✅<br>We list the dependencies in <code>eval/requirements.txt</code> which can be usable by Python package manager.</td></tr>
<tr><td>Exotic dependencies must have associated automation to download and build them</td><td>✅<br>We provide <code>dep.sh</code> to automatically download all dependencies.</td></tr>
<tr><td>OS-level dependencies must involve a VM/container, accompanied by a script to generate the VM/container</td><td>✅<br>We do not have any OS-level dependencies.</td></tr>
<tr><td>Proprietary dependencies must have associated instructions to obtain them, along with “mock” versions to demonstrate their use</td><td>✅<br>We do not have any proprietary dependencies.</td></tr>
</tbody>
</table>
<br>

(6) The artifact includes an **example input and configuration for each kind of experiment** in the paper:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody><tr><td>Authors are encouraged, but not required, to provide inputs, configurations, and outputs for all experiments described in the paper.</td><td>✅<br>1. We pack the inputs and configurations into the scripts.<br>2. We provide example running outputs of each script in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Sections 2</a> <a href="https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities">and 3 in README.md</a>.</td></tr></tbody>
</table>
<br>

(7) Others:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>Artifacts must be usable on other environments than the authors’, though software may require specific hardware such as one model of network card.</td><td>✅<br>1. We provide several options of hardware/software in <a href="https://github.com/LINC-BIT/VLASelect#121-hardware-requirements">Sections 1.2.1 and 1.2.2 in README.md</a>.<br>2. We have tested the artifact evaluation in a smaller environment than the paper, and provided the <a href="https://github.com/LINC-BIT/VLASelect/blob/main/Artifact%20Evaluation%20Report%20for%20VLASelect.pdf">evaluation report</a>.</td></tr>
<tr><td>Manual work such as writing configuration files must be minimized. There must be no redundant manual steps such as writing the same configuration values in multiple places, as this inevitably leads to human error.</td><td>✅<br>We provide 19 automatic scripts for evaluation reproduction, which do not include any manual work of writing additional files.</td></tr>
</tbody>
</table>

## 3. Reproduced

(1) The artifact includes a single script to run each experiment and output results, given the necessary input and configuration:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>The scripts must be documented, allowing researchers to ensure they correspond to the claims, merely producing the right output is not enough.</td><td>✅<br>We provide each script's documentation, example running outputs, and key observations in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Sections 2</a> <a href="https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities">and 3 in README.md</a>.</td></tr>
<tr><td>The scripts must handle common edge cases in a reasonable fashion, such as forgetting arguments or running the same script twice.</td><td>✅<br>Our scripts can handle common edge cases. For example, the script does not require any arguments. Running the same scripts multiple times just outputs multiple results.</td></tr>
</tbody>
</table>
<br>

(2) The artifact includes a **script to convert each experiment’s results into human-readable ones** as close to the paper presentation as possible:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>For simple results presentation such as tables, this and the previous script can be merged into one.</td><td>✅<br>The result presentation for one experiment only requires one script, as listed in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Sections 2</a> <a href="https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities">and 3 in README.md</a>.</td></tr>
<tr><td>The artifact may contain separate installation steps for the dependencies of plotting scripts, subject to the same criteria.</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#126-install-dependencies-for-plotting-scripts">Section 1.2.6 in README.md</a></td></tr>
</tbody>
</table>
<br>

(3) Others:
<table style="width: 100%; table-layout: fixed;">
<thead><tr><th width="60%">Checklist Item</th><th width="40%">Our Artifact</th></tr></thead>
<tbody>
<tr><td>The expected workflow for an evaluator or a researcher looking to reuse the artifact is to install the artifact using a handful of commands, run experiments with one command each, and plot data as necessary.</td><td>✅<br><a href="https://github.com/LINC-BIT/VLASelect#outline">Outline of README.md</a></td></tr>
<tr><td>In the absence of problems requiring debugging, active time must not exceed a few minutes.</td><td>✅<br>The minimum working example on each method takes 6-10 minutes to complete, as listed in <a href="https://github.com/LINC-BIT/VLASelect#2-evaluation-reproduction">Sections 2</a> <a href="https://github.com/LINC-BIT/VLASelect#3-supporting-various-vla-models-scaling-strategies-and-knowledge-exchange-granularities">and 3 in README.md</a>.</td></tr>
</tbody>
</table>
