# OFGQ: Hardware-Software Co-Design for Outlier-Aware Format-Adaptive Group Quantization of LLMs
## Overview

OFGQ combines software-level format adaptation with a hardware architecture specifically designed for efficient mixed-format FP4 execution.

At the software level, activation groups are quantized using one of three FP4 formats:

- **E3M0**
- **E2M1**
- **E1M2**

The format is selected at the **column-block level** through offline calibration. 

At the hardware level, OFGQ maps the different FP4 formats to a **bit-serial computation paradigm**, allowing the hardware execution latency to adapt to the selected format while avoiding the datapath inefficiency of conventional bit-parallel designs.

---

## Repository Structure

```text
.
├── OFGQ_quant/                 # Quantization and accuracy evaluation
│   ├── configs/                # Experiment configurations
│   ├── ofgq/                   # Core OFGQ implementation
│   │   ├── calibration/        # Offline format calibration
│   │   ├── models/             # Model integration
│   │   ├── quantization/       # FP4 / INT4 quantization
│   │   └── utils/
│   ├── scripts/
│   │   ├── calibrate.py        # Offline format selection
│   │   └── evaluate_ppl.py     # Perplexity evaluation
│   ├── analysis/
│   │   ├── compute_rcons.py    # Column-block distribution consistency
│   │   └── numerical_error_eval.py
│   └── README.md
│
├── OFGQ_rtl/                   # SystemVerilog RTL
│   ├── address_generator_ag.sv
│   ├── bit_serial_term_generator.sv
│   ├── bsg_group64.sv
│   ├── ofgq_pe.sv
│   ├── ofgq_cluster_4x16.sv
│   ├── ofgq_array_4cluster_16x16.sv
│   └── ag_bsg_top.sv
│
├── OFGQ_sim/                   # Performance / energy simulator
│   ├── main_ofgq.py
│   ├── main_ant.py
│   ├── main_olive.py
│   ├── main_tender.py
│   ├── ofgq_sim/
│   ├── tools/
│   ├── format_selection_schema.csv
│   └── README.md
│
└── README.md
```

The three main components can be used independently depending on the desired experiment.

| Component | Purpose |
|---|---|
| `OFGQ_quant` | Format calibration, FP4/INT4 quantization, PPL evaluation, and numerical analysis |
| `OFGQ_rtl` | RTL implementation of the core OFGQ accelerator datapath |
| `OFGQ_sim` | System-level performance and energy evaluation |

---

## Quantization

The quantization implementation is located in:

```text
OFGQ_quant/
```

### Environment

The code has been tested with:

```text
Python        3.9
PyTorch       2.6.0
CUDA          12.4
Transformers  4.52.4
Datasets      4.0.0
Accelerate    1.8.1
NumPy         2.0.2
PyYAML        6.0.2
tqdm          4.67.1
```

Test GPU:

```text
NVIDIA GeForce RTX 4090 D
```

### Installation

```bash
cd OFGQ_quant

pip install -r requirements.txt
pip install -e .
```

### Prepare Model and Dataset

Model and dataset files are not distributed with this repository.

Download the corresponding Hugging Face model and WikiText-2 dataset, and then modify:

```text
configs/llama_example.yaml
```

For example:

```yaml
model:
  path: "/path/to/model"

dataset:
  calibration_path: "/path/to/calibration.parquet"
  evaluation_path: "/path/to/test.parquet"
```

### Offline Format Calibration

Run:

```bash
python scripts/calibrate.py \
    --config configs/llama_example.yaml
```

The calibration process determines the preferred FP4 format for each column block and generates:

```text
outputs/best_formats.json
outputs/best_formats.txt
```

The JSON metadata is subsequently used during quantized inference.

### Perplexity Evaluation

Run:

```bash
python scripts/evaluate_ppl.py \
    --config configs/llama_example.yaml
```

The evaluation script loads the format-selection metadata generated during offline calibration and evaluates perplexity on WikiText-2.

For detailed configuration options, please refer to:

```text
OFGQ_quant/README.md
```

---

## Analysis

The repository includes additional scripts corresponding to the numerical analyses in the paper.

### Column-Block Distribution Consistency

Run:

```bash
cd OFGQ_quant

python analysis/compute_rcons.py \
    --config configs/rcons_llama_example.yaml
```

This script evaluates the column-block distribution consistency metric used to motivate column-block-level format adaptation.

### Numerical Error Analysis

Run:

```bash
python analysis/numerical_error_eval.py \
    --config configs/numerical_error_llama_example.yaml
```

The script evaluates both local and cascaded numerical errors for:

```text
E3M0
E2M1
E1M2
OFGQ
```

More details are provided in `OFGQ_quant/README.md`.

---

## RTL Implementation

The core hardware modules are provided in:

```text
OFGQ_rtl/
```

---

## Performance and Energy Simulation

The system-level simulator is located in:

```text
OFGQ_sim/
```

Detailed reproduction commands and hardware configuration parameters are provided in:

```text
OFGQ_sim/README.md
```
---

## Reproducing Paper Results

The main experiments can be mapped to the repository as follows:

| Experiment | Code |
|---|---|
| Offline FP4 format selection | `OFGQ_quant/scripts/calibrate.py` |
| WikiText-2 PPL | `OFGQ_quant/scripts/evaluate_ppl.py` |
| Column-block distribution consistency | `OFGQ_quant/analysis/compute_rcons.py` |
| Numerical error analysis | `OFGQ_quant/analysis/numerical_error_eval.py` |
| Accelerator RTL | `OFGQ_rtl/` |
| Performance and energy evaluation | `OFGQ_sim/` |
| Figures 15–18 | `OFGQ_sim/README.md` |
| Table 13 | `OFGQ_sim/README.md` |

---

## Citation

If you find this work useful in your research, please cite our paper:

```bibtex
@article{ofgq,
  title   = {Hardware-Software Co-Design for Outlier-Aware Format-Adaptive Group Quantization of LLMs},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

The complete citation information will be updated after publication.

---

## License

License information will be added in the final release.

The model weights and datasets used in the experiments are not distributed with this repository and remain subject to their respective licenses.

---

## Acknowledgements

We thank the authors and developers of the open-source models, datasets, and software frameworks used in this project.
