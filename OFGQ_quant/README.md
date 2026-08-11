# OFGQ Algorithm Code

This repository contains a cleaned implementation of the OFGQ software flow used in the provided experimental scripts.

The repository intentionally keeps model and dataset paths outside the source code. Edit `configs/llama_example.yaml` before running.

## Repository structure

```text
OFGQ_GitHub_ready/
├── configs/
│   └── llama_example.yaml
├── ofgq/
│   ├── calibration/
│   │   ├── collector.py
│   │   └── majority_vote.py
│   ├── models/
│   │   └── llama.py
│   ├── quantization/
│   │   ├── fp4.py
│   │   └── int4.py
│   └── utils/
│       ├── config.py
│       └── data.py
├── scripts/
│   ├── calibrate.py
│   └── evaluate_ppl.py
├── analysis/
│   ├── compute_rcons.py
│   ├── numerical_error_eval.py
│   └── README.md
├── outputs/
├── pyproject.toml
└── requirements.txt
```

## 1. Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Prepare local model and dataset

Download the Hugging Face model and WikiText-2 parquet files yourself.

Then edit:

```text
configs/llama_example.yaml
```

At minimum, set:

```yaml
model:
  path: "/your/local/model/path"

dataset:
  calibration_path: "/your/local/calibration.parquet"
  evaluation_path: "/your/local/test.parquet"
```

## 3. Offline calibration

```bash
python scripts/calibrate.py --config configs/llama_example.yaml
```

This creates:

```text
outputs/best_formats.json
outputs/best_formats.txt
```

`best_formats.json` is used directly during fixed-format inference.  
`best_formats.txt` is a human-readable summary of counts, percentages, and majority-vote results.

## 4. Evaluate WikiText-2 perplexity

```bash
python scripts/evaluate_ppl.py --config configs/llama_example.yaml
```

The evaluation script loads the fixed format metadata specified by:

```yaml
evaluation:
  metadata_path: "outputs/best_formats.json"
```


## 5. Column-block distribution consistency (R_cons)

The repository also includes the analysis code used to quantify the
column-block distribution consistency observation. The final-paper analysis
uses `X_qkv`, `Q`, `X_ao`, `X_fc1`, and `X_fc2`.

Edit the local paths in:

```text
configs/rcons_llama_example.yaml
```

Then run:

```bash
python analysis/compute_rcons.py --config configs/rcons_llama_example.yaml
```

The default analysis configuration uses 100 randomly sampled 2048-token
sequences, group size 64, and all decoder layers. The model layer count is
detected automatically. Results are written to `outputs/rcons/`.

## 6. Numerical error analysis

Edit `configs/numerical_error_llama_example.yaml` and set the local model, dataset, and metadata paths.

Run:

```bash
python analysis/numerical_error_eval.py --config configs/numerical_error_llama_example.yaml
```

The script reports local decoder-input MSE and cascaded decoder-output MSE for E3M0, E2M1, E1M2, and OFGQ. Results are written to `outputs/numerical_error/`. Layer numbers in the YAML file are one-based.

## Configuration

Important settings are all in YAML rather than hard-coded in Python:

```yaml
model:
  path:
  dtype:
  device_map:

dataset:
  calibration_path:
  evaluation_path:
  text_column:

quantization:
  num_bits:
  group_size:

calibration:
  num_samples:
  sequence_length:
  output_json:
  output_txt:

evaluation:
  sequence_length:
  max_samples:
  metadata_path:
```
