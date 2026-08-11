# OFGQ-Hardware-Software-Co-Design-for-Outlier-Aware-Format-Adaptive-Group-Quantization-of-LLM
## Environment

The code has been tested with the following environment:

- Python 3.9
- PyTorch 2.6.0
- CUDA 12.4
- Transformers 4.52.4
- Datasets 4.0.0
- Accelerate 1.8.1
- NumPy 2.0.2
- PyYAML 6.0.2
- tqdm 4.67.1

Hardware used for testing:

- NVIDIA GeForce RTX 4090 D
- NVIDIA Driver 580.76.05

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


CSV Reproduction Commands for Figures 15--18 and Table 13 in the OFGQ Paper
==========================================================

Common Parameters
-----------------
$FORMAT = "D:\private\ofgq_format_selection.csv"
$LPDDR5X = @(
  "--offchip-bandwidth-gbps", "68.3",
  "--offchip-energy-pj-per-bit", "4.0"
) + $QUANT
$HBM2E = @(
  "--offchip-bandwidth-gbps", "1640",
  "--offchip-energy-pj-per-bit", "3.9"
) + $QUANT

Figure 15: LPDDR5X, B=1, Prefill, L=2048, Six Models
----------------------------------------------------
$FIG15_MODELS = @("opt-6.7b", "opt-13b", "llama1-7b", "llama1-13b", "llama2-7b", "llama2-13b")
python .\main_ant.py    @LPDDR5X --model @FIG15_MODELS --mode prefill --sequence-length 2048 --output-dir .\reproduction_outputs\figure15
python .\main_olive.py  @LPDDR5X --model @FIG15_MODELS --mode prefill --sequence-length 2048 --output-dir .\reproduction_outputs\figure15
python .\main_tender.py @LPDDR5X --model @FIG15_MODELS --mode prefill --sequence-length 2048 --output-dir .\reproduction_outputs\figure15
python .\main_ofgq.py   @LPDDR5X --model @FIG15_MODELS --mode prefill --sequence-length 2048 --runtime-quantization    --output-dir .\reproduction_outputs\figure15
python .\main_ofgq.py   @LPDDR5X --model @FIG15_MODELS --mode prefill --sequence-length 2048 --no-runtime-quantization --output-dir .\reproduction_outputs\figure15
python -m tools.combine_results --add-average --inputs .\reproduction_outputs\figure15\ant_prefill_summary.csv .\reproduction_outputs\figure15\olive_prefill_summary.csv .\reproduction_outputs\figure15\tender_prefill_summary.csv .\reproduction_outputs\figure15\ofgq-rq_prefill_summary.csv .\reproduction_outputs\figure15\ofgq_prefill_summary.csv --output .\reproduction_outputs\figure15\figure15_source.csv

Final source data: .\reproduction_outputs\figure15\figure15_source.csv
Per-operation results: .\reproduction_outputs\figure15\*_prefill_layers.csv

Figure 16: LPDDR5X, LLaMA-1-7B, Prefill, L=2K/8K/16K
-------------------------------------------------------
python .\main_ant.py    @LPDDR5X --model llama1-7b --mode prefill --sequence-length 2048 8192 16384 --output-dir .\reproduction_outputs\figure16
python .\main_olive.py  @LPDDR5X --model llama1-7b --mode prefill --sequence-length 2048 8192 16384 --output-dir .\reproduction_outputs\figure16
python .\main_tender.py @LPDDR5X --model llama1-7b --mode prefill --sequence-length 2048 8192 16384 --output-dir .\reproduction_outputs\figure16
python .\main_ofgq.py   @LPDDR5X --model llama1-7b --mode prefill --sequence-length 2048 8192 16384 --no-runtime-quantization --output-dir .\reproduction_outputs\figure16
python -m tools.combine_results --inputs .\reproduction_outputs\figure16\ant_prefill_summary.csv .\reproduction_outputs\figure16\olive_prefill_summary.csv .\reproduction_outputs\figure16\tender_prefill_summary.csv .\reproduction_outputs\figure16\ofgq_prefill_summary.csv --output .\reproduction_outputs\figure16\figure16_source.csv

Final source data: .\reproduction_outputs\figure16\figure16_source.csv
Per-operation results: .\reproduction_outputs\figure16\*_prefill_layers.csv

Figure 17: LPDDR5X, B=1, Single-Token Decode, KV Sequence Length=2048, Six-Model Average
-----------------------------------------------------------------------------------------
$FIG17_MODELS = $FIG15_MODELS
python .\main_ant.py    @LPDDR5X --model @FIG17_MODELS --mode decode --sequence-length 2048 --output-dir .\reproduction_outputs\figure17
python .\main_olive.py  @LPDDR5X --model @FIG17_MODELS --mode decode --sequence-length 2048 --output-dir .\reproduction_outputs\figure17
python .\main_tender.py @LPDDR5X --model @FIG17_MODELS --mode decode --sequence-length 2048 --output-dir .\reproduction_outputs\figure17
python .\main_ofgq.py   @LPDDR5X --model @FIG17_MODELS --mode decode --sequence-length 2048 --no-runtime-quantization --output-dir .\reproduction_outputs\figure17
python -m tools.combine_results --add-average --inputs .\reproduction_outputs\figure17\ant_decode_summary.csv .\reproduction_outputs\figure17\olive_decode_summary.csv .\reproduction_outputs\figure17\tender_decode_summary.csv .\reproduction_outputs\figure17\ofgq_decode_summary.csv --output .\reproduction_outputs\figure17\figure17_source.csv

Final source data: .\reproduction_outputs\figure17\figure17_source.csv
Note: The row with model=average in figure17_source.csv is the arithmetic mean across the six models.

Figure 18: HBM2E, B=1, Prefill, L=2048, LLaMA-1-13B/30B/65B
----------------------------------------------------------------
$FIG18_MODELS = @("llama1-13b", "llama1-30b", "llama1-65b")
python .\main_ant.py    @HBM2E --model @FIG18_MODELS --mode prefill --sequence-length 2048 --output-dir .\reproduction_outputs\figure18
python .\main_olive.py  @HBM2E --model @FIG18_MODELS --mode prefill --sequence-length 2048 --output-dir .\reproduction_outputs\figure18
python .\main_tender.py @HBM2E --model @FIG18_MODELS --mode prefill --sequence-length 2048 --output-dir .\reproduction_outputs\figure18
python .\main_ofgq.py   @HBM2E --model @FIG18_MODELS --mode prefill --sequence-length 2048 --runtime-quantization    --output-dir .\reproduction_outputs\figure18
python .\main_ofgq.py   @HBM2E --model @FIG18_MODELS --mode prefill --sequence-length 2048 --no-runtime-quantization --output-dir .\reproduction_outputs\figure18
python -m tools.combine_results --add-average --inputs .\reproduction_outputs\figure18\ant_prefill_summary.csv .\reproduction_outputs\figure18\olive_prefill_summary.csv .\reproduction_outputs\figure18\tender_prefill_summary.csv .\reproduction_outputs\figure18\ofgq-rq_prefill_summary.csv .\reproduction_outputs\figure18\ofgq_prefill_summary.csv --output .\reproduction_outputs\figure18\figure18_source.csv

Final source data: .\reproduction_outputs\figure18\figure18_source.csv
Per-operation results: .\reproduction_outputs\figure18\*_prefill_layers.csv

Table 13: LPDDR5X, First Layer of LLaMA-2-7B, Prefill, L=2048
----------------------------------------------------------------
The Figure 15 commands generate the per-operation CSV files required for Table 13 under the same configuration. Aggregate them directly:
python -m tools.build_table13 --inputs .\reproduction_outputs\figure15\ant_prefill_layers.csv .\reproduction_outputs\figure15\olive_prefill_layers.csv .\reproduction_outputs\figure15\tender_prefill_layers.csv .\reproduction_outputs\figure15\ofgq-rq_prefill_layers.csv --model llama2-7b --sequence-length 2048 --output .\reproduction_outputs\table13\table13.csv

Final output: .\reproduction_outputs\table13\table13.csv
Expected four rows, rounded as in the paper:
ANT      Compute 214.09 ms, DRAM 101.89 ms, Layer 243.60 ms
OLIVE    Compute 190.56 ms, DRAM 101.89 ms, Layer 228.46 ms
TENDER   Compute 170.96 ms, DRAM 101.89 ms, Layer 208.86 ms
OFGQ-RQ  Compute  58.49 ms, Quantization 765 ns, DRAM 51.66 ms, Layer 78.86 ms

Note: Compute, Quantization, and DRAM overlap in the pipeline. Therefore, the three components cannot be added directly to obtain Layer.