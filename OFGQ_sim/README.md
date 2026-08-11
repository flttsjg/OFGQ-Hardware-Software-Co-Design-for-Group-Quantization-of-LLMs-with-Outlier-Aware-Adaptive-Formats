
CSV Reproduction Commands for Figures 15--18 and Table 13 in the OFGQ Paper
==========================================================

Prerequisites
-------------
1. Run all commands from the OFGQ_sim root directory.
2. Obtain the layer-wise LINEAR/Q/A format-selection results for all models to be evaluated, and merge them into one CSV file according to format_selection_schema.csv.
3. This public release generates only source-data CSV files for the figures and table. It does not generate plots.

CACTI and SRAM Parameter Notes
------------------------------
1. This repository does not include, build, or invoke CACTI. The code is a system-level analytical simulation framework. Reproducing Figures 15--18 and Table 13 with the commands below does not require a CACTI installation.
2. The default SRAM read and write energy values were previously obtained by the author using CACTI outside this repository. This public repository retains only the resulting numerical values. It does not include CACTI source code, executables, or a procedure for rerunning CACTI.
3. If results from another memory-characterization tool or hardware measurements are available, the SRAM energy values can be overridden at runtime. SRAM capacities, bank counts, port counts, and the minimum write granularity can also be changed through the corresponding command-line options.
4. This public release does not re-characterize memory-macro area, access time, cycle time, or read and write energy from the SRAM technology, capacity, and organization. To investigate a new physical SRAM configuration, use a legally obtained memory-characterization tool independently and then supply the applicable energy parameters to this simulator. Memory-macro area and circuit-level timing are outside the output scope of this framework.


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