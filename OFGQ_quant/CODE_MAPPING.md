# Mapping from the supplied experimental scripts

This file documents how the original files were reorganized.

| Supplied file | Clean repository destination |
|---|---|
| `quantize_bestfp4_result.py` | `ofgq/quantization/fp4.py` |
| `quantize_int4.py` | `ofgq/quantization/int4.py` |
| `generate_bestfp4_result.py` | `scripts/calibrate.py`, `ofgq/calibration/*`, `ofgq/models/llama.py` |
| `bestfp4_result_mant.py` | `scripts/evaluate_ppl.py`, `ofgq/models/llama.py` |

## Majority vote

The calibration implementation intentionally keeps the behavior requested for the paper code:

```text
sample 1 -> per-column-block MSE winner
sample 2 -> per-column-block MSE winner
...
sample N -> per-column-block MSE winner
                 |
                 v
      count format 0 / 1 / 2
                 |
                 v
          majority vote
                 |
                 v
      fixed offline format index
```

The intermediate text-file parsing in the original scripts has been replaced with direct structured JSON metadata. This changes only the storage format, not the majority-vote rule.
