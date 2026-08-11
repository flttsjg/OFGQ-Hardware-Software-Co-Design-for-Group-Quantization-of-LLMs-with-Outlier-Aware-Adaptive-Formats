from .fp4 import quantize_tensor as quantize_fp4
from .int4 import quantize_tensor as quantize_int4

__all__ = ["quantize_fp4", "quantize_int4"]
