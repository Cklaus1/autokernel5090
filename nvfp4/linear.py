"""NVFP4Linear: drop-in nn.Linear replacement with FP4 weight storage.

Stores weights in packed FP4 format (~4x compression vs FP16).
Uses NVFP4 tensor cores for M >= 128 (prefill), FP16 cuBLAS for M < 128 (decode).

Optimizations applied:
  - Pre-cached transposed FP4 weight view (avoids .t() per forward)
  - quantize_to_nvfp4_fast with CUDA v3 kernel (24µs vs 358µs Python)

Memory model:
  - w_fp4: [N, K//2] packed FP4 weights (permanent, ~4x smaller)
  - w_scale: flat FP8 block scales (permanent, small)
  - w_fp16_cache: [N, K] FP16 dequanted weights (lazy, for decode fallback)
    Set cache_fp16=False to skip this and dequant on-demand (saves memory,
    slower decode).
"""

import torch
import torch.nn as nn

from .quantize import quantize_to_nvfp4, quantize_to_nvfp4_fast, dequantize_nvfp4


class NVFP4Linear(nn.Module):
    """Linear layer with NVFP4 weight storage and hybrid compute."""

    def __init__(self, N: int, K: int, has_bias: bool, cache_fp16: bool = True):
        super().__init__()
        self.N = N
        self.K = K
        self.cache_fp16 = cache_fp16
        self.register_buffer("w_fp4", None)
        self.register_buffer("w_scale", None)
        self.register_buffer("w_fp16_cache", None)
        self.bias_p = nn.Parameter(torch.zeros(N)) if has_bias else None
        # Cached transposed view (set in from_linear or lazily)
        self._w_fp4_t = None

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, cache_fp16: bool = True
    ) -> "NVFP4Linear | None":
        """Convert an nn.Linear to NVFP4Linear.

        Returns None if the layer is incompatible (K not divisible by 16).
        """
        w = linear.weight.data
        N, K = w.shape
        if K % 16 != 0:
            return None

        mod = cls(N, K, linear.bias is not None, cache_fp16=cache_fp16)
        w_gpu = w.cuda().half()
        w_fp4, w_scale = quantize_to_nvfp4_fast(w_gpu, block_size=16)
        mod.w_fp4 = w_fp4
        mod.w_scale = w_scale
        mod._w_fp4_t = w_fp4.t()  # cache transposed view

        if cache_fp16:
            mod.w_fp16_cache = dequantize_nvfp4(w_fp4, w_scale, N, K, 16).half()

        if linear.bias is not None:
            mod.bias_p = nn.Parameter(linear.bias.data.cuda())

        del w_gpu
        return mod

    def _get_w_fp4_t(self) -> torch.Tensor:
        """Get transposed FP4 weight view, creating lazily if needed."""
        if self._w_fp4_t is None:
            self._w_fp4_t = self.w_fp4.t()
        return self._w_fp4_t

    def _get_fp16_weights(self) -> torch.Tensor:
        """Get FP16 weights, using cache if available."""
        if self.w_fp16_cache is not None:
            return self.w_fp16_cache
        return dequantize_nvfp4(self.w_fp4, self.w_scale, self.N, self.K, 16).half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, self.K).contiguous().half()
        M = x.shape[0]

        if M < 128:
            # Small M: FP16 cuBLAS (NVFP4 scale padding fails for M < 128)
            out = torch.nn.functional.linear(x, self._get_fp16_weights())
        else:
            # Large M: NVFP4 tensor cores
            a_fp4, a_scale = quantize_to_nvfp4_fast(x, block_size=16)
            out = torch._scaled_mm(
                a_fp4,
                self._get_w_fp4_t(),
                scale_a=a_scale,
                scale_b=self.w_scale,
                out_dtype=torch.float16,
            )

        if self.bias_p is not None:
            out = out + self.bias_p

        return out.reshape(*shape[:-1], self.N)


def convert_model(
    model: nn.Module,
    min_size: int = 64,
    cache_fp16: bool = True,
    verbose: bool = True,
) -> int:
    """Replace all compatible nn.Linear layers with NVFP4Linear.

    Args:
        model: PyTorch model to convert
        min_size: minimum weight dimension to convert
        cache_fp16: whether to cache FP16 weights for decode fallback
            Set False to save ~3x memory at cost of slower decode.
        verbose: print progress

    Returns:
        Number of layers converted
    """
    replacements = {}
    for name, m in model.named_modules():
        if (
            isinstance(m, nn.Linear)
            and min(m.weight.shape) >= min_size
            and m.weight.shape[1] % 16 == 0
        ):
            replacements[name] = m

    n = 0
    for name, old in replacements.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        new = NVFP4Linear.from_linear(old, cache_fp16=cache_fp16)
        if new is not None:
            setattr(parent, parts[-1], new)
            n += 1
            if verbose and n % 50 == 0:
                print(f"  {n}/{len(replacements)}...")
                torch.cuda.empty_cache()

    if verbose:
        print(f"  {n}/{len(replacements)} layers converted to NVFP4")
    return n
