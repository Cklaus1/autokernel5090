"""NVFP4Linear: drop-in nn.Linear replacement with FP4 weight storage.

Stores weights in packed FP4 format (~4x compression vs FP16).
Uses NVFP4 tensor cores for M >= 128 (prefill), FP16 cuBLAS for M < 128 (decode).

M-split optimization: for M >= 512, splits along M dimension and runs two
concurrent GEMMs on separate CUDA streams for ~2x throughput via SM overlap.

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

# Module-level CUDA streams for M-split (shared across all NVFP4Linear instances)
_stream1 = None
_stream2 = None

# Minimum M for M-split (M_half must be >= 128 for quantization padding)
_MSPLIT_MIN_M = 512


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

        if cache_fp16:
            mod.w_fp16_cache = dequantize_nvfp4(w_fp4, w_scale, N, K, 16).half()

        if linear.bias is not None:
            mod.bias_p = nn.Parameter(linear.bias.data.cuda())

        del w_gpu
        return mod

    def _get_fp16_weights(self) -> torch.Tensor:
        """Get FP16 weights, using cache if available."""
        if self.w_fp16_cache is not None:
            return self.w_fp16_cache
        return dequantize_nvfp4(self.w_fp4, self.w_scale, self.N, self.K, 16).half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global _stream1, _stream2
        shape = x.shape
        x = x.reshape(-1, self.K).contiguous().half()
        M = x.shape[0]

        if M < 128:
            # Small M: FP16 cuBLAS (NVFP4 scale padding fails for M < 128)
            out = torch.nn.functional.linear(x, self._get_fp16_weights())
            if self.bias_p is not None:
                out = out + self.bias_p
        elif M >= _MSPLIT_MIN_M:
            # Large M: M-split with concurrent CUDA streams (~2x throughput)
            if _stream1 is None:
                _stream1 = torch.cuda.Stream()
                _stream2 = torch.cuda.Stream()

            M_half = M // 2
            w_t = self.w_fp4.t()
            out = torch.empty(M, self.N, device=x.device, dtype=torch.float16)
            _default = torch.cuda.current_stream()

            # Pipeline: quantize + GEMM on each stream
            _stream1.wait_stream(_default)
            _stream2.wait_stream(_default)

            torch.cuda.set_stream(_stream1)
            a1_fp4, a1_scale = quantize_to_nvfp4_fast(x[:M_half], block_size=16)
            out[:M_half] = torch._scaled_mm(
                a1_fp4, w_t, scale_a=a1_scale, scale_b=self.w_scale,
                out_dtype=torch.float16,
            )

            torch.cuda.set_stream(_stream2)
            a2_fp4, a2_scale = quantize_to_nvfp4_fast(x[M_half:], block_size=16)
            out[M_half:] = torch._scaled_mm(
                a2_fp4, w_t, scale_a=a2_scale, scale_b=self.w_scale,
                out_dtype=torch.float16,
            )

            torch.cuda.set_stream(_default)
            _default.wait_stream(_stream1)
            _default.wait_stream(_stream2)

            if self.bias_p is not None:
                out = out + self.bias_p
        else:
            # Medium M (128-511): single NVFP4 GEMM
            a_fp4, a_scale = quantize_to_nvfp4_fast(x, block_size=16)
            out = torch._scaled_mm(
                a_fp4,
                self.w_fp4.t(),
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
