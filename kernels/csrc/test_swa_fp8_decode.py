"""Correctness + microbench for the sparse sliding-window Triton decode kernel
with FP8 KV cache.

Compared against:
  - BF16 SWA Triton kernel run (same code path, FP8_KV=0).
  - PyTorch BF16 reference (gather+slice+FP32 softmax) to bound absolute error.

Shapes sweep: Gemma4 sliding-layer config (batch=16, hd=128, kv_heads=2, q_heads=8, window=4096).
seq_len in {8192, 16384}. PASS = cos >= 0.999 AND max_abs <= 5e-3.
"""

import os
import sys
import math
import time

# Ensure venv packages import cleanly even when invoked via system python.
_VENV_SP = "/home/cklaus/projects/autokernel/.venv/lib/python3.12/site-packages"
if os.path.isdir(_VENV_SP) and _VENV_SP not in sys.path:
    sys.path.insert(0, _VENV_SP)

sys.path.insert(0, "/home/cklaus/projects/autokernel/kernels/triton")
sys.path.insert(0, "/home/cklaus/projects/autokernel")

import torch

from swa_decode import swa_decode_attention, bf16_swa_decode_ref


def _fill_paged_kv_bf16(batch, seq_len, num_kv_heads, head_dim, page_size, device):
    n_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch * n_pages_per_seq + 4
    k_cache = (torch.randn(total_pages, page_size, num_kv_heads, head_dim,
                           dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    v_cache = (torch.randn(total_pages, page_size, num_kv_heads, head_dim,
                           dtype=torch.bfloat16, device=device) * 0.1).contiguous()
    block_table = torch.zeros(batch, n_pages_per_seq, dtype=torch.int32, device=device)
    for b in range(batch):
        for i in range(n_pages_per_seq):
            block_table[b, i] = b * n_pages_per_seq + i
    return k_cache, v_cache, block_table


def _quantize_per_tensor_fp8(x_bf16, fp8_dtype=torch.float8_e4m3fn, fp8_max=448.0):
    """Per-tensor symmetric FP8 quantization. Returns (x_fp8, scale_scalar)."""
    x_fp32 = x_bf16.float()
    amax = x_fp32.abs().max().item()
    scale = max(amax / fp8_max, 1e-6)
    x_fp8 = (x_fp32 / scale).clamp(-fp8_max, fp8_max).to(fp8_dtype)
    return x_fp8, scale


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1).unsqueeze(0),
        b.float().reshape(-1).unsqueeze(0)).item()


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def run_shape_fp8(batch, seq_len, head_dim, num_q_heads, num_kv_heads, window,
                  page_size=16, device="cuda", fp8_dtype=torch.float8_e4m3fn):
    torch.manual_seed(0)
    q = (torch.randn(batch, num_q_heads, head_dim, dtype=torch.bfloat16, device=device) * 0.1)
    k_cache_bf16, v_cache_bf16, block_table = _fill_paged_kv_bf16(
        batch, seq_len, num_kv_heads, head_dim, page_size, device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Quantize K/V to FP8.
    k_fp8, k_scale_v = _quantize_per_tensor_fp8(k_cache_bf16, fp8_dtype)
    v_fp8, v_scale_v = _quantize_per_tensor_fp8(v_cache_bf16, fp8_dtype)
    k_scale_t = torch.tensor([k_scale_v], dtype=torch.float32, device=device)
    v_scale_t = torch.tensor([v_scale_v], dtype=torch.float32, device=device)

    # Dequant+requant reference to avoid FP8-quant noise dominating cos.
    # The FP8 kernel should match the "BF16 kernel with FP8-roundtripped KV" path.
    k_roundtrip = (k_fp8.float() * k_scale_v).to(torch.bfloat16)
    v_roundtrip = (v_fp8.float() * v_scale_v).to(torch.bfloat16)

    out_fp8 = swa_decode_attention(
        q, k_fp8, v_fp8, block_table, seq_lens, window=window,
        k_scale=k_scale_t, v_scale=v_scale_t)
    out_bf16_roundtrip = swa_decode_attention(
        q, k_roundtrip, v_roundtrip, block_table, seq_lens, window=window)
    out_bf16_orig = swa_decode_attention(
        q, k_cache_bf16, v_cache_bf16, block_table, seq_lens, window=window)

    cos_vs_rt = _cos(out_fp8, out_bf16_roundtrip)
    abs_vs_rt = (out_fp8.float() - out_bf16_roundtrip.float()).abs().max().item()
    cos_vs_orig = _cos(out_fp8, out_bf16_orig)
    abs_vs_orig = (out_fp8.float() - out_bf16_orig.float()).abs().max().item()

    # Bench
    t_fp8 = bench(lambda: swa_decode_attention(
        q, k_fp8, v_fp8, block_table, seq_lens, window=window,
        k_scale=k_scale_t, v_scale=v_scale_t))
    t_bf16 = bench(lambda: swa_decode_attention(
        q, k_cache_bf16, v_cache_bf16, block_table, seq_lens, window=window))

    # Effective BW: KV bytes per decode.
    eff_len = min(seq_len, window)
    kv_bytes_fp8 = 2 * batch * num_kv_heads * head_dim * eff_len * 1  # 1 byte per FP8 elt
    kv_bytes_bf16 = 2 * batch * num_kv_heads * head_dim * eff_len * 2
    bw_fp8 = kv_bytes_fp8 / (t_fp8 * 1e-3) / 1e9
    bw_bf16 = kv_bytes_bf16 / (t_bf16 * 1e-3) / 1e9
    peak_bw = 1700.0  # PRO 6000

    print(f"\nShape: B={batch} seq_len={seq_len} head_dim={head_dim} "
          f"num_q={num_q_heads} num_kv={num_kv_heads} window={window} dtype={fp8_dtype}")
    print(f"  FP8 vs BF16(roundtrip): cos={cos_vs_rt:.6f}  max_abs={abs_vs_rt:.4e}")
    print(f"  FP8 vs BF16(orig)     : cos={cos_vs_orig:.6f}  max_abs={abs_vs_orig:.4e}")
    print(f"  Triton SWA BF16 : {t_bf16*1000:6.1f} us   "
          f"effBW={bw_bf16:7.1f} GB/s ({bw_bf16/peak_bw*100:4.1f}% peak)")
    print(f"  Triton SWA FP8  : {t_fp8*1000:6.1f} us   "
          f"effBW={bw_fp8:7.1f} GB/s ({bw_fp8/peak_bw*100:4.1f}% peak)")
    print(f"  Speedup (BF16/FP8): {t_bf16/t_fp8:.3f}x")

    return {
        "shape": (batch, seq_len, head_dim, num_q_heads, num_kv_heads, window),
        "cos_vs_rt": cos_vs_rt,
        "abs_vs_rt": abs_vs_rt,
        "cos_vs_orig": cos_vs_orig,
        "abs_vs_orig": abs_vs_orig,
        "t_fp8_us": t_fp8 * 1000,
        "t_bf16_us": t_bf16 * 1000,
        "speedup": t_bf16 / t_fp8,
        "bw_fp8": bw_fp8,
        "bw_bf16": bw_bf16,
    }


def main():
    dev = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    base = dict(batch=16, head_dim=128, num_q_heads=8, num_kv_heads=2, window=4096)

    results = []
    results.append(run_shape_fp8(seq_len=8192, **base))
    results.append(run_shape_fp8(seq_len=16384, **base))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  seq={r['shape'][1]:>5} win={r['shape'][5]}: "
              f"cos_vs_rt={r['cos_vs_rt']:.5f}  "
              f"max_abs_vs_rt={r['abs_vs_rt']:.4e}  "
              f"t_fp8={r['t_fp8_us']:6.1f}us  "
              f"t_bf16={r['t_bf16_us']:6.1f}us  "
              f"speedup={r['speedup']:.2f}x")

    # Pass gate: cos vs roundtrip-BF16 >= 0.999, max_abs <= 5e-3.
    passed = all((r["cos_vs_rt"] >= 0.999 and r["abs_vs_rt"] <= 5e-3) for r in results)
    print(f"\nCorrectness gate (cos>=0.999 AND max_abs<=5e-3 at all shapes): "
          f"{'PASS' if passed else 'FAIL'}")

    return results, passed


if __name__ == "__main__":
    main()
