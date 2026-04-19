"""Correctness + microbench for the sparse sliding-window Triton decode kernel.

Compared against:
  - PyTorch reference (FP32 softmax after gather+slice).
  - FlashInfer BatchDecodeWithPagedKVCacheWrapper with window_left=window-1
    (dense KV read, masked softmax).

Gemma4 sliding layer config:
  batch=16, head_dim=128, num_kv_heads=2, num_q_heads=8, window=4096.
Sweep seq_len ∈ {4096, 8192, 16384}.
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
import flashinfer

from swa_decode import swa_decode_attention, bf16_swa_decode_ref


def _fill_paged_kv(batch, seq_len, num_kv_heads, head_dim, page_size, device, dtype):
    n_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch * n_pages_per_seq + 4
    k_cache = (torch.randn(total_pages, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device) * 0.1).contiguous()
    v_cache = (torch.randn(total_pages, page_size, num_kv_heads, head_dim,
                           dtype=dtype, device=device) * 0.1).contiguous()
    block_table = torch.zeros(batch, n_pages_per_seq, dtype=torch.int32, device=device)
    for b in range(batch):
        for i in range(n_pages_per_seq):
            block_table[b, i] = b * n_pages_per_seq + i
    return k_cache, v_cache, block_table


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1).unsqueeze(0),
        b.float().reshape(-1).unsqueeze(0)).item()


def correctness(batch, seq_len, head_dim, num_q_heads, num_kv_heads, window,
                page_size=16, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(batch, num_q_heads, head_dim, dtype=dtype, device=device) * 0.1
    k_cache, v_cache, block_table = _fill_paged_kv(
        batch, seq_len, num_kv_heads, head_dim, page_size, device, dtype)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    out_triton = swa_decode_attention(q, k_cache, v_cache, block_table, seq_lens,
                                      window=window)

    out_ref = bf16_swa_decode_ref(q, k_cache, v_cache, block_table, seq_lens,
                                  window=window)

    cos = _cos(out_triton, out_ref)
    max_abs = (out_triton.float() - out_ref.float()).abs().max().item()
    return cos, max_abs


def _flashinfer_sliding_decode(q, k_cache, v_cache, block_table, seq_lens,
                               num_q_heads, num_kv_heads, head_dim, page_size,
                               window):
    """FlashInfer decode with sliding-window mask (dense KV, masked softmax)."""
    batch = q.shape[0]
    device = q.device
    # Build indptr / indices from block_table + seq_lens.
    n_pages_per = (seq_lens + page_size - 1) // page_size  # [batch]
    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(n_pages_per, dim=0).to(torch.int32)
    # Flat page indices.
    indices_list = []
    for b in range(batch):
        indices_list.append(block_table[b, :int(n_pages_per[b].item())])
    indices = torch.cat(indices_list).to(torch.int32).contiguous()
    last_page_len = ((seq_lens - 1) % page_size + 1).to(torch.int32)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")

    # FlashInfer expects kv_cache packed [num_blocks, 2, page_size, num_kv_heads, head_dim]
    # OR separate k/v. The "NHD" kv_layout implies [max_num_pages, page_size, num_kv_heads, head_dim]
    # when k and v are passed as a tuple.
    wrapper.plan(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        window_left=window - 1,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )
    return wrapper, indptr, indices, last_page_len


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


def run_shape(batch, seq_len, head_dim, num_q_heads, num_kv_heads, window,
              page_size=16, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(batch, num_q_heads, head_dim, dtype=dtype, device=device) * 0.1
    k_cache, v_cache, block_table = _fill_paged_kv(
        batch, seq_len, num_kv_heads, head_dim, page_size, device, dtype)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)

    # Correctness vs PyTorch ref.
    out_triton = swa_decode_attention(q, k_cache, v_cache, block_table, seq_lens,
                                      window=window)
    out_ref = bf16_swa_decode_ref(q, k_cache, v_cache, block_table, seq_lens,
                                  window=window)
    cos_ref = _cos(out_triton, out_ref)
    abs_ref = (out_triton.float() - out_ref.float()).abs().max().item()

    # FlashInfer dense+mask run (apples-to-apples).
    try:
        wrapper, indptr, indices, last_page_len = _flashinfer_sliding_decode(
            q, k_cache, v_cache, block_table, seq_lens,
            num_q_heads, num_kv_heads, head_dim, page_size, window)
        kv_cache_tuple = (k_cache, v_cache)

        def run_fi():
            wrapper.run(q, kv_cache_tuple)

        out_fi = wrapper.run(q, kv_cache_tuple)
        cos_fi = _cos(out_triton, out_fi)
        abs_fi = (out_triton.float() - out_fi.float()).abs().max().item()

        t_fi = bench(run_fi)
    except Exception as e:
        print(f"  [FlashInfer run failed: {type(e).__name__}: {e}]")
        cos_fi = float("nan")
        abs_fi = float("nan")
        t_fi = float("nan")

    def run_triton():
        swa_decode_attention(q, k_cache, v_cache, block_table, seq_lens, window=window)

    t_triton = bench(run_triton)

    # Theoretical bandwidth: KV bytes touched per decode step.
    # Sparse: 2 (K+V) * batch * num_kv_heads * head_dim * window * 2 (BF16) bytes.
    eff_len = min(seq_len, window)
    kv_bytes_sparse = 2 * batch * num_kv_heads * head_dim * eff_len * 2
    # Dense-masked: 2 * batch * num_kv_heads * head_dim * seq_len * 2.
    kv_bytes_dense = 2 * batch * num_kv_heads * head_dim * seq_len * 2

    # Effective HBM bandwidth in GB/s.
    bw_triton = kv_bytes_sparse / (t_triton * 1e-3) / 1e9
    bw_fi = kv_bytes_dense / (t_fi * 1e-3) / 1e9 if not math.isnan(t_fi) else float("nan")

    # Per-SM peak BW for RTX PRO 6000 Blackwell ~1700 GB/s (GDDR7).
    pct_bw_triton = bw_triton / 1700.0 * 100.0
    speedup = t_fi / t_triton if not math.isnan(t_fi) else float("nan")

    print(f"\nShape: B={batch} seq_len={seq_len} head_dim={head_dim} "
          f"num_q={num_q_heads} num_kv={num_kv_heads} window={window}")
    print(f"  Correctness vs PyTorch ref: cos={cos_ref:.6f}  max_abs={abs_ref:.4e}")
    print(f"  Correctness vs FlashInfer:  cos={cos_fi:.6f}  max_abs={abs_fi:.4e}")
    print(f"  Triton SWA sparse : {t_triton*1000:6.1f} us   "
          f"effBW(sparse work)={bw_triton:7.1f} GB/s ({pct_bw_triton:4.1f}% peak)")
    print(f"  FlashInfer dense+mask: {t_fi*1000:6.1f} us   "
          f"effBW(dense work)={bw_fi:7.1f} GB/s")
    print(f"  Speedup vs FlashInfer: {speedup:.2f}x   "
          f"(theoretical: {seq_len/eff_len:.2f}x)")

    return {
        "shape": (batch, seq_len, head_dim, num_q_heads, num_kv_heads, window),
        "cos_ref": cos_ref,
        "cos_fi": cos_fi,
        "abs_ref": abs_ref,
        "abs_fi": abs_fi,
        "t_triton_us": t_triton * 1000,
        "t_fi_us": t_fi * 1000,
        "bw_triton_gbs": bw_triton,
        "bw_fi_gbs": bw_fi,
        "pct_bw_triton": pct_bw_triton,
        "speedup": speedup,
    }


def main():
    dev = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES will remap; caller pins GPU 1.
    print(f"Device: {torch.cuda.get_device_name(dev)}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Gemma4 sliding-layer config.
    base = dict(batch=16, head_dim=128, num_q_heads=8, num_kv_heads=2, window=4096)

    results = []
    # Primary kill-gate shape first.
    results.append(run_shape(seq_len=8192, **base))
    # Amplification shape.
    results.append(run_shape(seq_len=16384, **base))
    # Baseline shape where window == seq_len (no sparsity win, sanity check).
    results.append(run_shape(seq_len=4096, **base))

    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  seq={r['shape'][1]:>5} win={r['shape'][5]}: "
              f"cos_ref={r['cos_ref']:.5f}  "
              f"t_triton={r['t_triton_us']:6.1f}us  "
              f"t_fi={r['t_fi_us']:6.1f}us  "
              f"speedup={r['speedup']:.2f}x")

    # Kill-gate check: primary shape = seq=8192 window=4096.
    primary = results[0]
    passed = (primary["cos_ref"] > 0.999 and primary["speedup"] > 1.3)
    print(f"\nKill gate (seq=8192 win=4096 >=1.3x, cos>0.999): "
          f"{'PASS' if passed else 'KILL'}")

    return results, passed


if __name__ == "__main__":
    main()
