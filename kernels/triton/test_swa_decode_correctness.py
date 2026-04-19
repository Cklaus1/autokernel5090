"""Standalone correctness harness for swa_decode_attention.

Compares BF16 and FP8 kernel outputs against the PyTorch FP32 reference
(bf16_swa_decode_ref) using cosine similarity >= 0.999 as the pass criterion.

Run on a GPU machine:
    python kernels/triton/test_swa_decode_correctness.py

CPU-only environments will skip GPU-dependent tests gracefully.

Tag: W5_B2_swa_numwarps_tune
"""

import math
import sys
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Conditional import: skip if not on a CUDA machine
# ---------------------------------------------------------------------------
try:
    from kernels.triton.swa_decode import swa_decode_attention, bf16_swa_decode_ref
except ImportError:
    # Allow running from repo root without package install
    import importlib.util, pathlib
    _spec = importlib.util.spec_from_file_location(
        "swa_decode",
        pathlib.Path(__file__).parent / "swa_decode.py",
    )
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    swa_decode_attention = _mod.swa_decode_attention
    bf16_swa_decode_ref = _mod.bf16_swa_decode_ref

DEVICE = "cuda"
COS_THRESHOLD = 0.999


def _make_paged_kv(batch, seq_lens, page_size, num_kv_heads, head_dim, dtype, device):
    """Allocate a random paged KV cache and block table for the given sequences."""
    max_pages_needed = sum((sl + page_size - 1) // page_size for sl in seq_lens)
    # Add headroom so page IDs never collide
    pool_size = max_pages_needed + batch * 2

    k_cache = torch.randn(pool_size, page_size, num_kv_heads, head_dim, device=device)
    v_cache = torch.randn(pool_size, page_size, num_kv_heads, head_dim, device=device)

    if dtype == torch.bfloat16:
        k_cache = k_cache.bfloat16()
        v_cache = v_cache.bfloat16()
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        # Scale to [-1, 1] range before casting to FP8 to avoid saturation
        k_cache = (k_cache / k_cache.abs().max()).to(dtype)
        v_cache = (v_cache / v_cache.abs().max()).to(dtype)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")

    max_pages = max((sl + page_size - 1) // page_size for sl in seq_lens)
    block_table = torch.zeros(batch, max_pages, dtype=torch.int32, device=device)
    page_counter = 0
    for b, sl in enumerate(seq_lens):
        n_pages = (sl + page_size - 1) // page_size
        for p in range(n_pages):
            block_table[b, p] = page_counter
            page_counter += 1

    return k_cache, v_cache, block_table


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return float(F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item())


def run_case(
    batch, num_q_heads, num_kv_heads, head_dim, page_size, seq_lens, window,
    logits_soft_cap=0.0, fp8=False, label="",
):
    device = DEVICE
    kv_group_size = num_q_heads // num_kv_heads
    q = torch.randn(batch, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)

    kv_dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    k_cache, v_cache, block_table = _make_paged_kv(
        batch, seq_lens, page_size, num_kv_heads, head_dim, kv_dtype, device
    )
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)

    sm_scale = 1.0 / math.sqrt(head_dim)

    if fp8:
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        out_kernel = swa_decode_attention(
            q, k_cache, v_cache, block_table, seq_lens_t,
            window, sm_scale=sm_scale, logits_soft_cap=logits_soft_cap,
            k_scale=k_scale, v_scale=v_scale,
        )
        # For reference, dequantize FP8 -> BF16 (scale=1.0 so lossless within FP8 range)
        k_bf16 = k_cache.to(torch.bfloat16)
        v_bf16 = v_cache.to(torch.bfloat16)
        out_ref = bf16_swa_decode_ref(
            q, k_bf16, v_bf16, block_table, seq_lens_t,
            window, sm_scale=sm_scale, logits_soft_cap=logits_soft_cap,
        )
    else:
        out_kernel = swa_decode_attention(
            q, k_cache, v_cache, block_table, seq_lens_t,
            window, sm_scale=sm_scale, logits_soft_cap=logits_soft_cap,
        )
        out_ref = bf16_swa_decode_ref(
            q, k_cache, v_cache, block_table, seq_lens_t,
            window, sm_scale=sm_scale, logits_soft_cap=logits_soft_cap,
        )

    cos = _cosine(out_kernel, out_ref)
    status = "PASS" if cos >= COS_THRESHOLD else "FAIL"
    tag = f"fp8={fp8} cap={logits_soft_cap}"
    print(f"  [{status}] {label} | {tag} | cos={cos:.6f}")
    assert cos >= COS_THRESHOLD, (
        f"Cosine {cos:.6f} < {COS_THRESHOLD} for case: {label} {tag}"
    )


def main():
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device available (CPU-only environment).")
        sys.exit(0)

    torch.manual_seed(42)
    print(f"Running SWA decode correctness harness on {torch.cuda.get_device_name(0)}")
    print(f"Pass threshold: cos >= {COS_THRESHOLD}\n")

    # --- BF16 cases ---
    print("=== BF16 KV path ===")
    # Short sequence, window covers all
    run_case(
        batch=2, num_q_heads=8, num_kv_heads=2, head_dim=128,
        page_size=16, seq_lens=[64, 32], window=4096,
        label="short-seq full-window",
    )
    # Long sequence, window truncates
    run_case(
        batch=2, num_q_heads=8, num_kv_heads=2, head_dim=128,
        page_size=16, seq_lens=[512, 384], window=256,
        label="long-seq sparse-window",
    )
    # Gemma4-like shape: GQA=4, head_dim=256, 12K prompt, window=4096
    run_case(
        batch=1, num_q_heads=16, num_kv_heads=4, head_dim=256,
        page_size=16, seq_lens=[1024], window=512,
        label="gemma4-like GQA=4 hd=256",
    )
    # With logits soft cap
    run_case(
        batch=2, num_q_heads=8, num_kv_heads=2, head_dim=128,
        page_size=16, seq_lens=[128, 96], window=256,
        logits_soft_cap=50.0,
        label="soft-cap=50",
    )

    # --- FP8 KV path ---
    print("\n=== FP8 KV path (float8_e4m3fn, scale=1.0) ===")
    run_case(
        batch=2, num_q_heads=8, num_kv_heads=2, head_dim=128,
        page_size=16, seq_lens=[64, 32], window=4096,
        fp8=True, label="short-seq full-window",
    )
    run_case(
        batch=2, num_q_heads=8, num_kv_heads=2, head_dim=128,
        page_size=16, seq_lens=[256, 192], window=128,
        fp8=True, label="long-seq sparse-window",
    )

    print("\nAll cases PASSED.")


if __name__ == "__main__":
    main()
