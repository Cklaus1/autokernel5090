"""Profile FusenCache v3.1 selective decode path.

Instruments each phase of _decode_selective to find the bottleneck.
Runs outside vLLM to isolate the selective decode path directly.
"""
import os
os.environ["FUSEN_SELECTIVE"] = "1"
os.environ["FUSEN_HOT_WINDOW"] = "128"
os.environ["FUSEN_TOP_M"] = "8"

import time
import torch
from collections import defaultdict


def profile_selective_path():
    """Simulate the selective decode path with realistic data and time each phase."""

    device = torch.device("cuda")
    torch.cuda.synchronize()

    # Realistic parameters (Gemma 4 31B sliding layer)
    B = 1           # batch
    Hq = 32         # query heads
    Hk = 16         # kv heads
    D = 256         # head dim
    seq_len = 2048  # total cached tokens
    block_size = 16
    num_blocks = seq_len // block_size + 10
    chunk_size = 32
    hot_window = 128
    top_m = 8
    scale = 1.0 / (D ** 0.5)

    # Allocate realistic tensors
    query = torch.randn(B, Hq, D, device=device, dtype=torch.bfloat16)
    slot_size = D + D // 2  # v1 layout
    kv_cache = torch.randint(0, 255, (num_blocks, block_size, Hk, slot_size),
                              device=device, dtype=torch.uint8)
    v_scales = torch.ones(num_blocks * block_size, Hk,
                          device=device, dtype=torch.float16) * 0.1
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], device=device, dtype=torch.int32)

    # Landmarks (pre-computed, as v3.1 does)
    max_chunks = num_blocks * block_size // chunk_size + 1
    landmarks = torch.randn(max_chunks, Hk, D, device=device, dtype=torch.float16)

    # Warmup CUDA
    _ = torch.mm(torch.randn(256, 256, device=device),
                 torch.randn(256, 256, device=device))
    torch.cuda.synchronize()

    timings = defaultdict(list)
    num_iters = 50

    for iteration in range(num_iters):
        torch.cuda.synchronize()

        # ============================================================
        # Phase 1a: Compute physical chunk IDs
        # ============================================================
        t0 = time.perf_counter()
        torch.cuda.synchronize()

        hot_start = seq_len - hot_window
        num_chunks = hot_start // chunk_size
        remainder = hot_start % chunk_size

        chunk_starts = torch.arange(
            0, num_chunks * chunk_size, chunk_size, device=device)
        cblk = block_table[0, chunk_starts // block_size].long()
        coff = (chunk_starts % block_size).long()
        phys_ids = (cblk * block_size + coff) // chunk_size

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings["1a_chunk_ids"].append(t1 - t0)

        # ============================================================
        # Phase 1b: Gather landmarks
        # ============================================================
        lm = landmarks[phys_ids].float()  # (num_chunks, Hk, D)

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        timings["1b_gather_landmarks"].append(t2 - t1)

        # ============================================================
        # Phase 1c: GQA expand landmarks
        # ============================================================
        kv_groups = Hq // Hk
        lm_exp = lm.repeat_interleave(kv_groups, dim=1)

        torch.cuda.synchronize()
        t3 = time.perf_counter()
        timings["1c_gqa_expand"].append(t3 - t2)

        # ============================================================
        # Phase 1d: Score query vs landmarks
        # ============================================================
        qi = query[0].float()
        scores = torch.einsum('hd,chd->hc', qi, lm_exp) * scale

        torch.cuda.synchronize()
        t4 = time.perf_counter()
        timings["1d_score"].append(t4 - t3)

        # ============================================================
        # Phase 1e: TopK selection
        # ============================================================
        chunk_scores = scores.max(dim=0).values
        k = min(top_m, num_chunks)
        _, top_idx = chunk_scores.topk(k)

        torch.cuda.synchronize()
        t5 = time.perf_counter()
        timings["1e_topk"].append(t5 - t4)

        # ============================================================
        # Phase 2: Build position tensor
        # ============================================================
        offsets = torch.arange(chunk_size, device=device)
        selected_cold = (
            top_idx.unsqueeze(1) * chunk_size + offsets.unsqueeze(0)
        ).reshape(-1).to(torch.int32)

        if remainder > 0:
            rem = torch.arange(num_chunks * chunk_size, hot_start,
                               dtype=torch.int32, device=device)
            selected_cold = torch.cat([selected_cold, rem])

        hot_pos = torch.arange(hot_start, seq_len, dtype=torch.int32,
                               device=device)
        all_pos = torch.cat([selected_cold, hot_pos])
        all_pos, _ = all_pos.sort()

        # Pad into batch tensor
        positions = torch.zeros(B, all_pos.shape[0], dtype=torch.int32,
                                device=device)
        positions[0, :all_pos.shape[0]] = all_pos
        num_positions = torch.tensor([all_pos.shape[0]], dtype=torch.int32,
                                     device=device)

        torch.cuda.synchronize()
        t6 = time.perf_counter()
        timings["2_build_positions"].append(t6 - t5)

        # ============================================================
        # Phase 3: Triton selective kernel
        # ============================================================
        try:
            from vllm.v1.attention.ops.triton_fusencache_decode import (
                fusencache_selective_decode,
            )
            out = fusencache_selective_decode(
                query=query,
                kv_cache=kv_cache,
                v_scales=v_scales,
                positions=positions,
                num_positions=num_positions,
                block_table=block_table,
                scale=scale,
                num_kv_heads=Hk,
            )
            torch.cuda.synchronize()
            t7 = time.perf_counter()
            timings["3_triton_kernel"].append(t7 - t6)
        except Exception as e:
            t7 = time.perf_counter()
            timings["3_triton_FAILED"].append(t7 - t6)
            if iteration == 0:
                print(f"Triton selective kernel failed: {e}")

        # ============================================================
        # Phase 4: Total
        # ============================================================
        timings["total"].append(t7 - t0)

    # ============================================================
    # Report
    # ============================================================
    print(f"\n{'='*60}")
    print(f"FusenCache v3.1 Selective Decode Profile")
    print(f"{'='*60}")
    print(f"Config: B={B}, Hq={Hq}, Hk={Hk}, D={D}")
    print(f"        seq_len={seq_len}, hot_window={hot_window}")
    print(f"        chunk_size={chunk_size}, top_m={top_m}")
    print(f"        num_chunks={num_chunks}, attended={all_pos.shape[0]}/{seq_len}")
    print(f"        iterations={num_iters} (first is warmup)")
    print(f"\n{'Phase':<30} {'Mean (ms)':>10} {'Std':>10} {'% of total':>12}")
    print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*12}")

    # Skip first iteration (warmup)
    total_mean = sum(v for v in timings["total"][1:]) / (num_iters - 1) * 1000

    for phase in sorted(timings.keys()):
        if phase == "total":
            continue
        vals = [v * 1000 for v in timings[phase][1:]]  # ms, skip warmup
        mean = sum(vals) / len(vals)
        std = (sum((v - mean)**2 for v in vals) / len(vals)) ** 0.5
        pct = mean / total_mean * 100 if total_mean > 0 else 0
        print(f"{phase:<30} {mean:>10.3f} {std:>10.3f} {pct:>11.1f}%")

    vals_total = [v * 1000 for v in timings["total"][1:]]
    mean_total = sum(vals_total) / len(vals_total)
    std_total = (sum((v - mean_total)**2 for v in vals_total) / len(vals_total)) ** 0.5
    print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*12}")
    print(f"{'total':<30} {mean_total:>10.3f} {std_total:>10.3f} {'100.0%':>12}")

    print(f"\nPer-token decode cost: {mean_total:.3f} ms")
    print(f"Theoretical tok/s (decode only): {1000/mean_total:.1f}")

    # Also profile the landmark WRITE path
    print(f"\n{'='*60}")
    print(f"Landmark Write Path (per do_kv_cache_update call)")
    print(f"{'='*60}")

    k_float = torch.randn(8192, Hk, D, device=device)
    k_fp8 = k_float.to(torch.float8_e4m3fn)
    flat_slots = torch.arange(8192, device=device)
    chunk_ids = flat_slots // chunk_size

    landmarks_sum = torch.zeros(max_chunks, Hk, D, dtype=torch.float32,
                                device=device)
    landmarks_count = torch.zeros(max_chunks, Hk, dtype=torch.int32,
                                  device=device)

    # Python loop version (current)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(min(256, 8192)):  # sample 256 tokens
        cid = chunk_ids[t].item()
        landmarks_sum[cid] += k_fp8[t].float()
        landmarks_count[cid] += 1
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    python_ms = (t1 - t0) * 1000
    print(f"Python loop (256 tokens):    {python_ms:.3f} ms")
    print(f"  Extrapolated 8192 tokens:  {python_ms * 32:.1f} ms")

    # Vectorized version (proposed fix)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    k_f = k_fp8.float()  # (8192, Hk, D)
    cids = chunk_ids.long()
    # scatter_add approach
    n_chunks_actual = cids.max().item() + 1
    sums = torch.zeros(n_chunks_actual, Hk, D, device=device, dtype=torch.float32)
    # Use index_add_ on flattened tensor
    k_flat = k_f.reshape(8192, -1)  # (8192, Hk*D)
    sums_flat = sums.reshape(n_chunks_actual, -1)
    sums_flat.index_add_(0, cids, k_flat)
    counts = torch.zeros(n_chunks_actual, dtype=torch.int32, device=device)
    counts.scatter_add_(0, cids, torch.ones(8192, dtype=torch.int32, device=device))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    vectorized_ms = (t1 - t0) * 1000
    print(f"Vectorized (8192 tokens):    {vectorized_ms:.3f} ms")
    print(f"  Speedup: {python_ms * 32 / vectorized_ms:.0f}x")


if __name__ == "__main__":
    profile_selective_path()
