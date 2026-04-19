#!/usr/bin/env python3
"""Microbench: expert prefetch via L2 residency on GPU 1.

Prototype test for the proposal in plans/expert_prefetch_l2.md.

Runs 3 modes against a Gemma4 26B MoE-shaped workload:
  A) BASELINE — no prefetch
  B) PREFETCH — EMA-predicted top-16 warm-read on a second stream
  C) ORACLE   — prefetch the exact top-8 that WILL be used (upper bound)

Reports per-mode latency (μs), L2-hit-rate proxy, and speedup.

Runs on GPU 1 only. Target wall-clock < 60 s.
"""

import os
import sys
import time

# GPU 1 only
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "expert_prefetch_test.cu")
BUILD_DIR = "/tmp/build_expert_prefetch"
os.makedirs(BUILD_DIR, exist_ok=True)

if os.path.exists("/usr/local/cuda/bin/nvcc"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")

print(f"[BUILD] Compiling {SRC} ...")
t0 = time.time()
ext = load(
    name="expert_prefetch_test",
    sources=[SRC],
    extra_cuda_cflags=[
        "-gencode=arch=compute_120,code=sm_120",
        "-O3", "--use_fast_math", "-std=c++17",
    ],
    extra_cflags=["-O3", "-std=c++17"],
    build_directory=BUILD_DIR,
    verbose=False,
)
print(f"[BUILD] OK in {time.time()-t0:.1f}s")


def main():
    dev = torch.device("cuda:0")  # after CUDA_VISIBLE_DEVICES=1, "cuda:0" is physical GPU 1
    props = torch.cuda.get_device_properties(dev)
    L2_MB = props.L2_cache_size // (1024 * 1024)
    print(f"[INFO] Device: {props.name}  SM{props.major}{props.minor}  L2={L2_MB} MiB")

    # --- Gemma4 26B NVFP4 shape ---
    # Per-expert NVFP4 weight footprint: gate_up [2816,1408]+down[704,2816] ~2 MiB.
    E = 128
    BYTES_PER_EXPERT = 2 * 1024 * 1024   # 2 MiB
    TOP_K = 8
    TOP_P = 16                              # EMA-predicted set size

    # Build W = 128 * 2 MiB = 256 MiB
    n_bytes = E * BYTES_PER_EXPERT
    print(f"[INFO] W={n_bytes//(1024*1024)} MiB  (128 experts × 2 MiB)")
    W = torch.randint(-128, 127, (n_bytes,), dtype=torch.int8, device=dev)
    out = torch.empty(TOP_K * BYTES_PER_EXPERT, dtype=torch.int8, device=dev)
    # sink sized for the largest warmup grid (64 blocks.x × up to 128 experts)
    sink = torch.zeros(64 * 128, dtype=torch.int32, device=dev)

    # --- Routing simulation ---
    torch.manual_seed(0xBAE)
    # layer 1 true active
    layer1_true = torch.randperm(E, device=dev)[:TOP_K].to(torch.int32)
    # layer 0 disjoint (Jaccard ~0.14)
    layer0 = None
    for _ in range(30):
        cand = torch.randperm(E, device=dev)[:TOP_K].to(torch.int32)
        sa = set(cand.cpu().tolist()); sb = set(layer1_true.cpu().tolist())
        if len(sa & sb) <= 1:
            layer0 = cand
            break
    assert layer0 is not None

    # EMA predicted: simulate 75% recall — 6 of 8 true + 10 filler
    hit_count = 6
    l1_list = layer1_true.cpu().tolist()
    l0_list = layer0.cpu().tolist()
    remaining = [i for i in range(E) if i not in l1_list]
    torch.manual_seed(0xFEED)
    filler_idx = torch.randperm(len(remaining))[:TOP_P - hit_count].tolist()
    predicted_list = l1_list[:hit_count] + [remaining[i] for i in filler_idx]
    predicted = torch.tensor(predicted_list, dtype=torch.int32, device=dev)

    # Flush set: experts NOT in layer0 ∪ layer1 ∪ predicted, 128 MB = 64 experts,
    # and we pick 96 experts (192 MiB > L2=128) to guarantee L2 cold.
    hot_set = set(l0_list) | set(l1_list) | set(predicted_list)
    cold_candidates = [i for i in range(E) if i not in hot_set]
    # Repeat list if we don't have enough cold experts
    N_FLUSH = 96
    if len(cold_candidates) < N_FLUSH:
        # Tile to get enough coverage (each tile re-reads same experts;
        # cumulative read still exceeds L2 when N_FLUSH*2MB > L2).
        cold_ids = (cold_candidates * ((N_FLUSH // len(cold_candidates)) + 1))[:N_FLUSH]
    else:
        cold_ids = cold_candidates[:N_FLUSH]
    flush_ids = torch.tensor(cold_ids, dtype=torch.int32, device=dev)

    print(f"[INFO] layer0={l0_list}")
    print(f"[INFO] layer1={l1_list}")
    print(f"[INFO] predicted top-16 ∩ layer1_true = {hit_count}/{TOP_K} = "
          f"{hit_count/TOP_K*100:.0f}% recall")
    print(f"[INFO] flush set: {N_FLUSH} experts = {N_FLUSH*2} MiB (> L2 {L2_MB} MiB)")

    empty_pred = torch.tensor([0], dtype=torch.int32, device=dev)  # sentinel; mode=0 ignores
    ITERS = 300

    def timed(mode, name, pred_arg):
        # Re-run a few times, take median-like behavior via multiple iters internally
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        us = ext.run_scenario(W, out, BYTES_PER_EXPERT,
                              layer0, layer1_true, pred_arg, flush_ids,
                              sink, mode, ITERS)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        print(f"  [{name:>10}]  {us:8.2f} μs/iter  wall={wall:.2f}s  ({ITERS} iters)")
        return us

    print("\n=== Microbench ===")
    # Run multiple times to stabilize (warm caches of code, etc.)
    runs = 3
    base_us_list, pref_us_list, oracle_us_list = [], [], []
    for r in range(runs):
        print(f"-- run {r+1}/{runs} --")
        base_us_list.append(timed(0, "BASELINE", empty_pred))
        pref_us_list.append(timed(1, "PREFETCH", predicted))
        oracle_us_list.append(timed(2, "ORACLE", layer1_true))

    # Pick median to reduce noise
    base_us = sorted(base_us_list)[len(base_us_list) // 2]
    pref_us = sorted(pref_us_list)[len(pref_us_list) // 2]
    oracle_us = sorted(oracle_us_list)[len(oracle_us_list) // 2]

    # --- Speedup & L2 hit-rate proxy ---
    hbm_span = max(base_us - oracle_us, 1e-6)
    hit_proxy = max(0.0, min(1.0, 1.0 - (pref_us - oracle_us) / hbm_span))
    speedup = base_us / pref_us if pref_us > 0 else 0.0
    oracle_speedup = base_us / oracle_us if oracle_us > 0 else 0.0
    delta_pct = (base_us - pref_us) / base_us * 100.0

    print("\n=== Results (median of 3 runs) ===")
    print(f"  Baseline      : {base_us:.1f} μs    (runs: {[f'{x:.1f}' for x in base_us_list]})")
    print(f"  Prefetch(EMA) : {pref_us:.1f} μs    (Δ={delta_pct:+.2f}%  speedup={speedup:.3f}×)")
    print(f"  Oracle (top-8): {oracle_us:.1f} μs  (upper bound speedup={oracle_speedup:.3f}×)")
    print(f"  L2-hit-rate proxy (prefetch): {hit_proxy*100:.1f}%")
    print(f"  HBM-span (base - oracle)    : {hbm_span:.1f} μs")

    verdict = "PASS" if delta_pct >= 3.0 else "KILL"
    print(f"\n  Verdict: {verdict}  (gate ≥ 3% speedup)")

    # --- Append to results.tsv ---
    results_path = os.path.abspath(os.path.join(HERE, "..", "..", "results.tsv"))
    one_liner = (
        f"base={base_us:.0f}us pref={pref_us:.0f}us oracle={oracle_us:.0f}us "
        f"L2hit_proxy={hit_proxy*100:.0f}% speedup={speedup:.3f}x"
    )
    row = (
        f"expert_prefetch_test\tcuda_mem_advise\tkernel_bench\t0\t"
        f"{pref_us:.1f}\t{hit_proxy*100:.1f}\t{speedup:.3f}\t{verdict}\t0\t"
        f"{one_liner}\n"
    )
    with open(results_path, "a") as f:
        f.write(row)
    print(f"\n[RESULT] Appended row to {results_path}")
    print(f"  {row.strip()}")


if __name__ == "__main__":
    main()
