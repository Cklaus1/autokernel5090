#!/usr/bin/env python3
"""
Driver: orchestrates 1000+ config sweep by spawning one subprocess per (H, BLOCK_H).
Each subprocess compiles ~48 kernels (6 warps x 8 stages), preventing host OOM.

Also benchmarks CUTLASS FP4 GEMM at real Gemma4 shapes.
"""

import subprocess, json, csv, sys, time
from collections import defaultdict
from pathlib import Path

WORKER = "/tmp/sweep_worker.py"

H_VALUES = [2816, 704, 4096, 8192]
B_VALUES = [1, 32, 128, 512]
B_STR = ",".join(str(b) for b in B_VALUES)

# BLOCK_H values to sweep (power-of-2, 16 to 16384)
BH_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Standard autotune configs (from fused_norm_fp4.py _generate_autotune_configs)
STD_KEYS = set()
for bh in [256, 512, 1024, 2048, 4096]:
    for nw in [4, 8, 16]:
        for ns in [1, 2, 3]:
            STD_KEYS.add((bh, nw, ns))

def run_worker(H, BH, timeout=120):
    """Run worker subprocess for one (H, BLOCK_H). Returns list of result dicts."""
    try:
        r = subprocess.run(
            ["python3", WORKER, str(H), str(BH), B_STR],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout.strip())
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        pass
    return []


def run_cutlass_bench():
    """Benchmark CUTLASS FP4 GEMM shapes in a subprocess."""
    code = r'''
import json, torch, triton
from vllm._custom_ops import scaled_fp4_quant, cutlass_scaled_fp4_mm
shapes = [
    [1, 2816, 1408, "Gate/Up B=1"], [1, 704, 2816, "Down B=1"],
    [1, 2816, 8192, "QKV-sl B=1"], [1, 2816, 4096, "QKV-gl B=1"],
    [1, 4096, 2816, "O-gl B=1"], [1, 8192, 2816, "O-sl B=1"],
    [32, 2816, 1408, "Gate/Up B=32"], [32, 704, 2816, "Down B=32"],
    [32, 2816, 8192, "QKV-sl B=32"],
    [128, 2816, 1408, "Gate/Up B=128"], [128, 704, 2816, "Down B=128"],
    [128, 2816, 8192, "QKV-sl B=128"],
    [256, 2816, 1408, "Gate/Up B=256"], [512, 2816, 1408, "Gate/Up B=512"],
]
gs = torch.tensor([1.0], device="cuda", dtype=torch.float32)
results = []
for B, Hi, Ho, desc in shapes:
    r = {"desc": desc, "B": B, "H_in": Hi, "H_out": Ho}
    x = torch.randn(B, Hi, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(Ho, Hi, device="cuda", dtype=torch.bfloat16)
    try:
        xf, xs = scaled_fp4_quant(x, gs, is_sf_swizzled_layout=True)
        wf, ws = scaled_fp4_quant(w, gs, is_sf_swizzled_layout=True)
        def rc(xf=xf,wf=wf,xs=xs,ws=ws): return cutlass_scaled_fp4_mm(xf,wf,xs,ws,gs,gs,torch.bfloat16)
        rc(); torch.cuda.synchronize()
        r["cutlass_fp4_us"] = round(triton.testing.do_bench(rc, warmup=25, rep=100) * 1000, 2)
    except Exception as e:
        r["cutlass_fp4_us"] = None; r["error"] = str(e)[:80]
    wt = w.t().contiguous()
    def rb(x=x,wt=wt): return torch.mm(x, wt)
    try:
        rb(); torch.cuda.synchronize()
        r["cublas_bf16_us"] = round(triton.testing.do_bench(rb, warmup=25, rep=100) * 1000, 2)
    except: r["cublas_bf16_us"] = None
    results.append(r)
    del x, w; torch.cuda.empty_cache()
print(json.dumps(results))
'''
    with open("/tmp/cutlass_bench.py", "w") as f:
        f.write(code)
    try:
        r = subprocess.run(["python3", "/tmp/cutlass_bench.py"],
                           capture_output=True, text=True, timeout=300)
        if r.returncode == 0:
            return json.loads(r.stdout.strip())
    except:
        pass
    return []


def main():
    print("=" * 100)
    print("MASSIVE AUTOTUNE SWEEP: Fused RMSNorm + FP4 Quantization")
    print(f"BLOCK_H: {BH_VALUES}")
    print(f"H: {H_VALUES}, B: {B_VALUES}")
    print(f"Configs per (H,BH): 48 (6 warps x 8 stages)")
    total_combos = len(H_VALUES) * len(BH_VALUES)
    print(f"Total (H,BH) combos: {total_combos}, max configs: {total_combos * 48 * len(B_VALUES)}")
    print("=" * 100)

    all_results = []
    best_per = {}
    best_std_per = {}
    t_start = time.time()

    for H in H_VALUES:
        print(f"\n{'='*80}\nH = {H}\n{'='*80}")
        h_results = []

        for BH in BH_VALUES:
            if BH < 16:
                continue
            print(f"  BH={BH:>5}: ", end="", flush=True)
            t0 = time.time()
            results = run_worker(H, BH, timeout=180)
            dt = time.time() - t0

            if results:
                for r in results:
                    r["is_standard"] = (r["BLOCK_H"], r["num_warps"], r["num_stages"]) in STD_KEYS
                h_results.extend(results)
                all_results.extend(results)
                # Find best for this BH
                best_lat = min(r["latency_us"] for r in results)
                print(f"{len(results):>3} results, best={best_lat:.2f}us ({dt:.0f}s)")
            else:
                print(f"FAILED ({dt:.0f}s)")

        # Per-H summary
        if not h_results:
            continue
        for B in B_VALUES:
            b_results = [r for r in h_results if r["B"] == B]
            if not b_results:
                continue
            b_results.sort(key=lambda r: r["latency_us"])
            best = b_results[0]
            best_per[(B, H)] = best

            std_r = [r for r in b_results if r["is_standard"]]
            bstd = min(std_r, key=lambda r: r["latency_us"]) if std_r else None
            best_std_per[(B, H)] = bstd

            imp = ""
            if bstd:
                pct = (bstd["latency_us"] - best["latency_us"]) / bstd["latency_us"] * 100
                imp = f" ({pct:+.1f}% vs std {bstd['latency_us']:.2f}us)"
            print(f"\n  B={B}: BEST {best['latency_us']:.2f}us "
                  f"BH={best['BLOCK_H']} w={best['num_warps']} s={best['num_stages']}{imp}")

            print(f"  {'#':>3} {'us':>7} {'BH':>6} {'w':>3} {'s':>2} std")
            for rank, r in enumerate(b_results[:10], 1):
                m = "*" if r["is_standard"] else " "
                print(f"  {rank:>3} {r['latency_us']:>7.2f} {r['BLOCK_H']:>6} {r['num_warps']:>3} {r['num_stages']:>2}  {m}")

    # ---------------------------------------------------------------------------
    # CUTLASS FP4 GEMM
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("CUTLASS FP4 GEMM BENCHMARK")
    print(f"{'='*100}")

    cutlass_results = run_cutlass_bench()
    if cutlass_results:
        print(f"\n{'Desc':<22} {'B':>4} {'Hin':>5} {'Hout':>5} {'CUT_us':>8} {'cuB_us':>8} {'Ratio':>7}")
        print("-" * 65)
        for r in cutlass_results:
            cut = r.get("cutlass_fp4_us")
            cub = r.get("cublas_bf16_us")
            if cut and cub:
                print(f"{r['desc']:<22} {r['B']:>4} {r['H_in']:>5} {r['H_out']:>5} "
                      f"{cut:>8.2f} {cub:>8.2f} {cut/cub:>6.2f}x")
            elif r.get("error"):
                print(f"{r['desc']:<22} ERR: {r['error'][:50]}")

    # ---------------------------------------------------------------------------
    # Pareto analysis
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("PARETO-OPTIMAL CONFIGS")
    print(f"{'='*100}")

    for H in H_VALUES:
        cl = defaultdict(dict)
        for r in all_results:
            if r["H"] != H: continue
            key = (r["BLOCK_H"], r["num_warps"], r["num_stages"])
            cl[key][r["B"]] = r["latency_us"]

        comp = {k: v for k, v in cl.items() if len(v) == len(B_VALUES)}
        if not comp: continue

        pareto = []
        cl2 = list(comp.items())
        for i, (ki, li) in enumerate(cl2):
            dom = False
            for j, (kj, lj) in enumerate(cl2):
                if i == j: continue
                if all(lj.get(b, 1e9) <= li.get(b, 1e9) for b in B_VALUES) and \
                   any(lj.get(b, 1e9) < li.get(b, 1e9) for b in B_VALUES):
                    dom = True; break
            if not dom: pareto.append((ki, li))
        pareto.sort(key=lambda x: sum(x[1].values()))

        print(f"\n  H={H}: {len(pareto)} Pareto (of {len(comp)})")
        hdr = f"    {'BH':>6} {'w':>3} {'s':>2} " + " ".join(f"{'B='+str(b):>8}" for b in B_VALUES) + " std"
        print(hdr)
        for key, lats in pareto[:15]:
            m = " *" if key in STD_KEYS else "  "
            ls = " ".join(f"{lats.get(b, -1):>8.2f}" for b in B_VALUES)
            print(f"    {key[0]:>6} {key[1]:>3} {key[2]:>2} {ls}{m}")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"FINAL SUMMARY ({len(all_results)} benchmarks in {elapsed:.0f}s)")
    print(f"{'='*100}")
    print(f"\n{'Shape':>14} {'Best(us)':>10} {'Std(us)':>10} {'Gain':>8} {'Config':>35}")
    print("-" * 85)

    for H in H_VALUES:
        for B in B_VALUES:
            best = best_per.get((B, H))
            bstd = best_std_per.get((B, H))
            if best and bstd:
                pct = (bstd["latency_us"] - best["latency_us"]) / bstd["latency_us"] * 100
                cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
                tag = "(std)" if best["is_standard"] else "(NEW)"
                print(f"  B={B:>3},H={H:>4} {best['latency_us']:>9.2f} {bstd['latency_us']:>9.2f} "
                      f"{pct:>+7.1f}% {cfg} {tag}")
            elif best:
                cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
                print(f"  B={B:>3},H={H:>4} {best['latency_us']:>9.2f} {'N/A':>10} {'':>8} {cfg}")

    # Save files
    with open("/tmp/autotune_results.tsv", "w", newline="") as f:
        wr = csv.DictWriter(f, delimiter="\t",
                            fieldnames=["B","H","BLOCK_H","num_warps","num_stages","latency_us","is_standard"])
        wr.writeheader()
        for r in all_results: wr.writerow(r)

    if cutlass_results:
        with open("/tmp/cutlass_fp4_results.tsv", "w", newline="") as f:
            wr = csv.DictWriter(f, delimiter="\t",
                                fieldnames=["desc","B","H_in","H_out","cutlass_fp4_us","cublas_bf16_us"])
            wr.writeheader()
            for r in cutlass_results:
                wr.writerow({k: v for k, v in r.items() if k != "error"})

    with open("/tmp/autotune_summary.json", "w") as f:
        json.dump({
            "total_benchmarks": len(all_results),
            "elapsed_seconds": round(elapsed, 1),
            "best_per_shape": {f"B={k[0]},H={k[1]}": v for k, v in best_per.items()},
            "best_std_per_shape": {f"B={k[0]},H={k[1]}": v for k, v in best_std_per.items() if v},
            "cutlass_results": cutlass_results,
        }, f, indent=2)

    print(f"\nSaved: /tmp/autotune_results.tsv, /tmp/cutlass_fp4_results.tsv, /tmp/autotune_summary.json")


if __name__ == "__main__":
    main()
