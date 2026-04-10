#!/usr/bin/env python3
"""
Host-side driver for autotune sweep.
Calls docker exec for each (H, BLOCK_H) pair as a separate process.
Each worker process compiles 48 Triton kernels, preventing host OOM.
Restarts the container if it crashes.
"""

import subprocess
import json
import csv
import time
import sys
from collections import defaultdict
from pathlib import Path

CONTAINER = "fusen-bench-k4v4b64"  # Currently running container with GPU access
WORKER = "/tmp/sweep_worker.py"
RESULTS_DIR = Path("/root/projects/autokernel/profiling")
TSV_FILE = RESULTS_DIR / "autotune_results.tsv"
CUTLASS_FILE = RESULTS_DIR / "cutlass_fp4_results.tsv"

H_VALUES = [2816, 704, 4096, 8192]
B_VALUES = [1, 32, 128, 512]

# Standard 45-config autotune
STD_KEYS = set()
for bh in [256, 512, 1024, 2048, 4096]:
    for nw in [4, 8, 16]:
        for ns in [1, 2, 3]:
            STD_KEYS.add((bh, nw, ns))

# BLOCK_H values per H (1-4 iterations, avoids huge PTX from many iterations)
BH_FOR_H = {
    704:  [256, 512, 1024, 2048, 4096],
    2816: [1024, 2048, 4096, 8192],
    4096: [1024, 2048, 4096, 8192],
    8192: [2048, 4096, 8192, 16384],
}


def find_container():
    """Find a running vllm-built container."""
    global CONTAINER
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}} {{.Image}}"],
                       capture_output=True, text=True)
    for line in r.stdout.strip().split("\n"):
        if "vllm-built" in line:
            CONTAINER = line.split()[0]
            return CONTAINER
    # Try starting vllm-gemma4
    subprocess.run(["docker", "start", "vllm-gemma4"], capture_output=True)
    time.sleep(2)
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}} {{.Image}}"],
                       capture_output=True, text=True)
    for line in r.stdout.strip().split("\n"):
        if "vllm-built" in line:
            CONTAINER = line.split()[0]
            return CONTAINER
    return None


def ensure_container():
    """Make sure container is running and worker is deployed."""
    global CONTAINER
    # Check if current container is running
    if CONTAINER:
        r = subprocess.run(["docker", "exec", CONTAINER, "echo", "ok"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return True

    # Find or restart
    CONTAINER = None
    for name in ["vllm-fusen-k4v4", "vllm-gemma4"]:
        subprocess.run(["docker", "start", name], capture_output=True)
        time.sleep(2)
        r = subprocess.run(["docker", "exec", name, "echo", "ok"],
                          capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            CONTAINER = name
            # Deploy worker
            subprocess.run(["docker", "cp",
                          str(RESULTS_DIR / "sweep_worker.py"),
                          f"{CONTAINER}:/tmp/sweep_worker.py"],
                         capture_output=True)
            return True
    return False


def run_worker(H, BH, timeout=300):
    """Run one sweep worker. Returns list of result dicts or empty list."""
    if not ensure_container():
        print("    ERROR: No container available")
        return []

    try:
        b_str = ",".join(str(b) for b in B_VALUES)
        r = subprocess.run(
            ["docker", "exec", CONTAINER, "python3", WORKER, str(H), str(BH), b_str],
            capture_output=True, text=True, timeout=timeout
        )
        if r.returncode == 0 and r.stdout.strip():
            # stdout may have warnings before JSON, find the JSON line
            for line in r.stdout.strip().split("\n"):
                line = line.strip()
                if line.startswith("["):
                    return json.loads(line)
            return json.loads(r.stdout.strip())
        else:
            if r.stderr:
                # Check for OOM
                if "out of memory" in r.stderr.lower() or "killed" in r.stderr.lower():
                    print(f"    OOM/killed, restarting container...", flush=True)
                    CONTAINER_BACKUP = CONTAINER
                    ensure_container()
            return []
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT ({timeout}s)")
        return []
    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"    ERROR: {e}")
        return []


def main():
    print("=" * 100)
    print("MASSIVE AUTOTUNE SWEEP: Fused RMSNorm + FP4 Quantization")
    print("=" * 100)

    find_container()
    print(f"Container: {CONTAINER}")

    # Deploy worker
    subprocess.run(["docker", "cp",
                   str(RESULTS_DIR / "sweep_worker.py"),
                   f"{CONTAINER}:/tmp/sweep_worker.py"],
                  capture_output=True)

    total_configs = sum(len(BH_FOR_H[H]) * 48 * len(B_VALUES) for H in H_VALUES)
    print(f"Target: {total_configs} benchmarks ({sum(len(BH_FOR_H[H]) for H in H_VALUES)} worker calls)")
    print(f"H values: {H_VALUES}")
    print(f"B values: {B_VALUES}")

    all_results = []
    t_start = time.time()

    # Run sweep
    for H in H_VALUES:
        bh_list = BH_FOR_H[H]
        print(f"\n{'='*80}\nH = {H} (BLOCK_H: {bh_list})\n{'='*80}")

        for BH in bh_list:
            print(f"  BH={BH:>5}: ", end="", flush=True)
            t0 = time.time()
            results = run_worker(H, BH)
            dt = time.time() - t0

            if results:
                for r in results:
                    r["is_standard"] = (r["BLOCK_H"], r["num_warps"], r["num_stages"]) in STD_KEYS
                all_results.extend(results)
                best = min(results, key=lambda r: r["latency_us"])
                print(f"{len(results):>3} results, best={best['latency_us']:.2f}us "
                      f"(w={best['num_warps']},s={best['num_stages']}) [{dt:.0f}s]")
            else:
                print(f"FAILED [{dt:.0f}s]")

    elapsed = time.time() - t_start
    print(f"\nSweep complete: {len(all_results)} benchmarks in {elapsed:.0f}s")

    # ---------------------------------------------------------------------------
    # CUTLASS FP4 GEMM benchmark
    # ---------------------------------------------------------------------------
    print(f"\n{'='*100}")
    print("CUTLASS FP4 GEMM BENCHMARK")
    print(f"{'='*100}")

    cutlass_code = '''
import json, torch, triton
try:
    from vllm._custom_ops import scaled_fp4_quant, cutlass_scaled_fp4_mm
except ImportError:
    print("[]"); exit()
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
    cutlass_file = "/tmp/_cutlass_bench.py"
    with open(cutlass_file, "w") as f:
        f.write(cutlass_code)

    cutlass_results = []
    if ensure_container():
        subprocess.run(["docker", "cp", cutlass_file, f"{CONTAINER}:/tmp/cutlass_bench.py"],
                      capture_output=True)
        try:
            r = subprocess.run(
                ["docker", "exec", CONTAINER, "python3", "/tmp/cutlass_bench.py"],
                capture_output=True, text=True, timeout=300
            )
            if r.returncode == 0 and r.stdout.strip():
                for line in r.stdout.strip().split("\n"):
                    if line.strip().startswith("["):
                        cutlass_results = json.loads(line.strip())
                        break
        except:
            pass

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
                print(f"{r['desc']:<22} ERR: {r.get('error', '')[:50]}")
    else:
        print("  CUTLASS benchmark failed or not available")

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    if not all_results:
        print("\nNo results to analyze!")
        return

    print(f"\n{'='*100}")
    print(f"ANALYSIS ({len(all_results)} benchmarks)")
    print(f"{'='*100}")

    # Best per shape
    best_per = {}
    best_std_per = {}
    for H in H_VALUES:
        for B in B_VALUES:
            hr = [r for r in all_results if r["H"] == H and r["B"] == B]
            if not hr: continue
            hr.sort(key=lambda r: r["latency_us"])
            best_per[(B, H)] = hr[0]
            std_r = [r for r in hr if r["is_standard"]]
            best_std_per[(B, H)] = min(std_r, key=lambda r: r["latency_us"]) if std_r else None

    # Summary table
    print(f"\n{'Shape':>14} {'Best(us)':>10} {'Std(us)':>10} {'Gain':>8} {'Config':>40}")
    print("-" * 90)
    for H in H_VALUES:
        for B in B_VALUES:
            best = best_per.get((B, H))
            bstd = best_std_per.get((B, H))
            if not best: continue
            if bstd:
                pct = (bstd["latency_us"] - best["latency_us"]) / bstd["latency_us"] * 100
                cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
                tag = "(std)" if best["is_standard"] else "(NEW)"
                print(f"  B={B:>3},H={H:>4} {best['latency_us']:>9.2f} {bstd['latency_us']:>9.2f} "
                      f"{pct:>+7.1f}% {cfg} {tag}")
            else:
                cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
                print(f"  B={B:>3},H={H:>4} {best['latency_us']:>9.2f} {'N/A':>10} {'':>8} {cfg}")

    # Top 10 per shape
    print(f"\n--- TOP 10 PER SHAPE ---")
    for H in H_VALUES:
        for B in B_VALUES:
            hr = [r for r in all_results if r["H"] == H and r["B"] == B]
            if not hr: continue
            hr.sort(key=lambda r: r["latency_us"])
            print(f"\n  B={B}, H={H}:")
            print(f"  {'#':>3} {'us':>7} {'BH':>6} {'w':>3} {'s':>2} std")
            for rank, r in enumerate(hr[:10], 1):
                m = "*" if r["is_standard"] else " "
                print(f"  {rank:>3} {r['latency_us']:>7.2f} {r['BLOCK_H']:>6} {r['num_warps']:>3} {r['num_stages']:>2}  {m}")

    # Pareto analysis
    print(f"\n--- PARETO-OPTIMAL CONFIGS ---")
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
        for key, lats in pareto[:10]:
            m = " *" if key in STD_KEYS else "  "
            ls = " ".join(f"{lats.get(b, -1):>8.2f}" for b in B_VALUES)
            print(f"    {key[0]:>6} {key[1]:>3} {key[2]:>2} {ls}{m}")

    # Surprising findings
    print(f"\n--- SURPRISING CONFIGS ---")
    for H in H_VALUES:
        for B in B_VALUES:
            hr = [r for r in all_results if r["H"] == H and r["B"] == B]
            if not hr: continue
            hr.sort(key=lambda r: r["latency_us"])
            # Find non-standard configs in top 5
            novel_top = [r for r in hr[:5] if not r["is_standard"]]
            if novel_top:
                print(f"  B={B}, H={H}: Novel config in top 5:")
                for r in novel_top:
                    print(f"    {r['latency_us']:.2f}us BH={r['BLOCK_H']} w={r['num_warps']} s={r['num_stages']}")

    # ---------------------------------------------------------------------------
    # Save files
    # ---------------------------------------------------------------------------
    with open(TSV_FILE, "w", newline="") as f:
        wr = csv.DictWriter(f, delimiter="\t",
                            fieldnames=["B","H","BLOCK_H","num_warps","num_stages","latency_us","is_standard"])
        wr.writeheader()
        for r in all_results:
            wr.writerow(r)

    if cutlass_results:
        with open(CUTLASS_FILE, "w", newline="") as f:
            wr = csv.DictWriter(f, delimiter="\t",
                                fieldnames=["desc","B","H_in","H_out","cutlass_fp4_us","cublas_bf16_us"])
            wr.writeheader()
            for r in cutlass_results:
                wr.writerow({k: v for k, v in r.items() if k != "error"})

    with open(RESULTS_DIR / "autotune_summary.json", "w") as f:
        json.dump({
            "total_benchmarks": len(all_results),
            "elapsed_seconds": round(elapsed, 1),
            "best_per_shape": {f"B={k[0]},H={k[1]}": v for k, v in best_per.items()},
            "best_std_per_shape": {f"B={k[0]},H={k[1]}": v for k, v in best_std_per.items() if v},
            "cutlass_results": cutlass_results,
        }, f, indent=2)

    print(f"\nSaved:")
    print(f"  {TSV_FILE}")
    print(f"  {CUTLASS_FILE}")
    print(f"  {RESULTS_DIR / 'autotune_summary.json'}")


if __name__ == "__main__":
    main()
