#!/bin/bash
# Driver: run autotune sweep one (H, BH) at a time via docker exec
# Each worker is a separate python process -> no host RAM accumulation

set -e

RESULTS_DIR="/root/projects/autokernel/profiling"
TSV_FILE="$RESULTS_DIR/autotune_results.tsv"
CUTLASS_FILE="$RESULTS_DIR/cutlass_fp4_results.tsv"

# Ensure worker is in container
docker cp "$RESULTS_DIR/sweep_worker.py" vllm-fusen-k4v4:/tmp/sweep_worker.py

# Initialize TSV
echo -e "B\tH\tBLOCK_H\tnum_warps\tnum_stages\tlatency_us" > "$TSV_FILE"

# Key shapes for Gemma4
H_VALUES=(2816 704 4096 8192)
B_VALUES="1,32,128,512"

# BLOCK_H values that give 1-4 loop iterations for each H
# H=2816: BH=1024(3iter), 2048(2), 4096(1), 8192(1)
# H=704: BH=256(3), 512(2), 1024(1), 2048(1)
# H=4096: BH=1024(4), 2048(2), 4096(1), 8192(1)
# H=8192: BH=2048(4), 4096(2), 8192(1), 16384(1)
# Also include smaller BH for comparison

echo "=========================================="
echo "AUTOTUNE SWEEP: Fused RMSNorm + FP4"
echo "=========================================="

for H in "${H_VALUES[@]}"; do
    echo ""
    echo "===== H=$H ====="

    # Select BH values appropriate for this H
    case $H in
        704)  BH_VALUES=(256 512 1024 2048 4096) ;;
        2816) BH_VALUES=(512 1024 2048 4096 8192) ;;
        4096) BH_VALUES=(1024 2048 4096 8192) ;;
        8192) BH_VALUES=(2048 4096 8192 16384) ;;
    esac

    for BH in "${BH_VALUES[@]}"; do
        echo -n "  H=$H BH=$BH: "

        OUTPUT=$(docker exec vllm-fusen-k4v4 python3 /tmp/sweep_worker.py "$H" "$BH" "$B_VALUES" 2>/dev/null || echo "[]")

        if [ "$OUTPUT" = "[]" ] || [ -z "$OUTPUT" ]; then
            echo "FAILED"
            continue
        fi

        # Parse JSON and append to TSV
        COUNT=$(echo "$OUTPUT" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
for r in data:
    print(f\"{r['B']}\t{r['H']}\t{r['BLOCK_H']}\t{r['num_warps']}\t{r['num_stages']}\t{r['latency_us']}\")
print(len(data), file=sys.stderr)
" >> "$TSV_FILE" 2>&1 | tail -1)

        # Find best latency
        BEST=$(echo "$OUTPUT" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
best = min(data, key=lambda r: r['latency_us'])
print(f\"{len(data)} results, best={best['latency_us']:.2f}us (w={best['num_warps']},s={best['num_stages']})\")
" 2>/dev/null || echo "parse error")
        echo "$BEST"
    done
done

echo ""
echo "===== CUTLASS FP4 GEMM ====="
# Copy and run CUTLASS benchmark
cat > /tmp/cutlass_bench.py << 'PYEOF'
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
PYEOF

docker cp /tmp/cutlass_bench.py vllm-fusen-k4v4:/tmp/cutlass_bench.py
CUTLASS_OUT=$(docker exec vllm-fusen-k4v4 python3 /tmp/cutlass_bench.py 2>/dev/null || echo "[]")

if [ "$CUTLASS_OUT" != "[]" ] && [ -n "$CUTLASS_OUT" ]; then
    echo -e "desc\tB\tH_in\tH_out\tcutlass_fp4_us\tcublas_bf16_us" > "$CUTLASS_FILE"
    echo "$CUTLASS_OUT" | python3 -c "
import sys, json
data = json.loads(sys.stdin.read())
for r in data:
    cut = r.get('cutlass_fp4_us', '')
    cub = r.get('cublas_bf16_us', '')
    print(f\"{r['desc']}\t{r['B']}\t{r['H_in']}\t{r['H_out']}\t{cut}\t{cub}\")
    # Also print to stderr for display
    if cut and cub:
        ratio = cut/cub
        print(f\"  {r['desc']:<22} B={r['B']:>3} {r['H_in']:>5}x{r['H_out']:>5} CUTLASS={cut:.1f}us cuBLAS={cub:.1f}us ratio={ratio:.2f}x\", file=sys.stderr)
    elif r.get('error'):
        print(f\"  {r['desc']:<22} ERROR: {r['error'][:50]}\", file=sys.stderr)
" >> "$CUTLASS_FILE" 2>&1
fi

echo ""
echo "===== ANALYSIS ====="
python3 << 'PYEOF'
import csv
from collections import defaultdict

STD_KEYS = set()
for bh in [256, 512, 1024, 2048, 4096]:
    for nw in [4, 8, 16]:
        for ns in [1, 2, 3]:
            STD_KEYS.add((bh, nw, ns))

results = []
with open("/root/projects/autokernel/profiling/autotune_results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        row["B"] = int(row["B"]); row["H"] = int(row["H"])
        row["BLOCK_H"] = int(row["BLOCK_H"]); row["num_warps"] = int(row["num_warps"])
        row["num_stages"] = int(row["num_stages"]); row["latency_us"] = float(row["latency_us"])
        row["is_standard"] = (row["BLOCK_H"], row["num_warps"], row["num_stages"]) in STD_KEYS
        results.append(row)

print(f"\nTotal benchmarks: {len(results)}")

H_values = [2816, 704, 4096, 8192]
B_values = [1, 32, 128, 512]

print(f"\n{'Shape':>14} {'Best(us)':>10} {'Std(us)':>10} {'Gain':>8} {'Config':>40}")
print("-" * 90)

for H in H_values:
    for B in B_values:
        hr = [r for r in results if r["H"] == H and r["B"] == B]
        if not hr: continue
        hr.sort(key=lambda r: r["latency_us"])
        best = hr[0]
        std_r = [r for r in hr if r["is_standard"]]
        bstd = min(std_r, key=lambda r: r["latency_us"]) if std_r else None

        if bstd:
            pct = (bstd["latency_us"] - best["latency_us"]) / bstd["latency_us"] * 100
            cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
            tag = "(std)" if best["is_standard"] else "(NEW)"
            print(f"  B={B:>3},H={H:>4} {best['latency_us']:>9.2f} {bstd['latency_us']:>9.2f} {pct:>+7.1f}% {cfg} {tag}")
        else:
            cfg = f"BH={best['BLOCK_H']},w={best['num_warps']},s={best['num_stages']}"
            print(f"  B={B:>3},H={H:>4} {best['latency_us']:>9.2f} {'N/A':>10} {'':>8} {cfg}")

# Pareto analysis
print(f"\n--- PARETO-OPTIMAL CONFIGS ---")
for H in H_values:
    cl = defaultdict(dict)
    for r in results:
        if r["H"] != H: continue
        key = (r["BLOCK_H"], r["num_warps"], r["num_stages"])
        cl[key][r["B"]] = r["latency_us"]
    comp = {k: v for k, v in cl.items() if len(v) == len(B_values)}
    if not comp: continue
    pareto = []
    cl2 = list(comp.items())
    for i, (ki, li) in enumerate(cl2):
        dom = False
        for j, (kj, lj) in enumerate(cl2):
            if i == j: continue
            if all(lj.get(b, 1e9) <= li.get(b, 1e9) for b in B_values) and \
               any(lj.get(b, 1e9) < li.get(b, 1e9) for b in B_values):
                dom = True; break
        if not dom: pareto.append((ki, li))
    pareto.sort(key=lambda x: sum(x[1].values()))
    print(f"\n  H={H}: {len(pareto)} Pareto (of {len(comp)})")
    hdr = f"    {'BH':>6} {'w':>3} {'s':>2} " + " ".join(f"{'B='+str(b):>8}" for b in B_values) + " std"
    print(hdr)
    for key, lats in pareto[:10]:
        m = " *" if key in STD_KEYS else "  "
        ls = " ".join(f"{lats.get(b, -1):>8.2f}" for b in B_values)
        print(f"    {key[0]:>6} {key[1]:>3} {key[2]:>2} {ls}{m}")

# Top 15 per shape
print(f"\n--- TOP 15 PER SHAPE ---")
for H in H_values:
    for B in B_values:
        hr = [r for r in results if r["H"] == H and r["B"] == B]
        if not hr: continue
        hr.sort(key=lambda r: r["latency_us"])
        print(f"\n  B={B}, H={H}:")
        print(f"  {'#':>3} {'us':>7} {'BH':>6} {'w':>3} {'s':>2} std")
        for rank, r in enumerate(hr[:15], 1):
            m = "*" if r["is_standard"] else " "
            print(f"  {rank:>3} {r['latency_us']:>7.2f} {r['BLOCK_H']:>6} {r['num_warps']:>3} {r['num_stages']:>2}  {m}")

PYEOF

echo ""
echo "Results saved to:"
echo "  $TSV_FILE"
echo "  $CUTLASS_FILE"
