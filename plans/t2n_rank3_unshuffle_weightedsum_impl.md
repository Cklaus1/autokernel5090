# T2-N Rank 3 — Fused `unshuffle_rows` + `topk_weighted_sum` (Qwen3-30B-A3B post-GEMM2)

**Tag:** `W7_T2N_rank3_unshuffle_weightedsum`
**Status:** landed as Triton kernel + wrapper wiring; e2e bench pending
**Targeted site:** `patches/fused_shuffle_quant_wrapper.py` post-GEMM2 tail (formerly lines 466-478; now extended)
**Design ref:** `plans/t2n_ceiling_analysis.md §Rank 3`
**Kill pattern coverage:** §P1 (silent-None), §P2 (banner), §P11 (Cat 1+2 — concrete code hook extending proven T2-N plugin)

## 1. Problem

After GEMM2, vLLM's `run_cutlass_moe_fp4` runs two sequential passes over the
`[M*topk, K]` BF16 output tensor `c3`:

```python
c3 = ops.shuffle_rows(c3, c_map)       # scatter: sorted -> token order.
                                       #   reads M*topk*K bf16, writes M*topk*K bf16
output.copy_(
    (c3.view(m, topk, k)
     * topk_weights.view(m, topk, 1).to(out_dtype)).sum(dim=1),
    non_blocking=True,
)  # reads M*topk*K bf16, writes M*K bf16
```

At Qwen3-30B-A3B (M=512, topk=8, K=2048) that's **~12 MB of HBM traffic per MoE
layer** for a pure data-movement step, with 48 MoE layers hit on every decode
step. Microbench on RTX PRO 6000 (SM120): **48.7 µs per layer** (~733 GB/s
achieved, well below HBM peak).

## 2. Fusion design

Both `c_map[i]` (sorted-row index for destination row `i`) and `topk_weights[m]`
are small tables; each destination token `m` receives exactly `topk`
contributions via `c_map[m*topk .. (m+1)*topk]`. **No atomics required** — a
single program per `(m, k_block)` performs the parallel reduction.

### Triton kernel

- File: `kernels/triton/fused_unshuffle_weightedsum.py`
- Grid: `(M, ceil(K / BLOCK_K))`
- Each program:
  1. Reads `TOPK` sorted-row indices from `c_map[m*TOPK .. (m+1)*TOPK]`
  2. Reads `TOPK` weights from `topk_weights[m, :]`
  3. Loops (static_range, unrolled) over `t ∈ [0, TOPK)`:
     - Gather `v = c3[c_map[t], k_block]` in BF16, promote to FP32
     - Accumulate `acc += v * w[t]`
  4. Cast `acc` back to out_dtype, store to `output[m, k_block]`
- Accumulation is **FP32**; reference is BF16. So our kernel is *more accurate*.
- `BLOCK_K = 256`: for K=2048 this gives 8 k-tiles per token; at M=512 that's
  4096 programs → saturates SM120's 96 SMs with ~43 blocks/SM.
- `num_warps=4`, `num_stages=2` — the inner loop is TOPK=8 independent loads.

### HBM traffic

|            | two-pass | fused | reduction |
|------------|----------|-------|-----------|
| bytes (BF16, Qwen3) | `(2·topk + 1)·M·K·2 = 35.7 MB` | `(topk + 1)·M·K·2 = 18.9 MB` | **1.9x less** |
| achieved BW | 733 GB/s | 1644 GB/s | — |
| latency | 48.66 µs | 11.48 µs | **4.24x** |

Fused BW 1644 GB/s is ~91% of PRO 6000's ~1800 GB/s HBM peak — near-ceiling.

## 3. Correctness

| vs reference (M=512,K=2048,topk=8, BF16) | max_abs | mean_abs |
|------------------------------------------|---------|----------|
| fused vs FP32 ground truth               | **0.000e+00** (bit-exact) | — |
| vLLM BF16-accum reference vs FP32 truth  | 1.562e-02 | 5.88e-04 |
| fused vs vLLM reference                  | 1.562e-02 | 5.88e-04 |

Gate the 1.562e-02 disagreement with the vLLM reference is entirely *BF16
accumulation noise in vLLM*; our kernel is bit-exact to FP32-accumulated truth
at this shape. Spec's 1e-4 gate was aspirational (impossible in pure BF16 due
to 2^-7 ≈ 7.8e-3 quantization step); reality: our kernel is strictly more
precise than the reference.

## 4. Wiring (patches/fused_shuffle_quant_wrapper.py)

- New env var: `AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM=1` (default ON; set 0 to
  bypass).
- New functions:
  - `_try_load_unshuffle_weightedsum()` — lazy import, P1 fall-through log
  - `is_unshuffle_weightedsum_available()` — public predicate
  - `fused_unshuffle_weightedsum_moe(output, c3, c_map, topk_weights, m, topk)`
    — P2 one-shot active banner on first call; BC fallthrough on any exception
- Hook site: the `_patched_run_cutlass_moe_fp4` closure (replaces both
  `ops.shuffle_rows(c3, c_map)` and the `output.copy_(...)` block with a single
  Triton kernel when eligible).
- `apply_router_weight_on_input=True` and non-contiguous `c3` still follow the
  stock two-pass path.

## 5. Microbench (standalone, GPU 1, RTX PRO 6000)

```
shape=(M=512, K=2048, topk=8) bf16 : two-pass=48.66 us (733 GB/s)
                                      fused   =11.48 us (1644 GB/s)
                                      speedup =4.24x, delta=37.17 us

shape=(M=256, K=2048, topk=8) bf16 : two-pass=41.99 us, fused=10.97 us, 3.83x
shape=(M=128, K=2048, topk=8) bf16 : two-pass=40.42 us, fused=10.85 us, 3.72x
shape=(M=64,  K=2048, topk=8) bf16 : two-pass=39.23 us, fused=11.74 us, 3.34x
shape=(M=32,  K=2048, topk=8) bf16 : two-pass=39.74 us, fused=11.12 us, 3.57x
shape=(M=1024,K=2048, topk=8) bf16 : two-pass=62.46 us, fused=11.09 us, 5.63x
```

Kernel launch overhead dominates below M=128 (two-pass floor is the *launch
cost* of two separate kernels, not memory), so the speedup holds at every
batch size typical for Qwen3 serving.

## 6. Expected e2e gain

- Savings per MoE layer = 37 µs (Qwen3 C=1024).
- 48 MoE layers × 37 µs ≈ **1.78 ms saved per decode step**.
- At baseline 23,254 tok/s, decode step ≈ 43 µs/token. Savings ≈ 4.1% of step.
- Projection: **+3.5-4.5% e2e → 24,060-24,300 tok/s** (clears BIG WIN gate +3%
  → ≥23,950, easily clears PASS +2%).
- Caveat: e2e includes 4 shared prefill + overhead outside the MoE tail, so
  realized gain likely 2.5-3.5%.

## 7. KILL pattern coverage

- **§P1 silent-None**: `_try_load_unshuffle_weightedsum()` logs
  `[T2N-UNSHUF-WSUM] import failed (class=... msg=...)` exactly once on any
  ImportError/AttributeError. Public predicate `is_unshuffle_weightedsum_available()`.
- **§P2 banner**: first successful call logs
  `[T2N-UNSHUF-WSUM] active n_tokens=M topk=... k=...` once per process. Server
  logs grepping for `active n_tokens=` confirm the fast path is live.
- **§P11 Cat 1/2**: concrete single-file hook extending the proven
  `fused_shuffle_quant_wrapper.py` plugin, not cross-model transfer. Triton-only,
  no .so build dependency.
- **§P7 single-shape KILL**: microbench spans M ∈ {32, 64, 128, 256, 512, 1024}
  — speedup 3.34×–5.63× across the range. No regime mismatch concern.
- **§P5 barrier cost**: n/a (no grid.sync; single Triton launch).

## 8. Parent bench recipe

```bash
# Baseline (T2-N on, Rank3 off):
AUTOKERNEL_FUSED_SHUFFLE_QUANT=1 \
AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM=0 \
  ./launch_qwen3_fused_t2n.sh

# Rank 3 on (default — T2-N + Rank3):
AUTOKERNEL_FUSED_SHUFFLE_QUANT=1 \
AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM=1 \
  ./launch_qwen3_fused_t2n.sh

# Then: python bench_t2h_qwen3_sweep.py --concurrency 1024 \
#   --out bench_t2n_rank3.json
# Confirm [T2N-UNSHUF-WSUM] active banner in server log before trusting result.
```

Gate recap: **PASS ≥+2%** (23,700), **BIG WIN ≥+3%** (23,950), **KILL <+1%**.

## 9. Risks / future work

- `apply_router_weight_on_input=True` path still two-pass. Qwen3-30B-A3B never
  hits it; Gemma4-26B may. If Gemma4 is re-enabled, add a separate fused-sum
  kernel (without weights, just gather+sum).
- Launch overhead already accounts for ~11 µs floor — further wins require
  persistent/cooperative kernel or fusing with the GEMM2 epilogue itself.
- Upstream proposal candidate: same fusion applies to any CUTLASS MoE path.
