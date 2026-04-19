# T2-N Ceiling Analysis — Residual Non-Fused Ops

**Tag:** W5_16_t2n_ceiling_analysis  
**Date:** 2026-04-18  
**Baseline:** Qwen3-30B-A3B NVFP4, T2-N + fused-norm v2 = 23,254 gen tok/s @ C=1024  
**Prior T2-N-only peak:** 19,558 tok/s @ C=512  
**SM120 BW ceiling:** ~25-30k tok/s  
**Gap to close:** ~2-7k tok/s (9–27%)

---

## Source References

- Decoder layer forward: `workspace/qwen3_moe_copy.py` lines 413–433
- Attention forward: `workspace/qwen3_moe_copy.py` lines 341–358
- MoE block forward: `workspace/qwen3_moe_copy.py` lines 224–255
- Fused-norm v2 patched forward: `patches/wire_fused_norm_fp4_qwen3_v2.py` lines 249–314
- T2-N patched `run_cutlass_moe_fp4`: `patches/fused_shuffle_quant_wrapper.py` lines 402–478

---

## Op Coverage Table

Each row is one op in the hot-path decode forward (1 decoder layer, 48 total per Qwen3-30B-A3B).

| # | Op | File:Line | Fused-norm v2 | T2-N | Status | Est. % of step |
|---|---|---|---|---|---|---|
| 1 | `input_layernorm` (RMSNorm + residual add) | `qwen3_moe_copy.py:422-424` | YES — `fused_add_rms_norm_dynamic_fp4_quant` fuses norm+residual+FP4 quant | — | **FUSED** | ~0% (inside op 2) |
| 2 | `qkv_proj` (FP4 CUTLASS dense GEMM) | `qwen3_moe_copy.py:425` → `wire_fused_norm_fp4_qwen3_v2.py:278,240-243` | YES — matmul runs immediately after fused norm | — | **FUSED** | ~3-4% |
| 3 | `q_norm` / `k_norm` (RMSNorm per head, BF16) | `qwen3_moe_copy.py:349-354` → `wire_fused_norm_fp4_qwen3_v2.py:287-294` | NO | NO | **PLAIN** | ~1-2% |
| 4 | `rotary_emb` (RoPE, in-place BF16) | `qwen3_moe_copy.py:355` → `wire_fused_norm_fp4_qwen3_v2.py:296` | NO | NO | **PLAIN** | ~1-2% |
| 5 | `attn.attn` (paged attention / FlashInfer) | `qwen3_moe_copy.py:356` → `wire_fused_norm_fp4_qwen3_v2.py:297` | NO | NO | **PLAIN** (vendor kernel) | ~8-12% |
| 6 | `o_proj` (BF16 RowParallel GEMM) | `qwen3_moe_copy.py:357` → `wire_fused_norm_fp4_qwen3_v2.py:298` | NO | NO | **PLAIN** | ~5-7% |
| 7 | `post_attention_layernorm` (fused_add_rms_norm BF16) | `qwen3_moe_copy.py:431` → `wire_fused_norm_fp4_qwen3_v2.py:310` | NO (MoE norm feeds BF16 router + FP4 experts — net neutral to fuse FP4-only) | NO | **PLAIN** (vLLM `fused_add_rms_norm`) | ~1% |
| 8 | `gate` / router linear (BF16 ReplicatedLinear) | `qwen3_moe_copy.py:236` | NO | NO | **PLAIN** | ~1% |
| 9 | `get_cutlass_moe_mm_data` (CPU-side expert sort) | `fused_shuffle_quant_wrapper.py:430-433` | NO | NO | **PLAIN** (~3-5 µs CPU launch) | ~0.5% |
| 10 | `shuffle_rows` + `scaled_fp4_experts_quant` (GEMM1 input quant) | `fused_shuffle_quant_wrapper.py:436-438` | — | YES (fused kernel) | **FUSED** | ~1% |
| 11 | `cutlass_fp4_moe_mm` GEMM1 (W1: gate+up, FP4) | `fused_shuffle_quant_wrapper.py:444-447` | — | — | **PLAIN** (vendor kernel) | ~20-25% |
| 12 | `silu_and_mul_scaled_fp4_experts_quant` (act + GEMM2 quant) | `fused_shuffle_quant_wrapper.py:451-453` | NO | NO | **PLAIN** | ~3-5% |
| 13 | `cutlass_fp4_moe_mm` GEMM2 (W2: down, FP4) | `fused_shuffle_quant_wrapper.py:460-463` | — | — | **PLAIN** (vendor kernel) | ~15-20% |
| 14 | `shuffle_rows` (output unshuffle, c_map) | `fused_shuffle_quant_wrapper.py:466` | NO | NO | **PLAIN** | ~1-2% |
| 15 | topk-weight multiply + sum (output reduce) | `fused_shuffle_quant_wrapper.py:471-478` | NO | NO | **PLAIN** | ~1-2% |
| 16 | `shared_expert` (optional, Qwen3-30B has it) | `qwen3_moe_copy.py:237` | NO | NO | **PLAIN** (BF16 MLP, separate) | ~3-5% |

**Notes:**
- Rows 11 and 13 (`cutlass_fp4_moe_mm`) are the dominant BW consumers — they ARE theoretically optimal (vendor CUTLASS FP4 kernel, near hardware ceiling). Not a fusion target.
- Row 5 (`attn.attn`) is FlashInfer paged attention — also near-optimal; not a fusion target.
- Rows 3, 4, 12, 14, 15 are individually small but constitute meaningful aggregate overhead because each is a separate kernel launch (grid setup, stream ordering) on top of modest BW work.

---

## Top-3 Unfused Op Candidates

### Rank 1 — `silu_and_mul_scaled_fp4_experts_quant` (op 12)

**File:line:** `fused_shuffle_quant_wrapper.py:451-453` (calls `ops.silu_and_mul_scaled_fp4_experts_quant`)

**What it does:** After GEMM1 produces `c1` (BF16, shape `[M*topk, 2*N]`), this op applies SiLU+mul (the gated activation) AND immediately quantizes the result to NVFP4 for GEMM2 input. This is already a vLLM fused kernel — but it is a **separate kernel launch** from GEMM1.

**Fusion opportunity:** GEMM1 outputs BF16 to shared memory (or global memory in the epilogue). An SM-resident epilogue could perform SiLU+mul and FP4 quantization of the first GEMM's output tiles before they leave the SM, eliminating the full BF16 intermediate write of `[M*topk, 2*N]` (for Qwen3-30B-A3B: 512*8 * 2*2048 * 2 bytes = 16 MB at C=512). The subsequent `scaled_fp4_experts_quant` read would also be eliminated.

**Est. latency %:** ~4-6% of step (kernel launch overhead + BF16 write/read of large intermediate)

**Plugin hook sketch (similar to T2-N pattern):**
```python
# In patched run_cutlass_moe_fp4, replace:
#   ops.cutlass_fp4_moe_mm(c1, ...)          # GEMM1
#   int_fp4, int_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(c1, ...)
# With a custom CUTLASS epilogue kernel that fuses SiLU+mul+FP4-quant:
#   int_fp4, int_blockscale = fused_moe_gemm1_silu_quant(
#       rep_a_fp4, w1_fp4, rep_a_blockscale, w1_blockscale, w1_alphas,
#       a2_gscale, expert_offsets, blockscale_offsets, problem_sizes1
#   )
# Hook: monkey-patch run_cutlass_moe_fp4 (same pattern as T2-N wrapper in
# patches/fused_shuffle_quant_wrapper.py lines 402-480).
# The epilogue CUDA kernel writes FP4 nibbles + swizzled scales directly from
# GEMM1 output tiles, never materializing the BF16 [M*topk, 2N] tensor.
```

**Projected savings:** ~4-6% on step time. At 23,254 tok/s baseline, +4% → **+930 tok/s** (24,184 tok/s).

---

### Rank 2 — `q_norm` + `k_norm` + `rotary_emb` (ops 3+4)

**File:lines:**  
- `q_norm`: `qwen3_moe_copy.py:349-350`, wired in `wire_fused_norm_fp4_qwen3_v2.py:287-288`  
- `k_norm`: `qwen3_moe_copy.py:352-353`, wired in `wire_fused_norm_fp4_qwen3_v2.py:291-292`  
- `rotary_emb`: `qwen3_moe_copy.py:355`, wired in `wire_fused_norm_fp4_qwen3_v2.py:296`

**What they do:** After `qkv_proj` outputs BF16 `[M, H_total*head_dim]`, `qkv.split()` partitions it into q/k/v views, then q_norm and k_norm apply per-head RMSNorm (BF16 in-place, each head_dim=128), and rotary_emb applies RoPE in-place. Three separate kernel launches touching the same [M, 4096] q and [M, 512] k tensors (Qwen3-30B: 32 Q heads, 4 KV heads at head_dim=128).

**Fusion opportunity:** A single kernel can fuse: split QKV → q_norm(q) → k_norm(k) → rotary_emb(q,k). The q/k tensors are small (decode: M=1, so 4096 and 512 elements respectively), making launch overhead proportionally expensive. The fused kernel reads `qkv_proj` output once, applies all three ops, writes back.

**Est. latency %:** ~2-4% of step (3 kernel launches × small tensors, kernel launch overhead dominates at M=1 decode)

**Plugin hook sketch:**
```python
# In _patched_decoder_forward (wire_fused_norm_fp4_qwen3_v2.py line 279-296),
# after qkv, residual = self._fused_qkv_fn(hidden_states, residual):
# Replace:
#   q_by_head = attn.q_norm(q_by_head)    # separate kernel
#   k_by_head = attn.k_norm(k_by_head)    # separate kernel
#   q, k = attn.rotary_emb(positions, q, k)  # separate kernel
# With:
#   q, k = fused_qknorm_rope(
#       q, k, positions,
#       attn.q_norm.weight, attn.k_norm.weight, attn.q_norm.variance_epsilon,
#       attn.rotary_emb,
#   )
# Hook site: same file, _patched_decoder_forward, after the qkv split
# (lines 279-296). Add a lazy-built closure (similar to _fused_qkv_fn pattern)
# that holds q_norm/k_norm weights + rope params at closure time.
```

**Projected savings:** ~2-4% on step time. +3% → **+698 tok/s** (23,952 tok/s standalone).

---

### Rank 3 — `output unshuffle` + `topk-weight multiply+sum` (ops 14+15)

**File:lines:** `fused_shuffle_quant_wrapper.py:466` (`ops.shuffle_rows(c3, c_map)`) and `fused_shuffle_quant_wrapper.py:471-478` (the `c3.view * topk_weights.view * .sum(dim=1)` + `output.copy_` chain)

**What they do:** After GEMM2, `c3` is `[M*topk, K]` BF16 (sorted expert order). `shuffle_rows(c3, c_map)` unsorts it back to token order (writes `M*topk*K` BF16 to global memory). Then the topk-weight multiply and `sum(dim=1)` reduce over topk dimension to produce `[M, K]` output — this is a second pass over the same data.

**Fusion opportunity:** A single CUDA kernel can fuse: unshuffle (gather `c3[c_map[i]]`) + scale by `topk_weights[i]` + accumulate into `output[token_idx]`. No BF16 intermediate for the unshuffled buffer. Similar to T2-N's gather-fuse-write idea but on the output side.

**Est. latency %:** ~2-4% of step (shuffle_rows at M=512: ~7-10 µs; topk weighted sum: ~3-5 µs; both are small independent kernel launches)

**Plugin hook sketch:**
```python
# In fused_shuffle_quant_wrapper.py _patched_run_cutlass_moe_fp4 after GEMM2:
# Replace (lines 466-478):
#   c3 = ops.shuffle_rows(c3, c_map)
#   output.copy_((c3.view(m, topk, k) * topk_weights.view(...)).sum(dim=1), ...)
# With a fused unshuffle+weightedsum kernel:
#   fused_unshuffle_weightedsum(output, c3, c_map, topk_weights, m, topk, k)
# Hook site: same run_cutlass_moe_fp4 monkey-patch entry in
# fused_shuffle_quant_wrapper.py. Build as a companion .so alongside
# fused_shuffle_quant_sm120a.so (same csrc directory pattern).
```

**Projected savings:** ~2-3% on step time. +2.5% → **+581 tok/s** (23,835 tok/s standalone).

---

## Compound Upside Projection

If Rank 1 (GEMM1 epilogue fusion) lands:

| Stage | tok/s | vs 23,254 |
|---|---|---|
| Current (T2-N + fused-norm v2) | 23,254 | baseline |
| + Rank 1 (silu_quant epilogue) | ~24,100-24,500 | **+3.7-5.3%** |
| + Rank 2 (q/k-norm+rope) | ~24,600-25,000 | +5.8-7.5% |
| + Rank 3 (unshuffle+topk sum) | ~25,000-25,500 | +7.5-9.7% |

**Compound Rank 1 alone: +~930 tok/s (+4%), bringing the peak to ~24,184 tok/s.** This is the highest-leverage single op since it eliminates the 16 MB BF16 inter-GEMM write rather than just kernel launch overhead.

---

## Key Insight

The two existing plugins (T2-N and fused-norm v2) have captured the two highest-leverage fusion points: the BF16 norm→FP4 quant intermediate (fused-norm v2) and the BF16 shuffle intermediate before GEMM1 (T2-N). The remaining gap to the 25-30k ceiling is dominated by **two inter-GEMM BF16 intermediates** that neither plugin touches:

1. The `[M*topk, 2N]` BF16 buffer between GEMM1 and the SiLU+FP4-quant (Rank 1 — ~16 MB at C=512)
2. The `[M*topk, K]` BF16 buffer between GEMM2 and the unshuffle+topk-reduce (Rank 3 — ~4 MB)

Both are MoE-internal and require a custom CUTLASS epilogue or a lightweight fused scatter-reduce kernel respectively. Rank 1 is higher ROI because it combines a gated activation (arithmetic) with quantization (memory format transform) — the fused epilogue can amortize both over the same GEMM1 output tile while it is still in registers, avoiding a full global-memory round-trip.
