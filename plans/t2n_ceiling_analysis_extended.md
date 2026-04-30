# T2-N Ceiling Analysis — Extended (Ranks 4/5/6)

**Tag:** W7_T2N_ceiling_rank456
**Date:** 2026-04-18
**Baseline:** Qwen3-30B-A3B NVFP4, T2-N + fused-norm v2 = 23,254 gen tok/s @ C=1024
**SM120 BW ceiling:** ~25-30k tok/s (gap ~2-7k, 9-27%)
**Rank 1 (SiLU+quant epilogue):** KILL'd 2026-04-18 (vendor-CUTLASS-only; no Python hook possible, see §Rank1-RETRY below)
**Rank 2 (q_norm+k_norm+rotary):** in progress
**Rank 3 (unshuffle+weightedsum):** in progress

---

## Source References (verified cite-level)

- Decoder forward: `workspace/qwen3_moe_copy.py:413-433`
- Attention forward: `workspace/qwen3_moe_copy.py:340-358`
- MoE block forward: `workspace/qwen3_moe_copy.py:224-255`
- `SharedFusedMoE.forward` (calls router → experts): `.venv/.../fused_moe/shared_fused_moe.py:21-58`
- `FusedTopKRouter._compute_routing` → `ops.topk_softmax`: `.venv/.../fused_moe/router/fused_topk_router.py:17-32, 149-165`
- `Attention.forward` (decode path, split KV-cache-update op): `.venv/.../layers/attention/attention.py:398-500`
  - Crucially: `torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)` at **line 479-481** is a **separate kernel** from `unified_attention_with_output` on backends where `forward_includes_kv_cache_update` is False (FlashInfer decode path on SM120).
- `LogitsProcessor.forward` / `_get_logits`: `.venv/.../layers/logits_processor.py:54-104` — one BF16 matmul + optional TP all-gather, runs ONCE per forward (not per-layer; amortized over 48 layers).
- Model-level final `RMSNorm`: `workspace/qwen3_moe_copy.py:467, 509` — once per forward.
- Fused-norm v2 patched forward: `patches/wire_fused_norm_fp4_qwen3_v2.py:249-314`
- T2-N patched `run_cutlass_moe_fp4`: `patches/fused_shuffle_quant_wrapper.py:594-688`

---

## Full op-trace table (Rank 1-6 assignments)

Each row = one op in hot-path decode forward. `% step` is **per-decoder-layer** except where marked `(1×/fwd)` meaning once-per-forward and amortized across 48 layers.

| # | Op | Ref | Status | % step | Rank |
|---|---|---|---|---|---|
| A | `embed_tokens` (VocabParallelEmbedding lookup) | qwen3_moe_copy.py:486 | PLAIN, gather | <0.2% (1×/fwd) | — (amortized, not worth) |
| 1 | `input_layernorm` (RMS + residual add) | 422-424 | FUSED v2 | inside #2 | fused-norm v2 |
| 2 | `qkv_proj` (FP4 GEMM) | 425, v2:278 | FUSED v2 | ~3-4% | fused-norm v2 |
| 3 | `q_norm` (RMSNorm per head BF16) | 349, v2:287 | PLAIN | ~0.8-1.2% | **Rank 2** |
| 4 | `k_norm` (RMSNorm per head BF16) | 353, v2:293 | PLAIN | ~0.4-0.6% | **Rank 2** |
| 5 | `rotary_emb` (RoPE in-place) | 355, v2:296 | PLAIN | ~0.8-1.2% | **Rank 2** |
| 5b | `unified_kv_cache_update` (**separate op** on FlashInfer decode, BF16→FP8/BF16 KV write) | attention.py:479-481 | PLAIN (vendor) | ~1-2% | **Rank 4 (new)** |
| 6 | `attn.attn` (FlashInfer paged attn) | v2:297 | PLAIN (vendor) | ~8-12% | — (near-optimal) |
| 7 | `o_proj` (BF16 RowParallel GEMM) | v2:298 | PLAIN | ~5-7% | — (pure GEMM, no fusion partner) |
| 8 | `post_attention_layernorm` (RMS+residual, BF16 out) | 431 | PLAIN | ~1% | **Rank 5 (new)** |
| 9 | `gate` router linear (BF16 ReplicatedLinear) | 236 | PLAIN | ~0.5-0.8% | **Rank 5 (new)** |
| 10 | `ops.topk_softmax` (fused vendor kernel) | fused_topk_router.py:24 | FUSED (vendor) | ~0.4% | — (already fused upstream) |
| 11 | `get_cutlass_moe_mm_data` (CPU-side offsets) | wrapper:622 | PLAIN | ~0.3-0.5% | — (CPU, not GPU work) |
| 12 | `fused_shuffle_quant` GEMM1-input (T2-N) | wrapper:628 | FUSED T2-N | ~0.5% | T2-N |
| 13 | `cutlass_fp4_moe_mm` GEMM1 (W1 gate+up) | wrapper:636 | PLAIN (vendor) | ~20-25% | — (near-optimal) |
| 14 | `silu_and_mul_scaled_fp4_experts_quant` | wrapper:653 | PLAIN (vendor) | ~3-5% | Rank 1 **KILL** |
| 15 | `cutlass_fp4_moe_mm` GEMM2 (W2 down) | wrapper:662 | PLAIN (vendor) | ~15-20% | — (near-optimal) |
| 16 | `shuffle_rows` (output unshuffle c_map) | wrapper:668 | PLAIN | ~1-2% | **Rank 3** |
| 17 | topk-weight multiply + `.sum(dim=1)` (output reduce) | wrapper:672-678 | PLAIN | ~1-2% | **Rank 3** |
| 18 | `shared_expert` (BF16 MLP: gate_up + SiLU + down) | qwen3_moe_copy.py:237 via 123-131 | PLAIN (3 kernels) | ~3-5% | **Rank 6 (new)** |
| B | final model `RMSNorm` (after all layers) | qwen3_moe_copy.py:509 | PLAIN | <0.1% (1×/fwd) | — (amortized) |
| C | `lm_head` (BF16 ParallelLMHead matmul, H × vocab=151,936) | logits_processor.py:96 | PLAIN | ~0.8-1.5% (1×/fwd, ~0.02-0.03%/layer-equivalent) | — (1×/fwd, near-BW, no fusion partner) |
| D | `logits_processor` scale/soft_cap | logits_processor.py:66-72 | PLAIN | <0.1% (1×/fwd) | — |
| E | sampler argmax / topk | V1 sampler | PLAIN | ~0.1% (1×/fwd) | — |

---

## Rank 4 — `unified_kv_cache_update` + preceding `rotary_emb` write

**Hook site:** `patches/wire_fused_norm_fp4_qwen3_v2.py:296-297` (between `rotary_emb` and `attn.attn`).

**What it does:** On the FlashInfer decode backend, `attn_backend.forward_includes_kv_cache_update` is False, so `torch.ops.vllm.unified_kv_cache_update(key, value, layer_name)` runs as a **separate kernel launch** that scatter-writes `[M, num_kv_heads*head_dim]` of key/value into the paged KV cache (attention.py:479-481). At M=1 decode this is 512 elements for K and 512 for V — tiny, launch-overhead dominated.

**Fusion feasibility:** **Python-hookable**. The RoPE op writes K in-place already; the next op's only other reader of K is the KV-cache-update. A fused kernel can apply RoPE to K AND scatter-write the rotated K into the paged cache slot in one pass (eliminating one full K roundtrip). Additionally, the V scatter-write can be co-scheduled on the same launch. This is a sibling to Rank 2 — if Rank 2 lands we can extend its kernel to also emit the KV-cache scatter.

**Latency estimate:** ~1-2% of step (one kernel launch + small scatter; ~3-5 µs at M=1 on SM120 per launch × 48 layers).

**§P11 category:** **Cat 1 (bug-fix/code-level) PROCEED, P~0.7** — a concrete op visible in vLLM source; the kernel is small and contained; the fusion boundary is clean (write side of RoPE → scatter). Not cross-model cross-apply.

**Projected gain:** +1.5-2.5% step → **+350-600 tok/s** (23,600-23,850 standalone).

**Effort:** 2-3 days. Extends Rank 2 kernel. Requires paged-KV layout knowledge; FP8 KV cache adds a dequant-on-read dimension but no extra fusion work on the write side.

**One-line hook spec:** Extend Rank 2 `fused_qknorm_rope_kvwrite` kernel to also emit paged-KV scatter write for `key` (and raw `value`) into the `kv_cache` slot at the layer, replacing the subsequent `unified_kv_cache_update` torch op.

---

## Rank 5 — `post_attention_layernorm` + router-gate BF16 linear

**Hook site:** `patches/wire_fused_norm_fp4_qwen3_v2.py:310` (the `self.post_attention_layernorm(hidden_states, residual)` call) into the `gate(hidden_states)` call at `qwen3_moe_copy.py:236` inside the MoE block.

**What it does:** After attention, `post_attention_layernorm` applies RMSNorm+residual (two reads, one write of `[M, H=2048]` BF16). Immediately after, the first consumer of the normed output is `router_logits, _ = self.gate(hidden_states)` — a small BF16 `[M, H=2048] @ [H=2048, E=128]` matmul (262 KB weight, tiny output `[M, 128]`). Both are individually small but are launched as separate kernels. The normed output ALSO feeds the FP4 experts (via fused_shuffle_quant) AND, when `use_overlapped=True`, the shared expert in parallel.

**Fusion feasibility:** **Python-hookable**. A fused BF16 kernel can perform: (residual add) → RMSNorm → write normed BF16 to gmem once → route-GEMM (reading from registers/smem, producing router_logits directly). The normed output still needs to materialize in gmem for the downstream FP4 quant path, but the route-GEMM epilogue can be stolen into the last warp-group of the norm kernel (the route-weight matrix is small enough to live in smem tiled once per block at H=2048, E=128 → 512 KB — within SM120 smem budget when tiled along H).

**Latency estimate:** ~1-1.5% step (norm ~0.7% + gate ~0.5%; combined launch overhead saved at M=1 decode).

**§P11 category:** **Cat 1 PROCEED, P~0.55** — the concept is clean (fused norm+route-linear) but this is NOT the same fused-norm kernel as v2 (which produces FP4); a new BF16-norm-out + BF16-matmul kernel is required. Mid-effort, well-defined boundary. Downgraded from the `post_attention_layernorm` line in the original analysis which noted "net neutral to fuse FP4-only" — that caveat applies ONLY to fusing norm+experts (two consumers); fusing norm+router is NOT neutral because router has a single BF16 consumer chain.

**Projected gain:** +1.2-1.8% step → **+280-420 tok/s** (23,530-23,670 standalone).

**Effort:** 4-5 days. New CUDA kernel (BF16 RMSNorm epilogue with small GEMM tile); needs correctness against vLLM's `ops.topk_softmax` downstream consumer.

**One-line hook spec:** Monkey-patch `Qwen3MoeDecoderLayer.forward` to call a new `fused_postnorm_gate(hidden_states, residual, norm_weight, gate_weight) -> (hidden_normed, residual, router_logits)` kernel before handing `hidden_normed` + `router_logits` into `self.mlp.experts(...)`.

---

## Rank 6 — `shared_expert` MLP collapse (gate_up + SiLU + down)

**Hook site:** `workspace/qwen3_moe_copy.py:123-131` (`Qwen3MoeMLP.forward` used for the shared expert in Qwen3-30B-A3B; `shared_expert_intermediate_size=768` per Qwen3-30B config).

**What it does:** The shared expert is a 3-kernel BF16 MLP: `gate_up_proj` (BF16 `[M, 2048] @ [2048, 1536]` for the concatenated gate+up), `SiluAndMul` (BF16 activation over `[M, 1536]`), `down_proj` (BF16 `[M, 768] @ [768, 2048]`). At M=1 decode, each GEMM is launch-overhead-dominated — the intermediate BF16 `[M, 768]` and `[M, 1536]` writes/reads are small but each is a separate kernel launch. It runs in parallel with the MoE experts (`use_overlapped=True`) on a second stream, BUT competes for SM resources on SM120 (no fine-grain MPS) and dominates when the MoE main stream has brief idle windows.

**Fusion feasibility:** **Python-hookable**. Replace the 3-kernel BF16 MLP with a single fused gate_up+SiLU+down BF16 kernel (classic "FusedMLP" pattern, already implemented as a Triton kernel type in this repo — `kernels/fused_mlp` per CLAUDE.md). The shared expert is BF16-only (not quantized), so no FP4 scale juggling. At M=1 this collapses 3 launches → 1, and fuses the `[M, 1536]` SiLU intermediate into registers.

**Latency estimate:** ~3-5% step (the three shared-expert kernels consume meaningful time because each is ~5-10 µs at M=1 and they run serial-on-stream; overlapping with MoE hides some but not all of this).

**§P11 category:** **Cat 1 PROCEED, P~0.75** — shared-expert fusion is the classic FusedMLP win (we already have a kernel template for it); same-model, same-regime. The only risk is whether the overlap with MoE already hides enough of the cost at C=1024 to make the standalone improvement marginal. Worth banking.

**Projected gain:** +1.5-3% step → **+350-700 tok/s** (23,600-23,950 standalone). Lower end of range if overlap already hides 50% of the cost; upper end if the 3 separate launches each stall the CUDA graph.

**Effort:** 3-4 days. Reuse `kernels/fused_mlp` template; add a monkey-patch for `Qwen3MoeMLP.forward` that intercepts the shared-expert path only (not the dense-MLP-only layers, which would cause regressions at prefill).

**One-line hook spec:** Monkey-patch `Qwen3MoeMLP.forward` on the shared-expert instance (check `self.expert_gate is not None` OR patch only the layer's `self_attn.mlp.shared_expert` attribute) to call a fused BF16 `[gate_up_proj → SiLU+mul → down_proj]` kernel.

---

## Compound projection (if Ranks 2-6 all land mid-P)

Using mid-point of each range (assumes NOT all gains are independent — MoE-internal ranks share the same SM resources, so we apply a 15% overlap discount to the MoE-side ranks 3+6):

| Stage | tok/s | vs 23,254 |
|---|---|---|
| Current (T2-N + fused-norm v2) | 23,254 | baseline |
| + Rank 2 (q/k-norm+rope, +3%) | 23,950 | +3.0% |
| + Rank 3 (unshuffle+weightedsum, +2.5%, ×0.85 overlap) | 24,460 | +5.2% |
| + Rank 4 (KV-cache-update fusion, +2%) | 24,950 | +7.3% |
| + Rank 5 (postnorm+gate BF16, +1.5%) | 25,320 | +8.9% |
| + Rank 6 (shared-expert FusedMLP, +2.3%, ×0.75 overlap) | 25,760 | +10.8% |

**Compound projection: 25,700-25,900 tok/s** — lands at the lower end of the 25-30k SM120 BW ceiling. Remaining ~5-15% headroom is vendor-GEMM-only territory (ops 13, 15, 6, 7 = cutlass FP4 and FlashInfer attention).

If we additionally pessimize (Ranks 5+6 each deliver only lower-bound gain):

- Pessimistic: 23,254 → ~25,100 (+8%)
- Optimistic: 23,254 → ~26,200 (+13%)

---

## Rank 1 — RETRY candidacy

Rank 1 (SiLU+quant epilogue) was KILL'd because it required modifying the vendor CUTLASS FP4 MoE kernel's epilogue — **not Python-hookable**. A retry is ONLY viable as an **upstream PR to the vLLM CUTLASS MoE kernel** (`cutlass_moe.py` + the cutlass C++ kernel `csrc/moe/cutlass_moe_mm_fp4.cu`). 

**Recommendation:** File as **upstream PR request to vLLM** (cc'ing NVIDIA CUTLASS team). The 16 MB BF16 intermediate elimination is real and compounds with every MoE model on SM100/SM120, not just Qwen3-30B. Re-scope as UPSTREAM_RANK1 — not an autokernel patch plugin.

---

## Vendor-kernel-only items worth filing as upstream PR requests

| Item | Location | Why upstream | Projected e2e |
|---|---|---|---|
| **SiLU+FP4-quant GEMM1 epilogue** (old Rank 1) | `cutlass_moe.py` + CUTLASS FP4 MoE epilogue | Requires CUTLASS epilogue modification | +4-6% |
| **FP8 KV-cache + unified_attention fusion** (Rank 4 alt path) | FlashInfer backend | The split KV-update op is an artifact of the FlashInfer integration API | +1-2% |
| **`o_proj` + `post_attention_layernorm` epilogue fusion** | BF16 cuBLAS path | Would fuse op #7 write with op #8 read | +2-3% |

None of these are Python-hookable — they all require modifying the matmul kernel epilogue. Ranks 2/3/4/5/6 in this plan are the exhaustive set of **Python-hookable** fusion candidates remaining.

---

## Recommendation: next rank parent to implement (after Ranks 2/3 bench)

1. **Rank 4** (extends Rank 2 kernel) — highest P, lowest marginal effort once Rank 2 lands. Adds 1 day to the Rank 2 CUDA kernel. Ship Rank 4 as an "extension" to Rank 2's .so. **Do this FIRST after Rank 2 banks.**
2. **Rank 6** (shared-expert FusedMLP) — P~0.75, uses existing `kernels/fused_mlp` template, 3-4 day standalone effort. Ship independently of Ranks 2/3/4.
3. **Rank 5** (postnorm+gate) — P~0.55, 4-5 day effort, lower projected gain. Deprioritize until 4 and 6 have banked, then reassess headroom remaining.

**If only ONE more rank is implemented: Rank 4.** It extends existing work (Rank 2 kernel), has the clearest hook boundary, and is the lowest-risk Cat 1 bug-fix-style patch. Projected to push Qwen3-30B-A3B from 23,254 → ~24,400 standalone, ~24,950 combined with Ranks 2+3.

---

## Applied §P11 KILL_PATTERNS categorization summary

| Rank | §P11 category | P (projected) | Evidence type |
|---|---|---|---|
| 1 (KILL) | Cat 1 → vendor-only | N/A | Correctly KILL'd: no Python hook |
| 2 (IN-PROG) | Cat 1 | 0.7 | Code-level: 3 distinct kernel launches visible at v2:287-296 |
| 3 (IN-PROG) | Cat 1 | 0.65 | Code-level: 2 distinct kernels at wrapper:668-678 |
| **4 (NEW)** | **Cat 1** | **0.70** | Code-level: `unified_kv_cache_update` at attention.py:479-481 is a literal separate op |
| **5 (NEW)** | **Cat 1** | **0.55** | Code-level: two separate kernels (postnorm + gate.matmul); new kernel not an extension |
| **6 (NEW)** | **Cat 1** | **0.75** | Code-level: 3-kernel BF16 MLP at qwen3_moe_copy.py:123-131; reuses `kernels/fused_mlp` template |

No Cat 3 (cross-apply) or Cat 4 (literature) PROCEEDs in this analysis — all are directly cited from in-tree vLLM source.
