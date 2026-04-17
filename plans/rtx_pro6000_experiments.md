# RTX PRO 6000 Experiments — from the aigpu repo survey

**Date:** 2026-04-16
**Hardware:** 1× RTX PRO 6000 (96 GB) + 1× RTX PRO 6000 Max-Q (96 GB), Blackwell **SM120**, PCIe-only
**Source repos surveyed** (all under `/home/cklaus/projects/aigpu/`):
- `llm-compressor` — vLLM's quantization toolkit (NVFP4/MXFP4/FP8 block/AutoRound/SmoothQuant/model-free PTQ)
- `I-DLM` — Introspective Diffusion LM, claims 3.8× AR throughput with AR-equivalent quality
- `gpu_bitonic_sort` — cooperative CUDA recursive sort (Taelin)
- `ddtree` — block-diffusion draft-tree speculative decoding (DFlash draft + tree verify)
- `llama-cpp-turboquant` / `triattention` / `turboquant-gpu` — (already in tree, not newly cloned)

**Builds on:** `plans/future_work.md`, `plans/pro6000_projections.md`, `SESSION_FINAL_STATUS.md` (baseline 6,685 tok/s on 5090).

---

## ASI-0. Roofline constraint — what CAN'T be improved

Before running ANY kernel experiment, check where the operation sits on the SM120 roofline. `plans/sm120_attention_kernel.md` proves:

| Operation | Achieved utilization | Theoretical ceiling | Max possible gain |
|---|---|---|---|
| FA2 BF16 decode attention | 93% bandwidth | 100% bandwidth | **1.08× (8 %) — DOA** |
| NVFP4 matmul (cuBLASLt) | 75 % peak (1,261 TFLOPS) | 1,680 TFLOPS theoretical | **1.33×** |
| MoE routing + norms | ~15 % of step, Python-bound | ~0 % if fused | **0.15× step savings** |

**Implication for experiment ordering:** the ONLY attention speedup is **halving the data** (FP8 / FP4 KV → T2-I). BF16 attention experiments are at the roofline ceiling — they cannot move the needle. The plan's priority must be:

1. **Reduce total bytes moved** → FusenCache / TurboQuant / TriAttention / K2V2 / FP8 KV (T1-E, T1-F, T2-I, T2-L, T3-P)
2. **Increase batch to amortize weight reads** → batch tuning / scheduler / 96 GB headroom (T1-A)
3. **Reduce launches + overhead** → fused kernels / CUDA graphs / persistent (T2-F, T2-N, T3-S) — ~7 % ceiling, diminishing
4. **Everything else** → within noise unless it unlocks a new operating regime (spec decode, disaggregated, etc.)

---

## ASI-1. Disaggregated prefill/decode serving (1P1D) — MISSING from all prior passes

`plans/disaggregated_serving.md` is a **fully specified, zero-kernel-work configuration** that was never added to this plan. It is a third GPU-topology option alongside DP=2 (throughput) and SSD (latency):

| Mode | GPU 0 | GPU 1 | Optimizes for |
|---|---|---|---|
| DP=2 (T3-K) | prefill + decode | prefill + decode | Aggregate tok/s at C≥64 |
| SSD (T4-C) | verify | draft | Single-stream latency at C=1–4 |
| **1P1D (this)** | **prefill only** | **decode only** | **P99 TTFT + decode stability under mixed load** |

vLLM 0.18.1 supports this natively via `kv_role=kv_producer / kv_consumer` + `P2pNcclConnector` over PCIe. Full launch scripts are in `disaggregated_serving.md:57-98`.

**Projected impact** (from the spec):
- P99 TTFT at C=8: 640 ms (collocated) → 120 ms (disaggregated) = **5.3×**
- Decode tok/s under heavy prefill: 15 tok/s → 55 tok/s = **3.7×**
- KV transfer overhead: ~19 ms per request (536 MB @ 28 GB/s PCIe 4.0); halves to 10 ms with FP8 KV.

**When to use vs DP=2:** 1P1D dominates when the workload has **mixed prompt lengths AND latency SLA matters** (e.g., interactive serving, multi-turn chat). DP=2 dominates for pure throughput benchmarks. The user's actual workload profile (fusen_solver + multi-agent) is mixed → 1P1D is likely the better default.

- **What to do:** launch two vLLM instances per `disaggregated_serving.md:57-98`, put an nginx proxy in front, run `bench_serving.py` at C ∈ {4, 8, 16, 64, 128} with a bimodal prompt-length distribution (50 % short 256-tok, 50 % long 4K-tok). Compare P50/P99 TTFT and decode tok/s vs T1-A (single-GPU) and T1-G / T3-K (DP=2).
- **Run alongside T1-G:** day-0 measures both topologies in the same session.
- **Kill criterion:** if P99 TTFT under mixed load is < 1.5× better than DP=2 at C=64, the KV transfer overhead dominates — stick with DP=2.

---

## ASI-2. Early KV termination — free capacity multiplier, zero kernel work

`plans/early_kv_termination.md` documents that 40–60 % of generated tokens are never consumed by the client. KV blocks stay allocated until EOS. At C=256+ on PRO 6000, this is the difference between fitting and thrashing.

**Layer 1 — budget token hint + priority scheduling (config changes, 0 effort):**
Set `default_max_tokens` per task type in fusen_solver: code-gen=2048, Q&A=512, summarization=256, chat=512. This eliminates the longest tail. **Also enable `--scheduling-policy priority`** on the vLLM server (`plans/request_priority_queuing.md`) and map `max_tokens` → priority tier (≤50 → P1, ≤200 → P2, ≤500 → P3, >500 → P4). SJF approximation that improves P50 latency for short requests under mixed load — zero GPU work, just a CLI flag + 10-line priority mapper in fusen_solver.

**Layer 2 — output-driven early abort (thin middleware, 2 h):**
`EarlyAbortStreamWrapper` (code already in `early_kv_termination.md:142-163`) monitors the output stream for semantic completion signals (`</tool_call>`, `</answer>`, `</think>`) and calls `engine.abort(request_id)` immediately, freeing KV blocks on the GPU side. Synthesize a final chunk with `finish_reason: "early_abort"`.

**Layer 3 — multi-agent racing abort (fusen_solver change, 4 h):**
When fusen_solver launches N agents (best-of-N) and one wins, abort the other N-1 immediately. Currently they all run to EOS, wasting KV.

- **What to do:** implement Layer 1 (today), then Layer 2 + 3 in Week 1 alongside T1-A benchmarking. Measure KV block utilization with and without early abort at C=256, C=512.
- **Why PRO 6000:** at 96 GB the impact is proportional to concurrency. At C=512 with 128K context, 40 % KV reclaim = 200 more concurrent slots.
- **Kill criterion:** none — Layer 1 is free and correct by construction. Layer 2 requires testing that abort doesn't corrupt in-flight CUDA graph state.

---

## ASI-3. Interaction-effect compatibility matrix

Several experiments are **mutually exclusive or interact non-obviously**. The plan has 30 experiments but doesn't flag which can run together. Here is the compatibility map for KV-format and GPU-topology experiments:

### KV format compatibility (pick ONE per deployment)

| | FusenCache k4v4 | TurboQuant 2/3-bit | TriAttention | FP8 KV (native) | K2V2 (int2) |
|---|---|---|---|---|---|
| **fp8_decode_attention (T2-I)** | ❌ (wrong dtype) | ❌ (wrong dtype) | ✅ (different layer) | ✅ (native) | ❌ |
| **Prefix cache (T2-J)** | ⚠️ untested | ❌ (rotated keys) | ⚠️ untested | ✅ | ❌ |
| **Disaggregated 1P1D (ASI-1)** | ⚠️ (KV transfer must handle compressed format) | ❌ (no NCCL path for codebook) | ⚠️ untested | ✅ | ❌ |
| **DP=2 (T3-K)** | ✅ | ✅ | ✅ | ✅ | ✅ |

### GPU topology compatibility (pick ONE)

| | DP=2 | 1P1D (disaggregated) | SSD (async draft) | Single-GPU |
|---|---|---|---|---|
| Can stack with other? | No | No | No | N/A (default) |
| Best for | C≥64 throughput | Mixed workload + latency SLA | C=1–4 latency | Simplest baseline |

**Action:** before deploying any combination, verify the pair works. The plan should group experiments into **deployment configurations**:

- **Config A (throughput max):** T1-A baseline → T2-I (FP8 decode attn) → FP8 KV → DP=2 → T1-A batch sweep
- **Config B (mixed workload):** T1-A → ASI-1 (disaggregated) → FP8 KV → ASI-2 (early KV abort) → T1-F (TriAttention on prefill GPU)
- **Config C (long-context frontier):** T1-A → FusenCache k4v4 → T3-P (K2V2 quality gate) → T3-L/T (semantic eviction) → single-GPU

---

## ASI-4. Workload-topology decision matrix

| Workload profile | Dominant constraint | Best topology | Best KV format | Key experiments |
|---|---|---|---|---|
| **Batch throughput** (C≥128, 4K ctx) | KV capacity + batch overhead | DP=2 | FP8 KV or FusenCache | T1-A, T1-G, T3-K |
| **Interactive multi-turn** (C=8–32, 4K ctx, mixed prompt lengths) | P99 TTFT + decode stability | **Disaggregated 1P1D** | FP8 KV | ASI-1, ASI-2, T2-I |
| **Long reasoning** (C=1–4, 32K–128K ctx) | KV capacity per request | Single-GPU | FusenCache / TriAttention / K2V2 | T1-E/F, T3-P, T3-L |
| **Low-latency single-stream** (C=1, 4K ctx) | Per-step decode latency | SSD (async draft) | FP8 KV | T4-C, T3-R, T4-B |
| **Agent swarm** (fusen_solver, N=6–8 parallel agents) | KV capacity × N + TTFT | Disaggregated 1P1D | FP8 KV + ASI-2 Layer 3 | ASI-1, ASI-2, T2-I |

---

## 0. Hardware-truth guardrails

SM120 consumer Blackwell ≠ SM100 datacenter Blackwell. The PRO 6000 inherits SM120, so the following are **not available** and should not appear in any experiment here:

- TMEM (tensor memory) / `tcgen05`
- WGMMA
- Multi-SM clusters (cluster shape is 1×1×1)
- TMA multicast
- SM100-only cubins (e.g. DeepGEMM — fails with `CUDA_ERROR_NO_BINARY_FOR_GPU`, discovery #24)
- **I-DLM SGLang CuTe kernels** — `inference/sglang/sglang/jit_kernel/flash_attention/cute/interface.py:251` asserts `compute_capability in [9, 10, 11]`. SM120 is 12 → the assert fails. `blackwell_helpers.py` also uses `tcgen05.mma.*` PTX + TMEM pointers (lines 168, 189, 301) that SM120 does not have. **Any I-DLM run on PRO 6000 must force the SM90 fallback path (`flash_fwd_sm90.py`, selected at `interface.py:430-478`) or use a non-CuTe attention backend.**

What **is** available and matters:
- Native NVFP4 / FP6 / FP8 block-scaled MMA via CUTLASS 3.x / `torch._scaled_mm_v2`
- Native 2:4 sparse MMA silicon — **but vLLM dropped 2:4 support in PR #36799** (`llm-compressor/examples/sparse_2of4_quantization_fp8/README.md`); offline PPL only
- `cvt.rn.satfinite.e2m1x2.f32` (inline PTX only; not Triton)
- 96 GB GDDR7 per GPU — the *real* PRO 6000 lever vs the 32 GB 5090
- cuBLASLt FP8/FP4 fast paths (already dominant in autokernel benches)

---

## Tier 1 — Ship first (direct capacity / throughput wins)

### T1-A. PRO 6000 baseline at max-batch
Reproduce the 5090's 6,685 tok/s config, then let the 3× HBM eat.

| Step | Command / file | What to record |
|---|---|---|
| Reproduce 5090 peak on PRO 6000 | `bench_fusencache.py` k4v4b64 eager, Gemma-4 26B-A4B NVFP4 | tok/s @ C={1,32,64,128,256,512,1024} |
| Sweep scheduler headroom | `bench_concurrency_sweep.py` with `max_num_seqs ∈ {128,256,512,1024}`, `max_num_batched_tokens ∈ {2048,4096,8192,16384}` | Plateau point; find SM120's new knee |
| Max-context soak | 128 K-ctx, C=3–4 | OOM?, per-step latency, verify `plans/pro6000_projections.md` line 134 |

**Prereq (if running FusenCache k4v4):** `plans/fusencache_nvfp4_integration.md` documents 3 blockers that must be fixed first: (1) `logits_soft_cap` not in Triton decode kernel (3-line fix, `cap * tanh(score / cap)` after QK^T), (2) CUDA graph support declared `NEVER` (change to sync store mode), (3) sliding-window prefill mask missing for Gemma4's window=1024 layers. These are bugs, not experiments — fix them before T1-A or fall back to BF16 KV for the initial baseline.

**Success:** ≥ 10 k tok/s at C=512 on a single PRO 6000 without code changes. If not, T1-B is the unblocker.

### T1-B. Piecewise CUDA-graph race fix
Open blocker in `FUSENCACHE_VLLM_MAIN_STATUS.md:114-125`. Mixed-batch capture crashes `cudaErrorIllegalInstruction` at B≥48. The 5090 workaround was `FULL_DECODE_ONLY`, which costs ~30 %.

- **Approach:** no-clone metadata in piecewise path — don't re-allocate metadata tensors inside the captured region; bind into the graph's memory pool instead.
- **Touch points:** `vllm_patches/` (piecewise scheduler), `fusencache/` decode kernel launcher.
- **Validation:** capture succeeds at B=128; graph replay matches eager within 1 e-3 RMS.
- **Expected:** 6,685 → ~9 k tok/s on the *same* single-GPU stack, before any PRO 6000 capacity gain.

### T1-C. AutoRound-NVFP4 recalibrate Gemma-4 & Qwen3-MoE
Current autokernel quantization is plain RTN via modelopt. `llm-compressor` ships a learnable-rounding modifier that's 10–20 % better on sub-4-bit (per `examples/autoround/README.md`). Better quality → permits pushing `down_proj` / expert GEMMs to NVFP4 that we currently keep FP8 or BF16.

- **Entry:** `llm-compressor/src/llmcompressor/modifiers/autoround/base.py:35`
- **Template:** `llm-compressor/examples/autoround/quantization_w4a4_fp4/llama3.1_example.py`
- **Recipe:**
  ```python
  from llmcompressor.modifiers.autoround import AutoRoundModifier
  recipe = AutoRoundModifier(scheme="NVFP4A16", iters=200, batch_size=8)
  oneshot(model=model, recipe=recipe, dataset=ds, num_calibration_samples=256)
  ```
- **Runs:** Gemma-4 26B-A4B; Qwen3-30B-MoE. 256 calib samples of C4.
- **Metrics:** WikiText-2 / C4 PPL vs current RTN NVFP4 (autokernel: 701.4 on WikiText — sanity baseline); vLLM tok/s delta at C=256; if PPL improves, re-run with attention also NVFP4 to see if the 4× FlashInfer-FP8 Gemma4 penalty disappears.
- **Budget:** ~2–6 h calib per model on a single PRO 6000.

### T1-D. 200 B-class single-GPU NVFP4 (DeepSeek-V3 / Qwen3-235B-A22B)
The headline use of 96 GB. Use `llm-compressor`'s model-free PTQ, which skips `transformers` and streams safetensors.

- **Entry:** `llm-compressor/src/llmcompressor/entrypoints/model_free/process.py:65`
- **Prereq:** `reindex_fused_weights` CLI (colocate fused QKV before microscale quant — `examples/model_free_ptq/README.md:41-46`)
- **Template:** `llm-compressor/examples/model_free_ptq/deepseek_r1_nvfp4_fp8_block.py:18`
- **Validation:** PPL on WikiText-2; vLLM serve with `max_num_seqs=32`, C=8..64, record tok/s.
- **Expected novelty:** first autokernel run at 200B scale; forces a new batch / KV-budget regime (`pro6000_projections.md` predicts ~60× concurrency at 4K ctx for 235B NVFP4 on a single PRO 6000).

---

## Tier 2 — Moderate effort, likely useful

### T2-E. Mixed NVFP4 + FP8 per-layer scheme
Autokernel already knows: RedHat's all-NVFP4 Gemma underflows in QKV-fused scale (`max()` over heads). Plain BF16 attn works but wastes memory. The right answer is probably **attn NVFP4 with per-head scales, `down_proj` FP8**. `llm-compressor` supports regex-targeted multi-scheme recipes:

- **File:** `llm-compressor/examples/quantization_non_uniform/quantization_nvfp4_fp8.py:54-65`
- **Recipe sketch:**
  ```python
  scheme_fp8 = FP8_DYNAMIC.copy(); scheme_fp8["targets"] = ["re:.*down_proj.*"]
  scheme_nvfp4 = NVFP4.copy();     scheme_nvfp4["targets"] = ["re:.*self_attn.*", "re:.*up_proj.*", "re:.*gate_proj.*"]
  recipe = QuantizationModifier(config_groups={"g0": scheme_fp8, "g1": scheme_nvfp4})
  ```
- **Models:** Gemma-4 26B-A4B, Llama-3.3 70B, Qwen3-30B-MoE.
- **Metrics:** PPL; decode latency at B=1/32; VRAM; check for scale-underflow by diffing vs `fix_nvfp4_attn_to_bf16.py` outputs.

### T2-F. Wire the existing C++ fused norm+FP4-quant into vLLM (dense paths only)
Already built and benched (2.95× kernel-local, +12.9 % projected end-to-end, `SESSION_FINAL_STATUS_V2.md`). Never integrated because of the MoE shuffle-between-norm-and-quant data dependency. Integrate on the dense path first (KV proj, up/gate), defer MoE.

- **Kernel:** `kernels/fused_norm_fp4.py` + the SM120 PTX variant with `cvt.rn.satfinite.e2m1x2.f32`.
- **Patch surface:** vLLM's `custom_ops` registry; autokernel's plugin registry (`autokernel_v2/plugin_registry/`).
- **Metric:** per-step latency on dense layers only; end-to-end tok/s delta.
- **Risk:** verify numerics vs fused-RMSNorm + native `scaled_mm_v2` path to within 5 e-4 RMS.

### T2-G. I-DLM inference bake-off on PRO 6000
I-DLM is an AR-equivalent diffusion LM with a released Qwen3-8B-b2-allmasked checkpoint and an SGLang server. Claim: 3.8× AR throughput. Worth an honest head-to-head on our hardware.

- **Launch:** `I-DLM/inference/sglang/sglang/launch_server.py:14-28`, config `I-DLM/inference/configs/idlm_blockN4_config.yaml`
- **Core algorithm:** `I-DLM/inference/sglang/sglang/srt/dllm/algorithm/idlm_blockN.py:337-1037` (ISD: classify → forward → verify/sample → trim/assemble)
- **Checkpoint:** `I-DLM/training/model/Qwen3-8B-b2-allmasked/` (config.json:36-37 → `block_size=2`, `mask_token_id=151669`)
- **SM120 caveat (critical):** The CuTe kernels in `inference/sglang/sglang/jit_kernel/flash_attention/cute/` hard-assert compute-cap ∈ {9, 10, 11} and use `tcgen05` + TMEM. They will not load on PRO 6000. **Before running anything, patch `interface.py:430-478` to route SM120 → `flash_fwd_sm90.py` fallback, or disable the CuTe backend entirely and let SGLang pick FlashInfer / Triton.** Confirm on a throwaway run that no `No kernel available` or TMEM error appears. ISD itself (`idlm_blockN.py`) operates on post-attention logits and does *not* require the CuTe path — verified at `idlm_blockN.py:1-40`.
- **Experiments:**
  1. **Baseline BF16** on SM90-fallback path — run as-published on PRO 6000 after the patch. Record tok/s at C=1, 16, 32, 64 and accept-rate at `gen_block_size ∈ {2, 4}`.
  2. **NVFP4 weights** — quantize the Qwen3-8B-b2-allmasked weights via T1-C recipe; re-run.
  3. **Head-to-head** — AR baseline = Qwen3-8B vanilla, same prompts, same PRO 6000.
- **Metric:** tok/s, accept-rate, GSM8K / MATH-500 accuracy vs AR baseline (scripts already in `I-DLM/inference/eval/`).
- **Don't try:** "bump `gen_block_size` to 8 using 512 TMEM columns" — no TMEM on SM120. If you want block-N > 4, retrain with `block_size=15,gen_block_size=8` analogous to `Qwen3-8B-b3-allmasked` and live with the shmem budget.

### T2-H. FP8 KV on Qwen3-MoE (but not Gemma-4)
Autokernel's documented 4× slowdown with FP8 KV was **Gemma4-specific** — heterogeneous head dims (256/512) trip FlashInfer's FP8 attention path. Qwen3 has uniform head dims, so FP8 KV should deliver free 2× capacity.

- **Files to reuse:** existing vLLM FP8-KV patches in `vllm_patches/`, `fix_nvfp4_attn_to_bf16.py` recipe.
- **Measure:** tok/s and KV capacity at 4K / 32K / 128K ctx, C ∈ {64, 256, 512}.
- **Kill-criterion:** if < 1.2× vs BF16 KV at C=256, abandon — the gain must come from capacity, not kernel.

---

## Tier 3 — Speculative / research (do after Tier 1 lands)

### T3-I. DDTree block-diffusion draft-tree spec decode
Autokernel previously abandoned draft-model spec decode on Gemma-4 because pruned-layer drafts hit c≈0.83 (net slowdown). DDTree is a *different* acceptance profile: diffusion draft (DFlash) + tree verification.

- **Entry points:**
  - Tree build: `ddtree/ddtree.py:84-166`
  - KV compaction (C++ via `torch.utils.cpp_extension`): `ddtree/ddtree.py:30-74`
  - Tree-aware verify: `ddtree/ddtree.py:212-277`
  - DFlash attn with tree mask: `ddtree/model/dflash.py:58-102`
- **Port plan:** wire `ddtree.build_ddtree_tree` into vLLM's speculative-decode hook path; verify against target Gemma-4 26B.
- **Known cost & fix:** tree construction uses Python `heapq` on CPU — single-thread bottleneck at budget ≥ 1024. See T3-J.
- **Kill-criterion:** if effective draft cost ratio c > 0.6, abandon — this is autokernel's empirical speed-up/loss boundary.

### T3-J. GPU top-k kernel for tree expansion
Replace `heapq` in DDTree's expansion loop with a Blackwell-native batched top-k. This is also a reusable primitive for any tree-spec scheme (Eagle3, Medusa).

- **Reference:** `gpu_bitonic_sort/bitonic.cu:605-861` (recursive iterative evaluator) — **do not** copy the TMEM / cluster suggestions from the exploration agent; SM120 can't run them. Use it only as a correctness reference for partial-sort semantics.
- **Implementation:** Triton batched top-k with warp-level shuffle + persistent grid of `NUM_SMS` blocks. Early-terminate at depth `log(K)`.
- **Validation:** match CPU `heapq` selection on 10 k random logits to bit-exactness modulo tie-breaking.
- **Metric:** μs / tree-build at tree size {256, 1024, 4096} — compare to DDTree's CPU baseline and to autokernel's existing ngram path overhead.

### T3-K. DP=2 on the two PRO 6000s
Already designed in `plans/pro6000_projections.md` — use DP=2, not TP=2 (PCIe-only, no NVLink, 50–100 µs AllReduce × 60 layers = 3–6 ms/tok is too much). Stand up two independent vLLM servers behind a round-robin load balancer once Tier 1 is done.

- **Serve scripts to adapt:** `serve_best.sh`, `serve_gemma4_dp2.sh` (already drafted).
- **Expected aggregate:** 2× T1-A single-GPU throughput, minus ~5 % LB overhead.

---

## Tier 1b — Net-new KV-compression vectors from aigpu survey

Added after deeper review of `/home/cklaus/projects/aigpu/turboquant-gpu` and `/home/cklaus/projects/aigpu/triattention`. Both are Blackwell-compatible and orthogonal to FusenCache. Promote to Tier 1 if T1-A saturates before 10 k tok/s on PRO 6000.

### T1-E. TurboQuant KV compression (random orthogonal rotation + Lloyd-Max)
Random orthogonal rotation + Lloyd-Max vector quantization on the rotated distribution. Claims 5.02× KV compression with 2-bit keys (+ 1-bit QJL sign correction) and 3-bit values. cuTile-based, has a PyTorch fallback, **verified SM120-compatible** (README lists B200 / CUDA 13.0+; no SM≥100-only intrinsics).

- **Core kernels:** `turboquant-gpu/turboquant_gpu/compress.py:14-283` (five cuTile kernels: 2-bit K, 3-bit K, 3-bit V, 2-bit V, fused KV)
- **Host + auto-tune:** `turboquant-gpu/turboquant_gpu/host.py:27-229`
- **What to do:** wire `turboquant_compress_kv_3bit` into a vLLM custom attention backend alongside FusenCache k4v4 for comparison. Start with Qwen3-30B-MoE (homogeneous head dims → FlashInfer FP8 penalty doesn't apply).
- **Metric:** tok/s and KV footprint vs FusenCache k4v4b64 at 4K / 32K / 128K ctx, C ∈ {64, 256, 512}. Quality: PPL delta vs BF16 KV; cosine sim of reconstructed K, V.
- **Why PRO 6000:** 5× vs FusenCache's 4× is a modest per-token win, *but* the rotation happens once per block and compresses the K/V writes — we haven't tried rotation-based schemes at all.
- **Kill criterion:** if tok/s is within 5 % of FusenCache at equal batch and PPL regresses > 5 %, abandon — FusenCache is simpler.

### T1-F. TriAttention frequency-domain sparse KV selection
Per-head trigonometric token scoring: RoPE-inverted frequency statistics + geometric offsets pick which KV tokens matter for each head. Published 10.7× KV reduction and 2.5× throughput on AIME25-style long reasoning. Pure Triton, no CUDA kernels → SM120-portable.

- **Core algorithm:** `triattention/triattention/methods/triattention.py:59-150` (TriAttentionConfig, per-head/per-layer-per-head pruning)
- **Scoring:** `triattention/triattention/methods/pruning_utils.py:30-275` (RoPE inversion, rotation matrices, frequency stats)
- **Triton kernel:** `triattention/triattention/vllm/core/kernels/triton_scoring.py:38-193` (TrigTableCache precomputes cos/sin for all compression rounds)
- **vLLM plugin:** `triattention/triattention/vllm/runtime/` (already monkeypatches scheduler/worker for transparent KV compression)
- **Prereq:** Q/K frequency stats — shipped for Qwen3 and DeepSeek-R1-Distill. **Gemma-4 stats must be calibrated via `triattention/calibration/`.**
- **What to do:** (1) calibrate Gemma-4 frequency stats (one-time), (2) serve Gemma-4 26B with TriAttention plugin on vLLM, (3) run AIME25 + MATH-500 long-reasoning traffic.
- **Metric:** tok/s at long ctx (≥ 16K), KV memory footprint, AIME/MATH accuracy delta vs FusenCache-only baseline.
- **Cascade experiment (T1-F₂):** stack TriAttention (select top-K tokens) + TurboQuant / FusenCache (compress the residual). Ballpark 20–30× effective KV capacity at long ctx. Only do this if T1-E and T1-F individually land.
- **Kill criterion:** if accuracy drops > 2 % on AIME25 at the published 10× budget, reduce budget; if it still drops, abandon.

---

## Tier 2b — Wins latent in autokernel's own tree

Items already partially built or spec'd in this repo that haven't been landed.

### T2-I. Wire `kernels/fp8_decode_attention.py` into vLLM (bypasses FlashInfer FP8 penalty)
Autokernel's long-standing FP8-KV blocker on Gemma-4 is entirely a FlashInfer problem on heterogeneous head dims (256/512). The Triton split-K paged FP8 decode kernel in this repo does not call FlashInfer at all.

- **Kernel:** `kernels/fp8_decode_attention.py:1-100` (GQA-aware, paged, split-K, soft-cap). Tests pass in isolation.
- **Never wired:** no vLLM backend registration exists.
- **What to do:** write a ~50-line vLLM attention-backend wrapper in `autokernel_v2/plugins/` that routes decode to this kernel when `kv_cache_dtype == fp8`. Keep BF16 prefill on FA2. Compare vs FlashInfer FP8 on the same Gemma-4 NVFP4 model.
- **Metric:** tok/s at C=256, KV footprint, per-step latency. Goal: kill the 4× FlashInfer tax entirely (i.e., FP8 KV at ≥ 0.95× BF16 speed, doubled capacity).
- **Kill criterion:** if the kernel can't beat 0.9× of FA2 BF16 end-to-end, punt.

### T2-J. Prefix cache × FusenCache interaction test
vLLM's `--enable-prefix-caching` has never been run against FusenCache's compressed slot table. Both systems touch block tables; coexistence is unverified.

- **Evidence plugin:** `autokernel_v2/plugins/fusencache_kv.py` (KV registration only)
- **What to do:** 1-day integration. Serve 100 requests sharing a 1–2 K token system prompt with FusenCache k4v4 + `--enable-prefix-caching`. Check for slot-table / block-map collisions first (logs), then measure prefill latency and effective KV capacity.
- **Metric:** prefill latency delta, KV memory saved, end-to-end tok/s at C=256.
- **Why PRO 6000:** the savings are *multiplicative* (4× FusenCache × prefix reuse). A shared 2 K system prompt at 80 % hit rate at 96 GB is an enormous effective-capacity multiplier.
- **Kill criterion:** if enabling prefix caching crashes FusenCache at B ≥ 8, the block-table layouts are fundamentally incompatible — file a patch spec and abandon until vLLM's cache manager changes.

### T2-K. SpinQuant R1+R2 fused rotations + NVFP4
llm-compressor ships production-ready rotation transforms. R1+R2 fuse into weights (vLLM-consumable). R3+R4 need runtime activation hooks and are *not* production-ready — skip them.

- **File:** `llm-compressor/src/llmcompressor/modifiers/transform/spinquant/base.py:210-295`
- **Recipe:**
  ```python
  recipe = [
      SpinQuantModifier(rotations=["R1", "R2"], transform_type="hadamard"),
      QuantizationModifier(scheme="NVFP4", targets="Linear",
                           ignore=["lm_head", "re:.*router", "re:.*embed.*"]),
  ]
  ```
- **Expected:** +1–2 % PPL vs plain RTN NVFP4; same throughput. Combined with T1-C (AutoRound), compounds to 3–5 %.
- **Kill criterion:** if PPL improvement < 0.5 %, not worth the calibration time.

### T2-L. KV FP8 via llm-compressor `kv_cache_scheme` (writes compressed-tensors)
llm-compressor writes a `compressed-tensors` model with baked-in per-tensor static KV scales. vLLM reads this natively — no custom kernel, no FlashInfer surgery.

- **Entry:** `llm-compressor/src/llmcompressor/modifiers/quantization/quantization/base.py:37-48`
- **Example:** `llm-compressor/examples/quantization_kv_cache/llama3_fp8_kv_example.py:64-69`
- **What to do:** regenerate Qwen3-30B-MoE with `kv_cache_scheme={"type":"float","num_bits":8,"strategy":"tensor"}`. Serve with `vllm serve … --kv-cache-dtype=fp8`.
- **Overlap with T2-H:** T2-H was "turn on vLLM FP8 KV"; T2-L is "bake static scales in offline so the runtime path is deterministic." Run both and compare.
- **Metric:** PPL, tok/s, KV capacity. Goal: 2× capacity at ≥ 0.95× throughput on Qwen3 (i.e., what was supposed to work on Gemma-4 but didn't).

### T2-M. W4A4 NVFP4 (activation quant) on Gemma-4 / Qwen3-MoE
Autokernel currently runs W4A16 (weights NVFP4, activations BF16). llm-compressor supports full W4A4 NVFP4 with dynamic per-token per-group-16 activation scales — SM120 has native MMA for this.

- **Gemma-4 example:** `llm-compressor/examples/quantization_w4a4_fp4/gemma4_example.py:21-30` (MoE-aware via `CalibrationMoeMoE`)
- **Qwen3-VL-235B:** `llm-compressor/examples/quantization_w4a4_fp4/qwen3_vl_moe_w4a4_fp4.py:64-73` (overlaps with T1-D)
- **Expected:** +2–4 % tok/s vs W4A16 (activation fetch/quantize ops removed from the inner loop).
- **Kill criterion:** if PPL regresses > 3 % on WikiText-2, the activation-outlier handling in calibration isn't tight enough — try again with SpinQuant R1+R2 prepended.

---

## Tier 1c — DP=2 linearity measurement (prerequisite for T3-K)

### T1-G. DP=2 aggregate tok/s sanity check
`plans/pro6000_projections.md:113-120` projects perfect 2× DP=2 scaling; `specs/gemma4_nvfp4_pro6000_dp2.yaml:136-138` shows `kv_tokens_measured: 0` and `profiled: ''` — the projection is *untested*. This must pass before spending effort on T3-K's load-balancer deployment.

- **What to do:** bring up two containers (ports 8000 / 8001) on the two PRO 6000s, run `bench_dp2.py --quick --compare-single` against the T1-A single-GPU baseline. Record: aggregate tok/s at C ∈ {64, 256, 512}; per-request P50 TTFT; CPU utilisation on the driver host.
- **Pass:** aggregate ≥ 1.9× single-GPU at C=256. Then proceed with T3-K.
- **Kill:** aggregate < 1.25× at C=256 → DP=2 is throttled by CPU-side scheduling or an unanticipated PCIe cost, and T3-K is not free. Investigate before spending time on load-balancer work.

---

## Tier 2c — Latent autokernel wins (unfinished kernels + specs)

### T2-N. Fused shuffle+quant MoE kernel
`EXPERIMENT_DISCOVERIES.md:94-97` (Discovery #15) documents: "Shuffle (gather) can be folded INTO the quant kernel by adding a `dst2src_map` parameter (~5 lines CUDA). Norm CANNOT be moved past routing." Projected +2.3 % end-to-end, bit-identical output, **never built** — the note explicitly says "ruled out as low priority during the Gemma4 26B push."

- **What to do:** write a ~200-line C++ extension fusing `MoELayer.shuffle_rows()` + `quantize_to_nvfp4()` into one pass. Gate it as a vLLM custom op for NVFP4 models only (MoE shuffle path — `fix_nvfp4_attn_to_bf16.py` context).
- **Metric:** per-layer latency on Gemma-4 26B-A4B at B ∈ {1, 32, 64}; end-to-end tok/s delta at C=256. Validate bit-identical token streams vs baseline on GSM8K.
- **Why PRO 6000:** it's additive to T2-F (fused norm+FP4 on dense path). Together they remove two of the remaining launches in the MoE decode.
- **Kill:** > 10 % slower than two-op baseline OR any token mismatch vs eager path.

### T2-O. CUTLASS block-scaled MMA DSL end-to-end on Gemma-4 NVFP4
`test_cutlass_dsl_gemm.py`, `test_cutlass_dsl_mma.py`, `test_cutlass_cubin.py:48-80` — DSL compiles to PTX and validates codegen, but **no test runs a real-model forward pass.** CUTLASS 3.x SM120 codegen has block-scaled FP4 MMA support that autokernel currently accesses only via `torch._scaled_mm_v2` (which uses cuBLASLt). Custom CUTLASS may unlock kernels cuBLASLt doesn't (e.g., fused activation, non-standard tile shapes for MoE expert widths).
- **What to do:** (1) pick the best tile shape from the DSL tests (likely 16×8×64 block-scaled). (2) Compile to `.so` via CUTLASS host harness. (3) Register as a vLLM custom op only for Gemma-4 expert GEMMs. (4) Prefill 100 tokens; diff token-by-token against FlashInfer path. (5) If bit-identical, sweep M ∈ {32, 256, 2048}, K ∈ {5120}, N ∈ {expert up/down dims}.
- **Why PRO 6000:** same ISA as 5090 (no hardware advantage), but 96 GB removes any allocation-pressure blockers that might have stopped the 5090 prototype.
- **Kill:** > 5 e-3 RMS vs eager path, OR < 1.1× cuBLASLt FP4 at every swept shape.

---

## Tier 3c — Research / gate tests (new)

### T3-P. K2V2 int2-packed KV (8× compression, 128 K-context frontier)
`kv_cache_gen/RESULTS.md:112-122` swept k4v4b64 and k8v4 but never ran k2v2. `config.py:42` aliases "k4v2" exist; no k2v2 row in `sweep_full_results.tsv`. `sm120_attention_kernel.md:145` projects 3.4× decode at INT4 if quality is acceptable.
- **What to do:** `parse_spec("k2v2b16")` → run `kv_cache_gen.quality_eval.py` on a 1K-token Gemma-4 prompt. Gate: CosSim vs FP16 baseline. If ≥ 0.85, run end-to-end tok/s at 32 K and 128 K ctx, C=16–256.
- **Why PRO 6000:** 96 GB × 8× compression ≈ 1 M-token KV budget. Unlocks single-GPU 128 K ctx at C=16, or 32 K ctx at C=512 — configurations that are infeasible at k4v4.
- **Kill:** CosSim < 0.80 OR greedy-generation top-1 mismatch > 2 % on a held-out prompt.

### T3-Q. FusenSolver Mixture-of-Agents across DP=2 (hard↔strong, easy↔fast)
`plans/mixture_of_agents.md:1-150` fully specifies the policy; `fusen_solver/backends/multi_backend.py:106-127` exists but `REVIEW.md:BUG-04` (lines 32-38) documents "strategy routing ignored." Because this is an agent-orchestration layer on top of two independent servers, it is strictly a DP=2 feature.
- **What to do:** implement the router in `fusen_solver/integrations/cli.py`: keyword + strategy-tag classifier → easy work to GPU 1 (smaller/faster model, e.g. E2B or pruned Gemma), hard work to GPU 0 (full Gemma-4-26B NVFP4). Cascade fallback: fast-fail timeout → retry on strong backend.
- **Metric:** time-to-first-good-solution on a 5-category mix (bug fix, refactor, test gen, architecture, simple QA); utilisation of each GPU; acceptance rate of fast-backend outputs.
- **Why PRO 6000:** the heterogeneous-capacity policy only makes sense if both GPUs are independent (DP=2, no shared KV). TP=2 would serialise MoA.
- **Kill:** cascade fallback rate > 50 % (fast model unreliable) OR MoA slower than all-strong.

### T3-R. Self-drafting speculative decode (pruned Gemma-4 draft ↔ full Gemma-4 verify, same GPU)
Autokernel previously abandoned separate-GPU draft models and pruned-layer drafts on Gemma-4 (c ≈ 0.83, net slowdown — `EXPERIMENT_DISCOVERIES.md:#39, #42`). This variant is different: run both *draft* (expert-pruned 50–70 %) and *verifier* (full) on a single PRO 6000 with a shared input-embedding front-end, so draft KV overhead ≈ 0.
- **Evidence:** `tools/prune_experts.py:50-135` (importance-scoring + reindex) already implemented; `autokernel_v2/candidate_generator.py:228-243` proposes draft-via-small-model but assumes separate models.
- **What to do:** (1) prune Gemma-4 to 50 % experts and router-fine-tune for 4 epochs of C4 (256 samples) using llm-compressor's `LoRAModifier`. (2) In inference, use pruned as draft for K=2 tokens, full as verifier on the same GPU. (3) Measure empirical accept-rate and net tok/s.
- **Why PRO 6000:** 96 GB barely fits full+pruned KV budgets simultaneously; 5090 can't (17 GB model × 2 + KV > 32 GB). This variant is *only* feasible on PRO 6000.
- **Known risk:** `EXPERIMENT_DISCOVERIES.md:#25, #40, #41` documented 100 % non-prunable without fine-tuning on Gemma-4; 10 % expert prune → –41 % quality. Router fine-tune *might* recover, but this is speculative. Run the 30-min router-fine-tune + PPL gate before spending on integration.
- **Kill:** post-fine-tune PPL regresses > 3 % OR accept-rate < 0.70 at K=2.

### T3-S. Persistent-kernel MoE decode retest on PRO 6000
`test_persistent.py:1-68` + `EXPERIMENT_DISCOVERIES.md:222-226` (Discovery #35): cooperative groups work on SM120, `grid.sync()` ≈ 278 µs, deemed "marginal vs CUDA graphs" on 5090 and never integrated. PRO 6000 has more SMs and a larger register file; occupancy math changes.
- **What to do:** port the existing test into a vLLM custom MoE plugin (~100 lines wrapper). Benchmark Gemma-4 MoE decode at C={1, 32, 64} with `persistent=True` vs current CUDA-graph path.
- **Why PRO 6000:** larger register file per SM ⇒ less persistent-state pressure; more SMs ⇒ more work per `grid.sync()` hides the barrier cost better than on the 5090.
- **Kill:** end-to-end tok/s delta < +2 % OR launch overhead > 500 µs.

### T3-T. Semantic-eviction phase-1 score instrumentation (follow-on to T3-L)
T3-L is a 2 h attention-matrix shape test. If it passes (≥ 70 % low-mass tokens), this is the next step — measure the same signal *in production flow*, not on a logged snapshot.
- **Evidence:** `plans/semantic_kv_eviction.md:1-250` spec; `fusen_kv/backend.py:755-821` decode loop has the natural hook point. `moe_gen/spec.py` describes Gemma-4's 5 global layers vs 25 sliding-window layers — only the globals have unbounded KV growth.
- **What to do:** instrument FusenKVImpl to accumulate per-request per-global-layer softmax scores (lightweight Python accumulation, no eviction yet). Verify the distribution on live traffic matches the offline gate-test pattern.
- **Why PRO 6000:** eviction is the lever that turns "96 GB can serve C=512 at 32 K ctx" into "96 GB can serve C=1024." Don't build the evictor without live-traffic confirmation.
- **Kill:** live distribution is uniform (contradicts T3-L result), OR the lightweight attention pass costs > 10 % of decode latency — drop to cheaper heuristics (position-based H2O).

---

## Tier 3b — Cheap gate tests (30 min–2 h each, keep/kill before building)

Each is a 1-file hook script. Do not build the full optimization until the gate says "keep."

### T3-L. Semantic-KV-eviction gate (H2O / SnapKV shape test)
`plans/NEXT_20_RESEARCH_AREAS.md:58-70` proposes this. 2 h to *test the assumption*.
- **Hook:** FA2 attention call. Log the `(B, Hq, Q, K)` attention matrix for one layer every 1000 steps for 10 k tokens of real inference.
- **Compute:** for each KV position, cumulative attention weight across all queries. Fraction of positions with cumsum < 0.05.
- **Keep if:** ≥ 70 % of tokens have cumsum < 0.05 → eviction will give 3–10× KV reduction on long contexts.
- **Kill if:** < 50 %.

### T3-M. Router-prediction-cascade gate (30 min)
`plans/NOVEL_RESEARCH_PATHS.md:38-54`, `TOP_20_RANKED.md:99-116`. Simplest gate in the plan.
- **Hook:** `MoELayer.forward()` in Gemma-4 for layers {0, 5, 10, 15, 20, 25, 29}.
- **Log:** top-8 expert indices per token for 500 tokens.
- **Compute:** Jaccard and mutual information of layer-0 vs layer-N routing.
- **Keep if:** MI ≥ 0.5 for N > 5 → prefetching next-layer experts on the layer-0 decision is viable, 10–20 % MoE speedup ceiling.
- **Kill if:** MI < 0.3 → routing is independent across depth, no prefetch win.

### T3-N. Expert-output memoization gate (1–2 h)
`NOVEL_RESEARCH_PATHS.md:18-35`, `TOP_20_RANKED.md:61-77` (ranked #1 in novelty roadmap).
- **Hook:** `MoELayer.forward()` — capture the first 8 dims of each expert input for 1000 tokens.
- **Compute:** pairwise cosine similarity *within* each expert's input stream.
- **Keep if:** clusters with cosine ≥ 0.99 contain ≥ 20 % of activations → LRU cache on expert inputs saves a measurable fraction of expert GEMMs.
- **Kill if:** max intra-expert cosine < 0.95 (approx error exceeds FP4 quant noise).

### T3-O. L2 expert pinning frequency-profile (1 h)
Autokernel previously found expert L2 caching zero-gain — but that was *all 128 experts*. The real test is whether activations are power-law.
- **Hook:** MoE dispatch; count activations per expert for 1000 tokens.
- **Also capture:** `ncu --metrics l2__t_sectors_pipe_lsu_mem_global_op_ld.sum` baseline L2 hit rate on current MoE decode.
- **Keep if:** top-40 experts account for ≥ 80 % of activations **and** L2 hit rate < 60 % (headroom exists). 40 × ~1 MB NVFP4 = 40 MB, fits PRO 6000's ~48 MB L2 via `cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)`.
- **Kill if:** uniform activation OR L2 hit already > 70 %.

---

## Tier 4 — Conditional fallbacks (only if upstream items underperform)

### T4-B. LookaheadDecoding on Llama-3.3-70B only (Jacobi parallel decoding, no draft)
`https://github.com/hao-ai-lab/LookaheadDecoding` — ICML 2024. Parallel n-gram-pool-driven decoding: speculate K positions in parallel from a moving window of recent outputs, accept the longest matching prefix. No draft model, no tree, no extra weights. Reported 1.5–2.3× latency reduction (mostly batch=1).

**Critical scope limit (verified by web-fetch):** the upstream is **LLaMA-only**. Gemma-4 and Qwen3 are not supported without porting the model-side patches. So this only applies inside autokernel's plan when running Llama-3.3-70B (the T2-E target) — not on the Gemma-4 stack that drives most of the project's metrics.

- **Trigger:** T2-E (mixed NVFP4+FP8 on Llama-3.3-70B) lands and the resulting model is being served at C=1–4 (low-concurrency interactive use, where ngram-only accept is modest).
- **What to do:** install `lade` package, call `lade.augment_all()` + `lade.config_lade()`, set `USE_LADE=1`. Tune three params (LEVEL, WINDOW_SIZE, GUESS_SET_SIZE) per device. Benchmark vs plain ngram baseline on C=1, 4, 16.
- **Why PRO 6000:** Lookahead trades parallel-position compute for fewer sequential steps — worth it when compute headroom exists at low batch. 96 GB has the headroom.
- **Kill criterion:** Jacobi iteration adds < 15 % accept-rate over plain ngram on a 100-prompt held-out set, OR per-step latency increases > 20 % at C=16 due to Jacobi overhead, OR porting cost to Gemma-4 / Qwen3 exceeds the projected win.

### T4-C. Async DP-split speculative decoding (Tanishq SSD pattern)
`https://github.com/tanishqkumar/ssd` — async speculative decoding where draft and verifier run **in parallel on distinct GPUs**. The drafter doesn't wait for verifier acceptance; it speculates for *all* likely verification outcomes ahead of time.

**Trade-off vs T3-K (DP=2 throughput):** SSD requires the second PRO 6000 to be the *drafter*, not a parallel verifier. So it's an explicit choice — DP=2 doubles aggregate throughput at C=high, SSD reduces single-stream latency at C=low. Cannot run both at once on a 2-GPU box.

- **Trigger:** DP=2 (T3-K) is up but the workload is dominated by interactive single-stream traffic (latency, not aggregate tok/s, is the SLA).
- **What to do:** dedicate GPU 1 to a small drafter (e.g. Qwen3-1.7B or pruned Gemma-4 from T3-R, NVFP4-quantized to fit easily) and keep the full Gemma-4 26B-A4B verifier on GPU 0. Use Tanishq's repo as the orchestration reference. Measure single-stream latency vs T3-R's same-GPU self-draft.
- **Why PRO 6000 specifically:** the design assumes "distinct hardware" for draft and verify — exactly the dual-GPU PRO 6000 layout, no NVLink required since draft tokens are small.
- **Kill criterion:** single-stream latency improvement < 1.3× over T3-R (same-GPU self-draft), OR aggregate tok/s drops > 50 % vs DP=2 (i.e., the latency gain isn't worth giving up the DP=2 throughput option).

---

### T4-A. FusenDiffusion-Lite draft head (only if T3-I and T3-R both fail)
`plans/FUSENDIFFUSION_PLAN.md:68-130` proposes a 3–5 layer, ~180 M-param diffusion head trained on Gemma-4 26B hidden states; targeted acceptance α ≥ 0.6, projected single-user 1.8–2.3×. Estimated 25 h training on a single 5090; ~12–15 h on PRO 6000 DP=2.

**Why it's a fallback, not a Tier 1 item:** every other speculative-decode lever (T3-I DDTree, T3-R self-drafting, n-gram baseline) must fail first. The training cost is high and the win is bounded by acceptance rate.

- **Trigger:** T3-I lands but accept-rate α ≤ 0.5 *and* T3-R fails its quality gate.
- **What to do:** train 1-layer diffusion head on Gemma-4 hidden states from a 100 K-token corpus; target α ≥ 0.6 on a 100-prompt held-out set before any kernel work.
- **Kill criterion:** held-out α < 0.55 — n-gram + lookahead are already good enough; drop diffusion.

---

## Meta-tooling tracker (not perf experiments)

These improve the *search loop* that produces autokernel experiments. Track separately; do not confuse with PRO 6000 throughput levers.

- **M1. `imbue-ai/darwinian_evolver`** — generic evolutionary framework. Define organism = kernel-tuning params (block size, num_warps, split_k, pipeline depth), evaluator = `bench_*.py`, mutator = LLM-rewrite of the kernel/config. Could plug into `autokernel_v2/optimizer.py` as an alternative to the current candidate generator. **Defer until the existing optimizer's hit-rate plateaus** (per `EXPERIMENT_DISCOVERIES.md`, current keep-rate is ~56 % after 225 experiments — not yet plateaued).

---

## Saturation note

This plan has been walked four times: the four cloned aigpu repos, three sibling aigpu repos (turboquant-gpu, triattention, llama-cpp-turboquant), all of autokernel's `plans/`, `kernels/`, `kernels/csrc/`, `fusen_solver/`, `fusen_kv/`, `fusencache/`, `kv_cache_gen/`, `moe_gen/`, `nvfp4/`, `specs/`, `vllm_patches/`, `autokernel_v2/`, the 75+ top-level `test_*.py` / `exp_*.py` files, and all of `EXPERIMENT_DISCOVERIES.md` / `SESSION_FINAL_STATUS*` / `optimization_lessons*` / the 12 brainstorm docs. The remaining ~375 ideas in `COMPREHENSIVE_INFERENCE_MAP.md` are explicitly tagged "infeasible" or "needs months / retraining"; the 100_NOVEL_IDEAS V1–V5 set is LLM-generated speculation without code grounding. **Stop adding. Execute the existing tiers.** New experiments should be appended only when one of the listed kill criteria fires and produces concrete evidence for a new direction.

---

## Explicit "do not pursue" list

| Idea | Why not |
|---|---|
| TMEM-backed stacks (from bitonic-sort agent) | SM120 has no TMEM |
| CTA-cluster work-stealing | SM120 cluster shape fixed at 1×1×1 |
| `tcgen05` / WGMMA on PRO 6000 | Datacenter Blackwell only |
| DeepGEMM SM100 cubins | Binary-incompatible with SM120 (discovery #24) |
| SmoothQuant for *serving* | vLLM doesn't consume SmoothQuant-transformed compressed-tensors yet (`llm-compressor/experimental/mxfp4/README.md:4`); OK offline for PPL, not for tok/s |
| Re-fusing W4A16 dequant + GEMM | Autokernel proved split (Triton dequant + cuBLAS FP16) beats fused, 328 vs 15 TFLOPS |
| FP8 activations at K ≥ 5120 | Documented correctness failure (3-bit mantissa) |
| Expert-weight L2 pinning on MoE | 348 MB working set > 96 MB L2; measured zero gain |
| Triton FP8 attention rewrite | 35 % BW vs FA2's 93 %; Triton lacks `cp.async` pipelining; only a CUDA C++ rewrite has a chance, and FA2 already exists |
| bitonic-sort as MoE token router | Not the hot path (only 6 launches / MoE layer), and the repo is tree-recursive, not bitonic-network |
| 2:4 structured sparsity for serving | vLLM dropped support (PR #36799). `llm-compressor/examples/sparse_2of4_quantization_fp8/README.md` confirms. Offline PPL only |
| SpinQuant R3 / R4 (online activation rotations) | Runtime activation hooks not in vLLM's forward path |
| QuIP# U-rotate (non-mergeable) | Needs vLLM PR #22486 (not merged). Ship V-rotate only |
| DDTree tree budget > 256 | Mask is dense N×N bool (`ddtree/ddtree.py:153-206`); O(N²) memory + compute. 1024 → 1 M-element mask per forward |
| `llama-cpp-turboquant` integration | Misleading name — it's plain llama.cpp with SM120 defined, no turboquant kernels; adds nothing vs vLLM stack |
| I-DLM CuTe kernels on SM120 | Hard assert `cc ∈ {9,10,11}` at `interface.py:251`; uses `tcgen05` + TMEM. Must force SM90 fallback (see T2-G caveat) |
| Layer/expert pruning on Gemma-4 *without* fine-tune | Discovery #25/#40/#41: even 10 % prune → –41 % quality on early layers. Only try with router fine-tune (see T3-R) |
| N-gram spec decode on Gemma-4 MoE | Discovery #39/#42: –49 % throughput. Compute-bound MoE → draft cost > accept benefit |
| Cross-layer KV sharing | Discovery #18: Jaccard < 0.14 across layers. No hot set exists to share |
| All-C++ FusenCache with CUDA graphs `mode=none` | Discoveries #56, #58: crashes at C ≥ 32 with `illegalInstruction`. SM120 large-graph driver limit, not code-fixable |
| Fusen-solver reliability fixes (cascade timeouts, contiguity assertions) | Important, but not GPU-perf — out of scope for this plan; track in `production_hardening.md` |
| `apple/ml-ssd` | Name collision — it's self-distillation for code generation (training-time), not speculative decoding. Zero inference relevance |
| HY-SOAR (`hy-soar.github.io`) | Image diffusion (rectified-flow / SD 3.5-Medium), code "coming soon." No relevance to LLM inference on PRO 6000 |
| LookaheadDecoding on Gemma-4 / Qwen3 | Upstream is **LLaMA-only** (verified by web-fetch). Use only on Llama-3.3-70B (T4-B); porting to other arches is out of scope here |

---

## Suggested cadence

**Day 0 — gates + topology measurement (4–6 h total).**
- ASI-2 Layer 1 (5 min): set `default_max_tokens` per task type in fusen_solver — free KV, zero risk.
- T1-G (1 h): DP=2 linearity check — validates throughput topology.
- ASI-1 (2 h): disaggregated 1P1D bench (P99 TTFT at bimodal prompt distribution) — validates latency topology.
- T3-M (30 min), T3-O (1 h), T3-N (1–2 h), T3-L (2 h) — gate tests for later tiers.
- **Day-0 decision:** choose Config A (throughput), B (mixed), or C (long-ctx) from ASI-4 decision matrix as the default deployment target. All subsequent experiments run on that config.

**Week 1 — data reduction wins (roofline-guided: only experiments that reduce bytes moved).**
T1-A (baseline bench) → T2-I (FP8 decode attn — the single biggest attention optimization, per ASI-0) → T1-B (piecewise CG fix) → ASI-2 Layer 2 (early KV abort wrapper, 2 h). Should unblock FP8 KV on Gemma-4 (kills the 4× FlashInfer penalty) and net 9–11 k tok/s.

**Week 2 — quantization quality + topology deployment.**
T1-C (AutoRound), T2-K (SpinQuant R1+R2), T2-L (compressed-tensors KV-FP8), T2-H (FP8 KV on Qwen3), T2-M (W4A4 NVFP4). Parallel: T1-D calibration run for 200 B in the background; T2-J prefix-cache × FusenCache (check ASI-3 compat matrix first). Deploy winning Day-0 topology (DP=2 or 1P1D) as the steady-state config.

**Week 3 — new KV-compression vectors + 2nd topology.**
T1-E (TurboQuant rotation KV), T1-F (TriAttention), T3-P (K2V2 int2 quality gate). Try the *other* Day-0 topology to compare (e.g., if Week 2 deployed DP=2, try 1P1D this week). T2-G (I-DLM on SM90 fallback). ASI-2 Layer 3 (multi-agent racing abort, if fusen_solver workload justifies).

**Week 4 — launch-overhead + spec decode + whatever gate survived Day 0.**
T2-F (fused norm+FP4 on dense), T2-N (fused shuffle+quant MoE) — these are in the ~7% launch-overhead bucket per ASI-0. T3-I (DDTree), T3-R (self-drafting, only if T3-N gate passed). T2-O (CUTLASS DSL) as a low-priority research slot. Implement the top T3-L/T/M/N/O gate survivor.

---

## Metrics to log per experiment (uniform format)

```
exp_id, date, gpu ({PRO6000, PRO6000MaxQ, 5090-ref}), model, quant, batch, ctx,
concurrency, tok_s, p50_ttft_ms, p50_itl_ms, kv_tokens, vram_gb,
ppl_wikitext2, ppl_c4, notes
```

Append to `results.tsv` (schema already in repo). Commit the raw log under `vllm_server_*.log` pattern used by the rest of the project.
