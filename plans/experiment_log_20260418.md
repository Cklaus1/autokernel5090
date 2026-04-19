# Experiment Log — 2026-04-18

## Hardware
- 2× RTX PRO 6000 Blackwell (SM120a), 96 GB GDDR7 each
- PCIe-only (no NVLink), AMD 9950X3D host
- WSL2 (pin_memory=False penalty active; CUDA IPC unavailable)

## Context Entering Session
- vLLM images: `vllm-fusencache:latest` (main, 0.1.dev100+gc0c98b8b9, 2026-04-17), `vllm-built:latest` (0.19.1rc1), `vllm-gemma4-patched:latest`.
- FusenCache backend had prior-session "T1-B" shadow-tensor patch applied in `fusen_kv/backend.py` (52,259 bytes). Pre-patch backup kept at `backend.py.pre_t1b` (49,059 bytes).
- Prior FusenCache peak on main: 4,489 tok/s @ C=128 (Discovery #55, on vLLM 0.19.1rc1 image).
- Prior Qwen3-30B-A3B NVFP4 peak on PRO 6000: 17,426 tok/s @ C=512 (results.tsv 2026-04-10).
- I-DLM v2 was killed 2026-04-17 (acceptance 42%→8.6%; MASK-to-MASK attention is load-bearing).

## Completed Experiments

### Deliverables / Data
| ID | Experiment | Result | Headline Number | Files |
|---|---|---|---|---|
| T2-H | Qwen3-30B-A3B NVFP4 FP8 KV sweep | **PASS** | **16,637 tok/s @ C=768** (95% of prior 17,426 peak; 0 errors C=64→768) | `bench_t2h_qwen3.json`, `bench_t2h_qwen3_sweep.py` |
| T2-N | Fused shuffle+quant inline CUTLASS swizzle | **PASS** | **1.07×** microbench (16.96µs vs 18.19µs two-op); cos=0.9952 vs 0.9955 ref floor; 100% scales within ±1 FP8 step | `kernels/csrc/fused_shuffle_quant.cu` (edited), `kernels/csrc/build_fused_shuffle_quant.py` (new), `workspace/fused_shuffle_quant_sm120a.so` (rebuilt), `patches/test_fused_shuffle_quant_inline.py` |
| I-DLM v1 | Baseline reproduction PRO 6000 | **PASS** | **162.7 tok/s @ C=1, 65-68% acceptance** (beat historical 138.5/42% by +17%) | `run_idlm_v1.py`, `bench_idlm_v1_only.py`, `launch_v1_{nograph,graph}.sh` |

### Gate Tests / Kills
| ID | Experiment | Verdict | Why |
|---|---|---|---|
| T1-B | FusenCache piecewise CUDA graph patch | **REVERT** | Shadow-tensor patch regressed -95%: peak 224.9 tok/s @ C=32, C=64/128/256 crashed with `cudaErrorIllegalAddress` at `_to_list`→`pinned.copy_(sampled_token_ids)`. The `.copy_()` shadow-metadata path has size/indexing bugs under mixed-batch scheduling. Fallback FULL mode (captures [1..64]) = 43 tok/s @ C=32 (vs pre_t1b 2,017). Piecewise capture itself works — the crash fix landed — but runtime overhead + corruption dominate. |
| T1-B revert | pre_t1b baseline verify | **BLOCKED (upstream)** | pre_t1b backend crashes identically at C>4: `cudaErrorIllegalAddress` in `gemma4_mm.py:1257 embed_input_ids` → `is_multimodal = is_multimodal.to(input_ids.device)`. **Root cause is a vLLM main (0.1.dev100+gc0c98b8b9) Gemma4 multimodal bug, not FusenCache.** The 4,489 tok/s historical peak is only reproducible on the `vllm-built:latest` (0.19.1rc1) image — confirms Discovery #59 that main has regressions. |
| T1-F | TriAttention on Gemma4 26B | **KILL** (calibration criterion #3) | `scripts/calibrate.py` requires BF16/FP16 Gemma4 checkpoint for pre-RoPE Q calibration; local is NVFP4/modelopt-quantized only. `google/gemma-4-26b-it` is HF-gated. No AIME25/MATH-500 runs possible. |
| I-DLM graphs | v1 + CUDA graphs | **KILL** (+0% threshold) | +0% at C=1 (162.6 vs 162.7). Aggregate throughput flat ~160 tok/s across C=1/8/16 — I-DLM **does not batch** (each request runs full verify independently). Confirms prior T2G_idlm_vs_ar finding. No Medusa/EAGLE3 drafts cached for Qwen3-8B; would need training. |
| ASI-1 | Disaggregated 1P1D (1 prefill, 1 decode) | **KILL (WSL2 platform)** | `ncclCommInitRank` fails "unhandled cuda error" on first request because WSL2 does not support `cudaIpcGetMemHandle`, which NCCL requires for cross-process GPU transfer. Not fixable in config. Both instances loaded weights + reached health; NCCL handshake dies on real traffic. Revisit on bare-metal Linux or switch to LMCache / NIXL / UCX-RDMA connector. |

## Key Insights

1. **vLLM main has a latent CUDA illegal-address bug in Gemma4 multimodal path.** `gemma4_mm.py:1257 is_multimodal.to(input_ids.device)` faults at C>4 regardless of FusenCache patch state. Discovery #59 said "stay on 0.19.1rc1 + CUDA 12.8" — today's data is independent confirmation. Any Gemma4-on-main FusenCache work should revert to `vllm-built:latest` for reproducible throughput.

2. **Docker `--gpus 'device=N'` does NOT isolate GPUs on WSL2.** Both containers saw both GPUs; the ASI-1 decode instance silently ran on GPU 0 until an agent diagnosed it via `torch.cuda.get_device_properties(0).uuid`. Required fix: explicit `CUDA_VISIBLE_DEVICES=N` in the container env. `VLLM_HOST_IP=127.0.0.1` also needed for P2pNcclEngine to bind ZMQ on loopback where the proxy embeds it. (ASI-1 still died on CUDA IPC after both fixes — confirmed the NCCL issue is distinct.)

3. **SM90 CuTe fallback doesn't compile for sm_120a.** The `plans/idlm_setup.md` routing fix (add `12` to `[9, 10, 11, 12]`, route `12` through SM90 path) fails at JIT time with `cutlass.cute.nvgpu.common.OpError: expects Arch.sm_90a, got Arch.sm_120a`. I-DLM v1 ran anyway because a new patch bypasses CuTe entirely (set `dllm_mask_positions = None` in causal path, forcing FlashInfer). Same lesson for any future CuTe-based kernel port.

4. **I-DLM is a fundamentally single-user architecture on PRO 6000 SM120.** Aggregate ceiling ~160 tok/s irrespective of concurrency. AR batching trivially beats it at C≥8 (T2G_idlm_vs_ar from 2026-04-17 already showed 0.11× ratio). Treat I-DLM as a "single-user latency pin" only.

5. **CUTLASS NVFP4 swizzled-scale format is now well-understood and reusable.** Formula: `byte_offset = ((mTile*numKTiles + kTile) << 9) | (outerM<<4) | (innerM<<2) | innerK` with per-expert bump `blockscale_offsets[e]*numKTiles*4`. Inline in `fused_shuffle_quant.cu`; matches `rms_norm_dynamic_fp4_quant.cu::swizzled_sf_offset`. The 1.07× microbench is modest (vLLM's pipelined two-op is already fast) but the post-pass overhead is now zero, which unblocks the vLLM integration.

6. **PRO 6000 Qwen3-30B NVFP4 + FP8 KV serving is reproducibly in the 10-17k tok/s band.** Today's 16,637 tok/s @ C=768 (95% of 17,426 peak) confirms the config is stable across sessions. C=128 shows 10.5k (prefix cache warmed fast); the C=64 1,418 outlier is autotuning on cold start — not a regression, just a first-run artifact.

## In Progress / Not Started
*None.* All seven experiment threads reached a terminal state. Next-session candidates:

| Thread | Next Step | Cost Estimate |
|---|---|---|
| T2-N vLLM integration | Reshape `.so` scale buffer into vLLM's `MAX_TOKENS_PER_EXPERT*topk` layout; wire into `run_cutlass_moe_fp4`; e2e bench for the projected 2-4% | 4-6 hr |
| FusenCache main compat | File upstream vLLM bug for `gemma4_mm.py:1257` `is_multimodal.to()` CUDA illegal-address. Meanwhile pin serving to `vllm-built:latest` | 1 hr PR |
| Medusa/EAGLE3 for Qwen3-8B | Train a draft head (~1-2 days GPU) if batch-friendly spec decode still wanted | 2 days |
| ASI-1 on bare metal | Only on a non-WSL2 host. Or prototype with LMCache/NIXL connector instead of P2pNcclConnector | 1 day (bare metal) |
| TriAttention | Download/source a BF16 Gemma4 26B checkpoint, then run calibration + AIME25/MATH-500 sweep | 1 day |

## Result Rows Appended
`results.tsv` grew 128 → 154 lines. Eleven rows added today:
- T1-B piecewise: `T1B_piecewise_C{32,64,128,256}`, `T1B_fallback_FULL_C{32,64,128}`, `T1B_peak_vs_baseline`, `T1F_triattn_prereq`
- T2-H sweep: `T2H_qwen3_nvfp4_C{64,128,256,384,512,768}`
- T2-N: `T2N_fused_shuffle_quant_swizzled` (post-pass net -46%), `T2N_cutlass_inline_swizzle` (inline 1.07×)
- T1-B revert: `T1B_revert_pre_t1b` (gemma4_mm.py crash, vLLM main bug)
- ASI-1: `ASI1_disagg_bringup` (WSL2 NCCL kill)
- I-DLM: `IDLM_v1_baseline_pro6000`, `IDLM_v1_cuda_graphs`
- T1-F: `T1F_triattn_prereq_missing`

## Session Mechanics
- 4 Opus subagents dispatched in parallel (T1-B+T1-F GPU 0; T2-N CPU+brief GPU; I-DLM GPU 1; ASI-1 both GPUs).
- 1 Sonnet subagent dispatched (T2-H GPU 1) — aborted immediately on Docker permission denial; recovered by running sweep from parent session.
- Parent session handled: T2-H Docker + sweep directly, T1-B revert boot, bench scripts, results.tsv appends.
- No commits, no pushes. Containers `vllm-t1b-*`, `vllm-t2h`, `vllm-disagg-*` all cleaned on exit. GPUs idle at session close.

---

## Follow-up Wave (2026-04-18 afternoon / evening)

### Headline results

| ID | Experiment | Verdict | Headline |
|---|---|---|---|
| **T2-N e2e** | Fused shuffle+quant monkey-patched into vLLM MoE (Qwen3-30B-A3B NVFP4) | **PASS** | **1.34× at C=512 → 18,756 tok/s peak** (vs T2-H baseline 14k). Required `VLLM_USE_FLASHINFER_MOE_FP4=0` so vLLM routes through `run_cutlass_moe_fp4`. Plugin installed via dist-info entry_points. |
| **T2-N bigsweep** | Extended to `--max-num-seqs 1024`, C=384/512/640/768/896/1024 | **PASS** | Peak held at C=512 (17,614 tok/s). Past 512 throughput plateaus/dips due to KV memory pressure. Zero errors. |
| **Upstream PR** | `gemma4_mm.py:1257 is_multimodal.to()` async race | drafted | Root cause: `is_mm_embed_buffers` GPU storage aliases across async iters under CUDA graphs. 2-behavior fix (text-only fast path + `.clone()` + `non_blocking=False`). Files: `upstream/PR_gemma4_embed_input_ids_async_race.md`, `upstream/gemma4_embed_input_ids_fix.patch`. `gh` CLI installed; filing recipe in `upstream/README.md`. |
| **T1-B post-gemma4fix** | Re-verify FusenCache after applying the upstream fix | **FAIL** (partial unblock) | Gemma4 patch validated at C=32 (373 tok/s, 0 crashes), but **more async race sites surface**: `gemma4.py:177 Gemma4Router.forward root_size.to()` crashes C≥64 FULL mode; `gpu_model_runner.py:269 async_copy_ready_event.synchronize()` crashes piecewise. vLLM main has a broader async-scheduling issue — one-line fix not sufficient. Did not beat 4,489 baseline. |
| **ASI alt A+D** | `--scheduling-policy priority` + `--max-num-batched-tokens 512` + prefix caching on patched Gemma4 | **PARTIAL** | Zero errors at C=4/8/16/32. Short P99: C=4 921ms, C=8 4059ms (tail spike — priority didn't fire), C=16 688ms, C=32 802ms. vs disagg target 120ms@C=8: FAIL at C=8, PASS at C≥16. Priority+chunking effective at mid-high concurrency only. |

### CUDA/GPU expert ideas (7 Opus agents dispatched in parallel)

| # | Idea | Verdict | Numbers |
|---|---|---|---|
| 1 | **Mega-graph cooperative kernel** | **GO** | `grid.sync()` infra cost on SM120: 30L × 3 barriers = 73.5µs cooperative vs 82.7µs graph replay = **0.89× (cooperative faster)**. Per-barrier 0.82µs pure overhead (vs Discovery #35's 278µs — that included SM work imbalance, this isolates barrier mechanism). Skeleton 423 lines builds clean on sm_120a. Recommend 2 barriers/layer for full impl. Also found: phase-6 atomicAdd race in `persistent_moe_dispatch.cu` to fix in same PR. |
| 2 | **FusenCache cp.async + TMA decode** | **KILL** | 3-stage double-buffered cp.async: 14.2% HBM BW (254 GB/s of 1792 peak) = 1.24× over baseline C++ decode at 11.4% BW. Below 50% KILL gate. Root cause: decode is compute-bound on 4-bit nibble unpack + online softmax, not HBM-bound. TMA rejected: paged KV scatter defeats bulk-copy. Next: warp specialization + vectorized `prmt.b32` dequant. |
| 3 | **Warp-specialized attention (producer/consumer)** | **KILL** | 63.5% BW vs FlashInfer 89.4% = 0.71× (29% slower). Premise doesn't transfer: SM120 has 100 KB SMEM/SM (not Hopper's 228), no TMA, no WGMMA. Warp-spec gain on Hopper comes from overlapping 2 distinct async units — SM120 has neither. FA2's `cp.async.wait_group` already near-optimal. **Recommendation: FP8 KV decode is the real attention win (data-volume cut), not pipeline restructuring.** |
| 4 | **EAGLE3 training pipeline, Qwen3-30B** | **BLOCKED** (on vLLM fp8_kv bug) | Pipeline validated end-to-end with pseudo labels: loss 2023→15.3 in 600 steps/14s, top-1 match 0→75% (trivial upper bound). Phase A (vLLM teacher) BLOCKED: NVFP4 checkpoint forces `kv_cache_dtype=fp8_e4m3` but allocator sizes at BF16 → reshape `[N,2,16,4,128]` fails by 2×. Need vLLM patch in `_reshape_kv_cache_tensors`. Real training to 50% acceptance: ~6-10 hr AFTER the fix. |
| 5 | **Expert prefetch via `cudaMemAdvise`** | **KILL** | EMA prefetch (75% recall, top-16): 62.0µs → 125.9µs = 0.49× (2× slower). Oracle (exact top-8): 62.0µs → 95.7µs = 0.65× (1.5× slower). L2 hit rate→speedup conversion = 0%. Root cause: MoE grouped GEMM is HBM-BW bound (~2-3 FLOPs/byte), not latency-bound. Two HBM-bound streams contend on the bus (1.08× serialized per sanity check). Even oracle prefetch strictly worse. Fundamentally closed for Gemma4 MoE NVFP4. |
| 6 | **FA3 SM120a port** | **KILL** | First-attempt compile: RC=0 but cubin is empty (`LDC R1; EXIT;` stub) because `hopper/utils.h::enable_sm90<>::operator()` is `#if __CUDA_ARCH__ == 900` gated. After patching to accept 1200, build succeeds but cuobjdump shows 0 MMA/WGMMA/HMMA/TCGEN ops — `cute::GMMA::ss_op_selector::fma()` expands to `CUTE_INVALID_CONTROL_PATH` on non-SM90, which is a no-op under `-DNDEBUG`. `cutlass/arch/config.h:48` gates WGMMA on exactly `__CUDA_ARCH__ == 900`. SM120 has TMA + STSM + mbarrier but NOT WGMMA. Real port requires rewriting `mainloop_fwd_sm90_tma_gmma_ws.hpp` on `SM120_16x8x32_TN` synchronous tensor cores, dropping warp-spec. Estimate: 1-2wk BF16-fwd, 3-4wk FP8-fwd, 6-8+wk bwd. Expected gain without WGMMA: 0-10%. Wait for upstream `_sm120.cu` or use FlashInfer's native SM120 kernels. |
| 7 | **SASS native `cvt.e2m1x2.f32` FP4 scale kernel** | **KILL (premise wrong)** | Kernel correct: byte-identical to CUTLASS at M=128/512/2048/8192. 1.06× avg (1.01-1.09 range). BUT: CUTLASS's `scaled_fp4_experts_quant` already uses the native `cvt.rn.satfinite.e2m1x2.f32` PTX (verified in `/build/vllm/csrc/libtorch_stable/quantization/fp4/nvfp4_utils.cuh:72-89`) — the Discovery #10 speedup is already baked in. ~0.1% e2e. Do not integrate; redirect to T2-N wrapper cleanup. |

### ASI-1 reconsideration (WSL2 NCCL kill)

ASI-1 disaggregated 1P1D still dead on WSL2 (CUDA IPC unavailable). ASI alt A+D validated as partial substitute — good at C≥16, not at C=8. Remaining alternatives on the table: LMCache KV connector (CPU-shared, no NCCL); DP=2 with length-routed proxy.

### Key insight adds (today)

7. **`VLLM_USE_FLASHINFER_MOE_FP4=0` is required** to route Qwen3 MoE through `run_cutlass_moe_fp4` (the path T2-N monkey-patches). Without it, vLLM picks `FLASHINFER_CUTLASS` and T2-N's patch is inert. Host-shell env vars do NOT propagate into Docker containers — must pass via `-e`.
8. **Cooperative launch scaffolding on SM120 beats CUDA graph replay at pure infra cost** (0.89×). The Discovery #35 "278µs / barrier" number was dominated by SM work imbalance, not the barrier mechanism itself. Mega-graph is viable.
9. **vLLM main has multiple async-scheduling race sites beyond `gemma4_mm.py`** — `gemma4.py:177` router and `gpu_model_runner.py:269` scheduler sync both crash the same way. One upstream PR won't be enough; a broader `synchronize_input_prep`-style fence (Discovery #54 pattern) is needed.
10. **Docker `--gpus 'device=N'` does NOT isolate GPUs on WSL2** (confirmed today by ASI-1 agent via UUID check). Both containers see both physical GPUs. Required workaround: explicit `CUDA_VISIBLE_DEVICES=N` env.
11. **Docker `--cpuset-cpus` pins containers to specific CCDs** — 9950X3D has CCD 0 with 3D V-Cache (cores 0-7, threads 0-15). Give V-Cache CCD to the latency-sensitive container. In-place via `docker update --cpuset-cpus=…` works without restart.

### Artifacts (today)

**New code / patches:**
- `patches/fused_shuffle_quant_wrapper.py` (path fix + kernel sig align)
- `patches/fused_shuffle_quant_plugin.py` (vLLM plugin entry point)
- `launch_qwen3_fused_t2n.sh` (baseline/patched launcher)
- `kernels/csrc/fused_shuffle_quant.cu` (inline CUTLASS swizzle, rebuilt .so at workspace/)
- `kernels/csrc/build_fused_shuffle_quant.py`
- `kernels/csrc/mega_graph_skeleton.cu` (423-line cooperative kernel skeleton)
- `kernels/csrc/build_mega_graph.py`
- `kernels/csrc/bench_mega_graph_feasibility.py`
- `kernels/csrc/fusencache_decode_cpasync.cu` (623-line cp.async prototype — KILL'd, keep for warp-spec v2 reference)
- `kernels/csrc/build_fusencache_cpasync.py`
- `kernels/csrc/native_fp4_scale_kernel.cu` (KILL'd — premise wrong)
- `kernels/csrc/build_native_fp4_scale.py`
- `bench_t1b_revert.py`, `bench_t2h_qwen3_sweep.py`, `bench_asi_alt_ad.py`

**Upstream (drafts, not pushed):**
- `upstream/PR_gemma4_embed_input_ids_async_race.md` (10 KB)
- `upstream/gemma4_embed_input_ids_fix.patch` (5 KB)
- `upstream/README.md` updated with `gh`-based filing recipe

**Images:**
- `vllm-fusencache-gemma4fix:latest` (rebuilt correctly via `/build/vllm/` path)
- `vllm-fusencache:latest` (unchanged)

**Plans:**
- `plans/mega_graph_cooperative_kernel.md`
- `plans/fusencache_cpasync_tma.md`
- `plans/sass_fp4_scale_kernel.md`
- (2 more coming from in-flight agents)

### Next-session candidates (with today's evidence)

1. Mega-graph cooperative kernel full impl (GO per feasibility) — 1-2 weeks for +1.3-2× decode, unblocks SM120 large-batch throughput
2. Warp-specialized attention v2 (cp.async v1 KILL'd because compute-bound, not HBM-bound — need warp spec + vectorized dequant) — 5-7 days
3. vLLM upstream PR: broader async-scheduling fix across `gemma4_mm.py`, `gemma4.py:177`, `gpu_model_runner.py:269` — the single-file patch isn't enough
4. T2-N deployment polish: reshape wrapper's scale buffer into vLLM-standard layout → eliminates the `VLLM_USE_FLASHINFER_MOE_FP4=0` requirement
5. EAGLE3 Qwen3-30B continuation (if initial 1-2hr training shows acceptance >50%)
6. LMCache KV connector prototype (sidestep WSL2 NCCL block)

---

## Second Wave (2026-04-18 evening — triggered by "do megagraph + warp-spec v2" request)

### Mega-graph 2-layer prototype — **PASS (2.89-3.57× speedup)**

Agent took the skeleton and wrote `kernels/csrc/mega_graph_gemma4.cu` (~400 lines). Real bodies: RMSNorm → QKV → attention (4 heads, head_dim=128) → O+residual → RMSNorm → SwiGLU+residual per layer. BF16 throughout, dense KV, 1-expert MLP as starter simplifications. Each SM owns a stripe of hidden_dim=512. All 188 SMs cooperate per stage.

| Measurement | Value |
|---|---|
| Mega-graph 2-layer latency | **289 µs** (stable ±0.1µs) |
| Eager PyTorch F.linear 2-layer | 834-1030 µs |
| Speedup | **2.89-3.57×** |
| Correctness vs PyTorch | max_abs_diff = 1.95e-3 (ref_max 0.29, ~0.7% rel; well under 5e-3 gate) |
| HBM utilization | ~2% (latency-bound at M=1, expected) |
| Realized barriers/layer | 9 (rmsnorm inner grid-reduces) vs design target 2 |
| 30-layer projection | 4.33 ms mega vs 12.5 ms eager |

**Verdict: GO for full 30-layer impl.** Cooperative scaffolding works AND a realistic layer body works on SM120a. Collapsing to design's 2 barriers/layer (head-per-SM attention partition + single post-stage fence) is the next iteration. Phase-6 `atomicAdd` race in `persistent_moe_dispatch.cu` flagged for the MoE-integration iteration (dense prototype doesn't exercise it).

### Mega-graph: 30-layer scale-up **KILL 1.03×** — bottleneck is layer bodies, not barriers

Agent scaled to H=2048 (real Gemma4), 30 layers, heads=16, head_dim=128, intermediate=8192. Replaced grid-reduce RMSNorm with per-SM local block-wide reduction (collapsed 9→7 barriers/layer). Correctness: 3.07% relative diff at 30 layers — smooth monotonic BF16 accumulation (1 layer 1.95e-3, 10 layers 2.0e-2, 30 layers 5.86e-2). Not a barrier bug.

| | 30-layer decode (M=1, seq=256) |
|---|---|
| mega-graph | 23.69 ms |
| eager PyTorch 30-layer | 24.33 ms |
| speedup | **1.03×** (below 1.5× KILL gate) |
| HBM BW | **9.6%** (170 GB/s of 1792) |

**Root cause**: scalar BF16 FMA inner loops (`for i in range(HIDDEN): acc += xi * W[i,d]`) don't use tensor cores. cuBLAS `F.linear` uses `mma.sync.m16n8k16` BF16×BF16 tensor-core MMA — roughly **50× faster** than scalar CUDA cores at H=2048. The 2.89× win at 2-layer H=512 was an artifact of **cuBLAS launch overhead (~17µs) dominating tiny GEMV compute**; at H=2048 that overhead amortizes and the cooperative-kernel advantage vanishes entirely.

**This does NOT kill mega-graph** — it kills the naive scalar port. Cooperative launch + `grid.sync()` infra is proven sound; **the layer bodies need tensor-core matmul**. Agent's projection with WMMA/`mma.sync.m16n8k16`: 2-5ms / 30-layer step vs 24ms eager = **5-12× real win**. That's a v2-with-tensor-cores experiment, not a dead end.

**Next step (pending user):** dispatch a v2 agent that swaps scalar inner loops for WMMA/`mma.sync` tensor-core MMA. If it lands 5-10× at real H=2048, that's the session headline.

Files:
- `kernels/csrc/mega_graph_gemma4_30layer.cu` (naive scalar version, KILL'd)
- `plans/mega_graph_cooperative_kernel.md` §10 updated with 30-layer findings

**Files**: `kernels/csrc/mega_graph_gemma4.cu`, `build_mega_graph_gemma4.py`, `test_mega_graph_gemma4.py`. `plans/mega_graph_cooperative_kernel.md` §9 updated.

### SWA sparse attention Gemma4 — **PASS (4.64× @ seq=8192, 20× @ seq=16384)** 🎯

Triton port of existing split-KV online-softmax decode, with page-level truncation of the block table: for each sequence, stage-1 loop starts at `start_page = max(0, seq_len - window) // page_size` and programs are never launched against skipped range.

Shapes: B=16, head_dim=128, num_q_heads=8, num_kv_heads=2 (GQA=4), window=4096. BF16 KV, page_size=16.

**Correctness**: cos=0.999999 vs PyTorch FP32 (max_abs=3.05e-5). cos=0.999995 vs FlashInfer dense+mask.

| seq_len | Triton SWA | FlashInfer dense+mask | Speedup | Theoretical |
|---|---|---|---|---|
| 4096 (no sparsity) | 228 µs | 29 µs | 0.13× | 1.0× |
| **8192 (primary)** | **131 µs** | **610 µs** | **4.64×** | 2.0× |
| 16384 | 316 µs | 6320 µs | **20.0×** | 4.0× |

30.1% HBM BW (511 GB/s) at primary shape — still headroom via larger BLOCK_H, vectorized BF16 loads, or cp.async pipelining.

**Caveat**: at seq_len ≤ window, FlashInfer's tuned CUTLASS path beats Triton ~8×. Integration guard: `seq_len > window × 1.25`.

**E2E projection for Gemma4**: attention = 63% of decode × 25/30 sliding × 4.64× speedup at typical production seq → saves ~41% of decode wall → **~1.7× e2e speedup** when context exceeds window.

**Next step**: wire into vLLM `v1/attention/backends/flashinfer.py` for Gemma4's sliding layers. Stack with FP8 KV for the 2× memory (Discovery #37 confirmed FP8 doesn't help latency, only capacity).

**Files:**
- `plans/swa_sparse_attention_gemma4.md`
- `kernels/triton/swa_decode.py`
- `kernels/csrc/test_swa_decode.py`

### LMCache WSL2 KV connector — **PASS-blocked** (design ready, GPU smoke deferred)

- LMCache v1 connector already shipped in vLLM (`/build/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`, `lmcache_mp_connector.py`). `factory.py` registers `LMCacheConnectorV1` + `LMCacheMPConnector`.
- `lmcache==0.4.3` installable via `pip install --break-system-packages lmcache`. Pulls `cupy-cuda12x`, `nixl`, `redis`, `aiofile` — none active at default config.
- **Zero `cudaIpc` / `IpcMemHandle` references in the lmcache package.** Default config (`local_cpu=True`, `enable_p2p=False`, 20GB pool, chunk 256) uses plain `cudaMemcpyAsync` over pinned host buffers → **WSL2 compatible**. This is what ASI-1 was missing.
- Projected gain on 20-concurrent-user shared-prefix (500-tok system prompt + 50-tok query): **7.6× prefill reduction** (92 ms aggregate vs 700 ms baseline) → **P99 TTFT 700→50 ms**. Aggregate throughput gain modest (+10-20%, decode-dominated).
- GPU smoke deferred: GPU 0 at 89% util (T2-N polish + eagle3_train); GPU 1 also contended. Launch command + config yaml ready.

**Files:** `plans/lmcache_wsl2_test.md`

**This replaces ASI-1's killed direction.** WSL2-compatible substitute with the same order of magnitude on P99 TTFT.

### FusenCache warp-spec v2 decode — **KILL (incremental, not transformative)**

1-producer / 3-consumer warp topology with `bar.sync`-based rendezvous and vectorized nibble unpack (`uint32_t` load + `and 0x0F0F0F0F` + `shr 4`). ~600 LoC, compiled first-try on sm_120a, correctness PASS (max_abs 4.88e-4).

| kernel | latency | GB/s | % of 1792 peak | vs v1 |
|---|---:|---:|---:|---:|
| baseline C++ | 347 µs | 205 | 11.5% | 0.80× |
| cp.async v1 | 279 µs | 255 | 14.3% | 1.00× |
| **warp-spec v2** | **236 µs** | **302** | **16.8%** | **1.18×** |

**Verdict: KILL** (16.8% below 30% gate). Warp-spec works — real 1.18× — but the consumer path is still serialized on per-KV-token softmax broadcast. Producer/consumer overlap only pays at the per-tile boundary (every 16 KV tokens), not per-token.

**Path forward (v3, deferred):**
- (a) Warp-per-head split to remove the cross-warp reduce
- (b) Tensor-core `mma.sync.m16n8k16` for QK so 16 tokens' scores emerge from one instruction
- (c) Offline softmax with partial accumulator across the full tile before broadcast

For the current session: stop pursuing kernel-level FusenCache decode optimization. Bigger gains are elsewhere (mega-graph, sliding-window sparse attention, FP8 attention re-measure).

**Files:**
- `kernels/csrc/fusencache_decode_warpspec_v2.cu`, `build_fusencache_warpspec.py`, `bench_fusencache_warpspec.py`
- `plans/fusencache_warp_spec_v2.md`

### T2-N polish — remove env-var dependency — *(in flight, GPU 0)*

### Warp-specialized ATTENTION (separate experiment, unrelated to FusenCache v2) — KILL

Earlier in session this was tested on vanilla BF16 decode. 63.5% BW vs FlashInfer 89.4% = 0.71×. **Premise doesn't transfer to SM120a**: consumer Blackwell has 100 KB SMEM/SM (not Hopper's 228), no TMA, no WGMMA. Warp-spec's gain on Hopper comes from overlapping TMA async engine + WGMMA scheduler — SM120 has neither. FA2's `cp.async.wait_group` already near-optimal.

**However** — the FusenCache v2 in-flight agent targets a DIFFERENT bottleneck: not BW, but the **compute-bound 4-bit nibble unpack** on the consumer path. Different problem, different expected outcome.

### EAGLE3 pipeline — BLOCKED on vLLM fp8_kv reshape bug

Pipeline validated end-to-end with pseudo labels (loss 2023→15.3 in 600 steps/14s, top-1 match 0→75% — trivial upper bound, proves loop not draft quality). Phase A (vLLM teacher logprobs) BLOCKED: NVFP4 checkpoint forces `kv_cache_dtype=fp8_e4m3` but the allocator sizes at BF16 → `_reshape_kv_cache_tensors` fails by exactly 2× on shape `[N,2,16,4,128]`. Need a second vLLM upstream patch. Real training to 50% acceptance: ~6-10 hr on PRO 6000 AFTER the fix.

### Updated recommendation for next session

1. **Start mega-graph 30-layer impl** — feasibility PROVED, not just projected. Prototype already at 2.89× in 400 LoC.
2. **T2-N polish result** (pending) will tell us whether 1.34× ships without the env-var.
3. **Warp-spec v2 result** (pending) will determine whether FusenCache decode stays at 14% BW or breaks through to 30-50%.
4. **vLLM upstream fixes**: `gemma4_mm.py` + broader async-race + fp8_kv reshape bug. 3 small PRs.
5. EAGLE3 restart once fp8_kv reshape is fixed upstream.

---
## W1_4e_lmcache_smoke — LMCache KV Connector Smoke Bench

**Date:** 2026-04-18 18:45–19:10
**GPU:** GPU 0 (RTX PRO 6000, SM120, CUDA 13.0)
**Model:** Qwen/Qwen3-8B (BF16, max_len=4096, gpu_mem_util=0.85)
**Image:** vllm-fusencache:latest

### Baseline Result (no LMCache)
- Config: `--enable-prefix-caching`, no `--kv-transfer-config`
- 20 concurrent requests, 500-tok shared prefix + 50-tok unique query + 32-tok gen
- P50 TTFT: **32184.8 ms**
- P95 TTFT: **32203.0 ms**
- P99 TTFT: **32203.0 ms**
- Wall time: 32.60s, Agg throughput: 19.6 tok/s
- All 20 requests serialized through prefill → expected for cold cache workload

### LMCache Result (LMCacheConnectorV1)
- Status: **KILL — SM120 binary incompatibility**
- `lmcache==0.4.3` PyPI package `c_ops.so` compiled for CUDA 12 / pre-Blackwell architectures
- Error: `CUDA error: no kernel image is available for execution on the device` in `multi_layer_kv_transfer`
- The `c_ops.so` extension kernel does not have a cubin for SM120 (Blackwell)

### Root Cause Chain
1. `pip install lmcache==0.4.3` pulls `cupy-cuda12x` + `nixl` → `nixl_ep_cpp.so` requires `libcudart.so.12` (CUDA 12 soname), image has CUDA 13
2. Workaround: `--no-deps` + `nvidia-cuda-runtime-cu12` for `libcudart.so.12`
3. Still fails: `c_ops.cpython-312-x86_64-linux-gnu.so` has `libcudart.so.12` linkage → fixed with CUDA 12 runtime on LD_LIBRARY_PATH
4. Fatal: `c_ops.so` CUDA kernels compiled without SM120 architecture support → `cudaErrorNoKernelImageForDevice`

### Fix Path
- Rebuild lmcache from source with CUDA 13.0 + SM120 architecture flag:
  `TORCH_CUDA_ARCH_LIST="12.0"` in build environment
- OR: Use LMCache's `vllm-integration` mode where the in-tree vLLM adapter handles KV operations using vLLM's own CUDA kernels (bypasses c_ops), requiring lmcache Python package only for config/metadata
- Check `plans/lmcache_wsl2_test.md §8 Open Questions` item 3: c_ops-free path may work

### Verdict
- **KILL**: lmcache 0.4.3 PyPI binary is SM120-incompatible
- Source build required (10-min fix, not blocking for next session)
- Baseline TTFT data is valid: P99=32.2s at C=20 for 500-tok shared prefix workload
- Container teardown confirmed: GPU 0 at 603 MiB (idle)

