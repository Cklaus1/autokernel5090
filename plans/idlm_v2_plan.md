# I-DLM v2 Plan — Making Diffusion LM Competitive at Batch

## Bake-off Results (April 2026)

| Concurrency | I-DLM tok/s | AR tok/s | I-DLM/AR |
|---|---|---|---|
| C=1 | 110 | 61 | **1.80×** |
| C=4 | 147 | 135 | 1.09× |
| C=8 | 149 | 491 | 0.30× |
| C=16 | 153 | 991 | 0.15× |
| C=32 | 176 | 1,591 | 0.11× |

**I-DLM wins single-user by 1.8× but loses batch by 9×.**

## Root Cause: Why I-DLM Can't Batch

AR batches trivially: all requests predict the next token → one big matmul across the batch.

I-DLM can't batch because each request is at a different diffusion phase:
- Different numbers of committed vs MASK tokens
- Different attention masks (MASKs should NOT attend to each other)
- The verify step is per-request (compare draft vs target for each sequence independently)

The current code (`idlm_blockN.py:337-1037`) processes requests with `dllm_force_causal=True` — a shared causal mask for all requests. This is semantically wrong (MASKs contaminate each other) but necessary for batching with FlashAttention's standard causal kernel.

## The v2 Fix: `mask_mod` Kernel

### What
Replace `dllm_force_causal=True` with a per-request `mask_mod` callback that FlashAttention evaluates per-element:

```python
def idlm_mask_mod(b, h, q_idx, kv_idx):
    committed_end = committed_lengths[b]  # per-request
    if q_idx >= committed_end:  # query is a MASK token
        return kv_idx < committed_end  # attend only to committed
    else:
        return q_idx >= kv_idx  # standard causal
```

### Why This Enables Batching
- FlashAttention's `mask_mod` evaluates the mask function inside the kernel — no materialized N×N mask tensor
- Different requests can have different `committed_end` values → different mask patterns within the same batch
- SM120's persistent non-causal kernel path is actually FASTER than the causal path (no triangular load imbalance)
- The mask_mod is compiled into the attention kernel via CuTe DSL, so it's zero overhead per element

### Where
- `interface.py:430-478` — the SM90 forward path already supports `mask_mod` (line 113, 266)
- `flashinfer_backend.py:920-923` — replace `causal=True` with `mask_mod=idlm_mask_mod`
- `idlm_blockN.py:500-502` — replace `dllm_force_causal = True` with mask_mod construction

### Expected Impact
1. **Quality:** +5-15% acceptance rate (MASKs no longer contaminated by each other's KV → cleaner predictions)
2. **Batch throughput:** With per-request masks, all requests can be batched into one attention call regardless of diffusion phase → batch scaling like AR
3. **SM120 performance:** Persistent non-causal kernel runs ~10% faster than causal (no triangular tile skipping needed)

Combined: I-DLM v2 at C=32 should reach **0.5-0.8× of AR** (vs current 0.11×), and at C=1 should reach **2.0-2.5×** (vs current 1.8×, from better acceptance rate).

## Implementation Plan

### Phase 1: mask_mod prototype (1 day)
1. Write the `idlm_mask_mod` function that takes `committed_lengths` tensor
2. Test on a single request to verify correctness vs causal mask
3. Verify SM120 SM90 path dispatches to persistent non-causal kernel
4. Measure single-request quality (acceptance rate) and latency

### Phase 2: Batched verify (2 days)
1. Modify `idlm_blockN.py` verify step to work across batched requests
2. The current `_mega_packed.tolist()` sync is already batched — keep it
3. KV trim needs to handle per-request trim indices in parallel
4. Test at C=4, 8, 16, 32 — measure throughput scaling

### Phase 3: Fused classify + forward (1 day)
1. The classify step (cold start vs verify) currently branches per-request
2. Unify: all requests go through the same forward pass, mask_mod handles the per-request semantics
3. This eliminates the serial classify loop

### Phase 4: CUDA Graphs (1 day)
1. With mask_mod and unified forward, CUDA graph capture becomes possible
2. The mask_mod function is compiled into the kernel at capture time
3. `committed_lengths` is a graph input (updated per step without recapture)
4. Expected: +3-5× single-user, +2-3× batch (matching AR's CUDA graph benefits)

## Kill Criteria
- After Phase 1: if acceptance rate doesn't improve by ≥5%, the mask contamination theory is wrong → stop
- After Phase 2: if batch throughput at C=32 is still <0.3× of AR → architecture is fundamentally slower → stop
- After Phase 4: if total throughput doesn't exceed AR by ≥1.5× at C=1, the complexity isn't worth it

## Resource Requirements
- GPU 0 for development/testing
- Qwen3-8B-b2-allmasked checkpoint (already downloaded, 16 GB)
- SGLang fork with I-DLM support (already installed at /tmp/idlm_venv/)
- SM120 patches applied (interface.py already patched)

## What We'd Need from Upstream
- FlashAttention SM90 path to expose `mask_mod` through SGLang's API (currently only through CuTe DSL)
- OR: write a Triton attention kernel with per-request mask support (avoids CuTe dependency)

## Alternative: Skip I-DLM, Use Medusa/EAGLE3 Instead
If the mask_mod approach doesn't pan out, speculative decoding via Medusa or EAGLE3 provides similar single-user latency benefits (2-3×) with proven batch scaling. The downside: requires training a separate draft head (1-2 days) per target model. I-DLM's advantage is that the draft IS the model (no separate training needed).
