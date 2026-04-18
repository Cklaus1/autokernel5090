# I-DLM v2 Phase 1 Synthesis

## Agents Deployed: 4 (2 Opus, 2 Sonnet)

## Key Findings

### mask_mod is blocked (1A finding)
- CuTe `mask_mod` callable works on SM120 ✓
- BUT: `NotImplementedError` for varlen sequences (interface.py:329)
- AND: FlashInfer wrapper doesn't expose mask_mod parameter
- Two layers of abstraction block the "clean" path

### custom_mask bitmask is viable (1D finding)
- FlashInfer paged wrapper accepts `custom_mask` bitmask (line 663)
- Already used for spec decode tree masks
- I-DLM extend blocks are tiny (7 tokens) → 7×7 bitmask = trivial memory

### Mask builder is written (1B deliverable)
- `build_idlm_attn_mask_fast()` — vectorized, no Python loops
- Takes `extend_lens` + `committed_ends` → returns `[total, total]` bool mask
- Patch for `idlm_blockN.py:502` replaces `dllm_force_causal=True`

### Attention trace mapped (1D deliverable)
- Full call chain: idlm_blockN → model_runner → flashinfer_backend → kernel
- Primary injection: flashinfer_backend.py:920-932 (ragged wrapper)
- Secondary injection: flashinfer_backend.py:839-862 (paged wrapper)

## Implementation Plan (revised)

### Step 1: custom_mask bitmask approach (1-2h)
1. Add `dllm_attn_mask` field to `FlashInferAttnMetadata`
2. In `forward_extend()`, check for `dllm_attn_mask` and pass to wrapper
3. In `idlm_blockN.py`, build mask via `build_idlm_attn_mask_fast()`
4. Test single request correctness

### Step 2: If custom_mask doesn't work on ragged wrapper → Approach A
1. Bypass FlashInfer entirely for DLLM extend path
2. Reshape to 4D tensor (requires uniform block lengths)
3. Call CuTe `flash_attn_func()` directly with `mask_mod`

### Step 3: Benchmark acceptance rate
1. Run `bench_idlm_v2.py` comparing causal vs correct mask
2. Gate: acceptance rate improvement ≥5% → continue to Phase 2
