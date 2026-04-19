# FusenCache Decode Attention with cp.async + (evaluate TMA) — SM120a design doc

Date: 2026-04-17  Target: RTX PRO 6000 Blackwell Max-Q (SM120a, cc 12.0).
Spec: FusenCache k4v4b64 (4-bit K + 4-bit V, scale_block=64), head_dim=256,
GQA (Hq=16, Hk=8, group=2), paged KV (page_size=16 or 64 tokens/page).

## 1. Current kernel ceiling (the 35% BW problem)

Existing `fusencache_decode_attention.cu` (stage 1) loads packed K and V one
byte at a time per thread per `kv_idx`:

```
for (int i = tid; i < HALF_D; i += BLOCK_THREADS) {
    uint8_t k_packed = kv_cache[slot_base + i];   // 1-byte scalar load
    float   k_sc     = __half2float(scales[...]); // 2-byte scalar load
    ...
}
```

With HEAD_DIM=256, HALF_D=128, BLOCK_THREADS=128, that is exactly one 8-bit
load per thread per KV token, issued twice (once for K, once for V). The
loads are coalescable (contiguous byte lanes) but the generator can only
emit 32-bit sector fills at best. Measured BW on PRO 6000 for this kernel:
**~35 %** (Discovery #30). FA2 on the same GPU achieves 93 % on BF16 KV.

### Micro-measurement (sanity check, same GPU)

Config: B=16, Hq=16, Hk=8, D=256, seq_len=2048, page_size=16, num_splits=16.
Bytes read per decode step (K + V + scales, 4-bit so 0.5 B/dim each):
```
B * Hk * seq_len * (K_bytes_per_tok + V_bytes_per_tok + scale_bytes_per_tok)
= 16 * 8 * 2048 * (128 + 128 + 2*4*2)
= 16 * 8 * 2048 * 272
= 71.3 MiB
```
At 1792 GB/s peak, floor = 71.3 MiB / 1792 GB/s = **41.6 us**.
Existing kernel measured (warm, graph-captured, B=16 seq=2048): ~110–130 us
(see `fusencache_k4v4b64_bench.txt`). → **effective BW 32–38 %**. Confirms
the 35 % Discovery figure.

## 2. cp.async pipelining strategy

### Load width / instruction
Use `cp.async.cg.shared.global` with `cp_size=16` (16-byte cache-global). One
thread moves 16 bytes (= 32 packed 4-bit K values, = 64 half-dim 4-bit V
values). With 128 threads and HALF_D=128, one warp-collective issue loads
128 * 16 B = 2 KiB in a single instruction group — enough K for 16 KV tokens
per cp.async group, or V for 16 tokens.

Per KV token (K region): HEAD_DIM/2 = 128 B, so 128 B / 16 B = 8 threads are
enough. With a BLOCK_KV = 16-token tile and 128 threads, each thread issues
exactly one 16-B K-load and one 16-B V-load per tile.

### Tile layout (per-block)

Work on one (batch, head_group, split) tuple; iterate over KV in tiles of
BLOCK_KV = 16. For each tile:

1. **Issue stage N+1 loads** (cp.async.cg, 128-B tile of K for 16 tokens).
2. **Issue stage N+1 V loads** (same shape).
3. `cp.async.commit_group()` after each stage.
4. `cp.async.wait_group<N-1>()` → stage N ready. Dequant from SMEM, run
   QK^T, online softmax, PV accumulate.

We issue 2 groups per tile (K + V), tracked together as one pipeline stage.

### Shared-memory budget
Per pipeline stage: K tile 128 B × 16 tok = 2 KiB, V tile 2 KiB,
K scales 2 B × 2 sb × 16 = 64 B, V scales 64 B → ≈ 4.13 KiB / stage.

Stages × 4.13 KiB + query (2 heads × 256 × 4 B = 2 KiB) + reduction
scratch (0.5 KiB). On SM120 the optin smem is 101 376 B/block and 102 400
B/SM.

| stages | smem/block | blocks/SM (SMEM) |
|-------:|-----------:|-----------------:|
| 2      |  10.9 KiB  | 9 (SMEM) → limited by 128 thr reg |
| 3      |  15.0 KiB  | 6 |
| 4      |  19.2 KiB  | 5 |

So SMEM is not the bottleneck here (unlike FA2 with 96 KB BF16 tiles).
Occupancy will be bound by registers / threads, not SMEM.

### Pipeline depth decision: **stages = 3**
Rationale:
- stage=2 is the classic "one-behind" double buffer; exposes ~1 L2 miss
  latency (200 ns / 4 us) per tile.
- stage=3 fully hides a round trip from HBM (≈500–800 ns on Blackwell) by
  allowing 2 outstanding groups of cp.async, which is the upper bound
  advertised for SM80-class cp.async before diminishing returns.
- stage=4 gains little here because each tile is tiny (4 KiB) and the
  L2 prefetcher is already in the hot path; also costs an extra 4 KiB SMEM.

### Dequant
Dequant happens from SMEM on the consumer side. 4-bit unpack + FP32 FMA is
a handful of ops per byte and fits in the memory-latency shadow. The Q
vector is pre-converted to FP32 and kept in SMEM (same as current kernel),
so no extra traffic per-tile. Scales live in a tiny SMEM ringbuffer
alongside K/V.

## 3. TMA on SM120a (`cp.async.bulk.tensor.*`)

SM120a exposes TMA (bulk tensor copy) via PTX. However:

- TMA requires a **tensor map descriptor** (`cuTensorMapEncodeTiled`) that
  points at a **rectangular, 16-B-aligned, contiguous tile** in global
  memory. Each descriptor is set up on the host once per tensor.
- FusenCache paged KV is indirected via `block_table`. Two consecutive KV
  positions in a sequence can land in two completely different pages in
  global memory. **The per-token source is not contiguous.**
- We would need one TMA descriptor per physical block (!), or do scattered
  fixups, which negates the win.
- Q is contiguous per (batch, head) — but Q is 512 B for one head, a single
  16-B cp.async × 32 threads already moves it. TMA is overkill.

### TMA helps when
- Prefill with dense BF16 KV (contiguous sequence).
- Store-side of paged KV when pages are large (page_size ≥ 64) and fully
  filled — one TMA per page.
- Stage-2 reduction scratch if it ever becomes a 2D tile with head-dim tiles
  larger than 16 B × threads.

### Decision
**Decode side stays on cp.async 16-B loads.** TMA is not a win given the
paged indirection. Document revisit conditions: (a) page_size ≥ 64 *and*
KV written in large contiguous runs, (b) future KV layout that stripes
heads contiguously across a full page.

## 4. Integration with fusencache Python wrapper

The new kernel reuses the exact same TORCH_LIBRARY signature as the
existing `fusencache.decode_attention` (see `build_fusencache.py` line 124).
It ships as a separate op:

```
torch.ops.fusencache_cpasync.decode_attention(
    output, query, kv_cache, scales, block_table, seq_lens,
    mid_out, sm_scale, logits_soft_cap, num_kv_splits,
    head_dim, num_kv_heads, kv_group_size, page_size,
    k_bits, v_bits, scale_block_k, scale_block_v,
    k_offset, v_offset)
```

Restrictions for the prototype:
- k_bits = v_bits = 4 (FusenCache k4v4b64 only)
- head_dim = 256, scale_block = 64
- kv_group_size = 2 (Gemma 4 GQA layout)
- page_size arbitrary (loop over page-table indirection same as current)
- batch up to 32, one head group per block; single-token decode

Stage 2 (split reduction) is unchanged — reuse the existing
`decode_stage2<256, 256>` from the baseline kernel when integrating;
prototype inlines a small variant for standalone benching.

## 5. Go / no-go gates

- Compile must succeed on first two attempts (else stop and log blocker).
- BW achieved < 50 % of 1792 GB/s peak → **KILL**, log lesson.
- BW ≥ 60 % → keep, promote to production integration.
- Correctness: non-NaN output at small shapes; full numerical verify is a
  follow-up (the Triton reference already exists in `kv_cache_gen`).

## 6. Measured result (2026-04-17)

Built on first attempt (sm_120a, nvcc 12.8, torch 2.11+cu130). Ran on GPU 1.
Target config B=16, seq_len=2048, Hq=16, Hk=8, D=256, splits=16,
page_size=16, soft_cap=50.0, warmup=20, iters=200.

| kernel                         | elapsed  | GB/s  | % of 1792 GB/s |
|--------------------------------|---------:|------:|---------------:|
| baseline C++ (`fusencache_decode_attention`) | 347.7 us | 205  | 11.4 % |
| cp.async 3-stage prototype     | 280.7 us | 254  | 14.2 % |

Speedup cp.async vs baseline: **1.24×**. No NaNs.
Sanity re-runs at B=32 seq=2048 → 1.29×, B=32 seq=4096 → 1.34×.

Bytes-read floor at peak for this config: 39.8 us. Even the faster prototype
is ~7× above the floor.

### Interpretation

Both kernels are well below the 50 % BW gate. cp.async lifts BW modestly
but the per-tile compute (4-bit unpack + FP32 FMA + warp-level QK reduction
+ online softmax) is **serialized against the pipeline drain** — every KV
token requires a block-wide score broadcast before V accumulation proceeds.
cp.async only hides memory latency when the consumer can keep going; here
it stalls on the reduction.

### Go/no-go

**KILL for this idea as a standalone lift.** Next move to exceed 50 % BW is
not more cp.async; it is:

1. Warp-specialized producer/consumer (2 warps issue cp.async, 2 warps
   consume) — removes the block-wide barrier around score broadcast.
2. Vectorized 4-bit dequant via `prmt.b32` / DP4A-style unpack to collapse
   HALF_D dot into ~16 cycles per KV token.
3. Per-head (not per-block) softmax reduction using `shfl.sync` only.

Until those land, do **not** promote this prototype to production. Keep the
file on disk for reference and rebuild after the warp-spec pass.
