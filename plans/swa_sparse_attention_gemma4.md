# Sliding-Window-Attention Sparse Decode Kernel for Gemma4 26B

## Problem

Gemma4 26B has 30 decoder layers:
- 5 global layers (indices 5,11,17,23,29) — head_dim=256, full attention.
- 25 sliding layers — head_dim=128, window=4096 tokens.

FlashInfer's sliding-window decode path (`BatchDecodeWithPagedKVCacheWrapper.plan(..., window_left=4095)`)
still walks the full seq_len range of K/V pages on every decode step and applies the window
via a masked softmax. At seq_len >> window that wastes both HBM traffic and compute. For a
generated sequence of 16K tokens on a 25-of-30-layer sliding block, an efficient sparse path
should save roughly `1 - window/seq_len` of attention FLOPs and HBM reads.

## Goal

Write a decode-only attention kernel (M=1 per batch) that touches ONLY the most-recent
`window` K/V tokens. Page-scheduled so programs do not launch against the skipped range.
Plug-in replacement for FlashInfer's sliding-mask decode at the Gemma4 sliding-layer shape.

## Design decisions

### Language: Triton.

- Triton hits ~80% of cuBLAS/FlashInfer decode bandwidth in practice on SM120 for this
  shape class (see `kernels/fp8_decode_attention.py`, `kernels/csrc/fusencache_decode_attention.cu`).
- The bottleneck here is HBM bandwidth (decode M=1; no tensor-core advantage from CUDA).
- Iteration cost matters: 4-hour budget rules out a CUDA prototype + tuning cycle.
- Port the structure from the existing `fp8_decode_attention` kernel (split-K paged decode).

### KV layout: BF16 paged, `page_size=16` (FlashInfer default for standard KV).

- Matches what FlashInfer dense+mask uses — apples-to-apples bench.
- FusenCache uses `k4v4b64` (page_size=64 fused K/V FP4) which is out of scope this session.
- FP8 KV is Discovery #37's open ceiling — fold in after BF16 correctness is proven.

### Sparsity strategy: page-level truncation of the block table, NOT per-element mask.

Given `seq_len[b]` and `window=4096`:

1. `start_tok = max(0, seq_len[b] - window)`
2. `start_page = start_tok // page_size`
3. Loop `for page_idx in range(start_page, num_pages):`
4. Inside the first page (`page_idx == start_page`), apply a per-element left mask
   `abs_pos >= seq_len - window`. All subsequent pages need only the normal
   `abs_pos < seq_len` tail mask.

This skips `start_page` page loads entirely — that is where the speedup comes from.
Every iteration issues one page-table lookup and PAGE_SIZE*head_dim loads.

### Grid: split-KV decode, copied from `fp8_decode_attention`.

- Grid = (batch, num_head_blocks, num_kv_splits).
- BLOCK_H = 4 Q heads per program (GQA group is 4: 8 Q heads / 2 KV heads).
- Split-K pages are now computed over `window` pages, not full `seq_len`, so splits
  already lineup with the sparse range.
- Stage 2 merges splits (standard online softmax merge with LSE).

### Correctness target

Compare against FlashInfer `BatchDecodeWithPagedKVCacheWrapper` with
`window_left=window-1` on identical paged KV. Accept cos > 0.999, max abs < 5e-3
in BF16 (actual expected cos ~0.99999 because both compute the same attention — the
only difference is whether the skipped range is visited or masked out to -inf).

### Microbench shape

Gemma4 sliding layer config:
- B = 16 concurrent sequences
- head_dim = 128
- num_kv_heads = 2
- num_q_heads = 8 (GQA group = 4)
- window = 4096

Sweep seq_len ∈ {4096 (no sparsity win), 8192 (2x theoretical), 16384 (4x theoretical)}.

## Expected ratios

At seq_len=8192, window=4096:
- FLOPs saved: seq_len/window = 2x.
- HBM traffic saved on K/V: ~2x (we still load Q once, the output once, and
  4096/8192 = 50% of K/V pages).
- Overhead: one branch per sliding layer per decode; negligible.

Kill gate: <1.3x speedup at this shape and window fraction → mask path was already good
enough → drop this kernel from integration.

## Integration path (post-session)

- vLLM Gemma4 attention backend (`vllm/v1/attention/backends/flashinfer.py`
  or equivalent) currently forwards sliding layers to FlashInfer's sliding decode.
  Swap that call for this Triton kernel on sliding layers only (keep global layers on
  FlashInfer's 256-dim path).
- Gate via env var `AUTOKERNEL_SWA_SPARSE=1` so fallback remains one flag away.
