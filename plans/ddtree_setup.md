# DDTree Integration Plan

## What DDTree Does

DDTree (Diffusion Draft Tree) is a speculative decoding method from
"Accelerating Speculative Decoding with Block Diffusion Draft Trees."
It combines two ideas:

1. **Diffusion drafting (DFlash).** A small draft model based on Qwen3
   architecture generates candidate tokens via a denoising diffusion process
   rather than autoregressive sampling. The DFlash attention layer
   (`model/dflash.py`) concatenates context KV from the target model with
   noise KV from the draft model's own hidden states, applies RoPE, and runs
   standard flash/eager attention. This produces multiple candidate tokens in
   parallel per diffusion step.

2. **Tree-structured verification.** Instead of verifying a single draft
   sequence, DDTree builds a probability-weighted tree of candidate
   continuations and verifies the entire tree in one target-model forward pass.

### Tree Build (`ddtree.py:84-166`)

`build_ddtree_tree(draft_logits, budget)` constructs an optimal draft tree:

- Computes top-k log-probabilities per position from draft logits on GPU,
  then copies to CPU as NumPy arrays.
- Uses a Python `heapq` priority queue keyed by cumulative log-probability.
  Each heap entry represents a candidate node (token at a given depth and
  rank within its sibling set).
- Greedily pops the highest-probability node, records it in flat arrays
  (`node_token_ids`, `node_depths`, `parents`), builds `child_maps` for
  later traversal, and pushes two successors: the next sibling (same depth,
  rank+1) and the first child (depth+1, rank 0).
- Iterates until the node budget is exhausted.
- Builds a boolean visibility matrix (causal mask for the tree) so the
  target model can attend only along ancestor paths.

### Tree Verify (`ddtree.py:212-277`)

- `follow_verified_tree`: walks the tree using the target model's posterior
  logits. Starting from the root, it follows `child_maps` as long as the
  target model's sampled token matches a child, accumulating accepted indices.
  Returns accepted token indices plus the first rejected (bonus) token.
- `compact_dynamic_cache`: after verification, prunes the KV cache to keep
  only entries on the accepted path. Uses an inline C++ extension
  (`compact_tail_inplace`) for fast in-place compaction when available,
  falling back to `index_select` + `copy_`.

### DFlash Attention (`model/dflash.py:58-102`)

The `Qwen3DFlashAttention.forward` method:

- Projects hidden states to Q/K/V. Concatenates target-context K/V with
  draft-noise K/V along the sequence dimension.
- Applies QK-norm (RMSNorm per head) and RoPE.
- Updates the KV cache if provided.
- Dispatches to flash attention or eager attention based on config.
- The `is_causal=False` flag is critical: the tree mask is not a simple
  causal triangle, so a custom attention mask is passed explicitly.

## Port to vLLM Speculative Decode

### Hook point

vLLM's speculative decoding lives in `vllm/spec_decode/`. The integration
points are:

1. **Proposer interface.** Implement a `DDTreeProposer` that wraps the
   DFlash draft model. Must implement `get_spec_proposals()` returning
   candidate token IDs, probabilities, and the tree structure (parent array
   + visibility mask).

2. **Scorer / verifier.** vLLM's `Scorer` runs the target model on proposed
   tokens. DDTree needs to pass the tree attention mask into the target
   model's forward pass. This requires modifying the attention mask
   construction in the scorer to accept a non-causal (tree-shaped) mask
   instead of the default triangular mask.

3. **Acceptance / rejection.** Replace vLLM's default rejection sampler
   with `follow_verified_tree` logic. After scoring, walk the tree to find
   the longest accepted prefix, then call `compact_dynamic_cache` to prune
   rejected branches from the KV cache.

4. **KV cache compaction.** vLLM manages KV cache via block tables
   (PagedAttention). DDTree's `compact_dynamic_cache` assumes contiguous
   `DynamicCache` tensors. The port must either:
   - Run DDTree verification in a contiguous scratch buffer, then copy
     accepted KV back into paged blocks, or
   - Implement a paged-aware compaction that remaps block table entries
     for rejected branches.

### Sequence of work

1. Stand up DFlash draft model loading inside vLLM's model registry.
2. Implement `DDTreeProposer` conforming to vLLM's proposer ABC.
3. Modify scorer to inject tree visibility mask.
4. Implement tree-walk acceptance + KV compaction for paged KV.
5. Benchmark against baseline (no spec decode) on Gemma-4 26B.

## CPU heapq Bottleneck and GPU Top-k Fix

### The problem

`build_ddtree_tree` (line 116-149) runs entirely on CPU. The heapq loop
does `budget` iterations of `heappop` + up to 2 `heappush` calls, each
O(log N). For budget=256 this is fine (~50 us). For budget >= 1024, the
Python overhead dominates: heap operations are pure Python, per-element
NumPy scalar extraction (`float(top_log_probs_np[d, r])`) has overhead,
and the GIL serializes everything.

Profiling from `rtx_pro6000_experiments.md` T3-I confirms this is the
single-thread bottleneck at large budgets.

### The fix: GPU batched top-k (T3-J)

Replace the CPU heapq expansion with a Triton kernel:

- **Input:** `draft_logits` tensor (shape `[depth, vocab]`), already on GPU.
- **Output:** flat arrays of (token_id, depth, parent_index, cumulative
  log-prob) for the top-`budget` tree nodes, plus the visibility matrix.
- **Algorithm:** batched top-k using warp-level shuffle reductions.
  Persistent grid of `NUM_SMS` blocks. Early-terminate partial sort at
  depth `log2(K)` -- no need to fully sort.
- **Correctness reference:** `gpu_bitonic_sort/bitonic.cu:605-861`
  (recursive iterative evaluator). Use only for partial-sort semantics
  validation, not as an implementation template.
- **Reusable:** the same batched top-k primitive applies to Eagle3, Medusa,
  and any other tree-structured speculative decoding scheme.

## Kill Criterion

From autokernel's empirical data (T3-I):

> If effective draft cost ratio **c > 0.6**, abandon the approach.

The cost ratio c = (draft time + tree build time + verify overhead) /
(equivalent autoregressive time for accepted tokens). At c > 0.6, the
speculative decoding overhead eats more than 60% of the savings from
parallel verification, making net speedup negligible or negative.

Previous work on pruned-layer Gemma-4 drafts hit c ~ 0.83 (net slowdown).
DDTree's diffusion drafting should have a different acceptance profile, but
the 0.6 threshold remains the go/no-go gate.

## SM120 (Blackwell / RTX PRO 6000) Constraints

- **No TMEM / cluster features.** SM120 does not support the TMEM-based
  or cluster-level suggestions from the exploration agent's bitonic sort
  analysis. Do not copy those; they will not compile.
- **Warp shuffle is safe.** `__shfl_xor_sync` and `__shfl_down_sync` work
  on SM120. The Triton `tl.where` + `tl.sort` path also works.
- **Shared memory:** 228 KB per SM on SM120. The top-k kernel should stay
  within this. At budget=2048 with float32 keys + int32 values, the working
  set is 2048 * 8 = 16 KB per block -- well within limits.
- **Register pressure:** SM120 has 64K registers per SM. The persistent
  grid pattern (one block per SM) avoids register spill as long as the
  inner loop state fits in ~64 registers per thread.
- **FP16/BF16 compute:** draft logit top-k comparison can stay in FP32
  (promoted from FP16 draft output) since it is not compute-bound. The
  logsumexp normalization in tree build should use FP32 to avoid numerical
  issues with large logits.
