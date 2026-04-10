# Semantic KV Cache Eviction for FusenCache + Gemma 4

## Problem Statement

FusenCache at k4v4b64 gives ~166K token KV capacity on an RTX 5090 32GB.
For very long context workloads (32K+ tokens per sequence, multi-turn
conversations, RAG with large documents), even 166K slots can fill up —
especially with multiple concurrent sequences.

vLLM's current eviction is coarse: it preempts entire sequences, either
swapping them to CPU or discarding and recomputing. This wastes the full
KV cache of every sequence in the batch when any one sequence overflows.

**Better approach:** fine-grained per-token eviction within a sequence.
Keep the KV entries that matter most for future predictions; discard the
rest. This extends effective context without CPU swap overhead.

---

## Gemma 4 Architecture: Where Eviction Matters

Gemma 4 31B has 46 attention layers of two types:

| Layer Type      | Count | Window    | KV Scope     | Eviction Need |
|-----------------|-------|-----------|--------------|---------------|
| Sliding-window  | 25    | 1024 tok  | Local only   | **None** — already bounded |
| Global attention | 5    | Unlimited | Full context | **High** — unbounded growth |

The 5 global layers (at roughly 1/5th the total depth) are the only ones
that accumulate KV linearly with sequence length. The 25 sliding-window
layers self-limit to 1024 tokens of KV regardless of context length.

**Therefore: eviction applies only to the 5 global attention layers.**

With FusenCache k4v4b64 compression, each token's KV in a global layer is:
- Key: 4 bits/dim × 256 dims / 8 = 128 bytes per KV head
- Value: 4 bits/dim × 256 dims / 8 = 128 bytes per KV head
- For 16 KV heads: 4096 bytes = 4 KB per token per layer
- Across 5 global layers: 20 KB per token total

At 166K capacity: up to 8,300 tokens per sequence (if a single sequence
monopolizes all global-layer slots). With proper sharing across 5 global
layers, effective limit is ~33K tokens of global-attention KV.

---

## Prior Art: What We Borrow

### H2O (Heavy Hitter Oracle)
**Principle:** Track cumulative attention mass per KV position. Evict
lowest-mass entries. Heavy hitters (tokens that receive high attention
from many future queries) are retained.

**What works:** Cumulative softmax attention score is an excellent
importance proxy. Tokens that are consistently attended to are causally
important.

**What doesn't work for us:** H2O accumulates scores in FP32 across the
full sequence length, adding memory overhead proportional to seq_len.
Also assumes dense attention (not paged/blocked KV).

### SnapKV
**Principle:** Score KV entries by attention from the most recent
"observation window" (last K queries), not the full history.

**What works:** Recent queries are the best predictors of what the next
query will need. Old queries may have very different information needs.

**What works for us:** We already compute attention at every decode step.
The attention pattern from the last W decode steps is a natural importance
window that requires no extra storage — just accumulate the softmax outputs.

### ScissorHands
**Principle:** Tokens with similar attention patterns to recent tokens can
be pruned — only one representative of each "pattern cluster" is needed.

**What works:** Reduces redundancy in long contexts.

**Skip for now:** Clustering adds O(seq_len) compute per eviction check.
Complex to implement in paged KV. The gain over H2O is marginal in practice.

### StreamingLLM (Attention Sinks)
**Principle:** The first few tokens (BOS, system prompt start) receive
disproportionately high attention regardless of content. They act as
"attention sinks" — mathematical artifacts of softmax over long sequences.
These must never be evicted.

**What works:** Empirically validated across many model families.
**Our policy:** Always pin the first 16 tokens (configurable).

---

## Proposed Policy: FusenEvict

### Core Algorithm

**Score:** Each token `t` in the global-attention KV cache has an
importance score `s[t]` that accumulates attention weight it receives
from all subsequent decode steps:

```
s[t] += softmax(Q_current @ K[t] / sqrt(D)) for each decode step
```

**Eviction trigger:** When a global-layer KV cache reaches capacity
(configurable high-water mark, e.g. 90% full), evict the lowest-scoring
`evict_budget` tokens from each sequence above the mark.

**Protected set (never evict):**
1. First 16 tokens — attention sinks
2. Last 512 tokens — recency window (still building context)
3. Any token explicitly tagged as "anchor" (system prompt boundaries)

**Eviction budget:** Evict 10% of sequence length per trigger (amortizes
eviction cost across many decode steps).

### Score Accumulation in FusenCache

FusenCache already runs the full attention forward pass each decode step.
The softmax output `attn_weights` of shape `[B, Hq, seq_len]` is computed
but currently discarded. We tap into this to accumulate importance.

**Storage overhead:** One FP16 importance score per token per layer = 2
bytes/token for each of the 5 global layers. For a 32K sequence: 32K × 5
× 2 = 320 KB per sequence. Negligible vs. the KV cache itself.

Because scores are per-request (not shared across batch), they live in a
per-request side buffer attached to the sequence metadata.

### Score Normalization

Raw cumulative sums grow with sequence length. Normalize before eviction:

```
s_norm[t] = s[t] / (num_decode_steps - t)  # avg attention per step
```

This prevents recency bias: tokens seen by fewer steps don't get
artificially low cumulative scores just because they were written recently.

---

## Implementation Plan

### Phase 1: Score Tracking (no eviction, observation only)

**Goal:** Validate that cumulative attention scores correctly identify
important tokens before implementing eviction logic.

**Files to modify:**
- `fusen_kv/backend.py` — add `_attn_scores` side buffer per layer
- `fusen_kv/backend.py` — hook into `forward()` to accumulate scores after
  the decode attention call

**New metadata in FusenKVMetadata:**
```python
# Per-request importance scores for global layers: list[Tensor[seq_len]]
# None for sliding-window layers (never evict)
attn_score_buf: dict[int, torch.Tensor] | None = None  # req_id -> scores
```

**In FusenKVImpl.forward(), after decode:**
```python
if self.sliding_window is None and self._track_scores:
    # attn_weights: [B, Hq, S] — already computed inside decode kernel
    # Average across Q heads: [B, S]
    head_mean = attn_weights.mean(dim=1)
    # Accumulate per-request
    for b in range(B):
        seq_id = metadata.seq_ids[b]
        scores = self._get_or_create_score_buf(seq_id, seq_lens[b])
        scores[:seq_lens[b]] += head_mean[b, :seq_lens[b]]
```

**Obstacle:** The C++ decode kernel and Triton kernel don't currently
return `attn_weights` — they output only the final context vector. Two
options:
- Option A: Modify kernel to return attention weights (expensive — adds
  O(B × Hq × S) storage and kernel changes)
- Option B: Run a lightweight PyTorch attention only over the global-layer
  KV to compute scores (acceptable: global layers are 5/46 of total)
- Option C: Use a separate lightweight scoring pass with quantized KV that
  only computes attention mass, no context vector (most efficient)

**Recommendation: Option B** for Phase 1 (simplicity, correctness).
Option C for Phase 3 (optimization).

### Phase 2: Eviction Engine

**Goal:** Implement the actual eviction — removing low-score tokens from
the paged KV cache.

**Key challenge: paged KV cache fragmentation.**

vLLM's paged KV uses fixed-size blocks (16 tokens each). You cannot
delete individual tokens from the middle of a block without defragmenting.

**Two sub-strategies:**

**Strategy A: Block-aligned eviction (simple, ships first)**
- Evict entire 16-token blocks, not individual tokens
- Score of a block = average importance of its tokens
- When evicting, mark the block as free in the block table
- The sequence's block table now has "holes" — positions beyond the first
  gap are renumbered

This requires teaching the attention kernel about non-contiguous sequences:
instead of `seq_len` being a single integer, pass a list of valid block
indices. The decode kernel already uses a block table — this is a natural
extension.

**Implementation in vLLM v1:** `FusenKVMetadata.block_table` already maps
logical positions to physical blocks. An evicted block simply does not
appear in the block table. `seq_lens` decreases by `block_size` per
evicted block.

**Strategy B: Token-level virtual eviction (zero fragmentation)**
- Keep the physical block occupied
- Mark evicted tokens with a sentinel score = -inf in the scores buffer
- During attention, mask out sentinel positions in the attention logits
- This avoids all defragmentation but wastes physical slots for "ghost" tokens

**Recommendation: Strategy A** — it reclaims physical memory (the actual
goal) and the block table already supports non-contiguous mapping.

**New module: `fusen_kv/eviction.py`**

```python
class SemanticEvictionPolicy:
    def __init__(
        self,
        sink_tokens: int = 16,       # first N tokens always kept
        recency_window: int = 512,   # last M tokens always kept
        evict_fraction: float = 0.10, # evict 10% of seq when triggered
        hwm_fraction: float = 0.90,  # trigger at 90% capacity
        block_size: int = 16,
    ):
        ...

    def should_evict(self, seq_len: int, capacity: int) -> bool:
        return seq_len >= capacity * self.hwm_fraction

    def select_eviction_blocks(
        self,
        scores: torch.Tensor,    # [seq_len] FP16 importance scores
        seq_len: int,
        block_table: torch.Tensor,  # [num_blocks] physical block IDs
        block_size: int,
    ) -> list[int]:
        """Returns physical block IDs to evict."""
        # Compute block-level scores: mean over block tokens
        num_blocks = (seq_len + block_size - 1) // block_size
        block_scores = torch.zeros(num_blocks)
        for b in range(num_blocks):
            start = b * block_size
            end = min(start + block_size, seq_len)
            block_scores[b] = scores[start:end].mean()

        # Protect sink and recency blocks
        protected = set()
        sink_blocks = (self.sink_tokens + block_size - 1) // block_size
        for b in range(min(sink_blocks, num_blocks)):
            protected.add(b)
        recency_blocks = (self.recency_window + block_size - 1) // block_size
        for b in range(max(0, num_blocks - recency_blocks), num_blocks):
            protected.add(b)

        # Rank remaining blocks by score
        evict_count = max(1, int(num_blocks * self.evict_fraction))
        candidates = [(s, b) for b, s in enumerate(block_scores)
                      if b not in protected]
        candidates.sort()  # ascending score, worst first

        evict_logical = [b for _, b in candidates[:evict_count]]
        return [block_table[b].item() for b in evict_logical], evict_logical
```

**vLLM integration point:** The eviction policy runs in the scheduler,
not in the attention kernel. vLLM v1's scheduler calls into the block
manager to free blocks. We hook into the scheduler's `_preempt()` path
and replace it with our selective eviction for sequences that are still
running (just too long).

Specifically: intercept in `vllm/v1/core/scheduler.py` at the point where
it would otherwise call `self.block_manager.preempt_sequence()`, and
instead call `SemanticEvictionPolicy.evict(seq)` if the sequence is a
long-context candidate.

### Phase 3: Score Kernel (optional optimization)

Replace the Python loop score accumulation with a Triton kernel that
computes per-token attention mass in one fused pass:

```
Input:  Q [B, Hq, D], K [S, Hk, D] (dequantized from paged cache)
Output: attn_mass [B, S]  — sum of softmax weights across all Q heads
```

This kernel can be batched across the 5 global layers. Expected throughput:
~100 GFLOPS (small vs. the context vector kernel) so Python overhead
dominates anyway at decode time.

---

## Score Accumulation: Decay vs. No Decay

**Pure cumulative sum:** Simple. Older tokens naturally accumulate more
mass over time (they've been around for more decode steps). This creates
a mild recency bias in the wrong direction — old tokens appear more
important just because they've been "scored" more times.

**Fix:** Normalize by (number of decode steps since token was written):
```
importance[t] = sum_of_attention[t] / max(1, current_step - t)
```
This converts cumulative sum into average attention per step, making
scores comparable across tokens of different ages.

**Exponential decay:** An alternative that weights recent attention more:
```
score[t] = alpha * score[t] + (1 - alpha) * attn_weight[t]
           where alpha = 0.95 (configurable)
```
This is the H2O approach in practice. It naturally discounts stale scores
without normalization.

**Recommendation:** Start with exponential decay (alpha=0.95). It's
numerically stable, never overflows, and matches what H2O/SnapKV found
works best empirically. Switch to window-average if quality degrades.

---

## Eviction Correctness Concerns

### Attention Score Gaps
After eviction, the remaining tokens still sum-normalize via softmax.
The evicted tokens' attention weight is redistributed to surviving tokens.
This is mathematically incorrect (the model never saw this redistribution
during training) but empirically acceptable — same argument as KV
quantization, which also introduces error.

**Mitigation:** Evict conservatively (10% budget) and only low-score tokens.
The redistribution error is proportional to evicted mass, which should be
small.

### Position IDs After Eviction
Gemma 4 uses RoPE (rotary position embeddings). RoPE is applied to the
query and key at write time (keys stored post-RoPE in cache). After
eviction, position IDs of surviving tokens do not change — they retain
their original absolute positions. This is correct: the model can still
attend to position 5000 even if positions 500-1000 were evicted. RoPE
encodes absolute position, so gaps are fine.

### First-Token Bias (Attention Sinks)
Tokens 0-15 must never be evicted. Softmax over long sequences tends to
concentrate probability mass on the first few tokens as a "sink" (the
model has to put probability somewhere when no token is clearly important).
Evicting sinks causes catastrophic quality degradation (validated by
StreamingLLM paper). Our 16-token sink protection addresses this.

### Quality Regression Risk
Per H2O ablation: keeping top 20% of KV entries preserves >95% of
generation quality on most benchmarks. At 10% eviction per trigger, we
keep 90% of entries initially, degrading slowly. Quality should remain
acceptable for sequences up to 4× the protected budget.

**Validation plan:** After implementing, run perplexity evaluation using
`fusen_kv/eval_perplexity.py` on long-context benchmarks (if available)
or generate a suite of 32K-token prompts and measure output coherence.

---

## Gemma 4 Sliding Window Interaction

The 25 sliding-window layers already compute attention only over the last
1024 tokens. Their KV cache is bounded regardless of sequence length —
vLLM uses a circular buffer approach for sliding windows, automatically
overwriting old positions.

Semantic eviction is **not applied** to sliding-window layers. They never
overflow in the semantic sense: new KV slots replace old ones in a ring
buffer. Only global layers accumulate linearly.

This simplifies implementation: check `self.sliding_window is None` to
decide whether to score/evict.

---

## Configuration Interface

Expose via environment variables (consistent with existing FusenCache
conventions):

```bash
FUSEN_EVICT=1                  # Enable semantic eviction (default: 0)
FUSEN_EVICT_SINK=16            # Sink tokens to always preserve (default: 16)
FUSEN_EVICT_RECENCY=512        # Recency window tokens to preserve (default: 512)
FUSEN_EVICT_FRAC=0.10          # Fraction of seq to evict per trigger (default: 0.10)
FUSEN_EVICT_HWM=0.90           # High-water mark fraction (default: 0.90)
FUSEN_EVICT_DECAY=0.95         # EMA decay for importance scores (default: 0.95)
```

These map to `SemanticEvictionPolicy.__init__` parameters.

---

## Data Structures Summary

### Per-sequence, per-global-layer state

```python
class SeqEvictionState:
    scores: torch.Tensor        # [max_seq_len] FP16 importance scores
    step_count: int             # decode steps since this seq started
    # For EMA decay: score = decay * score + (1-decay) * attn_weight
```

### Integration with FusenKVImpl

```python
class FusenKVImpl(AttentionImplBase):
    ...
    # New fields:
    _eviction_policy: SemanticEvictionPolicy | None
    _seq_eviction_states: dict[int, SeqEvictionState]  # seq_id -> state
```

### Integration with vLLM Scheduler

The scheduler needs to:
1. Pass seq_id to FusenKVImpl so it can look up eviction state
2. Receive eviction decisions back (which physical blocks to free)
3. Call `block_manager.free_blocks(seq_id, evict_block_ids)`

This is the most invasive integration point. An alternative is to
implement eviction as a background thread that asynchronously frees blocks
between decode steps, avoiding changes to the scheduler hot path.

---

## Phased Delivery

| Phase | Deliverable | Complexity | Risk |
|-------|-------------|------------|------|
| 1 | Score tracking + logging (no eviction) | Low | Low |
| 2a | Block-aligned eviction, Python scoring | Medium | Medium |
| 2b | vLLM scheduler integration | Medium | Medium |
| 3 | Triton score kernel | High | Low |
| 4 | Per-head importance (separate scores per KV head) | Medium | Medium |
| 5 | Adaptive decay tuning based on perplexity feedback | High | High |

**Start with Phase 1:** Build confidence that scores correctly rank token
importance before touching eviction logic. Visualize score distributions
on real Gemma 4 inference traces to validate.

---

## Expected Impact

**Capacity:** At 10% eviction and 90% HWM, a sequence can grow to ~1.65M
effective token positions before the retained KV exceeds 166K slots.
(Practical limit lower due to recency window and sinks.)

**Throughput:** Eviction runs at most once per 10% of seq_len growth. For
a 32K sequence, that's every 3,200 decode steps — amortized overhead is
negligible.

**Memory:** Score buffers add ~2 bytes × 5 global layers × max_seq_len =
10 bytes/token. For 166K tokens: 1.6 MB. Negligible.

**Quality:** Depends on how well importance scores predict future attention.
H2O validates >95% quality retention at 20% KV budget on most benchmarks.
With our more conservative 90% retention, expect <1% quality degradation
for sequences under 3× the nominal capacity.

---

## Files to Create / Modify

| File | Action | Description |
|------|--------|-------------|
| `fusen_kv/eviction.py` | Create | `SemanticEvictionPolicy`, `SeqEvictionState` |
| `fusen_kv/backend.py` | Modify | Add score tracking hooks in `forward()`, add `_eviction_policy` field |
| `fusen_kv/score_kernel.py` | Create (Phase 3) | Triton kernel for fast attention mass computation |
| `fusen_kv/tests/test_eviction.py` | Create | Unit tests for policy logic and score accumulation |
| `fusen_kv/eval_eviction.py` | Create | End-to-end quality eval: perplexity vs. eviction rate curve |
