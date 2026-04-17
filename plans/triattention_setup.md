# TriAttention Setup Plan (T1-F)

Source: `plans/rtx_pro6000_experiments.md` lines 278-287.
Repo: `/home/cklaus/projects/aigpu/triattention/`

---

## What TriAttention Does

TriAttention compresses the KV cache using frequency-domain token scoring derived
from pre-RoPE Q/K statistics. The key insight (paper: arXiv 2604.04921) is that
pre-RoPE Q/K vectors in long-reasoning models concentrate around fixed per-head
centers; each head's distance preferences can be expressed as a trigonometric
series over RoPE frequencies. At runtime the method:

1. **Inverts RoPE** on incoming keys to recover pre-RoPE representations.
2. **Scores every cached token** per head using per-head frequency centers, norms,
   and geometric position offsets precomputed from calibration stats
   (`build_geometric_offsets`, `score_keys_for_round` in `pruning_utils.py`).
3. **Evicts low-scoring tokens** so the KV cache stays at the configured budget.
   Compression fires every `TRIATTN_RUNTIME_DIVIDE_LENGTH` new tokens (default 128)
   once the cache hits the budget.
4. **Preserves a window** of the most recent tokens (`TRIATTN_RUNTIME_WINDOW_SIZE`,
   default 128) unconditionally.

Pruning granularity is controlled by `TRIATTN_RUNTIME_PRUNING_MODE`:
- `per_head` — each KV head selects its own token subset independently.
- `per_layer_per_head` — independent selection per (layer, KV head) pair.

Published results on AIME25 with Qwen3-8B: **10.7x KV memory reduction**,
**2.5x throughput**, identical accuracy (40.8 vs 40.8 full attention).

The vLLM plugin (`triattention/vllm/runtime/`) monkeypatches the scheduler and
worker transparently; no model-code changes are needed.

---

## Setup Steps

### 1. Install TriAttention

```bash
cd /home/cklaus/projects/aigpu/triattention
pip install -e .
pip install flash-attn --no-build-isolation   # strongly recommended
```

The vLLM plugin is auto-discovered after installation; no explicit registration.

### 2. Calibrate Gemma-4 Frequency Stats (one-time, ~10-20 min)

Gemma-4 is not among the shipped models (Qwen3, DeepSeek-R1-Distill variants).
Stats must be generated before any serving or benchmarking.

```bash
# Obtain a calibration corpus (~32 k tokens of coherent text: book chapter,
# Wikipedia dump, or source code file — domain does not matter)
python scripts/calibrate.py \
    --model google/gemma-4-26b-it \
    --input /path/to/calibration_text.txt \
    --output triattention/calibration/gemma4_26b.pt
```

What the script does: single forward pass over the corpus, captures query states
from every attention layer, inverts RoPE, computes per-head frequency centers and
norms, writes a `.pt` stats file.

Tips:
- Provide text that approaches `--max-length` (default 32768 tokens) for full
  layer/head coverage.
- Avoid garbled or highly repetitive text.
- Calibration is domain-agnostic; math-domain inference after code-domain
  calibration works fine per the docs.

Save the output path; it becomes `TRIATTN_RUNTIME_SPARSE_STATS_PATH` at serve time.

### 3. Serve Gemma-4 26B with TriAttention on vLLM

```bash
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=triattention/calibration/gemma4_26b.pt
export TRIATTN_RUNTIME_KV_BUDGET=3072        # tune per benchmark (see §Benchmark Plan)
export TRIATTN_RUNTIME_PRUNING_MODE=per_head

vllm serve google/gemma-4-26b-it \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code \
    --enable-prefix-caching false \
    --max-num-batched-tokens 1024
```

Key flags:
- `--enable-prefix-caching false` — prefix caching is incompatible with KV
  compression (incorrect cache hits on evicted entries).
- `--max-num-batched-tokens 1024` — prevents a single large prefill from
  overshooting the budget before compression triggers.
- `--enforce-eager` — required for the monkeypatch hooks to fire correctly.

---

## Calibration Notes for Gemma-4

Gemma-4 uses a variant of RoPE. The `TriAttention.__init__` path auto-detects
`rope_scaling` / `rope_type` from `AutoConfig` and stores `rope_style` for use in
`invert_rope`. After calibration, validate the stats file loads cleanly:

```python
from triattention.methods.pruning_utils import load_head_frequency_stats
meta, stats = load_head_frequency_stats("triattention/calibration/gemma4_26b.pt", "cpu")
print(meta)          # should show rope_style, sampled_heads, head_dim
print(len(stats))    # should equal num_layers * num_kv_heads (or sampled subset)
```

If `validate_stats_metadata` raises a mismatch on `rope_type`, pass an explicit
override via `TriAttentionConfig.metadata_expectations` (Python API) or check
whether the vLLM plugin exposes an env-var override.

---

## Benchmark Plan

Run three phases against a **FusenCache-only baseline** at identical batch size.

### Phase 1 — Budget sweep (accuracy vs. memory)

Contexts: 16K and 32K tokens. Datasets: AIME25, MATH-500.

| KV Budget | Expected KV reduction | Goal |
|-----------|----------------------|------|
| 4096 | ~8x | High accuracy, moderate compression |
| 3072 | ~10.7x | Published AIME25 operating point |
| 2048 | ~14x | Aggressive; check accuracy delta |

Metric per run: tok/s (decode throughput), peak KV VRAM (MB), AIME/MATH accuracy.

```bash
# Example — AIME25, budget 3072
python scripts/cli.py run-one \
    --model google/gemma-4-26b-it \
    --dataset aime25 \
    --method triattention \
    --budget 3072
```

### Phase 2 — Throughput at long context (primary metric)

Fix budget at the accuracy-preserving point from Phase 1.
Sweep input lengths: 16K, 24K, 32K.
Record tok/s and KV footprint at each length vs. FusenCache baseline.

### Phase 3 — Combined TriAttention + FusenCache

If FusenCache is already deployed, stack TriAttention on top and measure
cumulative KV reduction and throughput gain.

---

## Kill Criterion

Abandon TriAttention (fall back to FusenCache-only) if **any** of the following:

1. **Accuracy regression > 5%** on AIME25 or MATH-500 at the budget needed to
   achieve meaningful throughput gain (the plans/rtx_pro6000_experiments.md
   threshold).
2. **Throughput within 5% of FusenCache-only** at equal batch size and equal PPL —
   FusenCache is simpler to operate and has no calibration overhead.
3. **Calibration failure**: `validate_stats_metadata` cannot reconcile Gemma-4's
   RoPE config with the stats file and no workaround is found within 2 attempts.
4. **OOM during calibration or serving** that cannot be resolved by reducing
   `--max-num-batched-tokens` or `TRIATTN_RUNTIME_KV_BUDGET`.

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Core algorithm | `/home/cklaus/projects/aigpu/triattention/triattention/methods/triattention.py` |
| Scoring / RoPE inversion | `/home/cklaus/projects/aigpu/triattention/triattention/methods/pruning_utils.py` |
| Triton scoring kernel | `/home/cklaus/projects/aigpu/triattention/triattention/vllm/core/kernels/triton_scoring.py` |
| vLLM monkeypatch | `/home/cklaus/projects/aigpu/triattention/triattention/vllm/runtime/` |
| Calibration script | `/home/cklaus/projects/aigpu/triattention/scripts/calibrate.py` |
| Shipped stats (Qwen3/DS) | `/home/cklaus/projects/aigpu/triattention/triattention/calibration/` |
| vLLM runtime stats | `/home/cklaus/projects/aigpu/triattention/triattention/vllm/stats/` |
| Calibration guide | `/home/cklaus/projects/aigpu/triattention/docs/calibration.md` |
