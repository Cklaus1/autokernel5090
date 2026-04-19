# EAGLE3 Draft Training for Qwen3-30B-A3B (NVFP4)

Status: pipeline-prep only (2 h budget). Full multi-day training is out of scope;
goal is to stand up an end-to-end training loop and validate it converges in the
right direction on a tiny slice.

## 1. Target model

| field | value |
| --- | --- |
| name | Qwen/Qwen3-30B-A3B (local: /root/models/Qwen3-30B-A3B-NVFP4) |
| architecture | Qwen3MoeForCausalLM (MoE, 128 experts, top-8) |
| hidden_size | 2048 |
| num_hidden_layers | 48 |
| num_attention_heads | 32 (q) / 4 (kv) |
| vocab_size | 151,936 |
| dtype on disk | NVFP4 (modelopt), KV cache FP8 |

Because weights are NVFP4 (packed uint8), we cannot back-prop through them on
bare transformers. We use **vLLM offline** as the teacher forward-pass driver
(it already handles NVFP4 cleanly — same code path used by vllm-ad). The draft
head is trained in a separate PyTorch process against teacher-emitted
top-k logprobs.

## 2. Draft (EAGLE3) head architecture

Follows the vLLM reference (`/build/vllm/vllm/model_executor/models/llama_eagle3.py`):

```
input_ids --> embed_tokens (vocab x hidden)         \
                                                      } concat (first layer only)
target_hidden_state  -----------------------------  /
|
v
[optional] input_norm over 3*hidden (if use_aux_hidden_state)
|
v
fc: Linear(2*hidden -> hidden)   [or 3*hidden -> hidden for aux]
|
v
Qwen3DecoderLayer x N_draft    (self-attn + MoE-free MLP)
|
v
RMSNorm
|
v
lm_head: Linear(hidden -> draft_vocab)
```

Initial config (small-scale pass):

| field | value | rationale |
| --- | --- | --- |
| N_draft (decoder layers) | 1 | EAGLE3 single-layer head, fastest drafts |
| hidden_size | 2048 | match target for residual injection |
| intermediate_size | 6144 | match target MLP width (no MoE in draft) |
| num_attention_heads | 32 / 4 kv | same GQA |
| use_aux_hidden_state | False (initial) | skip 3-layer hidden fusion (adds load) |
| draft_vocab_size | 151,936 | no vocab truncation for v1 |
| params (approx) | 2048*151936*2 + 2*2048*2048 + 4*2048*6144 ~= 725M | dominated by embed+lm_head tied to vocab |

If we tie embed_tokens and lm_head (weight sharing), size drops to ~370M.
For ~100-200M parameter target, we also truncate draft_vocab to 32k most-frequent
tokens (SpecForge convention). Plan B: use truncated vocab (32,768) → params ~150M.

## 3. Training loss

Distillation cross-entropy against teacher top-k logprobs:

L = sum_{t, k in top_K} softmax(teacher_logits_t / T)[k] * log_softmax(draft_logits_t)[k]

K=20 (vLLM `prompt_logprobs=20`). Temperature T=1.0 (teacher already post-softmax).

Acceptance-rate proxy: fraction of positions where argmax(draft_logits) == argmax(teacher_logits).
This correlates with vLLM's EAGLE3 acceptance rate but omits the tree-draft
rejection sampling bookkeeping.

## 4. Dataset

Public, open-licensed only. Primary: **allenai/tulu-3-sft-personas-instruction-following**
(ODC-BY). Fallback: **HuggingFaceH4/ultrachat_200k** (MIT). Stream a shuffled
10,000-example slice via `datasets.load_dataset(..., streaming=True)` or
prompt-fetch from the HF datasets-server REST API (avoids needing the full
`datasets` library inside the vLLM container).

Per-example: take the first user turn, truncate to 512 prompt tokens, generate
128 continuation tokens from the teacher → yields ~640 supervised positions per
example × 10K examples = ~6.4M training positions.

## 5. Two-phase pipeline

### Phase A — teacher label generation (vLLM offline on GPU 1)
1. Load Qwen3-30B-A3B-NVFP4 via `vllm.LLM(... gpu_memory_utilization=0.6 ...)`.
2. For each prompt: `llm.generate(prompt, SamplingParams(max_tokens=128, temperature=0.0, logprobs=20, prompt_logprobs=20))`.
3. Stream results to `/tmp/eagle3_labels/shard_{i:05d}.pt`:
   - `input_ids: int32[T]`
   - `topk_ids: int32[T, 20]`
   - `topk_logprobs: float16[T, 20]`
4. Throughput target: >= 5 examples/s * 640 tokens = 3200 tokens/s. With NVFP4
   serving on RTX PRO 6000 this is easily met.

### Phase B — draft head training (pure torch on GPU 1)
1. Load shards lazily. Shuffle at shard granularity.
2. Build draft head (see §2) in BF16. AdamW, lr=3e-4, cosine schedule, warmup=200.
3. Batch size 8 (sequences) × 640 tokens = 5,120 positions/step. ~1,250 steps/epoch.
4. 1-hour target: ~3-4K steps at ~1 step/s.
5. Checkpoint every 15 min to `/tmp/eagle3_ckpt_{minutes}.pt`.
6. Log loss + top-1 match vs. teacher (acceptance proxy) every 50 steps.

Note: because we don't yet feed `target_hidden_state` (Phase A only emits
logprobs, not hidden states), the draft for v1 reduces to an auto-regressive LM
distilled from the teacher's output distribution. This is weaker than real
EAGLE3 (which conditions on the target's last-layer hidden). It still
validates the pipeline and training stability. Upgrading to true EAGLE3
requires patching vLLM to expose `hidden_states` — see §7.

## 6. Success criteria

- **Pipeline PASS**: 1 h of training completes without OOM / crash, loss
  trends downward, top-1 match climbs above random (1/vocab). This is the
  agent-window goal.
- **Continuation PASS** (future work): held-out top-1 acceptance >= 50% after 1 h
  → commit to a 1-day full-scale run.
- **Continuation ABANDON**: acceptance <= 30% → the no-hidden-state shortcut
  is insufficient; must invest in exposing teacher hidden states from vLLM.

## 7. Follow-ups for real EAGLE3 quality

1. Fork `vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM.forward` to
   also return `hidden_states[-1]`; wire through `vllm.LLM.encode_prompt` or
   a custom sampler hook. Save as a fourth column per token.
2. Switch draft-head first layer to the concat(embed, hidden) form (§2).
3. Enable `use_aux_hidden_state=True` once 3-layer hidden fusion is available.
4. Run 1-2 day training at scale: 1M examples, bs=64, lr=3e-4, cosine.

## 8. Hardware plan

- GPU 0: occupied (vllm-ad) — untouched.
- GPU 1: all training work. Peak memory:
  * Phase A: ~40-50 GB (vLLM weights+KV).
  * Phase B: ~3 GB (head + grads + optim + activations for bs=8, ctx=640).
  * They do not overlap in time.
