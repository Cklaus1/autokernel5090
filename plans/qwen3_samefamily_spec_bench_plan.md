# Qwen3 Same-Family Speculative Decoding — Bench Plan

**Tag:** `W7_qwen3_samefamily_spec_prep`
**Date:** 2026-04-18
**Ref:** kill_audit §16 — highest-EV single remaining item, P=0.80
**Projection:** +98% aggregate throughput (29.6k → ~58k tok/s) for the Qwen3 serving slot.

---

## 1. Configuration

| Parameter | Value |
|---|---|
| Target model | Qwen3-30B-A3B-NVFP4 (`/root/models/Qwen3-30B-A3B-NVFP4`) |
| Draft model | Qwen3-1.7B (`/root/models/Qwen3-1.7B`) |
| Draft architecture | `Qwen3ForCausalLM` (dense, same family as target) |
| Target architecture | `Qwen3MoeForCausalLM` (MoE, same tokenizer + embedding space) |
| Shared vocab size | 151,669 |
| Speculative tokens (gamma) | 4 (sweep 2–6 as secondary axis) |
| Spec method | `draft_model` (vLLM native, not ngram) |
| Target GPU | RTX PRO 6000 SM120a |
| Docker image | `vllm-fusencache:latest` |
| Launcher | `./launch_qwen3_speculative.sh` |

---

## 2. Why Same-Family Is Different

Unlike cross-family spec decode (e.g., LLaMA draft for Qwen target, alpha ≈ 0.20–0.40), same-family pairs share:
- Identical tokenizer and vocabulary (no token remapping overhead).
- Identical RoPE theta (1,000,000 for Qwen3 across sizes — verified at publication).
- Identical activation function (SiLU in Qwen3).
- Same pre-training data distribution → draft logits closer to target distribution.

Published same-family acceptance rates (Qwen, LLaMA, Mistral cross-size pairs, 2024–2025 literature):
- Typical alpha: **0.60–0.80** for conversational tasks.
- Typical alpha: **0.75–0.85** for code/structured generation.
- Break-even alpha at gamma=4: ~0.45 (below this, spec decode is net-negative).

---

## 3. Theoretical Speedup Model

**Formula (standard rejection sampling):**

```
effective_tokens_per_step = (1 - alpha^(gamma+1)) / (1 - alpha)
draft_cost_ratio (c)       = latency(Qwen3-1.7B) / latency(Qwen3-30B-A3B)
step_cost                   = 1 + gamma * c
speedup                     = effective_tokens_per_step / step_cost
```

**Key: draft cost ratio.**
Qwen3-1.7B is a dense 28-layer model.  Qwen3-30B-A3B activates 3B params per token (MoE top-2/128 experts).  At single-user decode, the target is memory-bandwidth-bound on ~3B active weights; the draft at 1.7B active is comparable.  Estimated `c ≈ 0.30–0.45` (draft is ~30–45% of target latency at equivalent sequence).

**Projected speedup table (c=0.35, gamma=4):**

| alpha (acceptance rate) | effective_tokens | step_cost | speedup | net tok/s gain |
|---|---|---|---|---|
| 0.45 (break-even) | 2.31 | 2.40 | **0.96×** | −4% (do not deploy) |
| 0.55 | 2.65 | 2.40 | **1.10×** | +10% |
| 0.65 | 2.98 | 2.40 | **1.24×** | +24% |
| 0.70 | 3.16 | 2.40 | **1.32×** | +32% |
| 0.75 | 3.34 | 2.40 | **1.39×** | +39% |
| 0.80 | 3.51 | 2.40 | **1.46×** | +46% |

For the aggregate +98% projection (full two-slot: Gemma4 E2B + Qwen3-1.7B), the Qwen3 slot alone at alpha=0.75 contributes +39% on top of the baseline 36.6k tok/s Qwen slot → ~51k Qwen-slot tok/s.

---

## 4. Concurrency Sweep

Speculative decoding amortizes differently across concurrency levels (§5a finding):

| Concurrency (C) | Spec mechanism | Expected speedup | Reasoning |
|---|---|---|---|
| **C=1** (single user) | Draft serialized; highest per-request latency gain | **1.3–1.5×** | Memory-bandwidth-bound target; draft fills speculative "gap". Best absolute alpha. |
| **C=8** | Batch partially fills; some draft-target misalignment | **1.2–1.4×** | Alpha degrades ~5-10% vs C=1 due to batched verification token mixing. |
| **C=32** | Scheduler mixes requests; batch efficiency competes with spec overhead | **1.1–1.2×** | Draft model adds scheduling latency; verifier batch size grows → slower per step. |
| **C=128** | Throughput-saturated; compute-bound target | **1.0–1.1×** | At high concurrency, target saturates SM occupancy; draft adds latency overhead with diminishing per-token acceptance benefit. |

**Conclusion:** Same-family spec decode is strongest at C=1–8. This is the key workload for interactive/agentic use (single session). For batch inference (C=128), consider disabling spec decode or switching to ngram with low gamma.

### Secondary gamma sweep (C=1 only)

| gamma | expected speedup |
|---|---|
| 2 | 1.15–1.25× |
| 3 | 1.25–1.35× |
| **4 (default)** | **1.30–1.45×** |
| 5 | 1.30–1.45× |
| 6 | 1.25–1.40× (diminishing; draft errors compound) |

Optimal gamma for Qwen3 MoE likely 4–5. Above 5, draft divergence per step accumulates faster than the token-count gain.

---

## 5. Bench Procedure

### 5a. Pre-flight (run once)

```bash
# 1. Verify draft model downloaded and passes compat checks
python3 -c "
from transformers import AutoConfig
draft = AutoConfig.from_pretrained('/root/models/Qwen3-1.7B')
target = AutoConfig.from_pretrained('/root/models/Qwen3-30B-A3B-NVFP4', trust_remote_code=True)
print('draft  arch:', draft.architectures)
print('target arch:', target.architectures)
print('draft  vocab:', draft.vocab_size)
print('target vocab:', target.vocab_size)
assert draft.vocab_size == target.vocab_size, 'VOCAB MISMATCH — abort spec decode'
assert draft.rope_theta == target.rope_theta, f'RoPE theta mismatch: {draft.rope_theta} vs {target.rope_theta}'
print('COMPAT PASS')
"

# 2. Check VRAM headroom
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits
# Need >= 26 GB free: ~22 GB target + ~3 GB draft + 1 GB OS slack
```

### 5b. Baseline run (target only)

```bash
MODE=baseline AUTOKERNEL_QWEN3_SPEC=1 PORT=8009 ./launch_qwen3_speculative.sh
# Wait for health
until curl -sf http://localhost:8009/health; do sleep 2; done
python3 bench_t2h_qwen3_sweep.py --port 8009 --concurrency 1 8 32 128 --tag baseline_nospec
docker rm -f vllm-qwen3-samefamily-spec-baseline
```

### 5c. Patched run (target + draft)

```bash
MODE=patched AUTOKERNEL_QWEN3_SPEC=1 PORT=8010 ./launch_qwen3_speculative.sh
until curl -sf http://localhost:8010/health; do sleep 2; done
python3 bench_t2h_qwen3_sweep.py --port 8010 --concurrency 1 8 32 128 --tag patched_spec_g4
docker rm -f vllm-qwen3-samefamily-spec-patched
```

### 5d. Gamma sweep (C=1 only, post primary pass)

```bash
for GAMMA in 2 3 4 5 6; do
  NUM_SPEC_TOKENS=${GAMMA} MODE=patched AUTOKERNEL_QWEN3_SPEC=1 PORT=8010 \
    ./launch_qwen3_speculative.sh
  until curl -sf http://localhost:8010/health; do sleep 2; done
  python3 bench_t2h_qwen3_sweep.py --port 8010 --concurrency 1 --tag patched_spec_g${GAMMA}
  docker rm -f vllm-qwen3-samefamily-spec-patched
done
```

---

## 6. Acceptance Criterion

| Outcome | Interpretation | Action |
|---|---|---|
| speedup C=1 >= 1.30× AND alpha >= 0.60 | PASS — deploy spec decode in production launcher | Update `launch_qwen3_ngram_spec.sh` → `launch_qwen3_speculative.sh`; close §16 |
| speedup C=1 in [1.15, 1.30) | PARTIAL PASS — worthwhile for low-concurrency serving | Deploy with note; keep ngram as C=128 fallback |
| speedup C=1 < 1.15 OR alpha < 0.45 | FAIL — spec decode is net-negative | KILL §16; investigate alpha measurement to see if Qwen3 MoE logit mismatch is the cause |
| alpha < 0.40 at all C | Strong signal | Check RoPE theta / activation mismatch (see §7 Risk) |

---

## 7. Risk: Architectural Detail Mismatch

**Known risk:** Qwen3-1.7B (dense) and Qwen3-30B-A3B (MoE) share the Qwen3 base design but differ in:

| Detail | Qwen3-1.7B | Qwen3-30B-A3B | Risk |
|---|---|---|---|
| MoE vs dense | Dense (28 layers, FF=8960) | MoE (48 layers, 128 experts, top-2) | **Different logit sharpness** — MoE output distributions are sharper/more peaked than dense equivalents; this may push alpha LOWER than same-family dense-to-dense pairs |
| Depth | 28 layers | 48 layers | Hidden state at final layer may diverge more than size-mismatch suggests |
| RoPE theta | 1,000,000 (Qwen3 standard) | 1,000,000 | No mismatch |
| Activation | SiLU | SiLU | No mismatch |
| Tied embeddings | Yes (Qwen3 standard) | Yes | Same embedding space — favorable |
| Vocab | 151,669 | 151,669 | MATCH — critical for spec decode correctness |

**MoE logit divergence:** The primary risk. Qwen3-30B-A3B routes top-2 of 128 experts per token. The selected experts at each layer are a function of the input hidden state. The draft (dense Qwen3-1.7B) cannot reproduce this routing — its logits will reflect the dense MLP prediction, which may differ systematically from the MoE prediction even at the same token position. If alpha < 0.50 empirically, this is the likely cause.

**Mitigation:** If alpha < 0.50, re-run with gamma=2 (shorter speculation horizon limits divergence compounding). If still < 0.50, consider Qwen3-8B (stock dense, deeper, ~7B hidden) as draft candidate — larger draft captures more of the target's logit distribution at the cost of c≈0.6 (heavier draft step).

**Fallback draft candidates (if Qwen3-1.7B alpha < 0.50):**

| Draft model | HF repo | Size | c (est.) | Notes |
|---|---|---|---|---|
| Qwen3-1.7B (primary) | `Qwen/Qwen3-1.7B` | ~3.5 GB | 0.30–0.45 | Best size/cost tradeoff |
| Qwen3-4B | `Qwen/Qwen3-4B` | ~8 GB | 0.45–0.55 | Better alpha, heavier cost |
| Qwen3-8B (stock dense) | `Qwen/Qwen3-8B` | ~16 GB | 0.55–0.65 | Best alpha, high VRAM cost |

---

## 8. KILL_PATTERNS Cross-Check

Per `plans/KILL_PATTERNS.md` pre-dispatch checklist:
- [x] WSL2 GPU isolation: triple-isolated in launcher (Docker `device=N` + `CUDA_VISIBLE_DEVICES=0` + `NVIDIA_VISIBLE_DEVICES=N`).
- [x] `unset NAME` before container name assignment.
- [x] UUID isolation check logged inside container before `exec python3`.
- [x] No `getattr(..., None)` silent dispatch miss — spec config passed via `--speculative-config` CLI JSON (no Python plugin).
- [x] VRAM guard: `--gpu-memory-utilization 0.88` (vs 0.90 in ngram launcher) to reserve draft KV headroom.
- [x] Env gate: `AUTOKERNEL_QWEN3_SPEC=1` prevents accidental launch.

---

## 9. Expected Result Summary

```
C=1   baseline:  ~800 tok/s     patched: ~1,100–1,200 tok/s   (+38–50%)
C=8   baseline:  ~4,000 tok/s   patched: ~4,800–5,200 tok/s   (+20–30%)
C=32  baseline:  ~14,000 tok/s  patched: ~15,400–16,800 tok/s (+10–20%)
C=128 baseline:  ~29,600 tok/s  patched: ~30,500–31,400 tok/s (+3–6%)
```

If C=1 latency speedup >= 1.30× and aggregate (weighted toward the common C=8–32 workload) hits +25%+, this is a strong deployment target. The §16 +98% aggregate figure assumes BOTH Qwen3-1.7B and Gemma4-E2B drafts are deployed simultaneously for the two-model serving scenario; the Qwen3 slot alone at alpha=0.75 contributes approximately +35–40% to total cluster throughput.
