# Self-Drafting Speculative Decode Feasibility

**Proposal:** Prune Gemma4 26B to 50% experts, router-fine-tune with LoRA (4 epochs C4, 256 samples),
run pruned model as draft (K=2) and full model as verifier on the same PRO 6000 (96 GB).

---

## The Case Against

The pruning track record is definitive:

| Experiment | What was pruned | Result |
|---|---|---|
| Discovery #25 | 3 early layers (low residual scalars) | -41% quality |
| Discovery #41 | 3 middle layers individually | 0% coherent, all three |
| Discovery #40 | 5 experts in layer 0 (0.13% of slots) | 30% coherent — catastrophic |

The #40 result is the most damning. Five experts across a single layer — the absolute minimum possible
intervention — still produced catastrophic failure 70% of the time. The meta-discovery conclusion is
explicit: "Expert manipulation is completely off the table for Gemma4 without fine-tuning. No safe set
of experts to disable exists."

Removing 50% of experts (1,920 of 3,840 slots) is roughly 15,000x more aggressive than the smallest
failure. There is no evidence that any amount of expert removal is recoverable through a 30-minute LoRA
router fine-tune on 256 samples of C4.

## The Case For (Weakly)

- 96 GB fits both models simultaneously; this experiment is physically possible only on PRO 6000.
- LoRA router fine-tuning has not been tried. The discoveries condemned zero-fine-tune pruning; this
  adds a recovery step. The hypothesis is not physically impossible.
- 256-sample C4 fine-tune is cheap (~30 minutes). The cost of the gate test is low.
- If the pruned+fine-tuned draft achieves accept-rate >= 0.70, the verifier step (8.3 ms) becomes the
  bottleneck, and K=2 drafting could yield ~1.5-2x single-user throughput without additional hardware.

## The N-gram Analogy

Discoveries #39 and #42 ruled out n-gram spec decode completely: -11% at n=4, -49% at n=1, content
type irrelevant. The root cause was structural — compute-bound model, variable batch sizes breaking CUDA
graph reuse. Self-drafting avoids the variable-batch-size problem (draft and verifier share a single
forward pass structure), but it introduces a far harder problem: producing a coherent draft at all.
N-gram failed because overhead exceeded benefit. Self-drafting may fail at a more fundamental level —
the draft model may not be coherent enough to generate plausible tokens at all.

## Honest Assessment

The kill criterion (PPL regress > 3% OR accept-rate < 0.70) will almost certainly trigger on PPL
before the accept-rate is ever measured. Discovery #40 shows that even micro-scale expert pruning
produces 70% incoherent output. A 50% expert-pruned model starting from that baseline has no realistic
path to < 3% PPL regress through 256 samples of LoRA fine-tuning. Fine-tuning that aggressive a
structural change typically requires thousands of samples and multiple training stages.

The better alternatives already on the roadmap — EAGLE3 draft training (1-2 days GPU, designed for this
architecture) and FusenDiffusion (gate test already planned, 4-hour investment) — have a higher
probability of reaching the accept-rate target without modifying the verifier model at all.

## Recommendation

**Do not pursue.** The LoRA recovery hypothesis is not impossible, but it contradicts three independent
catastrophic failures at progressively smaller pruning magnitudes. The 30-minute gate test is cheap, but
interpreting a pass as evidence for the full 50% variant would require ignoring everything the pruning
track record shows.

If the router fine-tune approach is ever revisited, start at 5% expert pruning (not 50%), use a full
fine-tuning run (not 256 samples), and treat the PRO 6000 experiments plan's own kill criterion as a
hard stop, not a target to optimize around.

**Priority order for single-user throughput gain:** FusenDiffusion gate test → EAGLE3 draft training →
self-drafting (only if both prior approaches fail).
