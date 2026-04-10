# Measurement-First Methodology

**The #1 lesson from 35 experiments: measure before building.**

## The Problem

This session produced 35 discoveries. 60% of confident predictions were wrong:
- 12 out of 20 pre-confident predictions failed
- The biggest win (disable inductor = 2x) was accidental
- The most invested-in optimizations (fused MoE, FP8 attention) had least impact

## The Rule

**Never invest more than 1 day before measuring the critical assumption.**

```
For any optimization:
  1. Identify the critical assumption (the thing that must be true for it to work)
  2. Design the cheapest possible test of that assumption (hours, not days)
  3. Run the test
  4. Gate the full investment on the test result:
     - If assumption holds: invest fully
     - If assumption fails: pivot immediately, document why
```

## Decision Gates for All Future Work

### Kernel Optimizations
- **Gate:** Does a 50-line microbenchmark show improvement?
- **Test cost:** 1-2 hours
- **Example:** Before writing 800-line C++ FP8 attention kernel, benchmark a 50-line cp.async memory copy to verify we can achieve 90%+ BW on paged access

### Model Modifications (pruning, quantization, distillation)
- **Gate:** Does the modification preserve quality on 20 spot-check prompts?
- **Test cost:** 30 minutes
- **Example:** Before training a diffusion draft for 5 days, train a 1-layer head for 4 hours and measure acceptance rate

### System Changes (scheduler tuning, config changes)
- **Gate:** Does the change improve throughput on a 5-minute benchmark?
- **Test cost:** 15 minutes
- **Example:** Before committing to block_size=64, run a 5-minute benchmark at C=32

### Framework Integration
- **Gate:** Does the integration work end-to-end with 1 request?
- **Test cost:** 30 minutes
- **Example:** Before building FusenCache CUDA graph support, verify that ONE C++ kernel call captures in a CUDA graph

## Applied to Upcoming Projects

### Diffusion Adapter (FusenDiffusion)
```
Day 0: Critical assumption: MoE hidden states are predictable enough for drafting
  Test: Train 1-layer, 50M param head on 5K prompts (4 GPU-hours)
  Measure: acceptance rate on 100 diverse prompts
  Gate: > 50% → proceed to full training
         30-50% → try deeper head (3 layers)
         < 30% → abandon, pivot to n-gram

Day 1-2: If gate passed, train 3-layer head on 50K prompts
  Measure: acceptance rate, single-user tok/s
  Gate: > 2x speedup → proceed to batch optimization
         1-1.5x → marginal, deprioritize

Day 3-5: If gate passed, optimize for batch
  Measure: batch throughput at C=32
  Gate: > current 6,685 → ship it
         < current → only useful for single-user
```

### C++ FP8 Paged Decode
```
Hour 0: Critical assumption: cp.async achieves 90%+ BW on page-table-indirect loads
  Test: Write a 100-line kernel that cp.async loads pages via block_table
  Measure: bandwidth vs FA2's sequential loads
  Gate: > 80% BW → write full kernel
         < 50% BW → cp.async doesn't help for scattered access, abandon

Hour 1-4: If gate passed, write full FP8 decode kernel
  Measure: latency vs FA2 BF16
  Gate: < 200μs (vs FA2's 323μs) → integrate into vLLM
         > 300μs → Triton-level performance, not worth the C++ complexity
```

### PRO 6000 DP=2 Deployment
```
Hour 0: Critical assumption: both GPUs are visible and independent
  Test: nvidia-smi shows 2 GPUs, each can run a model independently
  Gate: works → proceed with DP=2 scripts
         doesn't → debug hardware/driver first

Hour 1: Launch both servers, benchmark each independently
  Gate: each GPU within 20% of RTX 5090 performance → DP=2 worth it
         > 50% slower per GPU (Max-Q throttling) → reconsider TP=2
```

## Tracking Template

For every optimization attempt, fill in:

```
## Optimization: [name]
Critical assumption: [what must be true]
Cheapest test: [what to measure, how]
Test cost: [hours]
Test result: [measured value]
Gate decision: [proceed / pivot / abandon]
If proceeded - full result: [measured after full build]
Accuracy of prediction: [predicted X, got Y, error Z%]
```

This accumulates into a dataset of prediction accuracy that improves future estimates.
