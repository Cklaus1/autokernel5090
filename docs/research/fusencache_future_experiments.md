# FusenCache — Future Experiments

Discovered through 64+ experiments across the FusenCache project.

## High Priority (Proven Impact)

### 1. Fix K8V4B16 Concurrent Serving
- **Bug:** CUDA index OOB in `_fc_scales` tensor during concurrent requests
- **Root cause:** Scale tensor indexing exceeds allocation when multiple requests
  use different block_table mappings simultaneously
- **Fix:** Pre-allocate `_fc_scales` at model init matching full KV cache size,
  or embed scales inline in the cache slot
- **Expected impact:** Unlock batch serving at 2.3x compression

### 2. On-Chip V Decompression (from turboquant-gpu)
- **Idea:** Don't dequant V to a full float tensor before attention. Instead,
  decompress V inside the Triton attention kernel during P×V accumulation.
- **Why:** Saves one full V tensor read/write per step
- **Expected impact:** 15-20% speed improvement for K8V4B16

### 3. Fused K+V Compress Kernel
- **Idea:** Quantize K (int8) and V (int4) in one Triton kernel launch
  instead of separate Python operations
- **Why:** Halves GPU launch overhead during store path
- **Expected impact:** 10-15% prefill speed improvement

### 4. Lloyd-Max Centroids for V (from turboquant-gpu)
- **Idea:** Instead of uniform 16-level quantization for V, use Lloyd-Max
  optimal centroids for the actual activation distribution
- **Why:** Lloyd-Max is provably optimal for known distributions. If V
  activations are approximately Gaussian, this gives better quality at same bits.
- **Expected impact:** Better V quality at 4-bit, possibly enabling V=3-bit

### 5. Random Rotation Before Quantization (from turboquant-gpu)
- **Idea:** Apply random orthogonal rotation to K/V before quantization.
  Makes coordinates approximately i.i.d. Gaussian, which is optimal for
  scalar quantization.
- **Why:** Could improve K quality from 0.5% to <0.2% error, or enable
  K=6-bit with same quality as current K=8-bit
- **Caution:** We proved TurboQuant's vLLM implementation was broken. But the
  underlying rotation trick IS mathematically sound — it was the implementation
  (bitpacking, head_dim mismatch) that failed, not the theory.
- **Expected impact:** Better quality at same bits, or same quality at fewer bits

## Medium Priority (Architectural)

### 6. Configurable Quality Knob
- **Idea:** Single env var `FUSEN_MODE=quality|balanced|compression` that selects:
  - quality: K8V4B16 (0.5% error, 2.3x, 23K tokens)
  - balanced: K8V4B32 (moderate, 2.5x)
  - compression: K6V4B32 (aggressive, 3.6x)
  - speed: FP8 native (2.6% error, 2.0x, 145 tok/s)
- **Why:** Different workloads need different tradeoffs

### 7. Adaptive Per-Layer Precision
- **Idea:** Use 8-bit K for early layers (more sensitive), 4-bit K for
  later layers (more tolerant)
- **Why:** Not all layers contribute equally to attention quality
- **Test method:** Run rapid_quant_test.py on each layer's real K/V
  to find per-layer sensitivity

### 8. V4c Selective Attention via vLLM Upstream PR
- **Idea:** Submit the landmark-based selective attention as a native
  vLLM feature, not a monkey-patch
- **Why:** Monkey-patching breaks torch.compile + CUDA graphs.
  Native integration would give 145 tok/s + selective for long context.
- **Blocker:** Requires vLLM upstream acceptance

## Lower Priority (Research)

### 9. 3-Bit V with Rotation
- **Idea:** If rotation makes V distribution Gaussian, Lloyd-Max 3-bit
  should give better quality than current uniform 4-bit without rotation
- **Expected:** 3-bit V = 25% less V storage, but needs rotation overhead
- **Risk:** We proved V<4-bit produces garbage without rotation. With rotation?

### 10. Integer Dot Product Accumulation
- **Idea:** Accumulate K×Q dot product in int32, scale once at the end
  (like llama.cpp does)
- **Why:** Reduces FP32 precision errors in the attention score computation
- **Expected impact:** Minor quality improvement, maybe 1-2%

### 11. Larger Block Sizes with Outlier Protection
- **Idea:** Use block=64 or block=128 (more compression) but protect
  top-2 outlier dims per block in FP8
- **Why:** Outliers dominate quantization error. Protecting them enables
  larger blocks without quality loss.
- **Reference:** KVQuant (Berkeley) showed this is critical for key quality

### 12. Multi-GPU Tensor Parallelism
- **Idea:** Test FusenCache with TP=2 across 2 GPUs
- **Why:** Doubles bandwidth, halves per-token latency
- **Blocker:** Needs 2 GPUs

### 13. Real Model KV Extraction for Quality Testing
- **Idea:** Extract actual K/V tensors from Gemma 4 31B during inference,
  save to disk, use for all future quality tests
- **Why:** Our rapid sweep used random data which doesn't predict real quality.
  Real K/V tensors would make the sweep reliable.
- **Blocked on:** transformers version (5.5+) needed to run Gemma 4 forward pass

## Methodology Improvements

### 14. Automated Experiment Loop
- **Idea:** AutoKernel-style loop for FusenCache: edit quantization config →
  run bench → keep if improved → log to TSV → repeat
- **Why:** 64 manual experiments is slow. Automation could run 200+ overnight.

### 15. Quality Metrics Beyond Top-1
- **Idea:** Measure perplexity, MMLU, coding accuracy (not just attention top-1)
- **Why:** We discovered that 100% attention top-1 doesn't predict coherent output.
  Need downstream task metrics.

### 16. Serving Stress Test
- **Idea:** 24-hour serving test at C=32 with various prompt lengths
- **Why:** Stability matters more than peak throughput for production
