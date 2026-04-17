# T2-G: I-DLM Inference Bake-Off on PRO 6000 — Setup Plan

## What is I-DLM?

I-DLM (Iterative Discrete Language Model) is an autoregressive-equivalent diffusion language model. Instead of generating one token at a time (standard AR), I-DLM generates N tokens per forward pass using speculative diffusion decoding:

- **Cold start:** Feeds `[t0, MASK, MASK, ..., MASK]` (1 real + 2(N-1) masked positions)
- **Verify round:** Feeds `[pending, spec0..specK, MASK, ..., MASK]` — verifies speculative tokens left-to-right using standard spec decoding criterion (`r = p(x)/q(x)`)
- **Claimed throughput:** 3.8x AR baseline

The core algorithm lives in `I-DLM/inference/sglang/sglang/srt/dllm/algorithm/idlm_blockN.py` (lines 337-1037: classify, forward, verify/sample, trim/assemble). It operates on post-attention logits and does NOT require CuTe flash attention kernels — any attention backend (FlashInfer, Triton) works.

**Checkpoint:** `I-DLM/training/model/Qwen3-8B-b2-allmasked/`
- Base model: Qwen3-8B with diffusion training
- `config.json`: `block_size=2`, `mask_token_id=151669`
- Custom model class: `SDARForCausalLM` (via `auto_map` in config.json)
- Attention implementation: `flex_attention`

**Config:** `I-DLM/inference/configs/idlm_blockN4_config.yaml`
```yaml
block_size: 7           # Must be 2*gen_block_size - 1
gen_block_size: 4       # Tokens per step (1 clean + 3 speculative)
confidence_threshold: 0.0
temperature: 1.0
top_k: 50
top_p: 0.95
use_spec_verify: true
```

---

## SM120 CuTe Kernel Blocker (CRITICAL — Must Fix Before Running)

The I-DLM repo ships a custom SGLang fork with CuTe flash attention kernels. These hard-assert compute capability membership and will crash on PRO 6000 (SM120, compute cap 12).

### The Problem

**File:** `inference/sglang/sglang/jit_kernel/flash_attention/cute/interface.py`

**Crash point 1 — Line 251:**
```python
assert compute_capability in [9, 10, 11], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x"
```
`_get_device_capability()` returns `torch.cuda.get_device_capability()[0]` which is `12` on PRO 6000. `12 not in [9, 10, 11]` -> immediate `AssertionError: No kernel available`.

**Crash point 2 — Lines 430-478:** The kernel routing has branches for SM90 (`compute_capability == 9`) and SM100/SM110 (`compute_capability in [10, 11]`), but nothing for SM120. Even if the assert is removed, the routing hits the `else` branch at line 478-480 and raises `ValueError`.

**Why SM100 path won't work on SM120:** `FlashAttentionForwardSm100` uses `tcgen05` (Blackwell datacenter tensor core gen 05) and TMEM (tensor memory). SM120 (RTX/PRO Blackwell) does NOT have TMEM or `tcgen05`. This is the same SM100 vs SM120 binary incompatibility documented in Discovery #24.

**Why SM90 path WILL work on SM120:** `FlashAttentionForwardSm90` uses Hopper-era TMA and warp-group MMA, which SM120 supports (SM120 is a superset of SM90 for these features). This is the same fallback strategy that FlashAttention-2 and FlashInfer use on consumer Blackwell.

### The Fix

Patch `interface.py` to route SM120 to the SM90 backend:

**Fix 1 — Line 251:** Add `12` to the allowed list:
```python
# BEFORE:
assert compute_capability in [9, 10, 11], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x"

# AFTER:
assert compute_capability in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
```

**Fix 2 — Lines 430-454:** Route SM120 to SM90 path. Change line 430 from:
```python
        if compute_capability == 9:
```
to:
```python
        if compute_capability in [9, 12]:  # SM120 uses SM90 fallback (no TMEM/tcgen05)
```

This routes PRO 6000 through `FlashAttentionForwardSm90` which uses Hopper-compatible instructions.

**Alternative fix (simpler but slower):** Disable CuTe entirely and let SGLang pick FlashInfer or Triton attention backend. Set environment variable or patch the `_flash_attn_fwd` function to raise ImportError, forcing the fallback path.

### Validation

After patching, run a throwaway single-request test and confirm:
- No `No kernel available` error
- No `TMEM` or `tcgen05` error
- No `CUDA_ERROR_NO_BINARY_FOR_GPU` error
- Attention output matches expected shape

---

## Step-by-Step Setup Instructions for PRO 6000

### Prerequisites
- PRO 6000 with CUDA 12.8+ driver
- Python 3.10+
- CUDA toolkit 12.8 (NOT 13.2, per Discovery #59)

### Step 1: Clone and Install I-DLM

```bash
cd /home/cklaus/projects/aigpu/I-DLM
# The repo is already cloned at this path

# Install the custom SGLang fork
cd inference/sglang
pip install -e .

# Install CuTe DSL (required for CuTe kernels)
pip install nvidia-cutlass-dsl==4.2.0
```

### Step 2: Apply SM120 Patch

```bash
# Patch interface.py
FILE="inference/sglang/sglang/jit_kernel/flash_attention/cute/interface.py"

# Line 251: Add 12 to allowed compute capabilities
sed -i 's/assert compute_capability in \[9, 10, 11\]/assert compute_capability in [9, 10, 11, 12]/' "$FILE"

# Line 430: Route SM120 to SM90 fallback
sed -i 's/if compute_capability == 9:/if compute_capability in [9, 12]:  # SM120 uses SM90 fallback/' "$FILE"
```

### Step 3: Download/Verify Checkpoint

The checkpoint is at `I-DLM/training/model/Qwen3-8B-b2-allmasked/`. Verify the safetensors index exists:
```bash
ls training/model/Qwen3-8B-b2-allmasked/model.safetensors.index.json
```

If actual weight files (`.safetensors`) are missing (only the index is present), download from the model source. The config references custom model code via `auto_map`:
- `AutoConfig` -> `configuration_sdar.SDARConfig`
- `AutoModelForCausalLM` -> `modeling_sdar.SDARForCausalLM`

These Python files are in the checkpoint directory and will be loaded automatically by `transformers`.

### Step 4: Smoke Test (Single Request)

```bash
cd /home/cklaus/projects/aigpu/I-DLM

# Launch server
python inference/sglang/sglang/launch_server.py \
  --model-path training/model/Qwen3-8B-b2-allmasked \
  --dtype bfloat16 \
  --dllm-algorithm idlm_blockN \
  --dllm-algorithm-config inference/configs/idlm_blockN4_config.yaml \
  --port 30000 \
  --trust-remote-code

# In another terminal, send a test request
curl -s http://localhost:30000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text": "The capital of France is", "sampling_params": {"max_new_tokens": 32}}'
```

Watch server logs for:
- "CuTe flash attention compiled for SM90" (good — SM120 routed to SM90)
- Any `No kernel available` or CUDA errors (bad — patch not applied correctly)

### Step 5: Verify CuTe Fallback Works

If CuTe SM90 fallback fails (e.g., JIT compilation error), disable CuTe and use FlashInfer:
```bash
# Option A: Environment variable (if SGLang supports it)
export SGLANG_ATTENTION_BACKEND=flashinfer

# Option B: Rename CuTe interface to force import failure
mv inference/sglang/sglang/jit_kernel/flash_attention/cute/interface.py \
   inference/sglang/sglang/jit_kernel/flash_attention/cute/interface.py.bak
```

---

## Benchmark Plan

### Experiment 1: Baseline BF16 (SM90 Fallback Path)

Run I-DLM as published on PRO 6000 after the SM120 patch.

```bash
# Sweep concurrency levels
for C in 1 16 32 64; do
  # Run benchmark at concurrency C
  # Record: tok/s, accept-rate
done

# Also sweep gen_block_size
for GBS in 2 4; do
  # Modify config: block_size = 2*GBS - 1, gen_block_size = GBS
  # Record: tok/s, accept-rate
done
```

**Metrics to record:**
- tok/s at C=1, 16, 32, 64
- Accept rate per gen_block_size (2 and 4)
- VRAM usage
- Per-step latency breakdown

### Experiment 2: NVFP4 Weights

Quantize the Qwen3-8B-b2-allmasked checkpoint to NVFP4 using the T1-C recipe from the autokernel project.

```bash
# Quantize checkpoint
# (Use the same quantization pipeline as Gemma4 26B NVFP4)

# Re-run the same sweep
for C in 1 16 32 64; do
  # Record: tok/s, accept-rate with NVFP4
done
```

**Key question:** Does NVFP4 quantization preserve the diffusion model's speculative acceptance rate? AR models are robust to quantization, but diffusion models may be more sensitive because spec verify depends on precise probability ratios.

### Experiment 3: Head-to-Head vs AR Baseline

Compare I-DLM against vanilla autoregressive Qwen3-8B on the same hardware, same prompts.

```bash
# AR baseline: standard Qwen3-8B (not the diffusion variant)
python -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B \
  --dtype bfloat16 \
  --port 30001

# Same prompts, same concurrency sweep
# Record: tok/s for both, calculate actual speedup ratio
```

**Eval scripts** (already available in `I-DLM/inference/eval/`):
- `eval_gsm8k.py` — math reasoning
- `eval_math500.py` — harder math
- `eval_humaneval.py` — code generation
- `eval_mmlu.py` — general knowledge
- `eval_ifeval.py` — instruction following

Run at minimum GSM8K and MATH-500 for quality comparison. The 3.8x throughput claim is meaningless if quality degrades.

---

## Checkpoint Details

**Model:** `Qwen3-8B-b2-allmasked`
- Architecture: SDAR (Semi-autoregressive Diffusion with Autoregressive)
- Base: Qwen3-8B
- `block_size`: 2 (trained with block size 2)
- `mask_token_id`: 151669
- `fuse_cross_entropy`: true (training optimization, not relevant for inference)
- `attn_implementation`: flex_attention

**Do NOT use gen_block_size > 4:** The experiments doc explicitly warns against `gen_block_size=8` — it would require 512 TMEM columns which don't exist on SM120. If block-N > 4 is desired, a retrained model with `block_size=15, gen_block_size=8` is needed.

---

## Dependency Issues and Compatibility Notes

### SGLang Version
The I-DLM repo ships its own SGLang fork. This is NOT upstream SGLang. Key differences:
- Custom DLLM (Diffusion LLM) algorithm support (`--dllm-algorithm` flag)
- Modified scheduler for speculative diffusion decoding
- CuTe flash attention integration (the source of the SM120 issue)

**Risk:** If we need a newer SGLang for other features (e.g., FlashInfer updates), the I-DLM fork may conflict. Keep the I-DLM SGLang in a separate venv.

### CUDA Compatibility
- Use CUDA 12.8 toolkit (proven stable on SM120 per Discovery #59)
- CUDA 13.2 caused regressions for vLLM/FusenCache; untested with I-DLM's SGLang fork
- nvidia-cutlass-dsl==4.2.0 required for CuTe kernel compilation

### PyTorch
- Need PyTorch 2.10+ with SM120 support
- `flex_attention` (used by the model) requires PyTorch 2.5+

### Potential Issues
1. **Model weight files:** The checkpoint directory has `model.safetensors.index.json` but actual `.safetensors` weight files may need downloading separately (check size of directory)
2. **Custom model code:** `modeling_sdar.py` and `configuration_sdar.py` in the checkpoint dir use `trust_remote_code=True` — the `--trust-remote-code` flag is required
3. **FlashInfer fallback:** If CuTe SM90 path has issues, FlashInfer is the fallback. Ensure FlashInfer is installed with SM120 support (`pip install flashinfer -i https://flashinfer.ai/whl/cu128/torch2.10/`)
4. **fused_verify_kernel:** The algorithm tries to import `fused_spec_verify` (line 34-37 of `idlm_blockN.py`). If this C++/CUDA extension isn't compiled for SM120, it falls back to pure PyTorch (`_HAS_FUSED_VERIFY = False`). This is safe but slower.

---

## Summary of Blockers

| Blocker | Severity | Fix | Status |
|---------|----------|-----|--------|
| CuTe assert line 251 (`compute_capability in [9,10,11]`) | FATAL | Add `12` to list | Patch ready |
| CuTe routing lines 430-478 (no SM120 branch) | FATAL | Route `12` to SM90 path | Patch ready |
| SM100/tcgen05/TMEM on SM120 | FATAL if not patched | SM90 fallback avoids this | Handled by above |
| Weight files potentially missing | BLOCKING | Download safetensors | Check directory size |
| fused_verify_kernel compilation | LOW | Falls back to PyTorch automatically | Non-blocking |
| CUDA 13.2 compatibility | UNKNOWN | Use CUDA 12.8 (proven) | Recommendation |
