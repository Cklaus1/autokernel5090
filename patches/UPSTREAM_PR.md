# FlashInfer PR: Auto-detect CUDA toolkit and compiler for Blackwell GPUs

## Target repo
https://github.com/flashinfer-ai/flashinfer

## Title
`[JIT] Auto-detect CUDA toolkit and compatible gcc for Blackwell (SM120) GPUs`

## Description

### Problem

FlashInfer's JIT compiler assumes the system default `nvcc` and `gcc` work for
all GPUs. On Blackwell (RTX 5090/5080, SM120), this fails because:

1. **SM120 requires CUDA 12.9+** for `compute_120a` / FP4 support. Systems
   often have CUDA 12.8 as default with 12.9 installed alongside.
2. **nvcc 12.9 rejects gcc-13+**. Many distros ship gcc-13 or gcc-14 as
   default, causing JIT compilation to fail with cryptic errors.

The result: FlashInfer works on Hopper (SM90) but silently fails on Blackwell
with "unsupported compute capability" or "unrecognized compiler" errors.

### Fix

Three changes to `flashinfer/jit/cpp_ext.py`:

1. **Auto-detect CUDA toolkit** — Query GPU compute capability via
   `nvidia-smi`, then search `/usr/local/cuda-*` for the newest toolkit that
   supports it. Only activates when needed (non-default GPU + multiple toolkits).

2. **Auto-detect compatible gcc** — When the selected nvcc rejects the system
   gcc, search for `gcc-12`, `gcc-11`, `gcc-10` as fallbacks.

3. **Match CXX to CC** — If gcc-N was auto-selected, use g++-N for linking
   consistency.

All changes preserve existing behavior on systems where defaults work:
- If `CUDA_HOME` / `CUDA_PATH` is set → used as-is (no change)
- If system nvcc supports the GPU → used as-is (no change)
- If system gcc works with nvcc → used as-is (no change)

New: `FLASHINFER_CUDA_HOME` env var for explicit override (highest priority).

### Testing

Tested on:
- RTX 5090 (SM120) with CUDA 12.8 system default + CUDA 12.9 installed
- 225+ kernel optimization experiments via AutoKernel
- vLLM 0.17.0 → 0.18.1 with FlashInfer 0.6.4 → 0.6.6
- Qwen3.5-9B NVFP4 achieving 170 tok/s single-user decode

### Impact

- **Blackwell users**: JIT "just works" without manual env var configuration
- **Non-Blackwell users**: Zero behavior change (auto-detection doesn't activate)
- **Multi-CUDA systems**: Correct toolkit selected automatically

### Author

Chris Klaus <cklaus@fusen.world>
Found during RTX 5090 kernel optimization with [AutoKernel](https://github.com/autokernel).

---

## Also consider: vLLM companion PR

The same auto-detection would benefit vLLM's own JIT compilation. Consider
filing a companion PR to vllm-project/vllm if this approach is accepted.
