# DeepGemm SM120 (RTX 5090) Compatibility Test

**Date:** 2026-04-09
**GPU:** RTX 5090, SM120 (12.0), 575W TDP
**Docker:** vllm-built (CUDA 12.8)

---

## Summary

**DeepGemm does NOT work on SM120 (RTX 5090).** There are two independent blockers:

1. The `deep_gemm` / `vllm.third_party.deep_gemm` package is not installed — `has_deep_gemm()` returns False.
2. Even if the package were present, all pre-compiled cubins are `sm100f` format (B100/B200 ISA) and fail with `CUDA_ERROR_NO_BINARY_FOR_GPU (209)` on SM120.

The "one-line fix" to `support_deep_gemm()` is necessary but **not sufficient**.

---

## Step 1: Where the SM120 Check Lives

File: `/build/vllm/vllm/platforms/cuda.py`, line 543

```python
@classmethod
def support_deep_gemm(cls) -> bool:
    """Currently, only Hopper and Blackwell GPUs are supported."""
    return cls.is_device_capability(90) or cls.is_device_capability_family(100)
```

`is_device_capability_family(100)` computes `cap.to_int() // 10 == 100 // 10`, i.e., `major == 10`.

- SM100 (B100/B200): `100 // 10 = 10` → matches
- SM120 (RTX 5090): `120 // 10 = 12` → does NOT match

**The fix:** add `or cls.is_device_capability_family(120)` — a true one-liner.

---

## Step 2: `has_deep_gemm()` Returns False

```
has_deep_gemm(): False
```

Neither `deep_gemm` (PyPI) nor `vllm.third_party.deep_gemm` (vendored) is present in this build.  
The `flashinfer` package has its own `flashinfer.deep_gemm` module which contains the SM100 FP8 GEMM infrastructure, but vLLM's `has_deep_gemm()` does not check for that — it only checks for the standalone `deep_gemm` package.

---

## Step 3: Binary Compatibility Test — SM100 Cubins on SM120

Even bypassing the Python check, `flashinfer.deep_gemm` contains 317 precompiled cubins all targeting `sm100f`.

```
cuLibraryLoadFromFile(sm100_cubin) → CUDA_SUCCESS   # file parses OK
cuLibraryGetKernel(symbol)         → CUDA_ERROR_NO_BINARY_FOR_GPU (209)
```

The cubin loads at the file-parse level but has no PTX fallback and no SM120 binary. The CUDA driver refuses to JIT-compile it for SM120 because there is no embedded PTX — only a device-specific binary.

**SM100 and SM120 are different ISA targets**, despite both being "Blackwell":
- SM100 = B100/B200 (HBM3e, NVLink4, TMEM, cluster-based tensor core ISA)
- SM120 = RTX 5090 (GDDR7, PCIe, consumer Blackwell, different memory architecture)

---

## Step 4: Monkey-Patch Result

```python
cuda_platform.CudaPlatformBase.support_deep_gemm = classmethod(lambda cls: True)
```

Even with this patch:
- `has_deep_gemm()` is still False (package not installed)
- `is_deep_gemm_supported()` stays False
- `DeepGemmMoE` import fails with ImportError (import path has changed between vLLM versions)

---

## Root Cause Analysis

| Check | Result | Fix Required |
|-------|--------|--------------|
| `support_deep_gemm()` SM120 check | False (wrong family check) | Add `is_device_capability_family(120)` |
| `has_deep_gemm()` package present | False (not installed) | Install `deep_gemm` package built for SM120 |
| SM100 cubin runs on SM120 | No — `CUDA_ERROR_NO_BINARY_FOR_GPU` | Need SM120-native cubins |

---

## What Would Actually Be Needed

To enable DeepGemm on SM120, all three must be addressed:

1. **Install `deep_gemm` built for SM120** — either pip package or vendored into vLLM
2. **Recompile the FP8 GEMM cubins targeting SM120** — the `sm100_fp8_gemm_1d1d.cuh` kernel uses SM100-specific hardware instructions (TMEM, specialized cluster patterns). These need recompilation with `--generate-code arch=compute_120,code=sm_120`
3. **Fix `support_deep_gemm()`** to include SM120 — trivial one-liner once the above exist

---

## Relevant File Paths

- `/build/vllm/vllm/platforms/cuda.py` — `support_deep_gemm()` at line 543
- `/build/vllm/vllm/utils/deep_gemm.py` — `is_deep_gemm_supported()`, `has_deep_gemm()`
- `/build/vllm/vllm/utils/import_utils.py` — `has_deep_gemm()`
- `/usr/local/lib/python3.12/dist-packages/flashinfer/deep_gemm.py` — flashinfer's SM100 FP8 GEMM runtime
- `/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins/` — precompiled SM100 cubins (317 kernels, all `sm100f`)

---

## Verdict

The audit's hypothesis was partially correct: `support_deep_gemm()` does incorrectly exclude SM120. But the fix is blocked by a more fundamental issue — there are no SM120-native DeepGemm binaries. The SM100 cubins use Blackwell server-class ISA extensions that are not forward-compatible to the RTX 5090's consumer Blackwell architecture.

**DeepGemm on RTX 5090 requires upstream work** (NVIDIA or flashinfer team) to compile SM120 cubins.
