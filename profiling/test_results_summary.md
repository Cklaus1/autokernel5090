# Test Results Summary
Date: 2026-04-09

## Suite 1: Two-Tier Brain Tests
**File:** `tools/test_two_tier.py`
**Result:** 32 passed, 0 failed
**Time:** 0.95s
**Status:** PASS

## Suite 2: Parallel Solver Tests
**File:** `fusen_solver/tests/test_solver.py`
**Result:** 35 passed, 0 failed
**Time:** 0.27s
**Status:** PASS

## Suite 3: AutoKernel v2 Auto-Config Tests
**File:** `tools/auto_config_test.py`
**Result:** 36 passed, 0 failed
**Time:** 0.08s
**Status:** PASS

## Suite 4: Per-Layer KV Selector Tests
**File:** `tools/test_select_kv_specs.py`
**Result:** 10 passed, 0 failed
**Time:** 0.96s
**Status:** PASS

## Suite 5: Fused Norm+FP4 Triton Kernel Tests (GPU, Docker)
**File:** `kernels/test_fused_norm_fp4.py`
**Runner:** `docker run vllm-built`
**Result:** 11 passed, 0 failed
**Time:** 78.96s (1:18) — includes Triton JIT compilation
**Status:** PASS

## Suite 6: FusenCache Decode C++ Kernel Tests (GPU, Docker)
**Files:** `kernels/csrc/build_fusencache.py` + `kernels/csrc/test_fusencache_decode.py`
**Runner:** `docker run vllm-built`
**Result:** 7/7 tests passed
**Status:** PASS
**Tests:** B1_D256_seq64_nosoftcap, B1_D256_seq64_softcap50, B4_D256_seq128, B8_D256_seq256_softcap50, B1_D256_seq512_split32, B16_D256_seq128_softcap50, B32_D256_seq64

## Suite 7: C++ FP8 Paged Decode Tests (GPU, Docker)
**Files:** `kernels/csrc/build_fp8_decode.py` + `kernels/csrc/test_fp8_paged_decode.py`
**Runner:** `docker run vllm-built`
**Result:** 5/5 tests passed
**Status:** PASS
**Tests:** basic_correctness, long_sequence, head_dim_128, soft_cap, gqa_group1

---

## Aggregate Summary

| Suite | Tests Run | Passed | Failed | Status |
|-------|-----------|--------|--------|--------|
| Two-Tier Brain | 32 | 32 | 0 | PASS |
| Parallel Solver | 35 | 35 | 0 | PASS |
| Auto-Config | 36 | 36 | 0 | PASS |
| Per-Layer KV Selector | 10 | 10 | 0 | PASS |
| Fused Norm+FP4 (GPU) | 11 | 11 | 0 | PASS |
| FusenCache Decode C++ (GPU) | 7 | 7 | 0 | PASS |
| FP8 Paged Decode C++ (GPU) | 5 | 5 | 0 | PASS |
| **TOTAL** | **136** | **136** | **0** | **ALL PASS** |

Note: vLLM server (port 8000) was excluded from all tests — CUDA graph capture was in progress during this run.
