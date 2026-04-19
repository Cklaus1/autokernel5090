# T1-B FusenCache Full-Mode CUDA Graph Crash — Deep Dive
**Tag:** `W6_T1B_full_mode_deepdive`
**Date:** 2026-04-18
**Category:** P11 Category 4 (silicon-level limit investigation, literature-grounded)

---

## §1 — Evidence Summary

**Confirmed crash:** `W5_T1B_fusen_full_mode_CRASH` row in `results.tsv` line 211: `cudaErrorIllegalInstruction` during graph replay at C>32. This is a different failure mode from the T1-B shadow-tensor bug (which was `cudaErrorIllegalAddress` during piecewise eager, fixed by W5). The full-mode crash survives the W5 fix.

**vLLM version in `vllm-built:latest`:** `0.1.dev100+gc0c98b8b9.d20260417`  
**vLLM version when 4,489 tok/s was banked:** `plans/mega_graph_cooperative_kernel.md §1.2` cites `vllm 0.19.1rc1` as the 4,489 peak version. The current image is a dev snapshot from 2026-04-17 — a different lineage entirely. **Version mismatch is confirmed.**

**Graph count at current config (`max_num_seqs=256`, `cudagraph_mode=full`):**
- `cudagraph_capture_sizes` = `[1,2,4]` + `range(8,256,8)` + `range(256,512+1,16)` = **51 sizes**
- `CUDAGraphMode.FULL` has `separate_routine=False` (verified above), so every token count gets exactly ONE graph
- Total: **51 full graphs captured**
- Each graph = 30 Gemma4 layers × ~15 nodes/layer ≈ **450 nodes/graph × 51 graphs = ~22,950 total graph nodes** submitted to the SM120 graph launcher across the capture session

The crash fires at C>32 (not during capture, but during replay). Discovery #56-58 (`plans/mega_graph_cooperative_kernel.md §1.1`) states: "SM120's graph launcher has a reduced bookkeeping budget for cooperative / large-graph replays." The `cudaErrorIllegalInstruction` at replay time — not capture time — is the canonical signature.

---

## §2 — Five Hypotheses Evaluated

### H1: Graph size exceeds SM120 bookkeeping budget
**Evidence: MOST LIKELY (P=0.60)**

The "crash at C>32 not C≤32" pattern directly maps to node-count scaling. At C=32 the batch is padded to the next capture size (C=32 is in the list; C=33 pads to 40), which executes a 30-layer, ~450-node graph. At C=32 itself, the graph runs — borderline. At C>32 the exact same graph structure runs at a larger batch, but the SM120 replay bookkeeping may scale with active-thread-count × node-count, pushing over the limit at C>32.

Key evidence from `vllm/config/compilation.py:663-664` (inside image):
```
[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_cudagraph_capture_size + 1, 16))
```
With `max_num_seqs=256`, `max_cudagraph_capture_size = min(256*2, 512) = 512`. The launch script uses `-cc.cudagraph_mode full` without passing `cudagraph_capture_sizes`, so it gets the full 51-entry list reaching C=512. This means the graph launcher is asked to handle 512-token graphs at replay time — Gemma4 30 layers × 512 batch is a very large replay.

**Config-only fix candidate:** Pass `-cc.cudagraph_capture_sizes '[1,2,4,8,16,32]'` to `launch_fusen_gemma4_t1b.sh`. This caps at C=32 (6 graphs × 450 nodes = 2,700 total nodes vs the current ~22,950), well within any plausible SM120 limit. Trades off high-concurrency throughput (no captured graph for C=64/128) for crash elimination. At C=32 the piecewise mode peaked at 751 tok/s; a C=32-capped full-mode graph should approach the 4,489 peak if that was the bottleneck.

**Alternative config fix:** Use `FULL_DECODE_ONLY` mode instead of `FULL`. This generates graphs only for pure-decode batches (separate_routine=True, smaller per-graph footprint) and falls back to eager for mixed batches. The plugin's `_patch_cudagraph_mode_override()` in `fusen_kv/plugin.py:319-358` actively suppresses this — it re-asserts FULL when vLLM would downgrade. If FULL_DECODE_ONLY avoids the crash, comment out the override patch.

### H2: Graph-captured memory allocator state corruption
**Evidence: POSSIBLE (P=0.25)**

Piecewise mode inserts stream events between chunks, resetting allocator state. Full mode accumulates across all 30 layers. The `_shadow_copy` fix (W5) only protects the eager (non-capturing) branch — inside `if not _capturing:`. During capture (`_capturing=True`, `fusen_kv/backend.py:1074-1079`), metadata is passed directly as originals with no isolation. If the FusenCache `k4v4b64` path allocates any dynamic buffer during capture that aliases a previously-captured graph's working memory, the replay at larger C could corrupt it.

The `_cpp_decode` path (`fusen_kv/backend.py:700-765`) allocates a fallback `mid_out`/`output` when `B > self._cpp_shared_mid_out.shape[0]` (line 731). During capture at C=256 (padded B=256), if `_max_B` was read as 256 but the shared buffer was only sized at 512 (via the max logic at lines 395-407), B=256 < 512 so the shared buffer is used. This path looks safe. More suspicious: at capture time with `_max_B=512` but `B_padded > 512` (e.g. C=512+1 would pad to next capture), the fallback allocation would happen inside a CUDA graph capture — which can leak allocator state. However, capture sizes max out at 512 and B never exceeds max_cudagraph_capture_size, so this edge case doesn't apply here.

### H3: Specific kernel incompatibility (bisect approach)
**Evidence: PLAUSIBLE (P=0.20)**

The FusenCache `k4v4b64` KV cache uses a custom `fusencache_decode.so` C++ kernel (`_FUSENCACHE_CPP_SO = "/tmp/build_fusencache/fusencache_decode.so"`). This kernel is loaded only if the `.so` exists. The comment at `backend.py:133-143` says: "ALWAYS was tested and crashes during mixed prefill+decode graph replay (Discovery #57)." This is a prior crash mode at the same location. The shadow-tensor fix (W5) addressed the piecewise mode — but the full-mode crash (Discovery #56-58) predates the W5 fix and was never addressed by it.

Without knowing whether the C++ `.so` is present in the container, it's unclear if the Triton or C++ decode kernel is executing. If `_HAS_CPP_DECODE=False` (Triton path), the graph-capture compatibility of the Triton kernel on SM120 is suspect (per `backend.py:134-136`: "Triton 3.6.0 CUDA graph replay crashes with 'illegal instruction'"). This would mean the crash is the known Triton-on-SM120 graph-capture incompatibility, not bookkeeping budget.

**This is the second-most likely root cause if the C++ kernel is NOT present in the container.** If `fusencache_decode.so` is missing from `/tmp/build_fusencache/`, the backend falls back to Triton decode, and the `_cudagraph_support` is set to `NEVER` (line 151-155). But then the `_patch_cudagraph_mode_override` in `plugin.py` suppresses the resulting `ValueError` and forces FULL mode anyway. The Triton kernel then gets captured and replays on SM120 where it crashes — this is exactly Discovery #57.

### H4: vLLM version mismatch
**Evidence: CONFIRMED contributing factor (P=0.45)**

The 4,489 tok/s peak was on `vllm 0.19.1rc1` (cited in `plans/mega_graph_cooperative_kernel.md §1.2`). Current image is `0.1.dev100+gc0c98b8b9.d20260417`. This is a completely different release branch. The dev snapshot likely includes changes to:
- `resolve_cudagraph_mode_and_sizes()` logic (seen to be more complex in current image)
- Graph capture infrastructure
- `AttentionCGSupport` enum handling
- `FULL` vs `FULL_DECODE_ONLY` automatic downgrade behavior

The `_patch_cudagraph_mode_override` in `plugin.py` was added specifically to fight this version's auto-downgrade. The patch forces FULL mode even when vLLM wants `FULL_DECODE_ONLY`, but FULL mode is exactly what trips the SM120 bookkeeping limit. In `0.19.1rc1`, the graph capture may have used a different (smaller) default capture list or different graph structure that fit under the limit.

### H5: WSL2 TDR watchdog timeout
**Evidence: LOW (P=0.10)**

WSL2 TDR limit is typically 2s. At C=32, 30-layer Gemma4 with FusenCache takes ~2ms/step (from §4.1 of `mega_graph_cooperative_kernel.md`). Even at C=512, a single step at ~40ms is well under 2s. `cudaErrorIllegalInstruction` is not the TDR signature (TDR produces `cudaErrorDeviceLostOrReset`). This hypothesis is rejected.

---

## §3 — Recommended Fix Sequence (cheapest first)

### Fix 1 (Config-only, 2 lines, effort ~5 min): Cap capture sizes at C=32
**Edit `launch_fusen_gemma4_t1b.sh`:**

Change the last line of the docker command from:
```bash
  -cc.mode none \
  -cc.cudagraph_mode full
```
To:
```bash
  -cc.mode none \
  -cc.cudagraph_mode full \
  -cc.cudagraph_capture_sizes '[1,2,4,8,16,32]'
```

This limits graph capture to 6 small graphs (max 32 tokens each), 450 nodes/graph × 6 = 2,700 total nodes. The SM120 bookkeeping limit is not publicly documented, but the crash pattern suggests it manifests when replaying graphs with large (C>32) batch dimensions. At C=32 this is the same size that previously worked. For C>32 the scheduler falls back to eager — which at C=64 with async scheduling may still achieve 2,000-3,000 tok/s vs piecewise's 751 tok/s. Test first.

**Risk:** If the crash is Triton-related (H3) and not bookkeeping-related, this fix won't help because the captured graph still uses the Triton kernel.

### Fix 2 (Config-only, 1 line, effort ~5 min): Switch to FULL_DECODE_ONLY and disable the override patch
If Fix 1 doesn't land: edit `fusen_kv/plugin.py:_patch_cudagraph_mode_override` to not re-assert FULL when vLLM resolves to FULL_DECODE_ONLY. Comment out line ~341-349 (the `if requested_mode == FULL and result == FULL_DECODE_ONLY` block). This lets vLLM use decode-only graphs, which are smaller and may avoid the SM120 limit while still capturing C=1..256 decode graphs.

**Risk:** FULL_DECODE_ONLY was noted as causing "3.4x throughput regression at C=128" in the plugin comment (line 308). This is likely the mechanism that makes FULL mode worth recovering.

### Fix 3 (Code change, ~30 lines, effort ~1 hr): Verify C++ kernel presence and conditional Triton fallback guard
Check whether `fusencache_decode.so` is present in the container at `/tmp/build_fusencache/`. If absent, `_HAS_CPP_DECODE=False` and the backend falls back to Triton with `_cudagraph_support=NEVER`. The `_patch_cudagraph_mode_override` then forces FULL mode with an incompatible Triton kernel — the exact Discovery #57 scenario.

**Diff (do not apply):**
```python
# fusen_kv/plugin.py: in _patch_cudagraph_mode_override, add guard:
# If we don't have the C++ kernels, FULL mode is NOT safe (Triton crashes).
# Only suppress the downgrade when C++ decode+store are both present.
from fusen_kv.backend import _HAS_CPP_DECODE, _HAS_CPP_STORE
if not (_HAS_CPP_DECODE and _HAS_CPP_STORE):
    logger.warning(
        "FusenKV: C++ kernels not available — NOT overriding cudagraph mode "
        "downgrade (Triton is not CUDA graph safe on SM120)"
    )
    return  # exit _patch_cudagraph_mode_override early, allow vLLM to downgrade
```

This would cause graceful degradation to FULL_DECODE_ONLY or PIECEWISE when C++ kernels are absent, rather than forcing FULL with a crashing Triton kernel.

### Fix 4 (Code change, ~50 lines, effort ~2 hrs): Bisect to identify crashing kernel
Enable `FUSEN_SYNC=1` and `FUSEN_DEBUG=1`, then run at C=33 (the first failing batch). The sync-after-each-kernel path in `backend.py:1103-1106` and `1162-1164` will pinpoint whether the crash is in `store_fn` or `decode_fn`. If it's `decode_fn` + Triton, confirms H3; if it faults before the first kernel, it's a graph-launcher issue (H1).

---

## §4 — Effort and P(lands) Assessment

| Fix | Effort | P(eliminates crash) | P(recovers 4,489 peak) |
|---|---|---|---|
| Fix 1: cap capture sizes at C=32 | 5 min | 0.40 | 0.30 |
| Fix 2: allow FULL_DECODE_ONLY | 10 min | 0.35 | 0.15 |
| Fix 1 + Fix 3 (C++ kernel guard) | 1 hr | 0.55 | 0.35 |
| Full bisect + targeted fix | 3-4 hrs | 0.65 | 0.40 |
| Accept piecewise + pivot | 0 min | 1.00 (no crash) | 0.00 |

**Calibration note (P11):** This is Category 4 (silicon-level limit, literature-grounded). The prior P(lands) was 0.10-0.20. After this deep-dive:
- H1 (bookkeeping) and H3 (Triton kernel incompatibility) are both viable and partially separable with the C++ kernel presence check.
- The version mismatch (H4) means the 4,489 peak was on different vLLM internals; even if the crash is fixed, the graph capture behavior may have changed enough that peak throughput is lower on the current image.
- Revised P(recovers 4,489 exactly): **0.15**. Revised P(improves over 751 tok/s): **0.40**.

---

## §5 — Verdict: Is 4,489 Recoverable?

**Probably not at exactly 4,489 tok/s. The floor is likely higher than 751 tok/s if Fix 1 lands.**

The 4,489 peak was measured on `vllm 0.19.1rc1` with a different CUDA graph infrastructure. The current image (`0.1.dev100+gc0c98b8b9`) has a substantially more complex `resolve_cudagraph_mode_and_sizes` function and the `_patch_cudagraph_mode_override` plugin fight that didn't exist in the original. Recovering the exact peak would require either:
1. Rolling back to `vllm 0.19.1rc1` image (archived in `_archive/vllm-0.18.1-patched-backup/`, close but not identical)
2. OR fixing the SM120 graph replay crash on the current image

Fix 1 (cap capture sizes at C=32) is the lowest-risk test. If the crash is bookkeeping-driven, C=32-capped full-mode graphs should work and yield some improvement over piecewise 751 tok/s — probably in the 2,000-3,500 range for C=32 (the original pre-T1B peak was 4,489 at C=32, but the new vLLM overhead may reduce this).

**Recommendation:**
1. **Run Fix 1 first** — 5-minute edit to `launch_fusen_gemma4_t1b.sh`, bench at C=4/8/16/32. If crash-free and ≥2,000 tok/s at C=32: partial win, log as `W6_T1B_capped_full_mode_PASS`.
2. **If Fix 1 fails at all C:** the crash is Triton-kernel (not bookkeeping). Apply Fix 3, recheck whether `fusencache_decode.so` is present in the container. If absent, the C++ kernel rebuild path is required (analogous to the LMCache SM120 rebuild — documented in `KILL_PATTERNS.md §4`).
3. **If both fail:** accept piecewise 751 tok/s as the ceiling for this vLLM version. Redirect effort to same-family spec decode (#16, projected +98% aggregate, P=0.80) which has higher EV and no silicon constraint.

**Recommended launch script edit for Fix 1** (`launch_fusen_gemma4_t1b.sh`, add one line):
```bash
  -cc.cudagraph_capture_sizes '[1,2,4,8,16,32]' \
```
Insert before the final closing `"` of the docker command, after `-cc.cudagraph_mode full`.
