# FP8 Decode Verify Bench Plan
**Tag:** W6_fp8_decode_verify_prep  
**Date:** 2026-04-18  
**Question:** Is the banked T2-I +71% throughput real, or is the custom Triton split-K FP8 decode kernel silently falling back to FlashInfer on every request?

---

## Background

W5_A1 instrumented `patches/fp8_decode_monkey_patch.py` with:
- WARNING-level init banner at `apply_patch()` time
- Per-request WARNING log at every FIDecode fallback (increments `_fallback_fire_count`)
- `atexit` banner printing total fallback count

The suspected failure mode: on SM120, `is_device_capability(100)` returns `False` → vLLM selects `FIDecode` metadata → `patched_forward` hits the FIDecode branch (lines 362-402) → falls through to `decode_wrapper.run()` → custom Triton kernel never executes. If this is happening on every decode request, the +71% T2-I result was measured while FlashInfer was doing all the work.

---

## Pre-bench hygiene (KILL_PATTERNS §3)

Before launching either mode:

```bash
# Clean GPU state — expect ≤1 GiB residual, no peer compute-apps
nvidia-smi

# Detect cross-GPU pid bleed (§P4)
nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
  | awk -F, '{print $1}' | sort | uniq -c \
  | awk '$1 > 1 {print "WARN: pid " $2 " on " $1 " GPUs"}'
# Must be empty before launching.

# Kill any leftover containers
docker ps | grep vllm
```

---

## Step 1 — Launch PATCHED mode

```bash
MODE=patched PORT=8009 ./launch_gemma4_fp8_decode_verify.sh
```

Wait for server ready:
```bash
until curl -sf http://localhost:8009/health; do sleep 5; done
echo "Server up"
```

Verify init banner (appears before first request):
```bash
docker logs vllm-fp8-decode-verify-patched 2>&1 | grep '\[FP8-decode\]'
# Expected lines:
#   [FP8-decode] INIT: patch applied to FlashInferImpl.forward ...
#   [FP8-decode] AutoKernel FP8 decode patch applied ...
# If these are absent: PYTHONSTARTUP did not fire — check container env vars.
```

Verify PYTHONSTARTUP fired:
```bash
docker exec vllm-fp8-decode-verify-patched env | grep -E 'PYTHONSTARTUP|AUTOKERNEL_FP8_DECODE'
# Expected:
#   PYTHONSTARTUP=/tmp/autokernel_fp8_decode_startup.py
#   AUTOKERNEL_FP8_DECODE=1
```

---

## Step 2 — Send 200 decode requests

Use the existing bench script, targeting port 8009:

```bash
PORT=8009 python3 bench_gemma4_nvfp4.py --num-requests 200 --max-tokens 128 --concurrency 8
```

If `bench_gemma4_nvfp4.py` doesn't accept a port flag, patch it inline or use:

```bash
python3 - <<'EOF'
import openai, time, concurrent.futures

client = openai.OpenAI(base_url="http://localhost:8009/v1", api_key="x")
MODEL = "gemma-4-26B-A4B-it-NVFP4"
PROMPT = "Summarize the capital cities of Europe in one sentence."
N = 200

def req(_):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=64,
    )
    return r.usage.completion_tokens

t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    tokens = list(ex.map(req, range(N)))
elapsed = time.time() - t0
print(f"{N} requests in {elapsed:.1f}s  "
      f"({sum(tokens)/elapsed:.0f} tok/s)  "
      f"total={sum(tokens)} tokens")
EOF
```

While requests are in flight, watch for FALLBACK warnings:
```bash
docker logs -f vllm-fp8-decode-verify-patched 2>&1 | grep '\[FP8-decode\] FALLBACK'
```

---

## Step 3 — Trigger atexit banner and read summary

```bash
docker kill -s SIGTERM vllm-fp8-decode-verify-patched
sleep 3
docker logs vllm-fp8-decode-verify-patched 2>&1 | grep 'FALLBACK SUMMARY'
```

---

## Step 4 — Launch BASELINE and record throughput

```bash
MODE=baseline PORT=8010 ./launch_gemma4_fp8_decode_verify.sh
until curl -sf http://localhost:8010/health; do sleep 5; done
PORT=8010 python3 bench_gemma4_nvfp4.py --num-requests 200 --max-tokens 128 --concurrency 8
```

Record both throughput numbers and the fallback count from step 3.

---

## Interpretation rubric

| FALLBACK SUMMARY result | Meaning | Action |
|---|---|---|
| `fallback_fire_count=0` | Kernel fired on all decode requests. +71% is attributable to the custom Triton split-K path (at least structurally — verify patched > baseline throughput). | Celebrate. File result. Update results.tsv with REAL tag. |
| `N time(s)` where N ≈ 200 | FIDecode path selected on every request. Custom kernel never ran. +71% banked result is the FlashInfer baseline mislabeled. | Update results.tsv: mark T2-I banked claim as UNVERIFIED/PHANTOM. Root cause: SM120 `is_device_capability(100)==False` → FIDecode. Fix path: port block_tables/seq_lens extraction from FIDecode wrapper plan, OR force TRTLLMDecode metadata selection. |
| `0 < N < 200` (partial) | Mixed metadata types. Some requests hit TRTLLMDecode (kernel fired), others hit FIDecode (fallback). | Investigate: which request shapes / batch sizes produce FIDecode vs TRTLLMDecode? Log `type(attn_metadata.decode).__name__` on first few requests. May be prefill-decode mix scheduling. |
| Init banner absent entirely | PYTHONSTARTUP failed silently OR patch was never imported. | Check: `docker exec CONTAINER env | grep PYTHONSTARTUP`; inspect `/tmp/autokernel_fp8_decode_startup.py` inside container. Fallback to strategy (c): inject `import fp8_decode_monkey_patch` directly into INNER_SCRIPT before `exec python3`. |

---

## Optional: assert-mode run (hard failure on any fallback)

To confirm the FIDecode diagnosis, run with assert mode — this makes the server crash on the first FIDecode request rather than silently routing:

```bash
AUTOKERNEL_FP8_DECODE_ASSERT_NO_FALLBACK=1 MODE=patched PORT=8011 \
  ./launch_gemma4_fp8_decode_verify.sh
```

If the container dies on first decode request with `RuntimeError: [FP8-decode] ASSERT_NO_FALLBACK: decode metadata is FIDecode`, the W5_A1 hypothesis is confirmed.

---

## Fix path (if fallback_count ≈ 200)

The FIDecode branch lacks `block_tables` / `seq_lens` because they're baked into the FlashInfer wrapper plan. Options:

1. **Extract from wrapper plan:** inspect `attn_metadata.decode.wrapper` internals for a `plan_info` or `indptr` that can reconstruct seq_lens. High-effort, version-fragile.
2. **Force TRTLLMDecode metadata:** investigate whether TRTLLM path can be enabled on SM120 by overriding the `is_device_capability(100)` check — e.g., `vllm.v1.attention.backends.flashinfer` monkey-patch to return `True` for the capability check.
3. **Rewrite kernel to accept FlashInfer wrapper API:** call into the Triton kernel from the FIDecode branch directly via the wrapper's `run()` hook. Most invasive.

Option 2 is the shortest path: a one-liner `is_device_capability` monkey-patch in the same `fp8_decode_monkey_patch.py` or a companion patch, forcing TRTLLMDecode selection.

---

## Risks

1. **Monkey-patch + existing plugin conflicts:** `AUTOKERNEL_SWA_SPARSE=0` and `AUTOKERNEL_FUSED_SHUFFLE_QUANT=0` are set in launcher to disable other patches. If `VLLM_PLUGINS` still loads SWA or T2-N, their `FlashInferImpl` modifications may compose unexpectedly with `fp8_decode_monkey_patch`. The launcher explicitly `unset VLLM_PLUGINS` to eliminate this.

2. **PYTHONSTARTUP timing:** PYTHONSTARTUP fires for every Python interpreter invocation, including subprocesses (EngineCore worker). This is intentional for the main server process but may produce spurious log lines from vLLM's worker subprocesses that import vllm but don't serve decode requests. Filter logs by main process PID if needed.

3. **PYTHONSTARTUP not honoured in all exec contexts:** some Python launchers use `-c` or `-m` which do not read PYTHONSTARTUP. Verify by checking init banner in logs. Fallback: write `import fp8_decode_monkey_patch` directly into the INNER_SCRIPT before `exec python3`, which is immune to this issue.

4. **SM120 `kv_cache.dtype` mismatch:** `fp8_decode_monkey_patch.py` asserts `k_cache.dtype == torch.float8_e4m3fn`. If the container's FlashInfer uses `float8_e5m2` for this model, the assert will fire. Watch for `AssertionError` in logs.

---

*Plan authored by W6_fp8_decode_verify_prep agent. Do not modify `patches/fp8_decode_monkey_patch.py` — already instrumented by W5_A1.*
