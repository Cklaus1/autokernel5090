# T2-N Ablation Harness — Usage & Interpretation Guide

**Tag:** `W8_t2n_ablation_harness`
**Script:** `/home/cklaus/projects/autokernel/bench_t2n_ablation.py`
**Output:** `/home/cklaus/projects/autokernel/bench_t2n_ablation_results.json`

---

## What it does

`bench_t2n_ablation.py` launches Qwen3-30B-A3B NVFP4 containers with
systematically varied env gates, runs the T2-H concurrency bench after
each launch, verifies banner visibility, and writes a JSON + text report
showing the incremental throughput contribution of every gate.

This eliminates the "is the banner firing?" silent-failure class that
burned the Rank 3 investigation (W8 postmortem).

---

## One-liner (parent usage)

```bash
# Full additive sweep, default concurrencies 512/768/1024:
python3 /home/cklaus/projects/autokernel/bench_t2n_ablation.py

# Faster smoke test (fewer concurrency levels, shorter bench):
python3 /home/cklaus/projects/autokernel/bench_t2n_ablation.py \
  --concurrencies 512,1024 --max-tokens 64 --n-requests 128

# Leave-one-out only (diagnose which single gate is a regression):
python3 /home/cklaus/projects/autokernel/bench_t2n_ablation.py --mode leave_one_out

# Both sweeps (full audit):
python3 /home/cklaus/projects/autokernel/bench_t2n_ablation.py --mode both

# Subset of gates only (ordered; T2N_CORE must come first for plugin wiring):
python3 /home/cklaus/projects/autokernel/bench_t2n_ablation.py \
  --gates T2N_CORE,FUSED_NORM_V2,RANK2_QKNORM_ROPE,RANK3_UNSHUF_WSUM

# Dry-run: print inner container scripts without launching anything:
python3 /home/cklaus/projects/autokernel/bench_t2n_ablation.py --dry-run
```

---

## Known env gates (full list)

| Short key | Env var | Notes |
|---|---|---|
| `T2N_CORE` | `AUTOKERNEL_FUSED_SHUFFLE_QUANT` | T2-N core (Rank 1 shuffle+quant); requires the `.so` |
| `FUSED_NORM_V2` | `AUTOKERNEL_FUSED_NORM_FP4_QWEN3` | Fused-norm v2 plugin; requires the norm `.so` |
| `RANK2_QKNORM_ROPE` | `AUTOKERNEL_FUSED_QKNORM_ROPE` | Fused q_norm+k_norm+RoPE (Triton) |
| `RANK3_UNSHUF_WSUM` | `AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM` | Fused unshuffle+weighted-sum (Triton) |
| `RANK4_KV_CACHE` | `AUTOKERNEL_FUSED_KV_CACHE_UPDATE` | Future |
| `RANK5_POST_ATTN_ROUTER` | `AUTOKERNEL_FUSED_POST_ATTN_ROUTER` | Future |
| `RANK6_SHARED_MLP` | `AUTOKERNEL_FUSED_SHARED_EXPERT_MLP` | Future |
| `PERSISTENT_BUF` | `AUTOKERNEL_FUSED_PERSISTENT_BUF` | Fix A smart gate (interaction with RANK3) |

`VLLM_USE_FLASHINFER_MOE_FP4` is hardwired to `0` in `ALWAYS_ENV` for all
combos (required for T2-N to fire).

`AUTOKERNEL_FUSED_MIN_TOKENS` is hardwired to `1` in `ALWAYS_ENV`
(Fix B threshold; not swept because it is a numeric threshold, not a
binary on/off gate — parent can override via `--gates` omission or by
editing `ALWAYS_ENV` before running).

---

## Additive sweep — how to read the contribution table

The additive sweep starts from an all-OFF baseline and adds one gate at a
time in rank order:

```
baseline  (all OFF)                  → 18,000 tok/s   (example)
+ T2N_CORE                           → 20,500 tok/s   +2,500  +13.9%
+ FUSED_NORM_V2                      → 21,800 tok/s   +1,300   +6.3%
+ RANK2_QKNORM_ROPE                  → 22,300 tok/s     +500   +2.3%
+ RANK3_UNSHUF_WSUM                  → 24,600 tok/s   +2,300  +10.3%
+ RANK4_KV_CACHE      (future)       → 24,600 tok/s       +0   +0.0%
...
```

Each row's `delta_tok_s` is the marginal throughput added by enabling that
one additional gate on top of all previously-enabled gates.  Because gates
interact (e.g. T2-N must be ON for Rank 2/3 to have any code path to
reach), the order matters — the table is prescriptive of deployment order,
not a Shapley-value decomposition.

### Silent-not-firing flag

If a gate's banner is missing from `docker logs` after the first warmup
request, the harness flags `silent_not_firing` in the JSON and prints
`MISSING` in the Banner column.  In that case the `delta_tok_s` for that
gate is unreliable (the gate was not actually active).

Rule: **do not credit a gate with a positive delta unless its banner is
verified.**  A `MISSING` banner + positive delta means something else
changed (e.g. PERSISTENT_BUF interaction, as in the W8 Rank-3 postmortem).

---

## Leave-one-out sweep — how to read it

The LOO sweep starts from all-ON and removes one gate at a time.  Each
`delta_tok_s` is negative and shows the cost of losing that gate from the
full stack:

```
all_on                     → 25,000 tok/s
- T2N_CORE                 → 18,000 tok/s   -7,000  -28.0%
- FUSED_NORM_V2            → 23,200 tok/s   -1,800   -7.2%
- RANK3_UNSHUF_WSUM        → 22,000 tok/s   -3,000  -12.0%
```

LOO is more accurate for mature stacks (where gate interactions are small)
because it measures each gate's value when everything else is working.
Use `--mode both` to cross-check additive vs LOO deltas; large discrepancies
indicate gate interaction effects.

---

## Banner verification rule (§P1 discipline)

Every gate that has a `BANNERS` entry must emit its banner to `docker logs`
before the bench starts.  The harness sends a single warmup request after
`/health` returns 200, waits 2 s for log flush, then greps.

Banners are now emitted via `print(..., flush=True)` in addition to
`logger.info` so they survive vLLM's logger-filtering allowlist (W8
Rank-3 fix).  If a banner is missing:

1. Check `docker exec <name> env | grep AUTOKERNEL` — verify the env var
   is set.
2. Check `docker logs <name> | grep -E 'import failed|ModuleNotFound'` —
   likely a PYTHONPATH issue.
3. Check `VLLM_USE_FLASHINFER_MOE_FP4` — must be `0` for T2-N entry point
   to be reached.

---

## Expected runtime

| Sweep | Combos | Est. time |
|---|---|---|
| Additive (8 gates) | 10 | ~80 min |
| Leave-one-out (8 gates) | 9 | ~72 min |
| Both | 19 | ~152 min |
| Smoke (4 gates, fast bench) | 6 | ~30 min |

Each combo: ~4 min container startup + ~4 min bench at C=512/768/1024
with max_tokens=128, n_requests=256.

---

## Output JSON schema

```
{
  "harness_version": "W8_t2n_ablation_harness",
  "timestamp": "...",
  "config": { "mode", "gates", "concurrencies", "max_tokens", "n_requests" },
  "additive_sweep": [
    {
      "combo_id": "baseline" | "add_01_T2N_CORE" | ...,
      "gate_values": { "T2N_CORE": "0"|"1", ... },
      "active_gates": [...],
      "healthy": true|false,
      "banner_results": { "T2N_CORE": true|false, ... },
      "silent_not_firing": [...],
      "bench": {
        "model": "...",
        "peak_gen_tok_s": 24923.0,
        "results": [ { "concurrency", "gen_tok_s", "p50_latency_s", ... } ]
      }
    },
    ...
  ],
  "leave_one_out_sweep": [ ... ],
  "additive_contributions": [
    {
      "combo_id": "...",
      "newly_added_gate": "RANK3_UNSHUF_WSUM",
      "peak_gen_tok_s": 24923.0,
      "delta_tok_s": 2300.0,
      "delta_pct": 10.29,
      "silent_not_firing": [],
      "banner_verified": true
    },
    ...
  ],
  "leave_one_out_contributions": [ ... ]
}
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `health timeout` on all combos | GPU/container issue unrelated to gates | Check `docker ps`, `nvidia-smi`, port conflicts |
| Banner `MISSING` for `RANK3_UNSHUF_WSUM` | PYTHONPATH missing `/autokernel` | Verify `ALWAYS_ENV["PYTHONPATH"]` contains `/autokernel` |
| Banner `MISSING` for `T2N_CORE` | `VLLM_USE_FLASHINFER_MOE_FP4=1` overriding | Hardwired to 0 in `ALWAYS_ENV`; check for host-level env override |
| Negative delta for a gate that should be positive | Gate interaction (e.g. PERSISTENT_BUF regresses at certain batch sizes) | Run LOO sweep to isolate; check `AUTOKERNEL_FUSED_MIN_TOKENS` threshold |
| All combos show same throughput | CUDA graph capture invalidating patches | Check `-cc.cudagraph_mode` setting; try `none` |

---

## Files

- `bench_t2n_ablation.py` — harness (this tag: W8)
- `bench_t2n_ablation_results.json` — output (written after each run)
- `plans/t2n_ablation_harness.md` — this document
