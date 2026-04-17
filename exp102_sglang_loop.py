#!/usr/bin/env python3
"""
Experiment 102+: SGLang optimization loop for Qwen3.5-9B NVFP4 + DFlash
Autokernel method: one change per experiment, bench, keep/revert, log.

Experiment plan:
  102: NVFP4 baseline (no spec decode) — establish SGLang NVFP4 performance
  103: DFlash on NVFP4 — the big test
  104+: Tune parameters (batch size, mem fraction, cuda graphs, etc.)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from sglang_bench import launch_server, kill_server, bench_decode, bench_batch, log_result

PORT = 30000
MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
DRAFT = "z-lab/Qwen3.5-9B-DFlash"
LOG_FILE = "sglang_results.tsv"

# Track best results for keep/revert decisions
best_decode = 0
best_batch32 = 0
best_config = None


def run_experiment(exp_num, tag, server_args, desc, timeout=600):
    global best_decode, best_batch32, best_config

    print(f"\n{'#'*60}")
    print(f"# EXP {exp_num}: {tag}")
    print(f"# {desc}")
    print(f"# Args: {' '.join(server_args[-10:])}")
    print(f"{'#'*60}")

    proc, ready, log_path = launch_server(PORT, server_args, timeout=timeout)

    if not ready:
        print(f"  FAIL: Server did not start")
        kill_server(proc)
        time.sleep(5)
        log_result(exp_num, tag, 0, 0, None, "FAIL", f"{desc} (server failed to start)", LOG_FILE)
        return False, 0, 0, None

    try:
        # Decode benchmark
        print("  Running decode benchmark...")
        decode_tok_s, accept_len, _ = bench_decode(PORT)
        print(f"  Decode: {decode_tok_s:.1f} tok/s" + (f" (accept: {accept_len:.2f})" if accept_len else ""))

        # Batch benchmark
        print("  Running batch benchmark...")
        batch_results = bench_batch(PORT, batch_sizes=[1, 8, 32])
        batch32 = batch_results.get(32, {}).get("total_tok_s", 0)
        batch8 = batch_results.get(8, {}).get("total_tok_s", 0)

        for bs, r in sorted(batch_results.items()):
            accept_str = f", accept={r['accept']:.2f}" if r.get("accept") else ""
            print(f"    batch={bs}: {r['total_tok_s']:.0f} tok/s ({r['per_user']:.1f}/user{accept_str})")

        # Keep/revert decision
        improved = decode_tok_s > best_decode * 1.01 or batch32 > best_batch32 * 1.01
        if improved or best_decode == 0:
            status = "KEEP"
            if best_decode > 0:
                print(f"  KEEP: decode {best_decode:.1f} -> {decode_tok_s:.1f}, batch32 {best_batch32:.0f} -> {batch32:.0f}")
            best_decode = max(best_decode, decode_tok_s)
            best_batch32 = max(best_batch32, batch32)
            best_config = server_args.copy()
        else:
            status = "REVERT"
            print(f"  REVERT: decode {decode_tok_s:.1f} vs best {best_decode:.1f}, batch32 {batch32:.0f} vs best {best_batch32:.0f}")

        log_result(exp_num, tag, decode_tok_s, batch32, accept_len, status, desc, LOG_FILE)
        return True, decode_tok_s, batch32, accept_len

    finally:
        kill_server(proc)
        time.sleep(8)  # Wait for GPU memory to free


def main():
    global best_decode, best_batch32

    # Common args for all experiments
    base_args = [
        "--model-path", MODEL,
        "--dtype", "bfloat16",
        "--trust-remote-code",
    ]

    # ═══════════════════════════════════════════════════════════
    # EXP 102: NVFP4 baseline — no spec decode, triton backend
    # ═══════════════════════════════════════════════════════════
    args_102 = base_args + [
        "--attention-backend", "triton",
        "--mem-fraction-static", "0.85",
    ]
    run_experiment(102, "nvfp4_baseline", args_102,
                   "NVFP4 9B baseline on SGLang (no spec decode, triton backend)")

    # ═══════════════════════════════════════════════════════════
    # EXP 103: NVFP4 + DFlash — the main test
    # Need SGLANG_ENABLE_SPEC_V2=1 for spec decode with radix cache on Mamba models
    # ═══════════════════════════════════════════════════════════
    os.environ["SGLANG_ENABLE_SPEC_V2"] = "1"
    args_103 = base_args + [
        "--attention-backend", "triton",
        "--mem-fraction-static", "0.85",
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", DRAFT,
        "--mamba-scheduler-strategy", "extra_buffer",
        "--max-running-requests", "32",
    ]
    ok, _, _, _ = run_experiment(103, "nvfp4_dflash", args_103,
                   "NVFP4 9B + DFlash spec decode (block_size=16)", timeout=900)

    if not ok:
        # Fallback: disable radix cache + no_buffer
        args_103b = base_args + [
            "--attention-backend", "triton",
            "--mem-fraction-static", "0.85",
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", DRAFT,
            "--mamba-scheduler-strategy", "no_buffer",
            "--disable-radix-cache",
            "--max-running-requests", "32",
        ]
        ok, _, _, _ = run_experiment(103, "nvfp4_dflash_nobuf", args_103b,
                       "NVFP4 + DFlash no_buffer + disable-radix-cache", timeout=900)

    if not ok:
        # Fallback 2: smaller draft tokens
        args_103c = base_args + [
            "--attention-backend", "triton",
            "--mem-fraction-static", "0.80",
            "--speculative-algorithm", "DFLASH",
            "--speculative-draft-model-path", DRAFT,
            "--mamba-scheduler-strategy", "extra_buffer",
            "--max-running-requests", "16",
            "--speculative-num-draft-tokens", "8",
        ]
        run_experiment(103, "nvfp4_dflash_small", args_103c,
                       "NVFP4 + DFlash small (8 draft tokens, 16 reqs)", timeout=900)

    # ═══════════════════════════════════════════════════════════
    # EXP 104: Try max-running-requests=64
    # ═══════════════════════════════════════════════════════════
    if best_decode > 0:
        args_104 = list(best_config)
        # Replace max-running-requests
        if "--max-running-requests" in args_104:
            idx = args_104.index("--max-running-requests")
            args_104[idx+1] = "64"
        else:
            args_104 += ["--max-running-requests", "64"]
        run_experiment(104, "max_reqs_64", args_104,
                       "Increase max-running-requests to 64")

    # ═══════════════════════════════════════════════════════════
    # EXP 105: Try mem-fraction-static=0.90
    # ═══════════════════════════════════════════════════════════
    if best_config:
        args_105 = list(best_config)
        idx = args_105.index("--mem-fraction-static")
        args_105[idx+1] = "0.90"
        run_experiment(105, "mem_frac_90", args_105,
                       "Increase mem-fraction-static to 0.90")

    # ═══════════════════════════════════════════════════════════
    # EXP 106: Try speculative-num-draft-tokens=8 (smaller blocks)
    # ═══════════════════════════════════════════════════════════
    if best_config and any("DFLASH" in str(a) for a in best_config):
        args_106 = list(best_config) + ["--speculative-num-draft-tokens", "8"]
        run_experiment(106, "draft_tokens_8", args_106,
                       "Reduce draft tokens from 16 to 8 (less overhead per step)")

    # ═══════════════════════════════════════════════════════════
    # EXP 107: Try page-size=64
    # ═══════════════════════════════════════════════════════════
    if best_config:
        args_107 = list(best_config) + ["--page-size", "64"]
        run_experiment(107, "page_size_64", args_107,
                       "Page size 64 (reduces paging overhead)")

    # ═══════════════════════════════════════════════════════════
    # EXP 108: Try MTP instead of DFlash (Qwen3.5 has built-in MTP)
    # ═══════════════════════════════════════════════════════════
    args_108 = base_args + [
        "--attention-backend", "triton",
        "--mem-fraction-static", "0.85",
        "--speculative-algorithm", "EAGLE",  # MTP uses EAGLE algorithm in SGLang
        "--speculative-num-draft-tokens", "3",
        "--mamba-scheduler-strategy", "extra_buffer",
        "--disable-radix-cache",
        "--max-running-requests", "32",
    ]
    run_experiment(108, "mtp3_sglang", args_108,
                   "MTP 3 tokens on SGLang (compare vs DFlash)")

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EXPERIMENT LOOP COMPLETE")
    print(f"Best decode: {best_decode:.1f} tok/s")
    print(f"Best batch32: {best_batch32:.0f} tok/s")
    print(f"Best config: {best_config}")
    print("=" * 60)

    # Print results table
    if os.path.exists(LOG_FILE):
        print(f"\nFull results in {LOG_FILE}:")
        with open(LOG_FILE) as f:
            print(f.read())


if __name__ == "__main__":
    main()
