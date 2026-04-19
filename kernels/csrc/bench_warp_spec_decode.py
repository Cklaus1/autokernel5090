#!/usr/bin/env python3
"""
Microbench: warp-specialized decode attention vs flashinfer single-decode.
B=1, seq_len=4096, head_dim=256, BF16 single-head. GPU 1 only, <60s.
"""
import os
import sys
import time
# Unbuffered stdout so we see incremental progress even on hang.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# GPU 1 only
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# venv flashinfer on python3
sys.path.insert(0, "/home/cklaus/projects/autokernel/.venv/lib/python3.12/site-packages")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
import build_warp_spec_decode as bw

HEAD_DIM = 256
SEQ_LEN  = 4096
# Multi-head to saturate the GPU (mimic Gemma4 decode: B*nheads ~= 128-512)
NUM_HEADS = 128
DTYPE = torch.bfloat16
PEAK_BW_GBS = 1792.0   # HBM3e RTX PRO 6000 Blackwell

def build_and_load():
    if not bw.load_library():
        bw.build_kernel()

def reference_decode(q, k, v, sm_scale):
    # q [H, d], k/v [H, seq, d]   or unbatched (H=1)
    if q.dim() == 1:
        s = (q.float() @ k.float().T) * sm_scale
        p = torch.softmax(s, dim=-1)
        return (p @ v.float()).to(q.dtype)
    # multi-head
    s = torch.einsum('hd,hsd->hs', q.float(), k.float()) * sm_scale
    p = torch.softmax(s, dim=-1)
    return torch.einsum('hs,hsd->hd', p, v.float()).to(q.dtype)

def time_it(fn, iters=100, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000.0   # us

def try_flashinfer_decode(q_h, k_seq, v_seq, sm_scale):
    """Multi-head single_decode. Returns (runnable, out tensor) or None."""
    try:
        import flashinfer
    except Exception as e:
        print(f"[WARN] flashinfer import failed: {e}")
        return None
    # single_decode_with_kv_cache layout:
    #   q [num_qo_heads, head_dim]
    #   k [kv_len, num_kv_heads, head_dim]  -- shared KV across heads
    # We need per-head independent KV to model "heads all load their own KV".
    # flashinfer's single_decode assumes GQA-shared KV, so for a fair
    # BW-bound compare we run num_heads independent calls. That would
    # sequentialize; instead we feed with num_qo_heads=num_kv_heads=H,
    # passing k[seq, H, d] permuted from our [H, seq, d].
    H = q_h.size(0)
    q_fi = q_h.contiguous()                           # [H, d]
    k_fi = k_seq.permute(1, 0, 2).contiguous()        # [seq, H, d]
    v_fi = v_seq.permute(1, 0, 2).contiguous()        # [seq, H, d]
    try:
        out_fi = flashinfer.single_decode_with_kv_cache(
            q_fi, k_fi, v_fi, sm_scale=sm_scale)
    except Exception as e:
        print(f"[WARN] flashinfer single_decode failed: {e}")
        return None
    def run():
        return flashinfer.single_decode_with_kv_cache(
            q_fi, k_fi, v_fi, sm_scale=sm_scale)
    return run, out_fi

def main():
    torch.manual_seed(0)
    build_and_load()

    device = torch.device("cuda")
    q = (torch.randn(NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device) * 0.1).contiguous()
    k = (torch.randn(NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=device) * 0.1).contiguous()
    v = (torch.randn(NUM_HEADS, SEQ_LEN, HEAD_DIM, dtype=DTYPE, device=device) * 0.1).contiguous()
    out_ws = torch.zeros(NUM_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)

    # --- Correctness (warp-spec vs torch float, on first 4 heads) ---
    torch.ops.warp_spec_decode.decode(out_ws, q, k, v, sm_scale)
    H_CHECK = min(4, NUM_HEADS)
    out_ref = reference_decode(q[:H_CHECK], k[:H_CHECK], v[:H_CHECK], sm_scale)
    err = (out_ws[:H_CHECK].float() - out_ref.float()).abs().max().item()
    rel = err / (out_ref.float().abs().max().item() + 1e-6)
    print(f"[CORR] max_abs_err={err:.4g}  rel={rel:.4g}")
    ws_correct = rel < 0.05     # BF16 accumulate tolerance

    # --- Warp-spec latency ---
    def run_ws():
        torch.ops.warp_spec_decode.decode(out_ws, q, k, v, sm_scale)
    ws_us = time_it(run_ws)

    # --- FlashInfer latency ---
    fi = try_flashinfer_decode(q, k, v, sm_scale)
    if fi is not None:
        run_fi, out_fi = fi
        fi_us = time_it(run_fi)
    else:
        run_fi, out_fi = None, None
        fi_us = float("nan")

    # --- BW calc ---
    # KV bytes moved (decode): K + V read = 2 * H * seq * head_dim * 2B
    kv_bytes = 2 * NUM_HEADS * SEQ_LEN * HEAD_DIM * 2
    bw_ws_gbs = kv_bytes / (ws_us * 1e-6) / 1e9
    pct_ws = bw_ws_gbs / PEAK_BW_GBS * 100.0
    if fi_us == fi_us:   # not NaN
        bw_fi_gbs = kv_bytes / (fi_us * 1e-6) / 1e9
        pct_fi = bw_fi_gbs / PEAK_BW_GBS * 100.0
        speedup = fi_us / ws_us
    else:
        bw_fi_gbs = pct_fi = float("nan")
        speedup = float("nan")

    print(f"\n=== Results (H={NUM_HEADS}, seq={SEQ_LEN}, d={HEAD_DIM}, BF16) ===")
    print(f"warp-spec: {ws_us:.2f} us   BW = {bw_ws_gbs:.1f} GB/s   ({pct_ws:.1f}% of {PEAK_BW_GBS:.0f} GB/s peak)")
    if fi_us == fi_us:
        print(f"flashinfer: {fi_us:.2f} us   BW = {bw_fi_gbs:.1f} GB/s   ({pct_fi:.1f}% of peak)")
        print(f"speedup: {speedup:.3f}x   (target FA2=93% BW)")
    print(f"corr: {'PASS' if ws_correct else 'FAIL'}  (rel={rel:.3g})")

    # --- Append TSV ---
    beats = pct_ws >= 93.0 if ws_correct else False
    status = "PASS" if (ws_correct and beats) else ("PASS" if ws_correct else "FAIL")
    # Per task: PASS/FAIL on correctness + BW% >= 93
    # But the ground rule says: if kernel-only BW < 93%, KILL for SM120
    verdict = "PASS" if (ws_correct and pct_ws >= 93.0) else "FAIL"
    desc = (f"BW% warp-spec={pct_ws:.1f} vs FA2 93%; fi={pct_fi:.1f}%; "
            f"speedup={speedup:.2f}x; "
            + ("meets BW floor -- gemma4 decode ~5-8% gain plausible" if verdict == "PASS"
               else "below FA2 93% BW floor; warp-spec gain doesn't translate on SM120a (no TMA, no WGMMA); KILL"))

    row = (
        f"warp_spec_decode\twarp_spec_attn\tkernel_bench\t0\t{ws_us:.1f}\t"
        f"{pct_ws:.1f}\t{speedup:.3f}\t{verdict}\t0\t{desc}\n"
    )
    results_path = "/home/cklaus/projects/autokernel/results.tsv"
    with open(results_path, "a") as f:
        f.write(row)
    print("\n[TSV] appended:")
    print(row.rstrip())
    return verdict


if __name__ == "__main__":
    v = main()
    print(f"\nVERDICT: {v}")
