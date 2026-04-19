"""FP8 vs BF16 attention remeasure on FlashInfer 0.6.7 TRTLLM SM120 backend.
Discovery #37 said FP8 can't beat FA2 BF16 on SM120 due to KV layout.
This microbench tests if FlashInfer 0.6.7's new TRTLLM SM120 backend fixed it.
Shape: head_dim=256, num_kv_heads=2, num_q_heads=8, seq_len=2048, B=1.
"""
import torch
import time
import flashinfer
from flashinfer.decode import (
    single_decode_with_kv_cache,
    trtllm_batch_decode_with_kv_cache,
)

torch.manual_seed(0)
device = torch.device("cuda:0")

ITERS = 200
WARMUP = 20
PEAK_BW_GB = 1792.0  # RTX PRO 6000 Blackwell Max-Q HBM peak


def sync_time_ms(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    lat = []
    # Use cuda events for precision
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record()
        fn()
        end[i].record()
    torch.cuda.synchronize()
    lat = [start[i].elapsed_time(end[i]) for i in range(iters)]  # ms
    lat.sort()
    return lat[len(lat) // 2]


def bench_single_decode(head_dim, num_kv, num_q, seq_len, kv_dtype, q_dtype=torch.bfloat16):
    q = torch.randn(num_q, head_dim, dtype=q_dtype, device=device)
    k_bf = torch.randn(seq_len, num_kv, head_dim, dtype=torch.bfloat16, device=device)
    v_bf = torch.randn(seq_len, num_kv, head_dim, dtype=torch.bfloat16, device=device)
    if kv_dtype == torch.float8_e4m3fn:
        k = k_bf.to(torch.float8_e4m3fn)
        v = v_bf.to(torch.float8_e4m3fn)
        k_scale = 1.0
        v_scale = 1.0
    else:
        k = k_bf.to(kv_dtype)
        v = v_bf.to(kv_dtype)
        k_scale = None
        v_scale = None

    def _run():
        return single_decode_with_kv_cache(
            q, k, v,
            kv_layout="NHD",
            k_scale=k_scale,
            v_scale=v_scale,
        )

    out = _run()
    assert out.shape == (num_q, head_dim), out.shape
    return sync_time_ms(_run)


def bench_trtllm_batch_decode(head_dim, num_kv, num_q, seq_len, kv_dtype, page_size=16):
    B = 1
    num_pages = (seq_len + page_size - 1) // page_size
    max_num_pages = num_pages * B + 1

    q = torch.randn(B, num_q, head_dim, dtype=torch.bfloat16, device=device)

    if kv_dtype == torch.float8_e4m3fn:
        cache_dtype = torch.float8_e4m3fn
    else:
        cache_dtype = kv_dtype

    # HND layout: [num_pages, num_kv, page_size, head_dim]
    k_cache = torch.randn(max_num_pages, num_kv, page_size, head_dim,
                          dtype=torch.bfloat16, device=device).to(cache_dtype)
    v_cache = torch.randn(max_num_pages, num_kv, page_size, head_dim,
                          dtype=torch.bfloat16, device=device).to(cache_dtype)

    block_tables = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, num_pages)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

    bmm1_scale = 1.0 / (head_dim ** 0.5)
    bmm2_scale = 1.0

    def _run():
        return trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=(k_cache, v_cache),
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
            kv_layout="HND",
            backend="auto",
        )

    out = _run()
    return sync_time_ms(_run)


def bytes_moved(head_dim, num_kv, seq_len, kv_dtype):
    kbytes = 1 if kv_dtype == torch.float8_e4m3fn else 2
    return 2 * seq_len * num_kv * head_dim * kbytes


def pct_bw(ms, bytes_):
    gbs = bytes_ / (ms * 1e-3) / 1e9
    return gbs, 100.0 * gbs / PEAK_BW_GB


def run_config(tag, head_dim, num_kv, num_q, seq_len):
    print(f"\n=== {tag}: head_dim={head_dim}, num_kv={num_kv}, num_q={num_q}, seq_len={seq_len} ===")

    results = {}

    try:
        ms = bench_single_decode(head_dim, num_kv, num_q, seq_len, torch.bfloat16)
        b = bytes_moved(head_dim, num_kv, seq_len, torch.bfloat16)
        gbs, pct = pct_bw(ms, b)
        print(f"single_decode BF16:  {ms*1000:7.1f} us  {gbs:6.0f} GB/s  {pct:5.1f}% peak  ({b/1e6:.2f} MB)")
        results["single_bf16"] = (ms, gbs, pct)
    except Exception as e:
        print(f"single_decode BF16 FAIL: {type(e).__name__}: {e}")
        results["single_bf16"] = None

    try:
        ms = bench_single_decode(head_dim, num_kv, num_q, seq_len, torch.float8_e4m3fn)
        b = bytes_moved(head_dim, num_kv, seq_len, torch.float8_e4m3fn)
        gbs, pct = pct_bw(ms, b)
        print(f"single_decode FP8:   {ms*1000:7.1f} us  {gbs:6.0f} GB/s  {pct:5.1f}% peak  ({b/1e6:.2f} MB)")
        results["single_fp8"] = (ms, gbs, pct)
    except Exception as e:
        print(f"single_decode FP8 FAIL: {type(e).__name__}: {e}")
        results["single_fp8"] = None

    try:
        ms = bench_trtllm_batch_decode(head_dim, num_kv, num_q, seq_len, torch.bfloat16)
        b = bytes_moved(head_dim, num_kv, seq_len, torch.bfloat16)
        gbs, pct = pct_bw(ms, b)
        print(f"trtllm_batch  BF16:  {ms*1000:7.1f} us  {gbs:6.0f} GB/s  {pct:5.1f}% peak")
        results["trtllm_bf16"] = (ms, gbs, pct)
    except Exception as e:
        print(f"trtllm_batch BF16 FAIL: {type(e).__name__}: {e}")
        results["trtllm_bf16"] = None

    try:
        ms = bench_trtllm_batch_decode(head_dim, num_kv, num_q, seq_len, torch.float8_e4m3fn)
        b = bytes_moved(head_dim, num_kv, seq_len, torch.float8_e4m3fn)
        gbs, pct = pct_bw(ms, b)
        print(f"trtllm_batch  FP8:   {ms*1000:7.1f} us  {gbs:6.0f} GB/s  {pct:5.1f}% peak")
        results["trtllm_fp8"] = (ms, gbs, pct)
    except Exception as e:
        print(f"trtllm_batch FP8 FAIL: {type(e).__name__}: {e}")
        results["trtllm_fp8"] = None

    print("--- Ratios (BF16 / FP8 => speedup from FP8) ---")
    if results.get("single_bf16") and results.get("single_fp8"):
        r = results["single_bf16"][0] / results["single_fp8"][0]
        print(f"  single_decode FP8 speedup vs BF16: {r:.2f}x")
    if results.get("trtllm_bf16") and results.get("trtllm_fp8"):
        r = results["trtllm_bf16"][0] / results["trtllm_fp8"][0]
        print(f"  trtllm_batch  FP8 speedup vs BF16: {r:.2f}x")

    fp8_opts = [v[0] for v in (results.get("single_fp8"), results.get("trtllm_fp8")) if v]
    bf16_opts = [v[0] for v in (results.get("single_bf16"), results.get("trtllm_bf16")) if v]
    if fp8_opts and bf16_opts:
        best_fp8 = min(fp8_opts)
        best_bf16 = min(bf16_opts)
        ratio = best_bf16 / best_fp8
        print(f"  BEST FP8 = {best_fp8*1000:.1f} us  BEST BF16 = {best_bf16*1000:.1f} us  ratio = {ratio:.2f}x")
        results["_best_ratio"] = ratio
        results["_best_fp8_us"] = best_fp8 * 1000
        results["_best_bf16_us"] = best_bf16 * 1000

    return results


def main():
    print("FlashInfer version:", flashinfer.__version__)
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
    print("Device:", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print(f"Compute capability: {cap[0]}.{cap[1]}")

    results_hd256 = run_config("head_dim=256 (global layer)", 256, 2, 8, 2048)
    results_hd128 = run_config("head_dim=128 (sliding layer)", 128, 2, 8, 2048)

    print("\n=== SUMMARY ===")
    for tag, r in (("hd=256", results_hd256), ("hd=128", results_hd128)):
        if "_best_ratio" in r:
            print(f"{tag}: FP8 {r['_best_fp8_us']:.1f}us vs BF16 {r['_best_bf16_us']:.1f}us = {r['_best_ratio']:.2f}x")

    ratios = [r.get("_best_ratio", 0) for r in (results_hd256, results_hd128)]
    best = max(ratios) if ratios else 0
    if best >= 1.3:
        verdict = "PASS"
    elif best <= 1.1:
        verdict = "FAIL"
    else:
        verdict = "MARGINAL"
    print(f"\nBest ratio across configs: {best:.2f}x => {verdict}")
    print(f"Discovery #37 {'NO LONGER HOLDS' if verdict=='PASS' else 'STILL HOLDS'} on FlashInfer 0.6.7")


if __name__ == "__main__":
    main()
