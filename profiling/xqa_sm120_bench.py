"""
XQA SM120 Benchmark: Tests FlashInfer XQA kernel on RTX 5090 (SM120)
vs standard FA2 BatchDecodeWithPagedKVCacheWrapper.

Run inside vllm-built container:
  docker run --rm --gpus all -v /root/projects/autokernel/profiling:/prof vllm-built python3 /prof/xqa_sm120_bench.py

Findings: XQA kernel WORKS on SM120 but is 2-100x SLOWER than FA2 for
Gemma4's GQA 8:1 pattern. XQA is optimized for datacenter GPUs (B200)
with different memory hierarchy. head_dim=512 is NOT supported (max 256).
"""
import torch
import time
from flashinfer.xqa import xqa
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

torch.manual_seed(42)

def benchmark_xqa_vs_fa2(batch_size, num_q_heads, num_kv_heads, head_dim, seq_len,
                          page_size=16, warmup=50, iters=200):
    num_pages_per_seq = seq_len // page_size
    total_pages = batch_size * num_pages_per_seq + 100

    query_xqa = torch.randn(batch_size, 1, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    k_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v_cache = torch.randn(total_pages, page_size, num_kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')

    block_tables = torch.zeros(batch_size, num_pages_per_seq, dtype=torch.int32, device='cuda')
    for i in range(batch_size):
        block_tables[i] = torch.arange(i * num_pages_per_seq, (i + 1) * num_pages_per_seq, dtype=torch.int32)

    seq_lens_xqa = torch.full((batch_size, 1), seq_len, dtype=torch.uint32, device='cuda')
    output_xqa = torch.empty(batch_size, 1, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    semaphores = torch.zeros(64 * 1024, dtype=torch.uint32, device='cuda')
    scale = 1.0 / (head_dim ** 0.5)

    # XQA benchmark
    for _ in range(warmup):
        xqa(q=query_xqa, k_cache=k_cache, v_cache=v_cache, page_table=block_tables,
            seq_lens=seq_lens_xqa, output=output_xqa, workspace_buffer=workspace,
            semaphores=semaphores, num_kv_heads=num_kv_heads, page_size=page_size,
            q_scale=scale, kv_scale=1.0)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        xqa(q=query_xqa, k_cache=k_cache, v_cache=v_cache, page_table=block_tables,
            seq_lens=seq_lens_xqa, output=output_xqa, workspace_buffer=workspace,
            semaphores=semaphores, num_kv_heads=num_kv_heads, page_size=page_size,
            q_scale=scale, kv_scale=1.0)
    torch.cuda.synchronize()
    xqa_time = (time.perf_counter() - start) / iters * 1e6

    # FA2 benchmark
    query_fi = torch.randn(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    kv_cache_fi = torch.stack([k_cache, v_cache], dim=1)
    workspace_fi = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device='cuda')
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_fi)
    indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device='cuda') * num_pages_per_seq
    indices = torch.arange(0, batch_size * num_pages_per_seq, dtype=torch.int32, device='cuda')
    last_page_len = torch.full((batch_size,), page_size, dtype=torch.int32, device='cuda')

    try:
        wrapper.plan(indptr, indices, last_page_len,
                     num_qo_heads=num_q_heads, num_kv_heads=num_kv_heads,
                     head_dim=head_dim, page_size=page_size,
                     q_data_type=torch.bfloat16, data_type=torch.bfloat16)
        output_fi = torch.empty(batch_size, num_q_heads, head_dim, dtype=torch.bfloat16, device='cuda')
        for _ in range(warmup):
            wrapper.run(query_fi, kv_cache_fi, out=output_fi)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            wrapper.run(query_fi, kv_cache_fi, out=output_fi)
        torch.cuda.synchronize()
        fa2_time = (time.perf_counter() - start) / iters * 1e6
    except Exception as e:
        fa2_time = float('inf')
        print(f'  FA2 error: {e}')

    speedup = fa2_time / xqa_time if xqa_time > 0 else 0
    print(f'  B={batch_size:3d} H={num_q_heads:2d}/{num_kv_heads} D={head_dim:3d} S={seq_len:5d}'
          f' | XQA: {xqa_time:7.1f}us  FA2: {fa2_time:7.1f}us  Speedup: {speedup:.2f}x')
    return xqa_time, fa2_time


if __name__ == '__main__':
    print('=== XQA vs FA2 Decode Benchmark on SM120 (RTX 5090) ===')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print()

    print('--- Gemma4 Sliding Window (head_dim=256, GQA 8:1) ---')
    for bs in [1, 4, 16, 64]:
        for seq_len in [1024, 4096, 8192]:
            benchmark_xqa_vs_fa2(bs, 8, 1, 256, seq_len)

    print()
    print('--- head_dim=512 (Gemma4 Global) ---')
    for bs in [1, 4]:
        try:
            benchmark_xqa_vs_fa2(bs, 8, 1, 512, 1024)
        except Exception as e:
            print(f'  B={bs} D=512: FAILED - {e}')
