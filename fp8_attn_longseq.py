"""Longer seq_len sweep to verify the 1.01x ratio isn't a launch-overhead artifact."""
import torch, flashinfer
from flashinfer.decode import trtllm_batch_decode_with_kv_cache

device = torch.device('cuda:0')

def sync_time_ms(fn, iters=200, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        start[i].record(); fn(); end[i].record()
    torch.cuda.synchronize()
    lat = [start[i].elapsed_time(end[i]) for i in range(iters)]
    lat.sort()
    return lat[len(lat)//2]

for seq_len in (4096, 8192, 16384):
    for hd in (256, 128):
        num_kv, num_q = 2, 8
        page_size = 16
        B=1
        num_pages = seq_len // page_size
        q = torch.randn(B, num_q, hd, dtype=torch.bfloat16, device=device)
        results = {}
        for tag, dt in (('BF16', torch.bfloat16), ('FP8', torch.float8_e4m3fn)):
            k = torch.randn(num_pages+1, num_kv, page_size, hd, dtype=torch.bfloat16, device=device).to(dt)
            v = torch.randn(num_pages+1, num_kv, page_size, hd, dtype=torch.bfloat16, device=device).to(dt)
            bt = torch.arange(num_pages, dtype=torch.int32, device=device).view(1,-1)
            sl = torch.tensor([seq_len], dtype=torch.int32, device=device)
            ws = torch.empty(256*1024*1024, dtype=torch.uint8, device=device)
            def _run():
                return trtllm_batch_decode_with_kv_cache(
                    query=q, kv_cache=(k,v), workspace_buffer=ws, block_tables=bt,
                    seq_lens=sl, max_seq_len=seq_len,
                    bmm1_scale=1.0/(hd**0.5), bmm2_scale=1.0, kv_layout='HND', backend='auto')
            _run()
            ms = sync_time_ms(_run)
            kb = 1 if dt==torch.float8_e4m3fn else 2
            bytes_ = 2*seq_len*num_kv*hd*kb
            gbs = bytes_/(ms*1e-3)/1e9
            print(f'hd={hd} seq={seq_len} {tag}: {ms*1000:7.1f}us {gbs:6.0f} GB/s {100*gbs/1792:5.1f}%')
            results[tag] = ms
        r = results['BF16']/results['FP8']
        print(f'  => FP8 speedup vs BF16 @ hd={hd} seq={seq_len}: {r:.2f}x')
