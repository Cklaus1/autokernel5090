"""Quick sanity check for swa_decode_attention on the Gemma4 26B shape.

Exercises the kernel against:
  - a non-contiguous K/V view (as produced by kv_cache[:, 0] inside the vLLM
    FlashInfer backend), to verify the plugin's zero-copy slice is safe.
Runs on CUDA_VISIBLE_DEVICES=1 only.
"""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kernels', 'triton'))

import torch
from swa_decode import swa_decode_attention, bf16_swa_decode_ref


def main():
    torch.manual_seed(0)
    batch, num_q, num_kv, head_dim, window = 8, 16, 8, 256, 1024
    seq_len = 4096
    page_size = 16

    q = torch.randn(batch, num_q, head_dim, device='cuda', dtype=torch.bfloat16) * 0.1
    n_pages_per_seq = (seq_len + page_size - 1) // page_size
    total_pages = batch * n_pages_per_seq + 4
    k_cache = (torch.randn(total_pages, page_size, num_kv, head_dim,
                           dtype=torch.bfloat16, device='cuda') * 0.1).contiguous()
    v_cache = (torch.randn(total_pages, page_size, num_kv, head_dim,
                           dtype=torch.bfloat16, device='cuda') * 0.1).contiguous()
    block_table = torch.zeros(batch, n_pages_per_seq, dtype=torch.int32, device='cuda')
    for b in range(batch):
        for i in range(n_pages_per_seq):
            block_table[b, i] = b * n_pages_per_seq + i
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device='cuda')

    kv = torch.stack([k_cache, v_cache], dim=1)  # [pages, 2, page, kv, d]
    k_view = kv[:, 0]
    v_view = kv[:, 1]
    print('k_view.is_contiguous():', k_view.is_contiguous(), 'stride:', k_view.stride())

    out_triton = swa_decode_attention(q, k_view, v_view, block_table, seq_lens, window=window)
    out_ref = bf16_swa_decode_ref(q, k_cache, v_cache, block_table, seq_lens, window=window)
    cos = torch.nn.functional.cosine_similarity(
        out_triton.float().reshape(-1).unsqueeze(0),
        out_ref.float().reshape(-1).unsqueeze(0)).item()
    max_abs = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f'Gemma4 shape batch={batch} seq={seq_len} win={window} head_dim={head_dim}')
    print(f'  cos={cos:.6f}, max_abs={max_abs:.6f}')
    ok = cos > 0.999 and max_abs < 5e-3
    print(f'  PASS' if ok else '  FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
