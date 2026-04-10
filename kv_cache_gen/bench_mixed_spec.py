"""Mixed-spec decode simulation: different KV cache specs for sliding vs global layers."""

import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec
from kv_cache_gen.generate import make_decode_fn, make_store_fn

# Gemma4 26B-A4B architecture
HIDDEN_SIZE = 2816
NUM_HEADS = 16
HEAD_DIM_SLIDING = 256
HEAD_DIM_GLOBAL = 512
NUM_KV_HEADS_SLIDING = 8
NUM_KV_HEADS_GLOBAL = 2
NUM_LAYERS = 30
MOE_INTERMEDIATE_SIZE = 704
NUM_EXPERTS = 128
TOP_K = 8
LAYER_TYPES = [
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
]

SPECS = {
    'k4v4kb64vb64': KVCacheSpec(name='k4v4kb64vb64', k_bits=4, k_sym_offset=7.5,
                                 k_scale_block=64, v_bits=4, v_sym_offset=7.5, v_scale_block=64),
    'k8v4kb32vb32': KVCacheSpec(name='k8v4kb32vb32', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=32, v_bits=4, v_sym_offset=7.5, v_scale_block=32),
    'k8v8kb64vb64': KVCacheSpec(name='k8v8kb64vb64', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=64, v_bits=8, v_sym_offset=127.5, v_scale_block=64),
    'k8v4kb16vb16': KVCacheSpec(name='k8v4kb16vb16', k_bits=8, k_sym_offset=127.5,
                                 k_scale_block=16, v_bits=4, v_sym_offset=7.5, v_scale_block=16),
}

# Mixed-spec combinations: (sliding_spec_name, global_spec_name)
COMBOS = [
    # Uniform (baselines)
    ('k4v4kb64vb64', 'k4v4kb64vb64'),
    ('k8v4kb32vb32', 'k8v4kb32vb32'),
    ('k8v8kb64vb64', 'k8v8kb64vb64'),
    # Mixed: high quality sliding + aggressive global
    ('k8v8kb64vb64', 'k4v4kb64vb64'),
    ('k8v4kb32vb32', 'k4v4kb64vb64'),
    ('k8v4kb16vb16', 'k4v4kb64vb64'),
    # Mixed: aggressive sliding + high quality global
    ('k4v4kb64vb64', 'k8v8kb64vb64'),
    ('k4v4kb64vb64', 'k8v4kb32vb32'),
]

BATCH_SIZES = [1, 32, 64, 128, 240]


def rmsnorm(x, weight, eps=1e-6):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (weight.float() * x.float() * torch.rsqrt(variance + eps)).half()


def simulate_moe(hidden, gate_w, expert_gate, expert_up, expert_down):
    h = hidden.half()
    logits = h @ gate_w
    topk_vals, _ = torch.topk(logits, TOP_K, dim=-1)
    topk_weights = F.softmax(topk_vals.float(), dim=-1).half()
    out = torch.zeros_like(h)
    for k in range(TOP_K):
        gate_out = h @ expert_gate.T
        up_out = h @ expert_up.T
        expert_out = F.gelu(gate_out) * up_out
        expert_out = expert_out @ expert_down.T
        out = out + topk_weights[:, k:k+1] * expert_out
    return out


def build_layer_weights(device='cuda'):
    """Build one set of shared weights (same perf characteristics per layer)."""
    return {
        'norm1_w': torch.ones(HIDDEN_SIZE, device=device, dtype=torch.float16),
        'norm2_w': torch.ones(HIDDEN_SIZE, device=device, dtype=torch.float16),
        'gate': torch.randn(HIDDEN_SIZE, NUM_EXPERTS, device=device, dtype=torch.float16) * 0.01,
        'expert_gate': torch.randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01,
        'expert_up': torch.randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01,
        'expert_down': torch.randn(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, device=device, dtype=torch.float16) * 0.01,
    }


def build_attn_weights(D, Hk, device='cuda'):
    q_size = NUM_HEADS * D
    kv_size = Hk * D
    return {
        'wq': torch.randn(HIDDEN_SIZE, q_size, device=device, dtype=torch.float16) * 0.01,
        'wk': torch.randn(HIDDEN_SIZE, kv_size, device=device, dtype=torch.float16) * 0.01,
        'wv': torch.randn(HIDDEN_SIZE, kv_size, device=device, dtype=torch.float16) * 0.01,
        'wo': torch.randn(q_size, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01,
    }


def build_kv_infra(spec, B, D, Hk, seq_len, device='cuda'):
    block_size = 16
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * B
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

    kv_cache = torch.zeros(total_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)
    store_fn = make_store_fn(spec)

    class L:
        pass
    layer = L()

    for b in range(min(B, 2)):
        key = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        val = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        slots = (torch.arange(seq_len, device=device, dtype=torch.int32)
                 + b * num_blocks_per_seq * block_size).clamp(max=total_blocks * block_size - 1)
        store_fn(key, val, kv_cache, slots, layer, Hk)
        del key, val

    block_table = torch.zeros(B, num_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        block_table[b] = torch.arange(num_blocks_per_seq, dtype=torch.int32) + b * num_blocks_per_seq

    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_block
    max_slots = total_blocks * block_size
    scales = torch.zeros(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device)
    if hasattr(layer, '_fc_scales'):
        n = min(layer._fc_scales.shape[0], max_slots)
        scales[:n] = layer._fc_scales[:n]

    seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)
    return kv_cache, scales, block_table, seq_lens


def estimate_vram(sliding_spec, global_spec, B):
    """Estimate VRAM for the full model at batch size B."""
    # KV cache
    cache = 0
    for lt in LAYER_TYPES:
        if lt == 'sliding':
            D, Hk, sl, spec = HEAD_DIM_SLIDING, NUM_KV_HEADS_SLIDING, 1024, sliding_spec
        else:
            D, Hk, sl, spec = HEAD_DIM_GLOBAL, NUM_KV_HEADS_GLOBAL, 8192, global_spec
        cache += B * ((sl // 16) + 1) * 16 * Hk * int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)
    # Weights (~30 layers * ~50MB)
    weights = 30 * 50 * 1024 * 1024
    return cache + weights


def benchmark_combo(sliding_spec, global_spec, B, warmup=2, trials=5):
    device = 'cuda'

    est = estimate_vram(sliding_spec, global_spec, B)
    free = torch.cuda.mem_get_info()[0]
    if est > free * 0.85:
        return None

    try:
        torch.cuda.empty_cache()
        shared_w = build_layer_weights(device)
        sliding_attn_w = build_attn_weights(HEAD_DIM_SLIDING, NUM_KV_HEADS_SLIDING, device)
        global_attn_w = build_attn_weights(HEAD_DIM_GLOBAL, NUM_KV_HEADS_GLOBAL, device)

        # Build per-layer decode functions and KV infra
        layer_data = []
        for lt in LAYER_TYPES:
            if lt == 'sliding':
                spec, D, Hk, seq = sliding_spec, HEAD_DIM_SLIDING, NUM_KV_HEADS_SLIDING, 1024
                attn_w = sliding_attn_w
                cfg = {'block_kv': 16, 'block_h': 8, 'num_warps': 2, 'num_kv_splits': 32}
            else:
                spec, D, Hk, seq = global_spec, HEAD_DIM_GLOBAL, NUM_KV_HEADS_GLOBAL, 8192
                attn_w = global_attn_w
                cfg = {'block_kv': 16, 'block_h': 8, 'num_warps': 2, 'num_kv_splits': 32}

            decode_fn = make_decode_fn(spec, **cfg)
            infra = build_kv_infra(spec, B, D, Hk, seq, device)
            layer_data.append((lt, D, Hk, attn_w, decode_fn, infra))

        hidden = torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01

        def run_step():
            h = hidden
            for lt, D, Hk, attn_w, decode_fn, (kvc, sc, bt, sl) in layer_data:
                normed = rmsnorm(h, shared_w['norm1_w'])
                q = (normed @ attn_w['wq']).reshape(B, NUM_HEADS, D)
                attn_out = decode_fn(q, kvc, sc, bt, sl, 1.0 / (D ** 0.5), Hk)
                h = h + (attn_out.reshape(B, -1).half() @ attn_w['wo'])
                normed2 = rmsnorm(h, shared_w['norm2_w'])
                moe_out = simulate_moe(normed2, shared_w['gate'],
                                       shared_w['expert_gate'], shared_w['expert_up'],
                                       shared_w['expert_down'])
                h = h + moe_out
            return h

        for _ in range(warmup):
            run_step()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(trials):
            run_step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms = (elapsed / trials) * 1000
        tps = B / (elapsed / trials)

        del layer_data, shared_w, sliding_attn_w, global_attn_w, hidden
        torch.cuda.empty_cache()
        return tps, ms

    except (torch.cuda.OutOfMemoryError, RuntimeError):
        torch.cuda.empty_cache()
        return None


def main():
    print("=" * 90)
    print("Mixed-Spec Decode Simulation: different KV cache per layer type")
    print("=" * 90)

    results = {}
    for s_name, g_name in COMBOS:
        label = f"{s_name} / {g_name}" if s_name != g_name else f"{s_name} (uniform)"
        s_spec, g_spec = SPECS[s_name], SPECS[g_name]
        s_comp = s_spec.compression_vs_bf16(HEAD_DIM_SLIDING)
        g_comp = g_spec.compression_vs_bf16(HEAD_DIM_GLOBAL)
        # Weighted average compression (25 sliding + 5 global)
        avg_comp = (25 * s_comp + 5 * g_comp) / 30

        print(f"\n--- Sliding: {s_name} ({s_comp:.1f}x) | Global: {g_name} ({g_comp:.1f}x) | Avg: {avg_comp:.1f}x ---")
        combo_results = []
        for B in BATCH_SIZES:
            r = benchmark_combo(s_spec, g_spec, B)
            if r:
                tps, ms = r
                print(f"  B={B:>3}: {tps:>8.1f} tok/s  ({ms:>7.1f} ms/step)")
                combo_results.append((B, tps, ms))
            else:
                print(f"  B={B:>3}: OOM/skip")
                combo_results.append((B, None, None))
        results[(s_name, g_name)] = combo_results

    # Summary table
    print("\n" + "=" * 90)
    print("Summary: tok/s by spec combo × batch size")
    print("=" * 90)
    header = f"{'Sliding':<16} {'Global':<16} {'AvgC':>5}"
    for B in BATCH_SIZES:
        header += f" {'B='+str(B):>8}"
    print(header)
    print("-" * (39 + 9 * len(BATCH_SIZES)))

    for (s_name, g_name), res in results.items():
        s_comp = SPECS[s_name].compression_vs_bf16(HEAD_DIM_SLIDING)
        g_comp = SPECS[g_name].compression_vs_bf16(HEAD_DIM_GLOBAL)
        avg = (25 * s_comp + 5 * g_comp) / 30
        row = f"{s_name:<16} {g_name:<16} {avg:>5.1f}x"
        for B, tps, ms in res:
            if tps:
                row += f" {tps:>8.0f}"
            else:
                row += f" {'OOM':>8}"
        print(row)


if __name__ == "__main__":
    main()
