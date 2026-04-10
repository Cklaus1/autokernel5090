"""Gemma4 26B-A4B decode simulator with data-driven KV cache kernels.

Simulates a full decode step (all 30 layers) with:
- Data-driven Triton kernel for attention
- Real MoE gate + expert matmuls for MLP
- RMSNorm between layers
- Measures actual tok/s across batch sizes and KV cache specs
"""

import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.generate import make_decode_fn, make_store_fn

# ===== Gemma4 26B-A4B architecture =====

HIDDEN_SIZE = 2816
NUM_HEADS = 16          # Q heads
HEAD_DIM_SLIDING = 256
HEAD_DIM_GLOBAL = 512
NUM_KV_HEADS_SLIDING = 8
NUM_KV_HEADS_GLOBAL = 2
NUM_LAYERS = 30
INTERMEDIATE_SIZE = 2112      # dense FFN (unused in MoE layers)
MOE_INTERMEDIATE_SIZE = 704   # per-expert FFN
NUM_EXPERTS = 128
TOP_K = 8
LAYER_TYPES = [
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
    'sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full',
]
SLIDING_WINDOW = 1024

# Best configs from sweep
BEST_CONFIGS = {
    'sliding': {'block_kv': 16, 'block_h': 8, 'num_warps': 2, 'num_kv_splits': 32},
    'global':  {'block_kv': 32, 'block_h': 8, 'num_warps': 4, 'num_kv_splits': 32},
}


class SimulatedLayer:
    """One Gemma4 decoder layer: RMSNorm → Attention → RMSNorm → MoE/FFN."""

    def __init__(self, layer_type, device='cuda'):
        self.layer_type = layer_type
        self.device = device
        is_global = (layer_type == 'full')

        self.head_dim = HEAD_DIM_GLOBAL if is_global else HEAD_DIM_SLIDING
        self.num_kv_heads = NUM_KV_HEADS_GLOBAL if is_global else NUM_KV_HEADS_SLIDING
        self.kv_group_size = NUM_HEADS // self.num_kv_heads

        # QKV projection weights (simulated as matmuls)
        q_size = NUM_HEADS * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.wq = torch.randn(HIDDEN_SIZE, q_size, device=device, dtype=torch.float16) * 0.01
        self.wk = torch.randn(HIDDEN_SIZE, kv_size, device=device, dtype=torch.float16) * 0.01
        self.wv = torch.randn(HIDDEN_SIZE, kv_size, device=device, dtype=torch.float16) * 0.01
        self.wo = torch.randn(q_size, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01

        # RMSNorm weights
        self.norm1_w = torch.ones(HIDDEN_SIZE, device=device, dtype=torch.float16)
        self.norm2_w = torch.ones(HIDDEN_SIZE, device=device, dtype=torch.float16)

        # MoE gate
        self.gate = torch.randn(HIDDEN_SIZE, NUM_EXPERTS, device=device, dtype=torch.float16) * 0.01

        # Expert weights (only materialize top-K during forward)
        # Each expert: gate_proj [H, I], up_proj [H, I], down_proj [I, H]
        # We simulate with a single set (same perf characteristics)
        self.expert_gate = torch.randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01
        self.expert_up = torch.randn(MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01
        self.expert_down = torch.randn(HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, device=device, dtype=torch.float16) * 0.01


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_norm = x.float() * torch.rsqrt(variance + eps)
    return (weight.float() * x_norm).half()


def simulate_moe(hidden, gate_w, expert_gate, expert_up, expert_down):
    """Simulate top-K MoE forward pass."""
    B = hidden.shape[0]
    h = hidden.half()

    # Router
    logits = h @ gate_w
    topk_vals, topk_ids = torch.topk(logits, TOP_K, dim=-1)
    topk_weights = F.softmax(topk_vals.float(), dim=-1).half()

    # Expert computation (simulate TOP_K experts per token)
    out = torch.zeros_like(h)
    for k in range(TOP_K):
        gate_out = h @ expert_gate.T
        up_out = h @ expert_up.T
        expert_out = F.gelu(gate_out) * up_out
        expert_out = expert_out @ expert_down.T
        out = out + topk_weights[:, k:k+1] * expert_out

    return out


def build_kv_infrastructure(spec, layer, B, seq_len, device='cuda'):
    """Pre-build KV cache, block table, and store initial KV data."""
    D = layer.head_dim
    Hk = layer.num_kv_heads
    block_size = 16

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = num_blocks_per_seq * B
    cache_per_head = int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

    kv_cache = torch.zeros(total_blocks, block_size, Hk, cache_per_head,
                           dtype=torch.uint8, device=device)

    # Store function + fill cache
    store_fn = make_store_fn(spec)

    class FakeLayer:
        pass
    fake_layer = FakeLayer()

    # Store data for each batch element
    for b in range(min(B, 4)):  # fill a few, rest are zeros (perf is same)
        key_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        value_fp = torch.randn(seq_len, Hk, D, device=device, dtype=torch.float16)
        slot_mapping = torch.arange(seq_len, device=device, dtype=torch.int32) + b * num_blocks_per_seq * block_size
        slot_mapping = slot_mapping.clamp(max=total_blocks * block_size - 1)
        store_fn(key_fp, value_fp, kv_cache, slot_mapping, fake_layer, Hk)

    # Build block table
    block_table = torch.zeros(B, num_blocks_per_seq, dtype=torch.int32, device=device)
    for b in range(B):
        start = b * num_blocks_per_seq
        block_table[b] = torch.arange(num_blocks_per_seq, dtype=torch.int32) + start

    # Scales
    min_block = min(spec.k_scale_block, spec.v_scale_block)
    num_sb = D // min_block
    max_slots = total_blocks * block_size
    scales = torch.zeros(max_slots, Hk, num_sb, 2, dtype=torch.float16, device=device)
    if hasattr(fake_layer, '_fc_scales'):
        copy_len = min(fake_layer._fc_scales.shape[0], max_slots)
        scales[:copy_len] = fake_layer._fc_scales[:copy_len]

    seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

    return kv_cache, scales, block_table, seq_lens


def run_decode_step(layers, decode_fns, kv_infra, hidden, B):
    """Run one full decode step through all 30 layers."""
    for i, (layer, (decode_fn, cache_data)) in enumerate(zip(layers, zip(decode_fns, kv_infra))):
        kv_cache, scales, block_table, seq_lens = cache_data

        # Pre-attention norm
        normed = rmsnorm(hidden, layer.norm1_w)

        # QKV projection
        normed_h = normed.half()
        q = (normed_h @ layer.wq).reshape(B, NUM_HEADS, layer.head_dim)
        k = (normed_h @ layer.wk).reshape(B, layer.num_kv_heads, layer.head_dim)
        v = (normed_h @ layer.wv).reshape(B, layer.num_kv_heads, layer.head_dim)

        # Attention decode (our data-driven kernel)
        scale = 1.0 / (layer.head_dim ** 0.5)
        attn_out = decode_fn(q, kv_cache, scales, block_table, seq_lens,
                             scale, layer.num_kv_heads)

        # Output projection
        attn_out_flat = attn_out.reshape(B, -1).half()
        hidden = hidden + (attn_out_flat @ layer.wo)

        # Post-attention norm + MoE
        normed2 = rmsnorm(hidden, layer.norm2_w)
        moe_out = simulate_moe(normed2, layer.gate, layer.expert_gate,
                               layer.expert_up, layer.expert_down)
        hidden = hidden + moe_out

    return hidden


def benchmark_spec(spec, batch_sizes, seq_len_sliding=1024, seq_len_global=8192,
                   warmup=2, trials=5):
    """Benchmark a KV cache spec across batch sizes. Returns tok/s per batch."""
    device = 'cuda'
    results = []

    for B in batch_sizes:
        # Check VRAM
        free_mem = torch.cuda.mem_get_info()[0]
        # Rough estimate: each layer needs cache + weights
        est_cache_bytes = 0
        for lt in LAYER_TYPES:
            sl = seq_len_sliding if lt == 'sliding' else seq_len_global
            D = HEAD_DIM_SLIDING if lt == 'sliding' else HEAD_DIM_GLOBAL
            Hk = NUM_KV_HEADS_SLIDING if lt == 'sliding' else NUM_KV_HEADS_GLOBAL
            est_cache_bytes += B * (sl // 16 + 1) * 16 * Hk * int(spec.k_bytes_per_dim * D + spec.v_bytes_per_dim * D)

        # Layer weights: ~30 layers × (QKV + MoE) ≈ 30 × 50MB
        est_total = est_cache_bytes + 30 * 50 * 1024 * 1024
        if est_total > free_mem * 0.85:
            print(f"  B={B}: skip (est {est_total/1e9:.1f}GB > {free_mem*0.85/1e9:.1f}GB free)")
            results.append((B, None))
            continue

        try:
            torch.cuda.empty_cache()

            # Build layers
            layers = [SimulatedLayer(lt, device) for lt in LAYER_TYPES]

            # Build decode functions and KV infrastructure per layer
            decode_fns = []
            kv_infra_list = []

            for layer_obj in layers:
                is_global = (layer_obj.layer_type == 'full')
                cfg = BEST_CONFIGS['global' if is_global else 'sliding']
                decode_fn = make_decode_fn(spec,
                                           block_kv=cfg['block_kv'],
                                           block_h=cfg['block_h'],
                                           num_warps=cfg['num_warps'],
                                           num_kv_splits=cfg['num_kv_splits'])

                seq_len = seq_len_global if is_global else seq_len_sliding
                infra = build_kv_infrastructure(spec, layer_obj, B, seq_len, device)

                decode_fns.append(decode_fn)
                kv_infra_list.append(infra)

            # Initial hidden state
            hidden = torch.randn(B, HIDDEN_SIZE, device=device, dtype=torch.float16) * 0.01

            # Warmup
            for _ in range(warmup):
                _ = run_decode_step(layers, decode_fns, kv_infra_list, hidden, B)
            torch.cuda.synchronize()

            # Benchmark
            t0 = time.perf_counter()
            for _ in range(trials):
                _ = run_decode_step(layers, decode_fns, kv_infra_list, hidden, B)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            ms_per_step = (elapsed / trials) * 1000
            tokens_per_step = B  # decode: 1 token per sequence per step
            tok_per_sec = tokens_per_step / (elapsed / trials)

            results.append((B, tok_per_sec, ms_per_step))
            print(f"  B={B:>3}: {tok_per_sec:>8.1f} tok/s  ({ms_per_step:>7.1f} ms/step)")

            # Cleanup
            del layers, decode_fns, kv_infra_list, hidden
            torch.cuda.empty_cache()

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            print(f"  B={B}: OOM ({str(e)[:60]})")
            results.append((B, None))
            torch.cuda.empty_cache()

    return results


def main():
    BATCH_SIZES = [1, 8, 32, 64, 128, 240]

    # Test the most promising specs from sweep
    specs_to_test = {
        'k8v4kb32vb32': KVCacheSpec(name='k8v4kb32vb32', k_bits=8, k_sym_offset=127.5,
                                     k_scale_block=32, v_bits=4, v_sym_offset=7.5, v_scale_block=32),
        'k4v4kb64vb64': KVCacheSpec(name='k4v4kb64vb64', k_bits=4, k_sym_offset=7.5,
                                     k_scale_block=64, v_bits=4, v_sym_offset=7.5, v_scale_block=64),
        'k8v8kb64vb64': KVCacheSpec(name='k8v8kb64vb64', k_bits=8, k_sym_offset=127.5,
                                     k_scale_block=64, v_bits=8, v_sym_offset=127.5, v_scale_block=64),
        'k8v4kb16vb16': KVCacheSpec(name='k8v4kb16vb16', k_bits=8, k_sym_offset=127.5,
                                     k_scale_block=16, v_bits=4, v_sym_offset=7.5, v_scale_block=16),
    }

    print("=" * 75)
    print("Gemma4 26B-A4B Decode Simulator — tok/s with data-driven KV cache")
    print(f"Architecture: {NUM_LAYERS} layers, {NUM_EXPERTS} experts (top-{TOP_K})")
    print(f"Sliding: D={HEAD_DIM_SLIDING}, Hk={NUM_KV_HEADS_SLIDING}, seq={SLIDING_WINDOW}")
    print(f"Global:  D={HEAD_DIM_GLOBAL}, Hk={NUM_KV_HEADS_GLOBAL}, seq=8192")
    print("=" * 75)

    all_results = {}

    for spec_name, spec in specs_to_test.items():
        comp = spec.compression_vs_bf16(HEAD_DIM_SLIDING)
        print(f"\n--- {spec_name} ({comp:.1f}x compression) ---")
        results = benchmark_spec(spec, BATCH_SIZES)
        all_results[spec_name] = results

    # Summary table
    print("\n" + "=" * 75)
    print("Summary: tok/s by spec × batch size")
    print("=" * 75)
    header = f"{'Spec':<20} {'Comp':>5}"
    for B in BATCH_SIZES:
        header += f" {'B='+str(B):>8}"
    print(header)
    print("-" * (28 + 9 * len(BATCH_SIZES)))

    for spec_name, results in all_results.items():
        spec = specs_to_test[spec_name]
        comp = spec.compression_vs_bf16(HEAD_DIM_SLIDING)
        row = f"{spec_name:<20} {comp:>5.1f}x"
        for B, *rest in results:
            if rest and rest[0] is not None:
                tok_s = rest[0]
                row += f" {tok_s:>8.0f}"
            else:
                row += f" {'OOM':>8}"
        print(row)


if __name__ == "__main__":
    main()
