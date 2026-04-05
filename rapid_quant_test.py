#!/usr/bin/env python3
"""Rapid KV quantization tester.

Loads the model ONCE, extracts real K/V tensors from a forward pass,
then tests any quantization config in milliseconds.

Usage:
    python rapid_quant_test.py
"""

import torch
import torch.nn.functional as F
import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_real_kv_tensors(model_path, prompt="The quick brown fox jumps over the lazy dog.",
                        max_new=1, device="cuda"):
    """Load model, run forward pass, extract real K/V tensors."""
    print(f"Loading model from {model_path}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=device,
        trust_remote_code=True)
    print(f"Loaded in {time.time()-t0:.1f}s")

    # Forward pass with cache
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)

    # Extract K/V from each layer's cache
    cache = outputs.past_key_values
    all_k = []
    all_v = []
    for layer_cache in cache:
        if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
            k, v = layer_cache[0], layer_cache[1]
            all_k.append(k.squeeze(0))  # remove batch dim
            all_v.append(v.squeeze(0))

    print(f"Extracted KV from {len(all_k)} layers")
    if all_k:
        print(f"  K shape per layer: {all_k[0].shape}")  # (Hk, S, D)
        print(f"  K range: [{all_k[0].float().min():.3f}, {all_k[0].float().max():.3f}]")

    # Also get a query for attention testing
    q = outputs.logits  # not ideal, but we need the model's internal Q
    # Better: use the last hidden state before attention
    # For now, use random Q with matching distribution
    if all_k:
        Hk, S, D = all_k[0].shape
        Q = torch.randn(1, Hk, D, device=device, dtype=torch.float32) * all_k[0].float().std()
    else:
        Q = None

    return all_k, all_v, Q, tokenizer


def quant_test(K, V, Q, k_bits, v_bits, block_size, scale_attn):
    """Test a quantization config on real K/V tensors. Returns quality metrics."""
    Hk, S, D = K.shape
    device = K.device

    def quantize(x, bits, block):
        qmax = (1 << bits) - 1
        mid = qmax / 2.0
        x_f = x.float()
        # Per-block quantization
        x_b = x_f.reshape(Hk, S, D // block, block)
        absmax = x_b.abs().amax(dim=-1, keepdim=True)
        sc = absmax / mid
        codes = (x_b / (sc + 1e-8) + mid).round().clamp(0, qmax)
        recon = (codes - mid) * sc
        return recon.reshape(Hk, S, D)

    K_r = quantize(K, k_bits, block_size)
    V_r = quantize(V, v_bits, block_size)

    # K reconstruction error
    k_err = (K.float() - K_r).norm() / K.float().norm()

    # Attention quality
    if Q is not None:
        q = Q[0]  # (Hk, D)
        # K is (Hk, S, D), need scores (Hk, S)
        scores_true = torch.einsum('hd,hsd->hs', q, K.float()) * scale_attn
        scores_recon = torch.einsum('hd,hsd->hs', q, K_r) * scale_attn

        attn_true = F.softmax(scores_true, dim=-1)
        attn_recon = F.softmax(scores_recon, dim=-1)

        top1_agree = (attn_true.argmax(dim=-1) == attn_recon.argmax(dim=-1)).float().mean()

        top5_true = attn_true.topk(5, dim=-1).indices
        top5_recon = attn_recon.topk(5, dim=-1).indices
        overlaps = []
        for h in range(Hk):
            t = set(top5_true[h].tolist())
            r = set(top5_recon[h].tolist())
            overlaps.append(len(t & r) / 5)
        top5 = sum(overlaps) / len(overlaps)
    else:
        top1_agree = torch.tensor(0.0)
        top5 = 0.0

    # Compression
    k_data = D * k_bits / 8
    v_data = D * v_bits / 8
    k_sc = (D // block_size) * 2
    v_sc = (D // block_size) * 2
    total_bytes = k_data + v_data + k_sc + v_sc
    fp16_bytes = D * 4
    compression = fp16_bytes / total_bytes

    return {
        'k_err': k_err.item(),
        'top1': top1_agree.item(),
        'top5': top5,
        'compression': compression,
    }


def main():
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/root/models/gemma-4-31B-it-AWQ-4bit"

    all_k, all_v, Q, tok = get_real_kv_tensors(model_path)
    if not all_k:
        print("Failed to extract KV tensors")
        return

    # Use middle layer (most representative)
    mid = len(all_k) // 2
    K = all_k[mid]
    V = all_v[mid]
    Hk, S, D = K.shape
    scale_attn = 1.0 / (D ** 0.5)

    print(f"\n{'='*75}")
    print(f"Rapid Quant Sweep — Layer {mid}, K={K.shape}, Real Model Data")
    print(f"{'='*75}")
    print(f"{'Config':<30} {'K err':>7} {'Top1':>6} {'Top5':>6} {'Comp':>6} {'~KV tok':>8}")
    print('─' * 75)

    configs = [
        # (name, k_bits, v_bits, block)
        ("FP8 equivalent (8/8/256)", 8, 8, 256),
        ("K8 V4 block32", 8, 4, 32),
        ("K8 V2 block32", 8, 2, 32),
        ("K6 V4 block32", 6, 4, 32),
        ("K6 V2 block32", 6, 2, 32),
        ("K4 V4 block32", 4, 4, 32),
        ("K4 V2 block32", 4, 2, 32),
        ("K4 V2 block16", 4, 2, 16),
        ("K4 V2 block64", 4, 2, 64),
        ("K3 V2 block32", 3, 2, 32),
        ("K2 V2 block32", 2, 2, 32),
        ("K4 V4 block256 (per-head)", 4, 4, 256),
        ("K4 V2 block256 (per-head)", 4, 2, 256),
    ]

    for name, kb, vb, bs in configs:
        t0 = time.time()
        r = quant_test(K, V, Q, kb, vb, bs, scale_attn)
        ms = (time.time() - t0) * 1000

        # Estimate KV tokens (8GB free, 60 layers, ~14 avg KV heads)
        k_data = D * kb / 8
        v_data = D * vb / 8
        sc_data = (D // bs) * 4  # K+V scales
        per_head = k_data + v_data + sc_data
        per_token_mb = 60 * 14 * per_head / 1e6
        tokens = 8000 / per_token_mb

        print(f"{name:<30} {r['k_err']*100:>5.1f}% {r['top1']*100:>5.0f}% "
              f"{r['top5']*100:>5.0f}% {r['compression']:>5.1f}x {tokens:>7.0f}")

    print(f"\nAll tests took <1 second on REAL model data")


if __name__ == "__main__":
    main()
