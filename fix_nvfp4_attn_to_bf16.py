"""Convert NVFP4 attention weights to BF16, following NVIDIA's pattern.

NVIDIA's Gemma4-31B-IT-NVFP4 excludes ALL self_attn layers from NVFP4 quantization,
keeping them in BF16. This avoids vLLM's scale fusion bug.

This script:
1. For self_attn projections (q_proj, k_proj, v_proj, o_proj):
   - Dequantizes: bf16 = fp4_dequant(weight_packed, weight_scale, weight_global_scale)
   - Stores as regular .weight tensor in BF16
   - Removes weight_packed, weight_scale, weight_global_scale, input_global_scale
2. For everything else (MoE experts, embeddings, norms): keeps as-is
3. Updates quantization_config to add self_attn to ignore list

NVFP4 dequant:
  Each byte in weight_packed holds 2 FP4-E2M1 values (low nibble, high nibble).
  FP4-E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit.
  block_scale (FP8-E4M3) is per group of 16 elements.
  dequant = fp4_to_float(nibble) * float(block_scale) * weight_global_scale
"""

import torch
import json
import os
import sys
import shutil
from safetensors import safe_open
from safetensors.torch import save_file


# FP4-E2M1 lookup table (all 16 values)
# Format: 1-bit sign, 2-bit exponent, 1-bit mantissa
# E2M1: bias=1, so exponent range is 0-3 (biased), -1 to 2 (unbiased)
FP4_E2M1_TABLE = torch.tensor([
    0.0,      # 0b0000: +0
    0.5,      # 0b0001: +0.5  (e=0, m=1 → 0.5)
    1.0,      # 0b0010: +1.0  (e=1, m=0 → 1.0)
    1.5,      # 0b0011: +1.5  (e=1, m=1 → 1.5)
    2.0,      # 0b0100: +2.0  (e=2, m=0 → 2.0)
    3.0,      # 0b0101: +3.0  (e=2, m=1 → 3.0)
    4.0,      # 0b0110: +4.0  (e=3, m=0 → 4.0)
    6.0,      # 0b0111: +6.0  (e=3, m=1 → 6.0)
    -0.0,     # 0b1000: -0
    -0.5,     # 0b1001: -0.5
    -1.0,     # 0b1010: -1.0
    -1.5,     # 0b1011: -1.5
    -2.0,     # 0b1100: -2.0
    -3.0,     # 0b1101: -3.0
    -4.0,     # 0b1110: -4.0
    -6.0,     # 0b1111: -6.0
], dtype=torch.float32)


def dequant_nvfp4(weight_packed, weight_scale, weight_global_scale, group_size=16):
    """Dequantize NVFP4 packed weights to float32.

    Args:
        weight_packed: [N, M//2] uint8 — 2 FP4 values per byte
        weight_scale: [N, M//group_size] float8_e4m3 — per-block scale
        weight_global_scale: scalar float32 — per-tensor scale
    Returns:
        [N, M] bfloat16 weight
    """
    N, packed_cols = weight_packed.shape
    M = packed_cols * 2  # 2 values per byte

    # Unpack FP4 nibbles
    packed = weight_packed.to(torch.int32)
    lo = packed & 0xF         # low nibble (first element)
    hi = (packed >> 4) & 0xF  # high nibble (second element)

    # Convert FP4 codes to float using lookup table
    fp4_table = FP4_E2M1_TABLE.to(weight_packed.device)
    lo_float = fp4_table[lo]   # [N, M//2]
    hi_float = fp4_table[hi]   # [N, M//2]

    # Interleave: [lo0, hi0, lo1, hi1, ...]
    values = torch.stack([lo_float, hi_float], dim=-1).reshape(N, M)

    # Apply per-block scales
    # weight_scale shape: [N, M//group_size] in FP8
    block_scale = weight_scale.float()  # [N, num_blocks]
    num_blocks = block_scale.shape[1]

    # Expand scales to match elements: each scale covers group_size elements
    block_scale_expanded = block_scale.unsqueeze(-1).expand(N, num_blocks, group_size).reshape(N, M)

    # Apply scales
    global_scale = weight_global_scale.float().item()
    result = values * block_scale_expanded * global_scale

    return result.to(torch.bfloat16)


def convert_checkpoint(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load all tensors
    sf_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.safetensors')])
    tensors = {}
    for sf in sf_files:
        print(f"Loading {sf}...")
        with safe_open(os.path.join(input_dir, sf), framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors")

    # Find all self_attn NVFP4 projection groups
    import re
    attn_projs = set()
    for key in tensors:
        m = re.match(r'(.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj))\.weight_packed', key)
        if m:
            attn_projs.add(m.group(1))

    print(f"Found {len(attn_projs)} attention projections to dequantize")

    # Dequantize each
    new_tensors = {}
    removed = set()

    for prefix in sorted(attn_projs):
        wp_key = f'{prefix}.weight_packed'
        ws_key = f'{prefix}.weight_scale'
        wgs_key = f'{prefix}.weight_global_scale'
        igs_key = f'{prefix}.input_global_scale'

        if wp_key not in tensors or ws_key not in tensors or wgs_key not in tensors:
            print(f"  SKIP {prefix}: missing tensors")
            continue

        wp = tensors[wp_key]
        ws = tensors[ws_key]
        wgs = tensors[wgs_key]

        print(f"  Dequant {prefix}: packed={wp.shape} scale={ws.shape} global={wgs.item():.1f}")
        bf16_weight = dequant_nvfp4(wp, ws, wgs)
        print(f"    → weight {bf16_weight.shape} bf16, range [{bf16_weight.min():.3f}, {bf16_weight.max():.3f}]")

        new_tensors[f'{prefix}.weight'] = bf16_weight
        removed.update([wp_key, ws_key, wgs_key, igs_key])

    # Build final tensor dict
    output_tensors = {}
    for key, tensor in tensors.items():
        if key in removed:
            continue
        if key.replace('.weight_packed', '.weight') in new_tensors:
            continue
        output_tensors[key] = tensor

    output_tensors.update(new_tensors)

    print(f"\nOriginal tensors: {len(tensors)}")
    print(f"Removed (NVFP4 attn): {len(removed)}")
    print(f"Added (BF16 attn): {len(new_tensors)}")
    print(f"Final tensors: {len(output_tensors)}")

    # Update config.json
    config_path = os.path.join(input_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

        # Add self_attn to ignore list in quantization_config
        qcfg = config.get('quantization_config', {})
        ignore_list = qcfg.get('ignore', [])

        # Add all self_attn patterns
        for layer_idx in range(30):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                pattern = f'model.language_model.layers.{layer_idx}.self_attn.{proj}'
                if pattern not in ignore_list:
                    ignore_list.append(pattern)

        qcfg['ignore'] = ignore_list
        config['quantization_config'] = qcfg

        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        print("Updated config.json with self_attn ignore list")

    # Copy other files
    for fname in os.listdir(input_dir):
        if fname.endswith('.safetensors') or fname == 'config.json':
            continue
        src = os.path.join(input_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # Save
    print(f"\nSaving to {output_dir}...")
    save_file(output_tensors, os.path.join(output_dir, 'model.safetensors'))

    # Size comparison
    orig_size = sum(os.path.getsize(os.path.join(input_dir, f))
                    for f in os.listdir(input_dir) if f.endswith('.safetensors'))
    new_size = os.path.getsize(os.path.join(output_dir, 'model.safetensors'))
    print(f"Original: {orig_size/1e9:.1f} GB")
    print(f"Converted: {new_size/1e9:.1f} GB")
    print(f"Delta: +{(new_size-orig_size)/1e9:.1f} GB (BF16 attn is larger than NVFP4)")
    print("\nDone!")


if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/root/models/gemma-4-26B-A4B-it-NVFP4-redhat'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/root/models/gemma-4-26B-A4B-it-NVFP4-bf16attn'
    convert_checkpoint(input_dir, output_dir)
