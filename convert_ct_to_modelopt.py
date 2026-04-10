"""Convert compressed-tensors NVFP4 checkpoint to modelopt format.

NVIDIA's modelopt format works on SM120 (RTX 5090). The compressed-tensors
format doesn't (broken scale handling in vLLM's CT MoE path).

The FP4 weight data is identical — only the scale naming convention differs:
  CT: weight_global_scale (divisor) → modelopt: weight_scale_2 = 1/weight_global_scale
  CT: input_global_scale (divisor)  → modelopt: input_scale = 1/input_global_scale
  CT: weight_packed                → modelopt: weight (same data)

Following NVIDIA's Gemma-4-31B-IT-NVFP4 pattern:
  - Exclude self_attn layers from NVFP4 (dequantize to BF16)
  - Use modelopt quant_method in config
"""

import torch
import json
import os
import sys
import shutil
from safetensors import safe_open
from safetensors.torch import save_file


# FP4-E2M1 lookup table for dequantization
FP4_E2M1_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def dequant_nvfp4(weight_packed, weight_scale, weight_global_scale):
    """Dequantize FP4 packed weight to BF16."""
    N, half_K = weight_packed.shape
    K = half_K * 2
    table = FP4_E2M1_TABLE
    packed = weight_packed.to(torch.int32)
    lo = table[packed & 0xF]
    hi = table[(packed >> 4) & 0xF]
    values = torch.stack([lo, hi], dim=-1).reshape(N, K)
    bs = weight_scale.float()
    bs_exp = bs.unsqueeze(-1).expand(N, K // 16, 16).reshape(N, K)
    gs = weight_global_scale.float().item()
    return (values * bs_exp * gs).to(torch.bfloat16)


def convert(input_dir, output_dir, exclude_attn=True):
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    sf_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.safetensors'))
    tensors = {}
    for sf in sf_files:
        print(f"Loading {sf}...")
        with safe_open(os.path.join(input_dir, sf), framework='pt', device='cpu') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    print(f"Loaded {len(tensors)} tensors")

    output_tensors = {}
    converted = 0
    dequantized = 0

    for key, tensor in tensors.items():
        # Skip vision tower — keep as-is
        if 'vision_tower' in key or 'audio_tower' in key or 'embed_vision' in key or 'embed_audio' in key:
            output_tensors[key] = tensor
            continue

        # Check if this is a self_attn NVFP4 weight that should be dequantized
        is_attn = 'self_attn' in key and any(p in key for p in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])
        is_nvfp4 = 'weight_packed' in key or 'weight_scale' in key or 'weight_global_scale' in key or 'input_global_scale' in key

        if exclude_attn and is_attn and is_nvfp4:
            # Dequantize attention to BF16
            if 'weight_packed' in key:
                prefix = key.replace('.weight_packed', '')
                ws_key = prefix + '.weight_scale'
                wgs_key = prefix + '.weight_global_scale'
                if ws_key in tensors and wgs_key in tensors:
                    bf16 = dequant_nvfp4(tensor, tensors[ws_key], tensors[wgs_key])
                    output_tensors[prefix + '.weight'] = bf16
                    dequantized += 1
                    print(f"  Dequant attn: {prefix} → BF16 {bf16.shape}")
            # Skip scale tensors for dequantized weights
            continue

        # Convert NVFP4 naming: CT → modelopt
        new_key = key

        if '.weight_packed' in key:
            # CT weight_packed → modelopt weight (same data)
            new_key = key.replace('.weight_packed', '.weight')
            output_tensors[new_key] = tensor
            converted += 1
            continue

        if '.weight_global_scale' in key:
            # CT weight_global_scale (divisor) → modelopt weight_scale_2 (1/divisor)
            new_key = key.replace('.weight_global_scale', '.weight_scale_2')
            output_tensors[new_key] = (1.0 / tensor.float()).to(torch.float32)
            converted += 1
            continue

        if '.input_global_scale' in key:
            # CT input_global_scale (divisor) → modelopt input_scale (1/divisor)
            new_key = key.replace('.input_global_scale', '.input_scale')
            output_tensors[new_key] = (1.0 / tensor.float()).to(torch.float32)
            converted += 1
            continue

        # Everything else passes through
        output_tensors[key] = tensor

    print(f"\nConverted {converted} NVFP4 tensors (CT→modelopt)")
    print(f"Dequantized {dequantized} attention projections to BF16")
    print(f"Output tensors: {len(output_tensors)}")

    # Update config.json
    config_path = os.path.join(input_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # Replace quantization_config with modelopt format
    # Following NVIDIA's hf_quant_config.json pattern
    config['quantization_config'] = {
        "quant_method": "modelopt",
        "kv_cache_quant_algo": None,
        "quantization": {
            "quant_algo": "NVFP4",
            "group_size": 16,
        }
    }

    # Add exclude list for attention layers if dequantized
    if exclude_attn:
        num_layers = config.get('text_config', {}).get('num_hidden_layers', 30)
        exclude = ["lm_head"]
        for i in range(num_layers):
            exclude.append(f"model.language_model.layers.{i}.self_attn*")
        config['quantization_config']['quantization']['exclude_modules'] = exclude

    out_config = os.path.join(output_dir, 'config.json')
    with open(out_config, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Updated config.json with modelopt quant_method")

    # Also write hf_quant_config.json (NVIDIA's format)
    hf_qconfig = {
        "producer": {"name": "modelopt", "version": "0.37.0"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": 16,
        }
    }
    if exclude_attn:
        hf_qconfig['quantization']['exclude_modules'] = exclude
    with open(os.path.join(output_dir, 'hf_quant_config.json'), 'w') as f:
        json.dump(hf_qconfig, f, indent=2)

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

    orig_size = sum(os.path.getsize(os.path.join(input_dir, f))
                    for f in os.listdir(input_dir) if f.endswith('.safetensors'))
    new_size = os.path.getsize(os.path.join(output_dir, 'model.safetensors'))
    print(f"Original: {orig_size/1e9:.1f} GB → Converted: {new_size/1e9:.1f} GB")
    print("Done!")


if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/root/models/gemma-4-26B-A4B-it-NVFP4-redhat'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt'
    exclude = '--no-exclude-attn' not in sys.argv
    convert(input_dir, output_dir, exclude_attn=exclude)
