"""Fix NVFP4 global scale mismatch for vLLM fused linear layers.

vLLM fuses QKV projections and takes max() of their global scales.
This causes underflow for projections with smaller scales.

This script rescales the weight_packed tensors so all projections in
a fused group use the same global scale (the max), compensating by
adjusting the per-block weight_scale tensors.

The math:
  Original: dequant = weight_packed * weight_scale * weight_global_scale
  After:    dequant = weight_packed * (weight_scale * old_global / new_global) * new_global
  Where new_global = max(all projections in fused group)

  The weight_packed bytes stay the same. Only weight_scale changes.
"""

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import os
import sys
import json
from collections import defaultdict


def fix_checkpoint(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Copy non-safetensor files
    for fname in os.listdir(input_dir):
        if not fname.endswith('.safetensors'):
            src = os.path.join(input_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src):
                import shutil
                shutil.copy2(src, dst)

    # Load all tensors
    sf_files = [f for f in os.listdir(input_dir) if f.endswith('.safetensors')]
    tensors = {}
    for sf in sf_files:
        with safe_open(os.path.join(input_dir, sf), framework='pt') as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(tensors)} tensors")

    # Find fused projection groups and fix scales
    # Group 1: QKV attention (q_proj, k_proj, v_proj)
    # Group 2: MoE gate+up (gate_proj, up_proj) — down_proj is separate in FusedMoE

    fixed_count = 0

    # Fix attention QKV
    layer_indices = set()
    for key in tensors:
        import re
        m = re.search(r'layers\.(\d+)\.self_attn', key)
        if m:
            layer_indices.add(int(m.group(1)))

    for layer_idx in sorted(layer_indices):
        prefix = f'model.language_model.layers.{layer_idx}.self_attn'

        for scale_type in ['weight_global_scale', 'input_global_scale']:
            proj_scales = {}
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                key = f'{prefix}.{proj}.{scale_type}'
                if key in tensors:
                    proj_scales[proj] = tensors[key].item()

            if len(proj_scales) < 2:
                continue

            max_scale = max(proj_scales.values())
            if all(abs(s - max_scale) < 1e-6 for s in proj_scales.values()):
                continue  # Already uniform

            print(f"  Layer {layer_idx} attn {scale_type}: {proj_scales} → unified to {max_scale}")

            for proj, old_scale in proj_scales.items():
                if abs(old_scale - max_scale) < 1e-6:
                    continue

                ratio = old_scale / max_scale
                # Adjust weight_scale to compensate
                ws_key = f'{prefix}.{proj}.weight_scale'
                if ws_key in tensors and scale_type == 'weight_global_scale':
                    tensors[ws_key] = (tensors[ws_key].float() * ratio).to(tensors[ws_key].dtype)

                is_key = f'{prefix}.{proj}.input_scale'
                if is_key in tensors and scale_type == 'input_global_scale':
                    tensors[is_key] = (tensors[is_key].float() * ratio).to(tensors[is_key].dtype)

                # Set global scale to the unified value
                tensors[f'{prefix}.{proj}.{scale_type}'] = torch.tensor(max_scale, dtype=torch.float32)
                fixed_count += 1

    # Fix MoE expert gate+up (but NOT down_proj — it has separate w2 scales in FusedMoE)
    for layer_idx in sorted(layer_indices):
        for expert_id in range(128):
            prefix = f'model.language_model.layers.{layer_idx}.experts.{expert_id}'

            for scale_type in ['weight_global_scale', 'input_global_scale']:
                proj_scales = {}
                for proj in ['gate_proj', 'up_proj']:  # Only gate+up are fused (w13)
                    key = f'{prefix}.{proj}.{scale_type}'
                    if key in tensors:
                        proj_scales[proj] = tensors[key].item()

                if len(proj_scales) < 2:
                    continue

                max_scale = max(proj_scales.values())
                if all(abs(s - max_scale) < 1e-6 for s in proj_scales.values()):
                    continue

                for proj, old_scale in proj_scales.items():
                    if abs(old_scale - max_scale) < 1e-6:
                        continue

                    ratio = old_scale / max_scale
                    ws_key = f'{prefix}.{proj}.weight_scale'
                    if ws_key in tensors and scale_type == 'weight_global_scale':
                        tensors[ws_key] = (tensors[ws_key].float() * ratio).to(tensors[ws_key].dtype)

                    tensors[f'{prefix}.{proj}.{scale_type}'] = torch.tensor(max_scale, dtype=torch.float32)
                    fixed_count += 1

    print(f"\nFixed {fixed_count} scale mismatches")

    # Save
    print(f"Saving to {output_dir}...")
    save_file(tensors, os.path.join(output_dir, 'model.safetensors'))
    print("Done!")


if __name__ == '__main__':
    input_dir = sys.argv[1] if len(sys.argv) > 1 else '/root/models/gemma-4-26B-A4B-it-NVFP4-redhat'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else '/root/models/gemma-4-26B-A4B-it-NVFP4-redhat-fixed'
    fix_checkpoint(input_dir, output_dir)
