#!/usr/bin/env python3
"""
Apply fused_add_rms_norm patch to Gemma4 decoder layer forward.

Run inside the Docker container BEFORE starting the vLLM server:
  python3 /patches/apply_gemma4_patch.py

This modifies /build/vllm/vllm/model_executor/models/gemma4.py in-place.
"""
import re

TARGET = '/build/vllm/vllm/model_executor/models/gemma4.py'

# Read the file
with open(TARGET, 'r') as f:
    content = f.read()

# Check if already patched
if 'fused_add_rms_norm' in content and 'PATCHED: fused add+norm' in content:
    print("Already patched, skipping.")
    exit(0)

# Add import for fused_add_rms_norm at the top of the imports section
# Find the layernorm import and add our import
old_import = 'from vllm.model_executor.layers.layernorm import RMSNorm'
new_import = '''from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.layernorm import fused_add_rms_norm as _fused_add_rms_norm'''

if old_import in content:
    content = content.replace(old_import, new_import, 1)
    print("Added fused_add_rms_norm import.")
else:
    print(f"WARNING: Could not find import line: {old_import}")

# Replace the forward method's post_attn_norm + add + pre_ff_norm pattern
# Original:
#   hidden_states = self.post_attention_layernorm(hidden_states)
#   hidden_states = hidden_states + residual
#   residual = hidden_states
#   hidden_states = self.pre_feedforward_layernorm(hidden_states)
#
# Patched:
#   hidden_states = self.post_attention_layernorm(hidden_states)
#   hidden_states, residual = _fused_add_rms_norm(
#       hidden_states, residual,
#       self.pre_feedforward_layernorm.weight.data,
#       self.pre_feedforward_layernorm.variance_epsilon,
#   )

old_pattern = """        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        # MLP runs unconditionally (same inputs for MoE and non-MoE)
        hidden_states = self.pre_feedforward_layernorm(hidden_states)"""

new_pattern = """        hidden_states = self.post_attention_layernorm(hidden_states)
        # PATCHED: fused add+norm eliminates separate add kernel + pre_ff_norm
        # fused_add_rms_norm: residual = hidden + residual; hidden = norm(residual)
        hidden_states, residual = _fused_add_rms_norm(
            hidden_states, residual,
            self.pre_feedforward_layernorm.weight.data,
            self.pre_feedforward_layernorm.variance_epsilon,
        )
        # hidden = pre_ff_norm(post_attn_norm(attn_out) + old_residual)
        # residual = post_attn_norm(attn_out) + old_residual"""

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern, 1)
    print("Patched post_attn_norm + add + pre_ff_norm -> fused_add_rms_norm.")
else:
    print("WARNING: Could not find the exact pattern to patch in forward().")
    print("The file may have been modified already or the code has changed.")

# Write back
with open(TARGET, 'w') as f:
    f.write(content)

# Clear __pycache__
import os
cache_dir = os.path.dirname(TARGET) + '/__pycache__'
if os.path.exists(cache_dir):
    import shutil
    shutil.rmtree(cache_dir)
    print("Cleared __pycache__.")

print("Patch applied successfully.")
print(f"Modified: {TARGET}")
