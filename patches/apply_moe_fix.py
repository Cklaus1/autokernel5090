"""Apply the MoE per-expert loop fix to bypass SM120 grouped GEMM bug."""
import re

# Fix 1: gemma4.py regex — only remap per-expert unfused names
path1 = '/build/vllm/vllm/model_executor/models/gemma4.py'
with open(path1) as f:
    code = f.read()

# Find the broad regex and make it conditional
old_pattern = r'name = re.sub\(r"\\\.experts\\\.\(\\d\+\)\\\."\, r"\.moe\.experts\.\\\1\."\, name\)'
# Try literal match instead
target = 'name = re.sub(r"\\.experts\\.(\\d+)\\.", r".moe.experts.\\1.", name)'
if target in code:
    replacement = '''# Only remap unfused per-expert NVFP4 names (not AWQ packed names)
                if re.search(r"\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)\\.", name):
                    name = re.sub(r"\\.experts\\.(\\d+)\\.", r".moe.experts.\\1.", name)'''
    code = code.replace(target, replacement)
    with open(path1, 'w') as f:
        f.write(code)
    print(f"Fixed {path1}: conditional expert remapping")
else:
    # Check for the lookbehind version
    target2 = 'name = re.sub(r"(?<!\\.moe)\\.experts\\.(\\d+)\\.", r".moe.experts.\\1.", name)'
    if target2 in code:
        replacement = '''# Only remap unfused per-expert NVFP4 names
                if re.search(r"\\.experts\\.\\d+\\.(gate_proj|up_proj|down_proj)\\.", name):
                    name = re.sub(r"\\.experts\\.(\\d+)\\.", r".moe.experts.\\1.", name)'''
        code = code.replace(target2, replacement)
        with open(path1, 'w') as f:
            f.write(code)
        print(f"Fixed {path1}: conditional expert remapping (from lookbehind)")
    else:
        print(f"WARNING: Could not find regex to patch in {path1}")
        for i, line in enumerate(code.split('\n')):
            if 'experts' in line and 'sub' in line:
                print(f"  Line {i+1}: {line.strip()}")


# Fix 2: cutlass_moe.py — add per-expert loop option
path2 = '/build/vllm/vllm/model_executor/layers/fused_moe/cutlass_moe.py'
with open(path2) as f:
    code = f.read()

if '_run_cutlass_moe_fp4_loop' in code:
    print(f"{path2}: already patched")
else:
    # Find run_cutlass_moe_fp4
    func_start = code.find('def run_cutlass_moe_fp4(')
    if func_start < 0:
        print(f"ERROR: run_cutlass_moe_fp4 not found in {path2}")
    else:
        # Find end of function
        func_end = code.find('\ndef ', func_start + 10)
        old_func = code[func_start:func_end]

        # Get the function signature
        sig_end = old_func.find(') -> ')
        if sig_end < 0:
            sig_end = old_func.find('):\n')

        # Rename original
        old_renamed = old_func.replace('def run_cutlass_moe_fp4(', 'def _run_cutlass_moe_fp4_grouped(')

        # New dispatcher + loop implementation
        loop_code = '''def run_cutlass_moe_fp4(
    a: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    a1_gscale: torch.Tensor,
    a2_gscale: torch.Tensor,
    activation: "MoEActivation" = MoEActivation.SILU,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    apply_router_weight_on_input: bool = False,
    workspace13: Optional[torch.Tensor] = None,
    workspace2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 MoE dispatch — grouped GEMM or per-expert loop."""
    import os
    if os.environ.get("VLLM_NVFP4_MOE_LOOP", "0") == "1":
        return _run_cutlass_moe_fp4_loop(
            a, w1_fp4, w1_blockscale, w1_alphas, w2_fp4, w2_blockscale,
            w2_alphas, topk_weights, topk_ids, a1_gscale, a2_gscale,
            activation, global_num_experts, expert_map,
            apply_router_weight_on_input, workspace13, workspace2)
    return _run_cutlass_moe_fp4_grouped(
        a, w1_fp4, w1_blockscale, w1_alphas, w2_fp4, w2_blockscale,
        w2_alphas, topk_weights, topk_ids, a1_gscale, a2_gscale,
        activation, global_num_experts, expert_map,
        apply_router_weight_on_input, workspace13, workspace2)


def _run_cutlass_moe_fp4_loop(
    a, w1_fp4, w1_blockscale, w1_alphas, w2_fp4, w2_blockscale,
    w2_alphas, topk_weights, topk_ids, a1_gscale, a2_gscale,
    activation, global_num_experts, expert_map,
    apply_router_weight_on_input, workspace13, workspace2,
) -> torch.Tensor:
    """Per-expert FP4 single-GEMM loop — SM120 grouped GEMM workaround."""
    from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant

    M, K = a.shape
    topk = topk_ids.shape[1]
    output = torch.zeros(M, K, device=a.device, dtype=a.dtype)

    for k_idx in range(topk):
        expert_ids = topk_ids[:, k_idx]
        weights = topk_weights[:, k_idx]

        for eid_tensor in expert_ids.unique():
            eid = eid_tensor.item()
            local_eid = eid
            if expert_map is not None:
                mapped = expert_map[eid].item()
                if mapped < 0:
                    continue
                local_eid = mapped

            mask = (expert_ids == eid)
            if not mask.any():
                continue

            tokens = a[mask]

            # Quantize activations to FP4
            a_scale = a1_gscale[local_eid:local_eid+1]
            a_fp4, a_bs = scaled_fp4_quant(tokens, a_scale)

            # GEMM 1: gate_up = tokens @ w1^T
            alpha1 = w1_alphas[local_eid:local_eid+1]
            gate_up = cutlass_scaled_fp4_mm(
                a_fp4, w1_fp4[local_eid], a_bs,
                w1_blockscale[local_eid], alpha1, a.dtype)

            # Activation (SiLU + Mul for gated)
            half = gate_up.shape[1] // 2
            intermediate = torch.nn.functional.silu(gate_up[:, :half]) * gate_up[:, half:]

            # Quantize intermediate
            a2_scale = a2_gscale[local_eid:local_eid+1]
            int_fp4, int_bs = scaled_fp4_quant(intermediate, a2_scale)

            # GEMM 2: down = intermediate @ w2^T
            alpha2 = w2_alphas[local_eid:local_eid+1]
            expert_out = cutlass_scaled_fp4_mm(
                int_fp4, w2_fp4[local_eid], int_bs,
                w2_blockscale[local_eid], alpha2, a.dtype)

            output[mask] += weights[mask].unsqueeze(1) * expert_out

    return output


'''

        code = code.replace(old_func, loop_code + old_renamed)
        with open(path2, 'w') as f:
            f.write(code)
        print(f"Fixed {path2}: added per-expert loop with VLLM_NVFP4_MOE_LOOP=1")

print("\nDone. Test with: VLLM_NVFP4_MOE_LOOP=1")
