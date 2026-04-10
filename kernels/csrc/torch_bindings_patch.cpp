// ============================================================
// Upstream PR Patch: Add to vLLM's csrc/torch_bindings.cpp
// ============================================================
//
// Step 1: Add these forward declarations near the top of the file
//         (after the existing rms_norm_per_block_quant declaration):

void rms_norm_dynamic_fp4_quant(
    torch::Tensor& result,
    torch::Tensor& result_scale,
    torch::Tensor const& input,
    torch::Tensor const& weight,
    torch::Tensor const& input_global_scale,
    double epsilon,
    bool is_sf_swizzled_layout);

void fused_add_rms_norm_dynamic_fp4_quant(
    torch::Tensor& result,
    torch::Tensor& result_scale,
    torch::Tensor& input,
    torch::Tensor const& weight,
    torch::Tensor& residual,
    torch::Tensor const& input_global_scale,
    double epsilon,
    bool is_sf_swizzled_layout);

// Step 2: Add these registrations inside TORCH_LIBRARY_EXPAND(_C, ops)
//         block, after the rms_norm_per_block_quant registration:
//
//   // Fused RMSNorm + dynamic FP4 block quantization (SM120+)
//   ops.def(
//       "rms_norm_dynamic_fp4_quant(Tensor! result, Tensor! result_scale, "
//       "Tensor input, Tensor weight, Tensor input_global_scale, "
//       "float epsilon, bool is_sf_swizzled_layout) -> ()");
//   ops.impl("rms_norm_dynamic_fp4_quant", torch::kCUDA,
//            &rms_norm_dynamic_fp4_quant);
//
//   // Fused residual-add + RMSNorm + dynamic FP4 block quantization (SM120+)
//   ops.def(
//       "fused_add_rms_norm_dynamic_fp4_quant(Tensor! result, "
//       "Tensor! result_scale, Tensor! input, Tensor weight, "
//       "Tensor! residual, Tensor input_global_scale, "
//       "float epsilon, bool is_sf_swizzled_layout) -> ()");
//   ops.impl("fused_add_rms_norm_dynamic_fp4_quant", torch::kCUDA,
//            &fused_add_rms_norm_dynamic_fp4_quant);
//
// Step 3: Add rms_norm_dynamic_fp4_quant.cu to CMakeLists.txt
//         In the FP4_ARCHS section (~line 914), add to SRCS:
//           "csrc/quantization/fused_kernels/rms_norm_dynamic_fp4_quant.cu"
//
// Step 4: Add fake tensor registration to vllm/_custom_ops.py:
//
//   if hasattr(torch.ops._C, "rms_norm_dynamic_fp4_quant"):
//       @register_fake("_C::rms_norm_dynamic_fp4_quant")
//       def _rms_norm_dynamic_fp4_quant_fake(
//           result, result_scale, input, weight, input_global_scale,
//           epsilon, is_sf_swizzled_layout):
//           pass
//
//       @register_fake("_C::fused_add_rms_norm_dynamic_fp4_quant")
//       def _fused_add_rms_norm_dynamic_fp4_quant_fake(
//           result, result_scale, input, weight, residual,
//           input_global_scale, epsilon, is_sf_swizzled_layout):
//           pass
