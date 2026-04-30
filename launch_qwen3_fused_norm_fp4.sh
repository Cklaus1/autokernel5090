#!/bin/bash
# Launch Qwen3-30B-A3B NVFP4 on GPU 1 with BOTH plugins enabled:
#   * T2-N fused shuffle+quant (patches run_cutlass_moe_fp4)
#   * fused_norm_fp4_qwen3   (patches Qwen3MoeDecoderLayer.forward to
#                             fuse input_layernorm -> qkv_proj via the
#                             native e2m1x2 rms_norm_dynamic_fp4_quant kernel)
#
# Installs two dist-info stubs inside the container so vLLM's
# load_general_plugins() picks up both entry points. The fused .so is
# loaded by the plugin at register() time.
set -e
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8003}"
NAME="${NAME:-vllm-fused-norm-fp4-qwen3-${MODE}}"

docker rm -f "${NAME}" 2>/dev/null || true

COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus 'device=1'
  --shm-size=8g --ipc=host
  -e CUDA_VISIBLE_DEVICES=0
  -v /root/models:/models:ro
  -v /home/cklaus/projects/autokernel:/autokernel
  -p "${PORT}:${PORT}"
  --entrypoint bash
)

VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model /models/Qwen3-30B-A3B-NVFP4
  --quantization modelopt
  --max-model-len 4096
  --max-num-seqs 512
  --trust-remote-code
  --port "${PORT}"
  --served-model-name Qwen3-30B-A3B-NVFP4
  --kv-cache-dtype fp8
  --gpu-memory-utilization 0.92
  -cc.mode none
  -cc.cudagraph_mode full
)

if [[ "${MODE}" == "patched" ]]; then
  echo "[fused_norm_fp4_qwen3 launch] starting patched server (T2-N + fused_norm_fp4_qwen3)"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true

# --- T2-N plugin dist-info ---------------------------------------------------
T2N_INFO=/usr/local/lib/python3.12/dist-packages/fused_shuffle_quant_plugin-0.1.dist-info
mkdir -p "$T2N_INFO"
cat > "$T2N_INFO/METADATA" <<META
Metadata-Version: 2.1
Name: fused-shuffle-quant-plugin
Version: 0.1
META
cat > "$T2N_INFO/entry_points.txt" <<EP
[vllm.general_plugins]
fused_shuffle_quant = fused_shuffle_quant_plugin:register
EP
cat > "$T2N_INFO/RECORD" <<REC
fused_shuffle_quant_plugin-0.1.dist-info/METADATA,,
fused_shuffle_quant_plugin-0.1.dist-info/entry_points.txt,,
fused_shuffle_quant_plugin-0.1.dist-info/RECORD,,
REC

# --- fused_norm_fp4_qwen3 plugin dist-info ----------------------------------
FN_INFO=/usr/local/lib/python3.12/dist-packages/fused_norm_fp4_qwen3-0.1.dist-info
mkdir -p "$FN_INFO"
cat > "$FN_INFO/METADATA" <<META
Metadata-Version: 2.1
Name: fused-norm-fp4-qwen3
Version: 0.1
META
cat > "$FN_INFO/entry_points.txt" <<EP
[vllm.general_plugins]
fused_norm_fp4_qwen3 = wire_fused_norm_fp4_qwen3_v2:register
EP
cat > "$FN_INFO/RECORD" <<REC
fused_norm_fp4_qwen3-0.1.dist-info/METADATA,,
fused_norm_fp4_qwen3-0.1.dist-info/entry_points.txt,,
fused_norm_fp4_qwen3-0.1.dist-info/RECORD,,
REC

# Activate both plugins.
# W8 fix (tag W8_rank3_silent_fire_debug): repo root /autokernel MUST be on
# PYTHONPATH so the Rank 3 import
#     from kernels.triton.fused_unshuffle_weightedsum import ...
# succeeds. Without this, the import in
# patches/fused_shuffle_quant_wrapper.py:_try_load_unshuffle_weightedsum
# raises ModuleNotFoundError, the P1 warning fires under logger name
# fused_shuffle_quant_wrapper (outside vLLM allowlist, easy to miss),
# and the two-pass fallback runs silently.
export PYTHONPATH=/autokernel:/autokernel/patches:${PYTHONPATH:-}
export AUTOKERNEL_FUSED_SHUFFLE_QUANT=1
export AUTOKERNEL_FUSED_SHUFFLE_QUANT_SO=/autokernel/workspace/fused_shuffle_quant_sm120a.so
export AUTOKERNEL_FUSED_NORM_FP4_QWEN3=1
export AUTOKERNEL_FUSED_NORM_FP4_SO=/autokernel/workspace/fused_rms_norm_fp4_cu13.so
export VLLM_PLUGINS=fused_shuffle_quant,fused_norm_fp4_qwen3
export VLLM_USE_FLASHINFER_MOE_FP4=0
export AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM=1
export AUTOKERNEL_FUSED_QKNORM_ROPE=1
export AUTOKERNEL_FUSED_PERSISTENT_BUF=0

exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
else
  echo "[fused_norm_fp4_qwen3 launch] starting baseline server (no plugins)"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
unset VLLM_PLUGINS
export AUTOKERNEL_FUSED_SHUFFLE_QUANT=0
export AUTOKERNEL_FUSED_NORM_FP4_QWEN3=0
exec python3 "$@"
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" vllm-fusencache:latest \
    -c "${INNER_SCRIPT}" bash "${VLLM_ARGS[@]}"
fi

echo "[fused_norm_fp4_qwen3 launch] container ${NAME} starting on port ${PORT}. Tail with:"
echo "    docker logs -f ${NAME}"
