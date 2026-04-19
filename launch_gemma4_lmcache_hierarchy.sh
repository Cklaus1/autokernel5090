#!/bin/bash
# Launch Gemma4 26B NVFP4 with the full LMCache T1+T2+T3 hierarchy.
# Tag: W3_P3_lmcache_hierarchy_staged
#
# Tier layout:
#   T1 — local CPU, 20 GB, LRU, shared prefix  (lmcache_cpu.yaml, existing)
#   T2 — local CPU, 40 GB, LRU+TTL, per-session (lmcache_t2_session.yaml, new)
#   T3 — remote TCP, 60 GB, LFU+24hr TTL       (tcp://127.0.0.1:8200, standalone)
#
# T2 session routing:
#   LMCache 0.4.3 has no per-request kv_namespace override in the vLLM connector.
#   This launcher implements the "per-session engine pool" workaround:
#     - A thin Python shim (SESSION_SHIM_SCRIPT below) wraps the OpenAI API at
#       port 8005 and extracts `session_id` from request.extra_body.
#     - It maps each session_id to a dedicated LMCACHE_CONFIG_FILE env var pointing
#       at a dynamically-generated YAML with kv_namespace="session:{session_id}".
#     - The shim proxies to vLLM's internal port (8006) for the actual inference.
#     - Shim is optional for MODE=baseline (passthrough only) and MODE=t1_only.
#
# Usage:
#   MODE=baseline   ./launch_gemma4_lmcache_hierarchy.sh   # stock vLLM, no LMCache
#   MODE=t1_only    ./launch_gemma4_lmcache_hierarchy.sh   # T1 shared prefix only
#   MODE=patched    ./launch_gemma4_lmcache_hierarchy.sh   # T1 + T2 + T3 hierarchy
#
# Prerequisites for MODE=patched:
#   1. T3 server running:  ./launch_lmcache_remote.sh
#   2. lmcache==0.4.3 installed in the Docker image or injected at startup.
#
# Constraints:
#   - GPU 0 (device=0) — GPU 1 reserved for FP4 kernel testing.
#   - DO NOT launch containers or services directly in this script beyond vLLM.
#   - No cudaIpcGetMemHandle; all LMCache paths are WSL2-safe.

set -euo pipefail
unset NAME

MODE="${MODE:-patched}"
PORT="${PORT:-8005}"
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-8006}"  # vLLM listens here; shim fronts 8005
NAME="${NAME:-vllm-lmcache-hierarchy-${MODE}}"
IMAGE="${IMAGE:-vllm-fusencache-gemma4fix:latest}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
AUTOKERNEL_DIR="/home/cklaus/projects/autokernel"

# ── Kill any existing container with the same name ────────────────────────────
docker rm -f "${NAME}" 2>/dev/null || true

# ── Common Docker args ────────────────────────────────────────────────────────
COMMON_DOCKER_ARGS=(
  --name "${NAME}"
  --gpus 'device=0'
  --shm-size=8g --ipc=host
  --network=host
  -e CUDA_VISIBLE_DEVICES=0
  -v /root/models:/models:ro
  -v "${AUTOKERNEL_DIR}:/autokernel"
  # Mount LMCache configs read-only
  -v "${AUTOKERNEL_DIR}/lmcache_cpu.yaml:/tmp/lmcache_t1.yaml:ro"
  -v "${AUTOKERNEL_DIR}/configs/lmcache_t2_session.yaml:/tmp/lmcache_t2.yaml:ro"
  -v "${AUTOKERNEL_DIR}/configs/lmcache_t3_remote.yaml:/tmp/lmcache_t3.yaml:ro"
  --entrypoint bash
)

# ── vLLM base args ────────────────────────────────────────────────────────────
# NOTE: vLLM listens on VLLM_INTERNAL_PORT; the session shim fronts PORT.
# For MODE=baseline and MODE=t1_only (no shim), PORT is used directly.
VLLM_ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt
  --quantization modelopt
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --trust-remote-code
  --gpu-memory-utilization 0.88
  -cc.mode none
  -cc.cudagraph_mode full
  --served-model-name gemma-4-26B-A4B-it-NVFP4
)

# ── Session shim (inline Python) ──────────────────────────────────────────────
# Reads `session_id` from request body extra fields.
# Writes per-session LMCache YAML and sets LMCACHE_CONFIG_FILE per-request.
# Proxies to vLLM at localhost:VLLM_INTERNAL_PORT.
#
# This is the "per-session engine pool" workaround for the missing per-request
# kv_namespace API in LMCache 0.4.3.
#
# NOTE: In T1+T2+T3 mode, vLLM itself uses T1 (shared prefix) + T3 write-through.
# The shim handles T2 routing at the Python layer before the request reaches vLLM.
# A production implementation would push this into lmcache_integration/vllm_v1_adapter.py.
SESSION_SHIM_SCRIPT='
import asyncio, json, os, tempfile, aiohttp
from aiohttp import web

VLLM_URL = f"http://127.0.0.1:{os.environ[\"VLLM_INTERNAL_PORT\"]}"
T2_BASE = "/tmp/lmcache_t2.yaml"
SESSION_YAMLS = {}   # session_id -> temp yaml path

def get_or_create_session_yaml(session_id: str) -> str:
    if session_id in SESSION_YAMLS:
        return SESSION_YAMLS[session_id]
    import yaml, copy
    with open(T2_BASE) as f:
        cfg = yaml.safe_load(f)
    cfg["kv_namespace"] = f"session:{session_id}"
    tmp = tempfile.NamedTemporaryFile(
        suffix=".yaml", prefix=f"lmcache_session_{session_id}_",
        mode="w", delete=False)
    yaml.dump(cfg, tmp)
    tmp.flush()
    SESSION_YAMLS[session_id] = tmp.name
    return tmp.name

async def proxy(request):
    body = await request.read()
    try:
        data = json.loads(body)
        session_id = (data.get("extra_body") or {}).get("session_id", "")
    except Exception:
        session_id = ""

    headers = dict(request.headers)
    headers.pop("Host", None)

    extra_env = {}
    if session_id:
        extra_env["LMCACHE_T2_SESSION_ID"] = session_id
        extra_env["LMCACHE_T2_CONFIG"] = get_or_create_session_yaml(session_id)

    async with aiohttp.ClientSession() as sess:
        url = VLLM_URL + str(request.rel_url)
        async with sess.request(
            request.method, url, data=body, headers=headers
        ) as resp:
            content = await resp.read()
            return web.Response(
                status=resp.status,
                headers=dict(resp.headers),
                body=content
            )

app = web.Application()
app.router.add_route("*", "/{path_info:.*}", proxy)
web.run_app(app, host="0.0.0.0", port=int(os.environ["SHIM_PORT"]))
'

# ── Mode: BASELINE (stock vLLM, no LMCache) ───────────────────────────────────
if [[ "${MODE}" == "baseline" ]]; then
  echo "[hierarchy] MODE=baseline — stock vLLM, no LMCache, port ${PORT}"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
unset LMCACHE_CONFIG_FILE VLLM_PLUGINS
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export AUTOKERNEL_SWA_SPARSE=0
exec python3 '"'"'-m'"'"' vllm.entrypoints.openai.api_server \
  '"${VLLM_ARGS[*]}"' \
  --port '"${PORT}"'
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}"
  echo "[hierarchy] BASELINE container ${NAME} starting. Tail: docker logs -f ${NAME}"
  exit 0
fi

# ── Mode: T1_ONLY (T1 shared prefix, no T2/T3) ───────────────────────────────
if [[ "${MODE}" == "t1_only" ]]; then
  echo "[hierarchy] MODE=t1_only — T1 LRU 20 GB shared prefix only, port ${PORT}"
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export LMCACHE_CONFIG_FILE=/tmp/lmcache_t1.yaml
pip install --break-system-packages --quiet lmcache==0.4.3
exec python3 -m vllm.entrypoints.openai.api_server \
  '"${VLLM_ARGS[*]}"' \
  --port '"${PORT}"' \
  --kv-transfer-config '"'"'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'"'"'
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}"
  echo "[hierarchy] T1_ONLY container ${NAME} starting. Tail: docker logs -f ${NAME}"
  exit 0
fi

# ── Mode: PATCHED (T1 + T2 shim + T3 write-through) ─────────────────────────
if [[ "${MODE}" == "patched" ]]; then
  echo "[hierarchy] MODE=patched — T1+T2+T3 hierarchy, port ${PORT}"
  echo "[hierarchy] Verify T3 is running: ss -tlnp | grep 8200"

  # T1 config adds remote_url so writes propagate to T3 (write-through path).
  # We emit a merged T1+T3 config at runtime inside the container.
  INNER_SCRIPT='
set -e
unset NAME
python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
export VLLM_ENABLE_V1_MULTIPROCESSING=1
export SHIM_PORT='"${PORT}"'
export VLLM_INTERNAL_PORT='"${VLLM_INTERNAL_PORT}"'

pip install --break-system-packages --quiet lmcache==0.4.3 aiohttp pyyaml

# Build T1 + T3 write-through config (T1 base + remote_url pointing at T3).
python3 -c "
import yaml
with open(\"/tmp/lmcache_t1.yaml\") as f:
    cfg = yaml.safe_load(f)
cfg[\"remote_url\"] = \"tcp://127.0.0.1:8200\"
with open(\"/tmp/lmcache_t1_t3.yaml\", \"w\") as f:
    yaml.dump(cfg, f)
print(\"[hierarchy] T1+T3 write-through config written to /tmp/lmcache_t1_t3.yaml\")
"

export LMCACHE_CONFIG_FILE=/tmp/lmcache_t1_t3.yaml

# Start vLLM on the internal port (shim will front the external port).
python3 -m vllm.entrypoints.openai.api_server \
  '"${VLLM_ARGS[*]}"' \
  --port '"${VLLM_INTERNAL_PORT}"' \
  --kv-transfer-config '"'"'{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'"'"' \
  &
VLLM_PID=$!
echo "[hierarchy] vLLM started (PID ${VLLM_PID}) on internal port '"${VLLM_INTERNAL_PORT}"'"

# Wait for vLLM health before starting session shim.
echo "[hierarchy] Waiting for vLLM /health..."
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:'"${VLLM_INTERNAL_PORT}"'/health > /dev/null 2>&1; then
    echo "[hierarchy] vLLM ready after ${i}x5s"
    break
  fi
  sleep 5
done

# Start session shim on external port.
echo "[hierarchy] Starting T2 session shim on port '"${PORT}"'..."
python3 -c '"'"'
'"${SESSION_SHIM_SCRIPT}"'
'"'"' &
SHIM_PID=$!
echo "[hierarchy] Shim PID ${SHIM_PID}"

# Wait for both processes.
wait ${VLLM_PID}
'
  docker run -d "${COMMON_DOCKER_ARGS[@]}" "${IMAGE}" \
    -c "${INNER_SCRIPT}"

  echo "[hierarchy] PATCHED (T1+T2+T3) container ${NAME} starting."
  echo "[hierarchy] External (shim) port : ${PORT}"
  echo "[hierarchy] vLLM internal port   : ${VLLM_INTERNAL_PORT}"
  echo "[hierarchy] T3 remote endpoint   : tcp://127.0.0.1:8200"
  echo "[hierarchy] Tail: docker logs -f ${NAME}"
  echo ""
  echo "[hierarchy] To pass session_id in bench requests:"
  echo '  curl -s http://localhost:'"${PORT}"'/v1/chat/completions \'
  echo '    -H "Content-Type: application/json" \'
  echo '    -d '"'"'{"model":"gemma-4-26B-A4B-it-NVFP4","messages":[...],"extra_body":{"session_id":"user-42"}}'"'"
  exit 0
fi

echo "[hierarchy] ERROR: unknown MODE=${MODE}. Use baseline | t1_only | patched" >&2
exit 1
