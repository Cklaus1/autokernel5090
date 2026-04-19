#!/bin/bash
# Disaggregated Prefill/Decode serving: Gemma4 26B NVFP4, 1P1D topology
#
# GPU 0 — prefill only (kv_producer, port 8100)
# GPU 1 — decode only  (kv_consumer, port 8101)
# Host  — proxy       (routes requests + embeds ZMQ addrs, port 8200)
#
# Transport: P2pNcclConnector over PCIe + ZMQ (no shared NCCL group at startup)
# KV cache dtype: BF16  (FP8 unsupported: FlashInfer rejects head_size=512 with fp8)
#
# Usage:
#   ./serve_disaggregated.sh            # start prefill + decode + proxy
#   ./serve_disaggregated.sh stop       # stop all three
#   ./serve_disaggregated.sh bench      # run quick latency benchmark
#
# Endpoints:
#   :8200  — proxy  (send ALL client requests here)
#   :8100  — prefill instance (internal)
#   :8101  — decode instance  (internal)
#
# See: plans/disaggregated_serving.md — full design spec
#      plans/rtx_pro6000_experiments.md ASI-1 — kill criterion + expected metrics
#
# Kill criterion (ASI-1): if P99 TTFT under mixed load is < 1.5x better than
# DP=2 at C=64, KV transfer overhead dominates — revert to serve_gemma4_dp2.sh

set -euo pipefail

IMAGE="vllm-built:latest"
MODEL_DIR="/root/models"
MODEL_PATH="/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_NAME="gemma-4-26B-A4B-it-NVFP4"

# HTTP ports
PREFILL_PORT=8100
DECODE_PORT=8101
PROXY_PORT=8200

# ZMQ ports for P2pNcclConnector KV transfer
# Each instance binds its own ZMQ ROUTER socket.
# In single-GPU-per-container mode, world_group.rank=0 so port_offset=0;
# we therefore give each container a different base kv_port.
KV_IP="127.0.0.1"
KV_PORT_PREFILL=14579   # prefill ZMQ bind
KV_PORT_DECODE=14580    # decode ZMQ bind

# 2 GB transfer buffer — enough for several large in-flight requests.
KV_BUFFER_SIZE=2000000000

# CPU pinning — one CCD per GPU on the 9950X3D (16C/32T).
CPUS_GPU0="0-15"    # CCD 0 → prefill (GPU 0)
CPUS_GPU1="16-31"   # CCD 1 → decode  (GPU 1)

CONTAINER_PREFILL="vllm-disagg-prefill"
CONTAINER_DECODE="vllm-disagg-decode"
PROXY_PID_FILE="/tmp/vllm-disagg-proxy.pid"
PROXY_LOG="/tmp/vllm-disagg-proxy.log"

# ------------------------------------------------------------------ helpers --

stop_servers() {
    docker rm -f "${CONTAINER_PREFILL}" "${CONTAINER_DECODE}" 2>/dev/null || true
    if [ -f "${PROXY_PID_FILE}" ]; then
        PID=$(cat "${PROXY_PID_FILE}")
        kill "${PID}" 2>/dev/null || true
        rm -f "${PROXY_PID_FILE}"
    fi
    # Also kill any leftover proxy processes
    pkill -f "vllm-disagg-proxy" 2>/dev/null || true
    echo "Disaggregated instances stopped."
}

wait_healthy() {
    local port="$1"
    local label="$2"
    local timeout=480
    printf "  Waiting for %s (port %s)" "${label}" "${port}"
    for i in $(seq 1 ${timeout}); do
        code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/health" 2>/dev/null || echo "000")
        if [ "${code}" = "200" ]; then
            printf " — ready in %ds\n" "${i}"
            return 0
        fi
        printf "\r  Waiting for %s (port %s) — %3ds / %ds" \
            "${label}" "${port}" "${i}" "${timeout}"
        sleep 1
    done
    printf "\n"
    echo "ERROR: ${label} failed to start within ${timeout}s"
    return 1
}

run_bench() {
    echo ""
    echo "=== Quick latency benchmark ==="
    echo "Concurrency 1, 4, 8 — bimodal prompt (50% 256-tok, 50% 4K-tok)"
    echo "Sending requests through proxy (port ${PROXY_PORT})"
    echo ""

    for concurrency in 1 4 8; do
        echo "--- C=${concurrency} ---"
        python3 - <<PYEOF
import asyncio, time, json, urllib.request, statistics

MODEL = "${MODEL_NAME}"
PORT  = ${PROXY_PORT}
C     = ${concurrency}

SHORT_PROMPT = "What is 2+2? Answer briefly." * 4          # ~256 tok
LONG_PROMPT  = ("Explain the history of computing in detail. " * 200)[:16000]  # ~4K tok

prompts = [SHORT_PROMPT if i % 2 == 0 else LONG_PROMPT for i in range(C)]

async def one_request(session_id, prompt):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "stream": False,
    }).encode()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    import urllib.error
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
            ttft = (time.perf_counter() - t0) * 1000
            return ttft, body.get("usage", {}).get("completion_tokens", 0)
    except urllib.error.URLError as e:
        return None, str(e)

async def run():
    tasks = [one_request(i, p) for i, p in enumerate(prompts)]
    t_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - t_start
    ttfts   = [r[0] for r in results if r[0] is not None]
    errors  = [r for r in results if r[0] is None]
    if errors:
        print(f"  Errors: {len(errors)} — {errors[0][1]}")
    if ttfts:
        print(f"  TTFT  p50={statistics.median(ttfts):.0f}ms  "
              f"p99={sorted(ttfts)[int(len(ttfts)*0.99)]:.0f}ms  "
              f"max={max(ttfts):.0f}ms")
        print(f"  Wall  {elapsed*1000:.0f}ms  requests={len(ttfts)}")

asyncio.run(run())
PYEOF
    done

    echo ""
    echo "Expected targets (disaggregated_serving.md §6):"
    echo "  C=8 P99 TTFT: 640ms (collocated) → 120ms (disaggregated)"
    echo "  Decode tok/s under heavy prefill: 15 → 55 tok/s"
    echo ""
    echo "Kill criterion (ASI-1): abort if P99 TTFT < 1.5x better than DP=2 at C=64."
}

# ------------------------------------------------------------------ dispatch --

if [ "${1:-}" = "stop" ]; then
    stop_servers
    exit 0
fi

if [ "${1:-}" = "bench" ]; then
    run_bench
    exit 0
fi

# ------------------------------------------------------------------ launch ---

stop_servers

# Ensure msgpack is installed in the image (P2pNcclConnector dependency).
echo "=== Checking msgpack dependency ==="
if ! docker run --rm --entrypoint python3 "${IMAGE}" -c "import msgpack" 2>/dev/null; then
    echo "  msgpack missing — installing into ${IMAGE} ..."
    CID=$(docker run -d --entrypoint bash "${IMAGE}" -c "pip install --break-system-packages msgpack -q && echo done")
    docker wait "${CID}"
    docker commit "${CID}" "${IMAGE}"
    docker rm "${CID}"
    echo "  msgpack installed — image updated."
else
    echo "  msgpack already present — OK."
fi

# Ensure proxy dependencies are available on the host
echo ""
echo "=== Checking proxy dependencies (quart, aiohttp) ==="
python3 -c "import quart, aiohttp" 2>/dev/null || \
    pip3 install --break-system-packages --ignore-installed quart aiohttp -q 2>/dev/null || \
    python3 -m pip install --break-system-packages --ignore-installed quart aiohttp -q 2>/dev/null
python3 -c "import quart, aiohttp; print('  proxy deps OK.')"
echo ""

# Prerequisite reminder (non-fatal)
echo "=== PRE-FLIGHT CHECKS ==="
echo "  ZMQ ports: ${KV_IP}:${KV_PORT_PREFILL} (prefill), ${KV_IP}:${KV_PORT_DECODE} (decode)"
echo "  KV dtype: BF16 (FP8 disabled — FlashInfer rejects head_size=512 with fp8 for Gemma4)"
echo ""

# ------------------------------------------------------------------
# Instance 0 — Prefill GPU (GPU 0, port 8100)
# ------------------------------------------------------------------
echo "=== Starting prefill instance (GPU 0, port ${PREFILL_PORT}, ZMQ :${KV_PORT_PREFILL}) ==="
# WSL2 GPU isolation fix (KILL_PATTERNS.md §P4): --gpus alone leaks; must also set
# NVIDIA_VISIBLE_DEVICES (host mapping) + CUDA_VISIBLE_DEVICES=0 (container-internal view).

docker run -d \
    --name "${CONTAINER_PREFILL}" \
    --network=host \
    --gpus '"device=0"' \
    --memory=80g \
    --cpuset-cpus="${CPUS_GPU0}" \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "${MODEL_DIR}:/models:ro" \
    --entrypoint bash \
    "${IMAGE}" -c \
    'python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
exec python3 -m vllm.entrypoints.openai.api_server \
        --model '"${MODEL_PATH}"' \
        --quantization modelopt \
        --port '"${PREFILL_PORT}"' \
        --served-model-name '"${MODEL_NAME}"' \
        --gpu-memory-utilization 0.80 \
        --max-model-len 28672 \
        --kv-transfer-config '"'"'{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"'"${KV_IP}"'","kv_port":'"${KV_PORT_PREFILL}"',"kv_buffer_size":'"${KV_BUFFER_SIZE}"'}'"'"' \
        -cc.mode none \
        -cc.cudagraph_mode full'

# ------------------------------------------------------------------
# Instance 1 — Decode GPU (GPU 1, port 8101)
# ------------------------------------------------------------------
echo "=== Starting decode instance  (GPU 1, port ${DECODE_PORT}, ZMQ :${KV_PORT_DECODE}) ==="
# WSL2 GPU isolation fix: GPU 1 host device mapped to CUDA index 0 inside this container.
# NVIDIA_VISIBLE_DEVICES=1 selects host GPU 1; CUDA_VISIBLE_DEVICES=0 re-indexes it to 0
# inside the container so vLLM sees exactly one GPU and cannot touch the prefill GPU.

docker run -d \
    --name "${CONTAINER_DECODE}" \
    --network=host \
    --gpus '"device=1"' \
    --memory=80g \
    --cpuset-cpus="${CPUS_GPU1}" \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v "${MODEL_DIR}:/models:ro" \
    --entrypoint bash \
    "${IMAGE}" -c \
    'python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'"'"'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}'"'"', flush=True)
" 2>&1 || true
exec python3 -m vllm.entrypoints.openai.api_server \
        --model '"${MODEL_PATH}"' \
        --quantization modelopt \
        --port '"${DECODE_PORT}"' \
        --served-model-name '"${MODEL_NAME}"' \
        --gpu-memory-utilization 0.80 \
        --max-model-len 8192 \
        --max-num-seqs 128 \
        --kv-transfer-config '"'"'{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"'"${KV_IP}"'","kv_port":'"${KV_PORT_DECODE}"',"kv_buffer_size":'"${KV_BUFFER_SIZE}"'}'"'"' \
        -cc.mode none \
        -cc.cudagraph_mode full'

# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------
echo ""
echo "=== Waiting for both instances to become healthy ==="
echo "    (Both load full Gemma4 26B weights — CUDA graph capture takes 5-8 min)"
echo ""

wait_healthy "${PREFILL_PORT}" "prefill (GPU 0)" || {
    echo "  Prefill failed. Logs:"
    docker logs "${CONTAINER_PREFILL}" 2>&1 | tail -20
    exit 1
}
wait_healthy "${DECODE_PORT}"  "decode  (GPU 1)" || {
    echo "  Decode failed. Logs:"
    docker logs "${CONTAINER_DECODE}" 2>&1 | tail -20
    exit 1
}

# ------------------------------------------------------------------
# Proxy server
# ------------------------------------------------------------------
# The P2pNcclConnector requires request_ids to contain:
#   ___prefill_addr_IP:PORT___decode_addr_IP:PORT_UUID
# This proxy embeds those addresses and routes:
#   1. Sends prefill-only request (max_tokens=1) to prefill instance
#   2. Sends full request (full decode) to decode instance
# ------------------------------------------------------------------
echo ""
echo "=== Starting disaggregated proxy (port ${PROXY_PORT}) ==="

python3 - <<PROXY_SCRIPT &
import asyncio, uuid, os, sys
import aiohttp
from quart import Quart, make_response, request

PREFILL_URL  = "http://127.0.0.1:${PREFILL_PORT}"
DECODE_URL   = "http://127.0.0.1:${DECODE_PORT}"
PREFILL_ZMQ  = "${KV_IP}:${KV_PORT_PREFILL}"
DECODE_ZMQ   = "${KV_IP}:${KV_PORT_DECODE}"
PROXY_PORT   = ${PROXY_PORT}
TIMEOUT      = aiohttp.ClientTimeout(total=3600)

app = Quart("vllm-disagg-proxy")

@app.route("/health")
async def health():
    return "", 200

@app.route("/v1/completions",      methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        data = await request.get_json()
        prefill_req = dict(data)
        prefill_req["max_tokens"] = 1
        prefill_req.pop("max_completion_tokens", None)

        req_id = (
            f"___prefill_addr_{PREFILL_ZMQ}___decode_addr_"
            f"{DECODE_ZMQ}_{uuid.uuid4().hex}"
        )
        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": req_id,
        }

        async with aiohttp.ClientSession(timeout=TIMEOUT) as sess:
            # Step 1: prefill only
            async with sess.post(
                PREFILL_URL + request.path,
                json=prefill_req,
                headers=headers,
            ) as resp:
                await resp.read()  # consume (discard prefill output)

            # Step 2: decode
            async def generate():
                async with sess.post(
                    DECODE_URL + request.path,
                    json=data,
                    headers=headers,
                ) as resp:
                    async for chunk in resp.content.iter_chunked(1024):
                        yield chunk

            response = await make_response(generate())
            response.timeout = None
            return response

    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}, 500

# Write PID for stop_servers
with open("${PROXY_PID_FILE}", "w") as f:
    f.write(str(os.getpid()))

app.run(host="0.0.0.0", port=PROXY_PORT)
PROXY_SCRIPT

PROXY_BGPID=$!
echo "  Proxy PID: ${PROXY_BGPID} → ${PROXY_PID_FILE}"
echo "${PROXY_BGPID}" > "${PROXY_PID_FILE}"

# Wait for proxy to be ready
for i in $(seq 1 30); do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:${PROXY_PORT}/health" 2>/dev/null || echo "000")
    if [ "${code}" = "200" ]; then
        echo "  Proxy ready in ${i}s."
        break
    fi
    sleep 1
done

# ------------------------------------------------------------------
# Startup summary
# ------------------------------------------------------------------
echo ""
echo "=== DISAGGREGATED SERVING READY ==="
echo ""
echo "  Topology:   1P1D (1 prefill GPU 0, 1 decode GPU 1)"
echo "  Transport:  P2pNcclConnector + ZMQ over PCIe"
echo "  KV dtype:   BF16  (~19 ms transfer per request)"
echo "  Proxy:      port ${PROXY_PORT}  (embeds ZMQ addrs in request_id)"
echo ""
echo "  GPU 0 :${PREFILL_PORT}  prefill-only  max_len=28672  mem_util=0.90"
echo "  GPU 1 :${DECODE_PORT}  decode-only   max_len=8192   mem_util=0.85  max_seqs=128"
echo ""
echo "  Send ALL client requests to the PROXY:"
echo "    http://localhost:${PROXY_PORT}/v1/chat/completions"
echo ""
echo "Quick test:"
echo "  curl http://localhost:${PROXY_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":50}'"
echo ""
echo "Benchmark:"
echo "  ./serve_disaggregated.sh bench"
echo ""
echo "Stop:"
echo "  ./serve_disaggregated.sh stop"
echo ""
echo "Logs:"
echo "  docker logs -f ${CONTAINER_PREFILL}   (prefill GPU 0)"
echo "  docker logs -f ${CONTAINER_DECODE}    (decode GPU 1)"
echo "  tail -f ${PROXY_LOG}                  (proxy)"
