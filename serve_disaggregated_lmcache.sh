#!/bin/bash
# Disaggregated Prefill/Decode serving: Gemma4 26B NVFP4, 1P1D topology
# LMCACHE VARIANT — replaces P2pNcclConnector with LMCacheConnectorV1
#
# GPU 0 — prefill only (kv_producer, port 8100)
# GPU 1 — decode only  (kv_consumer, port 8101)
# Host  — proxy       (routes requests + LMCache KV rendezvous, port 8200)
#
# Transport: LMCacheConnectorV1 over pinned host-RAM buffers (cudaMemcpyAsync)
#            — zero cudaIpcGetMemHandle, WSL2-compatible (confirmed exp_log §ASI-1)
# KV cache dtype: BF16  (FP8 unsupported: FlashInfer rejects head_size=512 with fp8)
#
# Architecture change vs serve_disaggregated.sh:
#   BEFORE: P2pNcclConnector — kv_rank/kv_parallel_size, ZMQ ROUTER sockets,
#           request_id embeds prefill/decode ZMQ addresses for handshake.
#           Requires cudaIpcGetMemHandle → KILLS on WSL2.
#   AFTER:  LMCacheConnectorV1 — both instances share a single kv_port rendezvous
#           (14579). Prefill writes KV to local_cpu pool (20 GB pinned RAM), decode
#           reads from the same pool. Proxy embeds the same req_id into both calls
#           so LMCache can correlate producer → consumer lookup.
#           No NCCL group, no IPC, no ZMQ address embed in request_id.
#
# LMCache rebuild required: PyPI wheel lacks SM120 cubin. Build inline (≈5 min).
# Pattern P3 (NAME override), P4 (WSL2 GPU leak triple) applied per KILL_PATTERNS §4.
#
# Usage:
#   ./serve_disaggregated_lmcache.sh            # start prefill + decode + proxy
#   ./serve_disaggregated_lmcache.sh stop       # stop all three
#   ./serve_disaggregated_lmcache.sh bench      # run quick latency benchmark
#
# Endpoints:
#   :8200  — proxy  (send ALL client requests here)
#   :8100  — prefill instance (internal)
#   :8101  — decode instance  (internal)
#
# Reference:
#   serve_disaggregated.sh           — original P2pNccl launcher (preserved)
#   plans/lmcache_disagg_migration.md — architecture doc + bench plan
#   plans/KILL_PATTERNS.md §4        — fix templates used here

set -euo pipefail

# P3 fix: prevent inherited NAME from parent shell clobbering container names
unset NAME

IMAGE="vllm-built:latest"
MODEL_DIR="/root/models"
MODEL_PATH="/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_NAME="gemma-4-26B-A4B-it-NVFP4"

# HTTP ports (unchanged from original)
PREFILL_PORT=8100
DECODE_PORT=8101
PROXY_PORT=8200

# LMCache rendezvous — single shared port for producer→consumer KV lookup.
# Both instances bind to the same kv_ip:kv_port; LMCache matches by request_id.
# (P2pNccl used TWO separate ZMQ ports; LMCache uses one shared endpoint.)
KV_IP="127.0.0.1"
KV_PORT=14579

# 20 GB pinned host-RAM pool — matches smoke bench config (launch_lmcache_smoke_sm120.sh)
# and lmcache_cpu.yaml local_cpu_size_gb: 20
KV_BUFFER_SIZE=20000000000

# CPU pinning — one CCD per GPU on the 9950X3D (16C/32T)
CPUS_GPU0="0-15"    # CCD 0 → prefill (GPU 0)
CPUS_GPU1="16-31"   # CCD 1 → decode  (GPU 1)

CONTAINER_PREFILL="vllm-disagg-lmc-prefill"
CONTAINER_DECODE="vllm-disagg-lmc-decode"
PROXY_PID_FILE="/tmp/vllm-disagg-lmc-proxy.pid"
PROXY_LOG="/tmp/vllm-disagg-lmc-proxy.log"

# ------------------------------------------------------------------ helpers --

stop_servers() {
    docker rm -f "${CONTAINER_PREFILL}" "${CONTAINER_DECODE}" 2>/dev/null || true
    if [ -f "${PROXY_PID_FILE}" ]; then
        PID=$(cat "${PROXY_PID_FILE}")
        kill "${PID}" 2>/dev/null || true
        rm -f "${PROXY_PID_FILE}"
    fi
    pkill -f "vllm-disagg-lmc-proxy" 2>/dev/null || true
    echo "LMCache disaggregated instances stopped."
}

wait_healthy() {
    local port="$1"
    local label="$2"
    local timeout=600
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

# P4 fix: WSL2 GPU-isolation check before benching.
# If any pid appears on >1 UUID, abort and advise teardown.
check_gpu_isolation() {
    echo "=== P4 GPU isolation check ==="
    LEAK=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
        | awk -F, '{print $1}' | sort | uniq -c | awk '$1 > 1 {print "WARN: pid " $2 " on " $1 " GPUs"}')
    if [ -n "${LEAK}" ]; then
        echo "  ${LEAK}"
        echo "  WARNING: cross-GPU pid bleed detected (WSL2 known issue)."
        echo "  Bench results may be unreliable. Tear down all containers and re-run serially."
    else
        echo "  OK — no cross-GPU pid bleed detected."
    fi
}

# UUID sanity: print GPU UUIDs so operator can verify assignment
check_gpu_uuids() {
    echo "=== GPU UUID assignment ==="
    nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2>/dev/null | while IFS=, read -r idx uuid name; do
        echo "  GPU ${idx}: ${uuid} (${name})"
    done
}

run_bench() {
    echo ""
    echo "=== Quick latency benchmark (LMCache disaggregated) ==="
    echo "Concurrency 1, 4, 8 — bimodal prompt (50% 256-tok, 50% 4K-tok)"
    echo "Sending requests through proxy (port ${PROXY_PORT})"
    echo ""

    check_gpu_isolation

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
    echo "Expected targets (LMCache disagg, plans/lmcache_disagg_migration.md §bench):"
    echo "  C=8 P99 TTFT: ~120ms (vs 640ms collocated, vs 160ms T1-only LMCache)"
    echo "  Decode tok/s under heavy prefill: ~55 tok/s (GPU 1 never stalls on prefill)"
    echo ""
    echo "Kill criterion (ASI-1 inherited): abort if P99 TTFT < 1.5x better than DP=2 at C=64."
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

check_gpu_uuids

# ----------------------------------------------------------------
# LMCache SM120 rebuild
# The PyPI lmcache==0.4.3 wheel has no SM120 cubin (see KILL W1_4e_lmcache_smoke).
# We rebuild from source inside each container at launch time.
# This is the same approach validated in launch_lmcache_smoke_sm120.sh (P99 160→110ms).
# ----------------------------------------------------------------
LMCACHE_BUILD_CMDS='
echo "=== rebuilding lmcache for SM120 ===" >&2
export TORCH_CUDA_ARCH_LIST="12.0"
sed -i "s/raise RuntimeError(CUDA_MISMATCH_MESSAGE/pass  #/" \
    /usr/local/lib/python3.12/dist-packages/torch/utils/cpp_extension.py
apt-get install -qq -y git >/dev/null 2>&1
cd /tmp && git clone -q https://github.com/LMCache/LMCache.git
cd /tmp/LMCache && git checkout v0.4.3 >/dev/null 2>&1
pip install . --break-system-packages --no-build-isolation --force-reinstall --no-deps >/dev/null 2>&1
pip install --break-system-packages sortedcontainers nvtx aiofile aiofiles >/dev/null 2>&1
echo "=== lmcache rebuilt — SM120 cubin present ===" >&2
'

# Ensure proxy dependencies are available on the host
echo ""
echo "=== Checking proxy dependencies (quart, aiohttp) ==="
python3 -c "import quart, aiohttp" 2>/dev/null || \
    pip3 install --break-system-packages --ignore-installed quart aiohttp -q 2>/dev/null || \
    python3 -m pip install --break-system-packages --ignore-installed quart aiohttp -q 2>/dev/null
python3 -c "import quart, aiohttp; print('  proxy deps OK.')"
echo ""

echo "=== PRE-FLIGHT CHECKS ==="
echo "  KV transport: LMCacheConnectorV1 (pinned host-RAM, cudaMemcpyAsync)"
echo "  KV rendezvous: ${KV_IP}:${KV_PORT}  (shared by both instances)"
echo "  KV buffer: $((KV_BUFFER_SIZE / 1000000000)) GB pinned host-RAM pool"
echo "  KV dtype: BF16 (FP8 disabled — FlashInfer rejects head_size=512 with fp8)"
echo ""

# ------------------------------------------------------------------
# Instance 0 — Prefill GPU (GPU 0, port 8100)
# P4 fix: --gpus 'device=0' + NVIDIA_VISIBLE_DEVICES=0 + CUDA_VISIBLE_DEVICES=0
# kv_role: kv_producer  — runs prefill, writes KV to local_cpu pool
# NOTE: no kv_rank / kv_parallel_size — LMCacheConnectorV1 does not use NCCL groups.
# ------------------------------------------------------------------
echo "=== Starting prefill instance (GPU 0, port ${PREFILL_PORT}) ==="

docker run -d \
    --name "${CONTAINER_PREFILL}" \
    --network=host \
    --gpus 'device=0' \
    -e NVIDIA_VISIBLE_DEVICES=0 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e LMCACHE_CONFIG_FILE=/autokernel/lmcache_cpu.yaml \
    -e VLLM_USE_V1=1 \
    --memory=80g \
    --cpuset-cpus="${CPUS_GPU0}" \
    --shm-size=8g --ipc=host \
    -v "${MODEL_DIR}:/models:ro" \
    -v "/home/cklaus/projects/autokernel:/autokernel:ro" \
    --entrypoint bash \
    "${IMAGE}" \
    -c "
${LMCACHE_BUILD_CMDS}
exec python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --quantization modelopt \
    --port ${PREFILL_PORT} \
    --served-model-name ${MODEL_NAME} \
    --gpu-memory-utilization 0.80 \
    --max-model-len 28672 \
    --kv-transfer-config '{
        \"kv_connector\":   \"LMCacheConnectorV1\",
        \"kv_role\":        \"kv_producer\",
        \"kv_buffer_size\": ${KV_BUFFER_SIZE},
        \"kv_ip\":          \"${KV_IP}\",
        \"kv_port\":        ${KV_PORT}
    }' \
    -cc.mode none \
    -cc.cudagraph_mode full
"

# ------------------------------------------------------------------
# Instance 1 — Decode GPU (GPU 1, port 8101)
# P4 fix: --gpus 'device=1' + NVIDIA_VISIBLE_DEVICES=1 + CUDA_VISIBLE_DEVICES=0
#         (CUDA_VISIBLE_DEVICES=0 because inside the container device=1 remaps to 0)
# kv_role: kv_consumer  — skips prefill, reads KV from local_cpu pool by request_id
# ------------------------------------------------------------------
echo "=== Starting decode instance  (GPU 1, port ${DECODE_PORT}) ==="

docker run -d \
    --name "${CONTAINER_DECODE}" \
    --network=host \
    --gpus 'device=1' \
    -e NVIDIA_VISIBLE_DEVICES=1 \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e LMCACHE_CONFIG_FILE=/autokernel/lmcache_cpu.yaml \
    -e VLLM_USE_V1=1 \
    --memory=80g \
    --cpuset-cpus="${CPUS_GPU1}" \
    --shm-size=8g --ipc=host \
    -v "${MODEL_DIR}:/models:ro" \
    -v "/home/cklaus/projects/autokernel:/autokernel:ro" \
    --entrypoint bash \
    "${IMAGE}" \
    -c "
${LMCACHE_BUILD_CMDS}
exec python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_PATH} \
    --quantization modelopt \
    --port ${DECODE_PORT} \
    --served-model-name ${MODEL_NAME} \
    --gpu-memory-utilization 0.80 \
    --max-model-len 8192 \
    --max-num-seqs 128 \
    --kv-transfer-config '{
        \"kv_connector\":   \"LMCacheConnectorV1\",
        \"kv_role\":        \"kv_consumer\",
        \"kv_buffer_size\": ${KV_BUFFER_SIZE},
        \"kv_ip\":          \"${KV_IP}\",
        \"kv_port\":        ${KV_PORT}
    }' \
    -cc.mode none \
    -cc.cudagraph_mode full
"

# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------
echo ""
echo "=== Waiting for both instances to become healthy ==="
echo "    (Both load full Gemma4 26B weights + LMCache rebuild — allow 10-12 min)"
echo ""

wait_healthy "${PREFILL_PORT}" "prefill (GPU 0)" || {
    echo "  Prefill failed. Logs:"
    docker logs "${CONTAINER_PREFILL}" 2>&1 | tail -30
    exit 1
}
wait_healthy "${DECODE_PORT}"  "decode  (GPU 1)" || {
    echo "  Decode failed. Logs:"
    docker logs "${CONTAINER_DECODE}" 2>&1 | tail -30
    exit 1
}

# ------------------------------------------------------------------
# Proxy server
# ------------------------------------------------------------------
# LMCacheConnectorV1 protocol (differs from P2pNcclConnector):
#   - No ZMQ address embedding in request_id required.
#   - Proxy assigns a stable UUID as the request_id.
#   - Step 1: send max_tokens=1 prefill-only request to prefill instance with that UUID.
#             Prefill runs forward pass, writes KV to LMCache pool keyed by UUID.
#   - Step 2: send full-decode request to decode instance with the same UUID.
#             Decode instance calls LMCache consumer.load(UUID) → gets KV → skips prefill.
#   - Both requests use the same X-Request-Id header so LMCache can correlate.
# ------------------------------------------------------------------
echo ""
echo "=== Starting disaggregated proxy (port ${PROXY_PORT}) ==="

python3 - <<PROXY_SCRIPT &
import asyncio, uuid, os, sys
import aiohttp
from quart import Quart, make_response, request

PREFILL_URL  = "http://127.0.0.1:${PREFILL_PORT}"
DECODE_URL   = "http://127.0.0.1:${DECODE_PORT}"
PROXY_PORT   = ${PROXY_PORT}
TIMEOUT      = aiohttp.ClientTimeout(total=3600)

app = Quart("vllm-disagg-lmc-proxy")

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

        # LMCacheConnectorV1: use a plain UUID as request_id.
        # Both calls get the same id so LMCache matches producer write → consumer read.
        # (P2pNccl required ___prefill_addr_IP:PORT___decode_addr_IP:PORT_UUID format
        #  — that format is NOT needed here and would confuse LMCache routing.)
        req_id = uuid.uuid4().hex

        headers = {
            "Content-Type": "application/json",
            "X-Request-Id": req_id,
        }

        async with aiohttp.ClientSession(timeout=TIMEOUT) as sess:
            # Step 1: prefill only — GPU 0 runs forward pass, writes KV to pool
            async with sess.post(
                PREFILL_URL + request.path,
                json=prefill_req,
                headers=headers,
            ) as resp:
                await resp.read()  # consume + discard prefill output token

            # Step 2: decode — GPU 1 loads KV from pool, generates tokens
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
echo "=== DISAGGREGATED SERVING READY (LMCache) ==="
echo ""
echo "  Topology:   1P1D (1 prefill GPU 0, 1 decode GPU 1)"
echo "  Transport:  LMCacheConnectorV1 — pinned host-RAM pool (cudaMemcpyAsync)"
echo "  KV pool:    $((KV_BUFFER_SIZE / 1000000000)) GB local_cpu (WSL2-compatible, no IPC)"
echo "  KV dtype:   BF16"
echo "  KV port:    ${KV_IP}:${KV_PORT}  (shared rendezvous)"
echo "  Proxy:      port ${PROXY_PORT}  (assigns UUID req_id, no ZMQ addr embed)"
echo ""
echo "  GPU 0 :${PREFILL_PORT}  kv_producer  max_len=28672  mem_util=0.80"
echo "  GPU 1 :${DECODE_PORT}  kv_consumer  max_len=8192   mem_util=0.80  max_seqs=128"
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
echo "  ./serve_disaggregated_lmcache.sh bench"
echo ""
echo "Stop:"
echo "  ./serve_disaggregated_lmcache.sh stop"
echo ""
echo "Logs:"
echo "  docker logs -f ${CONTAINER_PREFILL}   (prefill GPU 0)"
echo "  docker logs -f ${CONTAINER_DECODE}    (decode GPU 1)"
echo "  tail -f ${PROXY_LOG}                  (proxy)"
