#!/bin/bash
# Disaggregated Prefill/Decode serving: Gemma4 26B NVFP4, 1P1D topology
#
# GPU 0 — prefill only (kv_producer, port 8100)
# GPU 1 — decode only  (kv_consumer, port 8101)
#
# Transport: P2pNcclConnector over PCIe (zero external deps)
# KV cache dtype: FP8 (~10 ms transfer overhead vs ~19 ms for BF16)
#
# Usage:
#   ./serve_disaggregated.sh            # start both instances
#   ./serve_disaggregated.sh stop       # stop both instances
#   ./serve_disaggregated.sh bench      # run quick latency benchmark
#
# Endpoints:
#   GPU 0 :8100 — prefill instance  (send ALL client requests here)
#   GPU 1 :8101 — decode instance   (internal; P2pNccl streams tokens back)
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

# Ports
PREFILL_PORT=8100
DECODE_PORT=8101

# NCCL KV transfer rendezvous
KV_IP="127.0.0.1"
KV_PORT=14579
# 2 GB transfer buffer — enough for several large in-flight requests.
# Reduce to 1000000000 if OOM occurs during KV buffer allocation on 26B NVFP4.
KV_BUFFER_SIZE=2000000000

# CPU pinning — one CCD per GPU on the 9950X3D (16C/32T).
# Prevents L3 V-Cache thrashing between the two CCDs.
CPUS_GPU0="0-15"    # CCD 0 → prefill (GPU 0)
CPUS_GPU1="16-31"   # CCD 1 → decode  (GPU 1)

CONTAINER_PREFILL="vllm-disagg-prefill"
CONTAINER_DECODE="vllm-disagg-decode"
LOG_PREFILL="/tmp/vllm-disagg-prefill.log"
LOG_DECODE="/tmp/vllm-disagg-decode.log"

# ------------------------------------------------------------------ helpers --

stop_servers() {
    docker rm -f "${CONTAINER_PREFILL}" "${CONTAINER_DECODE}" 2>/dev/null || true
    echo "Disaggregated instances stopped."
}

wait_healthy() {
    local port="$1"
    local label="$2"
    local timeout=240
    printf "  Waiting for %s (port %s)" "${label}" "${port}"
    for i in $(seq 1 ${timeout}); do
        if curl -sS "http://localhost:${port}/health" > /dev/null 2>&1; then
            printf " — ready in %ds\n" "${i}"
            return 0
        fi
        printf "\r  Waiting for %s (port %s) — %3ds / %ds" \
            "${label}" "${port}" "${i}" "${timeout}"
        sleep 1
    done
    printf "\n"
    echo "ERROR: ${label} failed to start within ${timeout}s"
    echo "Check logs: docker logs ${CONTAINER_PREFILL} / ${CONTAINER_DECODE}"
    docker logs "${CONTAINER_PREFILL}" 2>&1 | tail -20
    docker logs "${CONTAINER_DECODE}"  2>&1 | tail -20
    exit 1
}

run_bench() {
    echo ""
    echo "=== Quick latency benchmark ==="
    echo "Concurrency 1, 4, 8 — bimodal prompt (50% 256-tok, 50% 4K-tok)"
    echo "Sending requests to prefill instance (port ${PREFILL_PORT})"
    echo ""

    for concurrency in 1 4 8; do
        echo "--- C=${concurrency} ---"
        python3 - <<PYEOF
import asyncio, time, json, urllib.request, statistics

MODEL = "${MODEL_NAME}"
PORT  = ${PREFILL_PORT}
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

# Prerequisite reminder (non-fatal; operator must verify manually)
echo ""
echo "=== PRE-FLIGHT CHECKS (manual — not automated here) ==="
echo "  1. P2P access:  nvidia-smi topo -m   (GPUs should show PIX or NV*)"
echo "  2. NCCL P2P bw: nccl-tests p2p_bw --minbytes 512M --maxbytes 512M -g 2"
echo "  3. KV port ${KV_PORT} must be free:  ss -ltn | grep ${KV_PORT}"
echo ""

# ------------------------------------------------------------------
# Instance 0 — Prefill GPU (GPU 0, port 8100)
# ------------------------------------------------------------------
# Design choices (disaggregated_serving.md §3):
#   max-model-len 32768  — prefill GPU handles long contexts
#   gpu-memory-utilization 0.90  — maximize KV cache for prompt storage
#   No --max-num-seqs cap — prefill is compute-bound, not seq-count-bound
# ------------------------------------------------------------------
echo "=== Starting prefill instance (GPU 0, port ${PREFILL_PORT}, cores ${CPUS_GPU0}) ==="

docker run -d \
    --name "${CONTAINER_PREFILL}" \
    --gpus '"device=0"' \
    --memory=80g \
    --cpuset-cpus="${CPUS_GPU0}" \
    -v "${MODEL_DIR}:/models:ro" \
    -p "${PREFILL_PORT}:8100" \
    --entrypoint python3 \
    "${IMAGE}" \
    -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --quantization modelopt \
        --port 8100 \
        --served-model-name "${MODEL_NAME}" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --kv-cache-dtype fp8 \
        --kv-transfer-config '{
            "kv_connector":     "P2pNcclConnector",
            "kv_role":          "kv_producer",
            "kv_rank":          0,
            "kv_parallel_size": 2,
            "kv_ip":            "'"${KV_IP}"'",
            "kv_port":          '"${KV_PORT}"',
            "kv_buffer_size":   '"${KV_BUFFER_SIZE}"'
        }' \
        -cc.mode none \
        -cc.cudagraph_mode full

# ------------------------------------------------------------------
# Instance 1 — Decode GPU (GPU 1, port 8101)
# ------------------------------------------------------------------
# Design choices (disaggregated_serving.md §3):
#   max-model-len 8192   — decode GPU only needs active generation context
#   max-num-seqs 128     — optimize for high batch decode throughput
#   gpu-memory-utilization 0.85  — headroom for inbound KV DMA buffers
# ------------------------------------------------------------------
echo "=== Starting decode instance  (GPU 1, port ${DECODE_PORT}, cores ${CPUS_GPU1}) ==="

docker run -d \
    --name "${CONTAINER_DECODE}" \
    --gpus '"device=1"' \
    --memory=80g \
    --cpuset-cpus="${CPUS_GPU1}" \
    -v "${MODEL_DIR}:/models:ro" \
    -p "${DECODE_PORT}:8101" \
    --entrypoint python3 \
    "${IMAGE}" \
    -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --quantization modelopt \
        --port 8101 \
        --served-model-name "${MODEL_NAME}" \
        --gpu-memory-utilization 0.85 \
        --max-model-len 8192 \
        --max-num-seqs 128 \
        --kv-cache-dtype fp8 \
        --kv-transfer-config '{
            "kv_connector":     "P2pNcclConnector",
            "kv_role":          "kv_consumer",
            "kv_rank":          1,
            "kv_parallel_size": 2,
            "kv_ip":            "'"${KV_IP}"'",
            "kv_port":          '"${KV_PORT}"',
            "kv_buffer_size":   '"${KV_BUFFER_SIZE}"'
        }' \
        -cc.mode none \
        -cc.cudagraph_mode full

# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------
echo ""
echo "=== Waiting for both instances to become healthy ==="
echo "    (Both load full Gemma4 26B weights — expect 60-120s startup)"
echo ""

# Decode instance connects to prefill via NCCL rendezvous; start prefill first
# and let decode follow.  Both health checks run sequentially to avoid a race
# where decode hangs waiting for the NCCL group that prefill hasn't formed yet.
wait_healthy "${PREFILL_PORT}" "prefill (GPU 0)"
wait_healthy "${DECODE_PORT}"  "decode  (GPU 1)"

# ------------------------------------------------------------------
# Startup summary
# ------------------------------------------------------------------
echo ""
echo "=== DISAGGREGATED SERVING READY ==="
echo ""
echo "  Topology:   1P1D (1 prefill, 1 decode)"
echo "  Transport:  P2pNcclConnector over PCIe  [KV ip=${KV_IP} port=${KV_PORT}]"
echo "  KV dtype:   FP8  (~10 ms transfer per request at 536 MB BF16 → 268 MB FP8)"
echo ""
echo "  GPU 0 :${PREFILL_PORT}  prefill-only  max_len=32768  mem_util=0.90"
echo "  GPU 1 :${DECODE_PORT}  decode-only   max_len=8192   mem_util=0.85  max_seqs=128"
echo ""
echo "  Send ALL client requests to the PREFILL instance:"
echo "    http://localhost:${PREFILL_PORT}/v1/chat/completions"
echo ""
echo "  Decode instance (port ${DECODE_PORT}) is internal —"
echo "  P2pNccl transfers KV from GPU 0 → GPU 1 after prefill completes."
echo ""
echo "Quick test:"
echo "  curl http://localhost:${PREFILL_PORT}/v1/chat/completions \\"
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
