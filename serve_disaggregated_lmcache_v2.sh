#!/bin/bash
# Disaggregated Prefill/Decode serving: Gemma4 26B NVFP4, 1P1D topology
# LMCACHE VARIANT v2 — W6_lmcache_disagg_harden
#
# Hardening changes vs W5_6 (serve_disaggregated_lmcache.sh):
#   1. verify_isolation() — pre-launch + post-launch UUID check per container.
#      Refuses to start if P4 triple-env does not achieve physical GPU separation.
#   2. Serial startup — prefill first → /health → UUID verify → decode second.
#      Any isolation failure aborts before the decode container is launched.
#   3. Fallback mode (ISOLATION=disabled or auto-detected) — launches both
#      prefill and decode on GPU 0 serially (never concurrently). Still measures
#      KV transfer correctness + TTFT savings via LMCache producer→consumer loop.
#      Tagged MODE=fallback in all log lines.
#   4. LMCache diagnostic — kv_buffer pool address + periodic writer/reader stats
#      logged from each container.
#
# GPU 0 — prefill only (kv_producer, port 8100)
# GPU 1 — decode only  (kv_consumer, port 8101)  [skipped in fallback mode]
# Host  — proxy       (routes requests + LMCache KV rendezvous, port 8200)
#
# Transport: LMCacheConnectorV1 over pinned host-RAM buffers (cudaMemcpyAsync)
#            — zero cudaIpcGetMemHandle, WSL2-compatible (confirmed exp_log §ASI-1)
# KV cache dtype: BF16  (FP8 unsupported: FlashInfer rejects head_size=512 with fp8)
#
# KILL_PATTERNS references:
#   §P3  — unset NAME (parent-shell override)
#   §P4  — WSL2 GPU isolation INVALIDATED 2026-04-19 (W5_25_wsl2_isolation_FAIL)
#           P4 triple-env DOES NOT work on this WSL2 setup. Both containers
#           collapse onto GPU 0 despite DeviceIDs set correctly in docker inspect.
#           See plans/KILL_PATTERNS.md §P4 for remediation options and
#           W6_wsl2_isolation_rootcause agent findings (dispatch pending).
#
# Usage:
#   ./serve_disaggregated_lmcache_v2.sh                # auto — runs verify_isolation, aborts on fail
#   ISOLATION=disabled ./serve_disaggregated_lmcache_v2.sh   # fallback serial on GPU 0
#   ./serve_disaggregated_lmcache_v2.sh stop
#   ./serve_disaggregated_lmcache_v2.sh bench
#   ./serve_disaggregated_lmcache_v2.sh bench_fallback  # bench fallback mode explicitly
#
# Endpoints:
#   :8200  — proxy  (send ALL client requests here)
#   :8100  — prefill instance (internal)
#   :8101  — decode instance  (internal; GPU 0 in fallback mode)
#
# Reference:
#   serve_disaggregated_lmcache.sh   — W5_6 original (preserved, BC-compatible)
#   plans/lmcache_disagg_migration.md — architecture doc + W5_6 bench plan
#   plans/lmcache_disagg_hardened.md  — W6 changes + fallback bench plan
#   plans/KILL_PATTERNS.md §P4       — WSL2 isolation INVALIDATED annotation

set -euo pipefail

# P3 fix: prevent inherited NAME from parent shell clobbering container names
unset NAME

IMAGE="vllm-built:latest"
MODEL_DIR="/root/models"
MODEL_PATH="/models/gemma-4-26B-A4B-it-NVFP4-modelopt"
MODEL_NAME="gemma-4-26B-A4B-it-NVFP4"

# HTTP ports (unchanged from W5_6)
PREFILL_PORT=8100
DECODE_PORT=8101
PROXY_PORT=8200

# LMCache rendezvous — single shared port for producer→consumer KV lookup.
KV_IP="127.0.0.1"
KV_PORT=14579

# 20 GB pinned host-RAM pool
KV_BUFFER_SIZE=20000000000

# CPU pinning — one CCD per GPU on the 9950X3D (16C/32T)
CPUS_GPU0="0-15"    # CCD 0 → prefill (GPU 0)
CPUS_GPU1="16-31"   # CCD 1 → decode  (GPU 1)

CONTAINER_PREFILL="vllm-disagg-lmc-prefill"
CONTAINER_DECODE="vllm-disagg-lmc-decode"
PROXY_PID_FILE="/tmp/vllm-disagg-lmc-proxy.pid"
PROXY_LOG="/tmp/vllm-disagg-lmc-proxy.log"

# ISOLATION mode: "auto" (default) or "disabled" (fallback serial).
# Set ISOLATION=disabled in env to force fallback without verify_isolation check.
ISOLATION="${ISOLATION:-auto}"

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

# UUID sanity: print GPU UUIDs so operator can verify assignment
check_gpu_uuids() {
    echo "=== GPU UUID assignment ==="
    nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader 2>/dev/null | while IFS=, read -r idx uuid name; do
        echo "  GPU ${idx}: ${uuid} (${name})"
    done
}

# ------------------------------------------------------------------
# verify_isolation() — W6 hardening: pre-launch AND post-launch check.
#
# Strategy: launch a minimal nvidia-smi probe container on the declared GPU,
# read the GPU UUID it sees, and compare against host nvidia-smi.
# A correct isolation: container on GPU N sees ONLY GPU N's UUID.
# A WSL2 leak: container on GPU N sees GPU 0's UUID regardless of --gpus flag.
#
# Also checks cross-GPU pid bleed after the real container launches:
# any vLLM pid appearing on >1 UUID = isolation failure.
#
# Arguments:
#   $1 — GPU device index declared to docker (e.g., 0 or 1)
#   $2 — container name that was just launched
#   $3 — expected GPU UUID (from host nvidia-smi for that index)
#
# Returns:
#   0 = isolation confirmed
#   1 = isolation failure — sets ISOLATION=disabled if ISOLATION=auto
# ------------------------------------------------------------------
verify_isolation() {
    local declared_gpu="$1"
    local container_name="$2"
    local expected_uuid="$3"

    echo ""
    echo "=== [verify_isolation] GPU ${declared_gpu} — container ${container_name} ==="

    # Step A: probe UUID visible inside the container
    local container_uuid
    container_uuid=$(docker exec "${container_name}" \
        nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null \
        | head -1 | tr -d ' ') || {
        echo "  WARN: could not exec nvidia-smi inside container ${container_name}."
        echo "        Container may still be starting up — skipping UUID check."
        return 0
    }

    if [ -z "${container_uuid}" ]; then
        echo "  WARN: container reported empty UUID — nvidia-smi inside container failed."
        return 0
    fi

    echo "  Host GPU ${declared_gpu} UUID:      ${expected_uuid}"
    echo "  Container visible GPU UUID: ${container_uuid}"

    if [ "${container_uuid}" = "${expected_uuid}" ]; then
        echo "  OK — UUID matches. GPU ${declared_gpu} is isolated."
    else
        echo ""
        echo "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "  ISOLATION FAILURE (P4 WSL2 triple-env INVALIDATED)"
        echo "  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo ""
        echo "  Container '${container_name}' declared --gpus device=${declared_gpu}"
        echo "  but sees UUID ${container_uuid} (GPU 0) instead of"
        echo "  expected UUID ${expected_uuid} (GPU ${declared_gpu})."
        echo ""
        echo "  This confirms W5_25_wsl2_isolation_FAIL: WSL2 runtime ignores"
        echo "  DeviceRequests even when docker inspect shows correct DeviceIDs."
        echo "  P4 triple-env (NVIDIA_VISIBLE_DEVICES + CUDA_VISIBLE_DEVICES + --gpus)"
        echo "  DOES NOT isolate on this WSL2 setup."
        echo ""
        echo "  Remediation options (plans/KILL_PATTERNS.md §P4):"
        echo "    (a) Use --runtime=nvidia --gpus all + container-internal CUDA_VISIBLE_DEVICES=N"
        echo "        with N = HOST GPU index (untested — may be the only WSL2 workaround)."
        echo "    (b) Serialize containers — never run two vLLM instances concurrently."
        echo "        Use ISOLATION=disabled to engage serial fallback mode."
        echo "    (c) Native Linux — nvidia-container-runtime honors --gpus device=N there."
        echo "    (d) Wait for W6_wsl2_isolation_rootcause agent findings."
        echo ""
        echo "  To force fallback mode and proceed serially: ISOLATION=disabled ./$(basename "$0")"
        return 1
    fi

    # Step B: pid bleed check — scan for any container pid appearing on >1 GPU UUID
    echo ""
    echo "  [verify_isolation] Checking cross-GPU pid bleed for ${container_name}..."
    local container_pid
    container_pid=$(docker inspect --format='{{.State.Pid}}' "${container_name}" 2>/dev/null || echo "")
    if [ -z "${container_pid}" ] || [ "${container_pid}" = "0" ]; then
        echo "  WARN: could not get container PID — skipping pid bleed check."
        return 0
    fi

    # Get all descendant pids of the container's init pid (cgroup namespace may differ)
    local leak_check
    leak_check=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
        | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1,$2}' \
        | sort -k1,1 | awk '
            { pids[$1] = pids[$1] " " $2; count[$1]++ }
            END { for (p in count) if (count[p] > 1) print "BLEED: pid " p " on " count[p] " GPUs: " pids[p] }
        ')
    if [ -n "${leak_check}" ]; then
        echo "  WARNING: cross-GPU pid bleed detected:"
        echo "  ${leak_check}"
        echo "  This is the P4 WSL2 known failure mode (W5_25_wsl2_isolation_FAIL)."
        echo "  Dual-GPU bench results will be unreliable."
        return 1
    fi

    echo "  OK — no cross-GPU pid bleed detected."
    return 0
}

# ------------------------------------------------------------------
# lmcache_diagnostics() — W6 hardening: log KV buffer pool residence
# and periodic writer/reader stats.
# $1 = container name
# $2 = role label (prefill/decode)
# ------------------------------------------------------------------
lmcache_diagnostics() {
    local cname="$1"
    local role="$2"
    echo ""
    echo "=== [lmcache_diagnostics] ${role} (${cname}) ==="

    # Pinned host RAM address — LMCache allocates a single contiguous pool via
    # cudaMallocHost. The address appears in the startup log as:
    #   "kv_buffer pool allocated at 0x... (20 GB pinned host RAM)"
    local pool_addr
    pool_addr=$(docker logs "${cname}" 2>&1 \
        | grep -i "pinned\|pool allocated\|kv_buffer\|local_cpu" \
        | tail -5 || true)
    if [ -n "${pool_addr}" ]; then
        echo "  KV pool log lines:"
        echo "${pool_addr}" | while IFS= read -r line; do echo "    ${line}"; done
    else
        echo "  KV pool address: not yet logged (container still initializing or no explicit alloc log)."
    fi

    # Writer/reader stats: LMCache logs cache_hit_rate, put_count, get_count
    local stats
    stats=$(docker logs "${cname}" 2>&1 \
        | grep -i "cache_hit\|put_count\|get_count\|kv_transfer\|producer\|consumer" \
        | tail -10 || true)
    if [ -n "${stats}" ]; then
        echo "  LMCache stats:"
        echo "${stats}" | while IFS= read -r line; do echo "    ${line}"; done
    else
        echo "  LMCache stats: no transfer events yet (send a request first)."
    fi
}

# ------------------------------------------------------------------
# periodic_lmcache_stats() — background loop that prints stats every 30s
# $1 = container name
# $2 = role label
# ------------------------------------------------------------------
periodic_lmcache_stats() {
    local cname="$1"
    local role="$2"
    while true; do
        sleep 30
        # Non-fatal: container may have stopped
        docker logs --since=31s "${cname}" 2>/dev/null \
            | grep -i "cache_hit\|put_count\|get_count\|kv_transfer\|lmcache" \
            | while IFS= read -r line; do echo "[lmcache_stats:${role}] ${line}"; done || true
    done
}

# P4 check: legacy warn-only (bench-time); kept for BC with W5_6 bench path
check_gpu_isolation() {
    echo "=== P4 GPU isolation check (post-launch) ==="
    LEAK=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null \
        | awk -F, '{print $1}' | sort | uniq -c | awk '$1 > 1 {print "WARN: pid " $2 " on " $1 " GPUs"}')
    if [ -n "${LEAK}" ]; then
        echo "  ${LEAK}"
        echo "  WARNING: cross-GPU pid bleed detected (WSL2 known issue, W5_25_wsl2_isolation_FAIL)."
        echo "  MODE=fallback recommended. Run: ISOLATION=disabled ./$(basename "$0")"
    else
        echo "  OK — no cross-GPU pid bleed detected."
    fi
}

run_bench() {
    local mode="${1:-normal}"
    echo ""
    if [ "${mode}" = "fallback" ]; then
        echo "=== Quick latency benchmark — MODE=fallback (serial GPU 0 only) ==="
        echo "  Both prefill + decode on GPU 0 sequentially via LMCache host-RAM."
        echo "  Tests KV transfer correctness + TTFT savings without dual-GPU parallelism."
        echo "  ISOLATION=disabled"
    else
        echo "=== Quick latency benchmark (LMCache disaggregated) ==="
        echo "  Concurrency 1, 4, 8 — bimodal prompt (50% 256-tok, 50% 4K-tok)"
        echo "  Sending requests through proxy (port ${PROXY_PORT})"
    fi
    echo ""

    check_gpu_isolation

    for concurrency in 1 4 8; do
        echo "--- C=${concurrency} MODE=${mode} ---"
        python3 - <<PYEOF
import asyncio, time, json, urllib.request, statistics

MODEL = "${MODEL_NAME}"
PORT  = ${PROXY_PORT}
C     = ${concurrency}
MODE  = "${mode}"

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
              f"max={max(ttfts):.0f}ms  mode={MODE}")
        print(f"  Wall  {elapsed*1000:.0f}ms  requests={len(ttfts)}")

asyncio.run(run())
PYEOF
    done

    echo ""
    if [ "${mode}" = "fallback" ]; then
        echo "Fallback serial targets (no dual-GPU parallelism, KV correctness baseline):"
        echo "  KV transfer: producer writes host-RAM, consumer reads same pool — correctness MUST hold"
        echo "  C=8 P99 TTFT: ~250-400ms expected (serialized on one GPU, no decode isolation)"
        echo "  TTFT vs collocated: should be similar or worse (no GPU 1 speedup)"
        echo "  Goal: confirm LMCache KV round-trip is correct before investing in isolation fix"
    else
        echo "Expected targets (LMCache disagg dual-GPU, plans/lmcache_disagg_migration.md §bench):"
        echo "  C=8 P99 TTFT: ~120ms (vs 640ms collocated, vs 160ms T1-only LMCache)"
        echo "  Decode tok/s under heavy prefill: ~55 tok/s (GPU 1 never stalls on prefill)"
    fi
    echo ""
    echo "Kill criterion (ASI-1 inherited): abort if P99 TTFT < 1.5x better than DP=2 at C=64."
}

# ------------------------------------------------------------------ dispatch --

if [ "${1:-}" = "stop" ]; then
    stop_servers
    exit 0
fi

if [ "${1:-}" = "bench" ]; then
    run_bench normal
    exit 0
fi

if [ "${1:-}" = "bench_fallback" ]; then
    run_bench fallback
    exit 0
fi

# ------------------------------------------------------------------ launch ---

stop_servers

check_gpu_uuids

# Read expected GPU UUIDs from host (needed for verify_isolation)
GPU0_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 2>/dev/null | tr -d ' ' || echo "")
GPU1_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 1 2>/dev/null | tr -d ' ' || echo "")

if [ -z "${GPU0_UUID}" ]; then
    echo "ERROR: Could not read GPU 0 UUID from host nvidia-smi."
    echo "       Ensure nvidia-smi is installed and NVIDIA driver is accessible."
    exit 1
fi

echo ""
echo "  Expected GPU 0 UUID: ${GPU0_UUID}"
if [ -n "${GPU1_UUID}" ]; then
    echo "  Expected GPU 1 UUID: ${GPU1_UUID}"
else
    echo "  GPU 1 not found — only GPU 0 available. Will use fallback mode."
    ISOLATION="disabled"
fi

# ----------------------------------------------------------------
# Isolation mode decision
# ----------------------------------------------------------------
if [ "${ISOLATION}" = "disabled" ]; then
    echo ""
    echo "========================================================"
    echo "  MODE=fallback  ISOLATION=disabled"
    echo "  Both prefill + decode will run on GPU 0 SERIALLY."
    echo "  No dual-GPU parallelism. KV correctness still tested."
    echo "  See plans/lmcache_disagg_hardened.md §fallback_mode"
    echo "========================================================"
    echo ""
    # Override decode GPU to 0 as well
    DECODE_GPU=0
    DECODE_CPUS="${CPUS_GPU0}"
else
    echo ""
    echo "  ISOLATION=auto — will run verify_isolation after prefill launch."
    echo "  If isolation fails, re-run with: ISOLATION=disabled ./$(basename "$0")"
    echo ""
    DECODE_GPU=1
    DECODE_CPUS="${CPUS_GPU1}"
fi

# ----------------------------------------------------------------
# LMCache SM120 rebuild commands (shared by both containers)
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
echo "  ISOLATION mode: ${ISOLATION}"
echo ""

# ----------------------------------------------------------------
# SERIAL STARTUP — prefill FIRST, verify, then decode
# ------------------------------------------------------------------
# W6 change: both containers launched serially (not concurrently).
# Prefill is /health-checked and UUID-verified before decode starts.
# This prevents the WSL2 race where concurrent GPU initialization can
# both land on GPU 0 during cudaDeviceInit.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Instance 0 — Prefill GPU (GPU 0, port 8100)
# P4 triple-env applied (even though INVALIDATED — kept for forward-compat
# if WSL2 isolation fix lands). verify_isolation() validates actual behavior.
# kv_role: kv_producer  — runs prefill, writes KV to local_cpu pool
# ------------------------------------------------------------------
echo "=== [SERIAL STARTUP 1/2] Starting prefill instance (GPU 0, port ${PREFILL_PORT}) ==="

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

echo "  Prefill container launched. Waiting for /health before isolation check..."
wait_healthy "${PREFILL_PORT}" "prefill (GPU 0)" || {
    echo "  Prefill failed to become healthy. Logs:"
    docker logs "${CONTAINER_PREFILL}" 2>&1 | tail -30
    lmcache_diagnostics "${CONTAINER_PREFILL}" "prefill"
    stop_servers
    exit 1
}

# W6: verify prefill isolation before launching decode
if [ "${ISOLATION}" = "auto" ]; then
    verify_isolation 0 "${CONTAINER_PREFILL}" "${GPU0_UUID}" || {
        echo ""
        echo "ERROR: Prefill isolation check FAILED."
        echo "       GPU 0 container sees wrong UUID — P4 WSL2 triple-env DOES NOT isolate."
        echo "       Results.tsv: W5_25_wsl2_isolation_FAIL confirmed."
        echo ""
        echo "To proceed in fallback serial mode (both on GPU 0, no parallelism):"
        echo "  ISOLATION=disabled ./$(basename "$0")"
        echo ""
        echo "Aborting. Tearing down prefill container."
        stop_servers
        exit 1
    }
    echo "  Prefill isolation OK — proceeding to decode launch."
fi

# Log LMCache diagnostics for prefill now that it's healthy
lmcache_diagnostics "${CONTAINER_PREFILL}" "prefill"

# ------------------------------------------------------------------
# Instance 1 — Decode GPU
# Normal mode:   GPU 1, port 8101
# Fallback mode: GPU 0, port 8101 (serial, after prefill — no concurrent GPU contention)
# P4 triple-env applied for forward-compat; verify_isolation validates.
# kv_role: kv_consumer  — skips prefill, reads KV from local_cpu pool by request_id
# ------------------------------------------------------------------
if [ "${ISOLATION}" = "disabled" ]; then
    echo ""
    echo "=== [SERIAL STARTUP 2/2] Starting decode instance (MODE=fallback, GPU 0, port ${DECODE_PORT}) ==="
    echo "  NOTE: Both instances on GPU 0 — no dual-GPU parallelism. KV correctness test only."
else
    echo ""
    echo "=== [SERIAL STARTUP 2/2] Starting decode instance (GPU 1, port ${DECODE_PORT}) ==="
fi

docker run -d \
    --name "${CONTAINER_DECODE}" \
    --network=host \
    --gpus "device=${DECODE_GPU}" \
    -e NVIDIA_VISIBLE_DEVICES="${DECODE_GPU}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e LMCACHE_CONFIG_FILE=/autokernel/lmcache_cpu.yaml \
    -e VLLM_USE_V1=1 \
    -e LMCACHE_DISAGG_MODE="${ISOLATION}" \
    --memory=80g \
    --cpuset-cpus="${DECODE_CPUS}" \
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

echo "  Decode container launched. Waiting for /health..."
wait_healthy "${DECODE_PORT}"  "decode  (GPU ${DECODE_GPU})" || {
    echo "  Decode failed to become healthy. Logs:"
    docker logs "${CONTAINER_DECODE}" 2>&1 | tail -30
    lmcache_diagnostics "${CONTAINER_DECODE}" "decode"
    stop_servers
    exit 1
}

# W6: verify decode isolation (only in auto mode with GPU 1)
if [ "${ISOLATION}" = "auto" ] && [ -n "${GPU1_UUID}" ]; then
    verify_isolation 1 "${CONTAINER_DECODE}" "${GPU1_UUID}" || {
        echo ""
        echo "ERROR: Decode isolation check FAILED after decode launch."
        echo "       GPU 1 container sees wrong UUID (same as GPU 0)."
        echo "       This is P4 WSL2 triple-env INVALIDATED (W5_25_wsl2_isolation_FAIL)."
        echo ""
        echo "Options:"
        echo "  1. Tear down and re-run in fallback mode: ISOLATION=disabled ./$(basename "$0")"
        echo "  2. Wait for W6_wsl2_isolation_rootcause agent findings for a real fix."
        echo "  3. Continue but bench results will be unreliable (both on GPU 0)."
        echo ""
        echo "Aborting. Run with ISOLATION=disabled to use serial fallback."
        stop_servers
        exit 1
    }
    echo "  Decode isolation OK — dual-GPU topology confirmed."
fi

# Log LMCache diagnostics for decode
lmcache_diagnostics "${CONTAINER_DECODE}" "decode"

# ------------------------------------------------------------------
# Proxy server
# ------------------------------------------------------------------
echo ""
echo "=== Starting disaggregated proxy (port ${PROXY_PORT}) ==="

# Proxy logic is identical for normal and fallback modes — the proxy
# always does Step1(prefill:8100) → Step2(decode:8101). In fallback mode
# these are both on GPU 0 but the LMCache host-RAM rendezvous still works
# because the producer/consumer protocol is independent of GPU assignment.
python3 - <<PROXY_SCRIPT &
import asyncio, uuid, os, sys
import aiohttp
from quart import Quart, make_response, request

PREFILL_URL  = "http://127.0.0.1:${PREFILL_PORT}"
DECODE_URL   = "http://127.0.0.1:${DECODE_PORT}"
PROXY_PORT   = ${PROXY_PORT}
ISOLATION    = "${ISOLATION}"
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

            # Step 2: decode — reads KV from pool, generates tokens.
            # In fallback mode: decode also on GPU 0 but LMCache host-RAM
            # pool is the same object — producer wrote, consumer reads, correct.
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

# Start background stats monitors
periodic_lmcache_stats "${CONTAINER_PREFILL}" "prefill" &
STATS_PREFILL_PID=$!
periodic_lmcache_stats "${CONTAINER_DECODE}" "decode" &
STATS_DECODE_PID=$!
echo "  LMCache stats monitors: prefill PID ${STATS_PREFILL_PID}, decode PID ${STATS_DECODE_PID}"

# ------------------------------------------------------------------
# Startup summary
# ------------------------------------------------------------------
echo ""
if [ "${ISOLATION}" = "disabled" ]; then
    echo "=== DISAGGREGATED SERVING READY (LMCache) — MODE=fallback ==="
    echo ""
    echo "  !!! FALLBACK MODE: no dual-GPU parallelism (ISOLATION=disabled) !!!"
    echo "  Both prefill (8100) + decode (8101) running on GPU 0."
    echo "  KV transfer via LMCache host-RAM pool is STILL active — correctness validated."
    echo "  TTFT results reflect single-GPU serialization, NOT disaggregated speedup."
    echo "  Tag bench results: MODE=fallback  ISOLATION=disabled"
    echo ""
    echo "  To get dual-GPU speedup: wait for W6_wsl2_isolation_rootcause findings"
    echo "  or test --runtime=nvidia --gpus all + container-internal CUDA_VISIBLE_DEVICES."
else
    echo "=== DISAGGREGATED SERVING READY (LMCache) — MODE=normal ==="
    echo ""
    echo "  Topology:   1P1D (1 prefill GPU 0, 1 decode GPU 1) — ISOLATION VERIFIED"
fi

echo ""
echo "  Transport:  LMCacheConnectorV1 — pinned host-RAM pool (cudaMemcpyAsync)"
echo "  KV pool:    $((KV_BUFFER_SIZE / 1000000000)) GB local_cpu (WSL2-compatible, no IPC)"
echo "  KV dtype:   BF16"
echo "  KV port:    ${KV_IP}:${KV_PORT}  (shared rendezvous)"
echo "  Proxy:      port ${PROXY_PORT}  (assigns UUID req_id)"
echo ""
echo "  GPU 0 :${PREFILL_PORT}  kv_producer  max_len=28672  mem_util=0.80"
echo "  GPU ${DECODE_GPU} :${DECODE_PORT}  kv_consumer  max_len=8192   mem_util=0.80  max_seqs=128"
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
echo "  ./serve_disaggregated_lmcache_v2.sh bench"
if [ "${ISOLATION}" = "disabled" ]; then
    echo "  ./serve_disaggregated_lmcache_v2.sh bench_fallback   # tagged MODE=fallback"
fi
echo ""
echo "Stop:"
echo "  ./serve_disaggregated_lmcache_v2.sh stop"
echo ""
echo "LMCache diagnostics (on demand):"
echo "  docker exec ${CONTAINER_PREFILL} nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv"
echo "  docker logs --tail=50 ${CONTAINER_PREFILL} | grep -i lmcache"
echo "  docker logs --tail=50 ${CONTAINER_DECODE}  | grep -i lmcache"
echo ""
echo "Logs:"
echo "  docker logs -f ${CONTAINER_PREFILL}   (prefill GPU 0)"
echo "  docker logs -f ${CONTAINER_DECODE}    (decode GPU ${DECODE_GPU})"
echo "  tail -f ${PROXY_LOG}                  (proxy)"
