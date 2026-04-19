#!/bin/bash
# WSL2 GPU isolation test script
# Tag: W6_wsl2_isolation_rootcause
# Date: 2026-04-18
#
# Tests 5 isolation approaches (a)-(e) with a 2-container workload.
# Each container prints its UUID; success = distinct UUIDs on GPU 0 and GPU 1.
#
# Usage:
#   sudo ./test_wsl2_gpu_isolation.sh [a|b|c|d|e|all]
#   Default: runs approach (e) (highest likelihood of working on this WSL2 config)
#
# REQUIREMENTS:
#   - Docker running
#   - nvidia-container-toolkit installed
#   - At least one Docker image with Python + PyTorch (uses vllm-built:latest)
#   - For approach (c): /dev/nvidia0 and /dev/nvidia1 must exist in WSL2 namespace
#
# DOES NOT start vLLM or load models — just runs UUID checks. CPU-equivalent time: ~5s.

set -euo pipefail

IMAGE="${TEST_IMAGE:-vllm-built:latest}"
APPROACH="${1:-e}"

# --- Shared UUID check snippet (runs inside both containers) ---
UUID_CHECK='python3 -c "
import subprocess, sys
try:
    import torch
    n = torch.cuda.device_count()
    if n == 0:
        print(\"[GPU-ISOLATION-CHECK] ERROR: no CUDA devices visible\", flush=True)
        sys.exit(1)
    uuid0 = torch.cuda.get_device_properties(0).uuid
    uuids = [torch.cuda.get_device_properties(i).uuid for i in range(n)]
    print(f\"[GPU-ISOLATION-CHECK] visible={n} uuid0={uuid0} all_uuids={uuids}\", flush=True)
    if n > 1:
        print(\"[GPU-ISOLATION-WARN] visible > 1: WSL2 cgroup isolation not working\", flush=True)
except Exception as e:
    print(f\"[GPU-ISOLATION-CHECK] EXCEPTION: {e}\", flush=True)
    sys.exit(1)
"'

cleanup() {
    docker rm -f wsl2_test_gpu0 wsl2_test_gpu1 2>/dev/null || true
}
trap cleanup EXIT
cleanup

run_uuid_check() {
    local NAME=$1
    shift
    docker run --rm --name "${NAME}" "$@" \
        --entrypoint bash "${IMAGE}" -c "${UUID_CHECK}" 2>&1
}

collect_results() {
    local label=$1
    local out0=$2
    local out1=$3

    echo ""
    echo "=== APPROACH ${label} RESULTS ==="
    echo "--- GPU 0 container ---"
    echo "${out0}"
    echo "--- GPU 1 container ---"
    echo "${out1}"

    uuid0=$(echo "${out0}" | grep -oP 'uuid0=\K\S+' || echo "NOT_FOUND")
    uuid1=$(echo "${out1}" | grep -oP 'uuid0=\K\S+' || echo "NOT_FOUND")
    visible0=$(echo "${out0}" | grep -oP 'visible=\K\d+' || echo "?")
    visible1=$(echo "${out1}" | grep -oP 'visible=\K\d+' || echo "?")

    echo ""
    echo "--- ISOLATION VERDICT ---"
    echo "  GPU 0 container: visible=${visible0}, UUID=${uuid0}"
    echo "  GPU 1 container: visible=${visible1}, UUID=${uuid1}"

    if [ "${uuid0}" = "NOT_FOUND" ] || [ "${uuid1}" = "NOT_FOUND" ]; then
        echo "  VERDICT: FAIL — could not read UUID from one or both containers"
    elif [ "${uuid0}" = "${uuid1}" ]; then
        echo "  VERDICT: FAIL — both containers show SAME UUID (${uuid0})"
        echo "           => isolation not working; both landed on same physical GPU"
    else
        echo "  VERDICT: PASS — distinct UUIDs confirmed"
        echo "           => GPU isolation working for this approach"
        if [ "${visible0}" != "1" ] || [ "${visible1}" != "1" ]; then
            echo "  WARNING: one or both containers sees >1 GPU (cgroup isolation absent)"
            echo "           Isolation relies on CUDA_VISIBLE_DEVICES only — fragile."
        fi
    fi
    echo "=============================="
}

# ============================================================
# Approach (a): --runtime=nvidia + --gpus all + CUDA_VISIBLE_DEVICES=N
# Both containers get all GPUs; rely on CUDA env var for device selection.
# Risk: CUDA_VISIBLE_DEVICES=0 in both → both land on GPU 0.
# This approach replicates the KNOWN FAILURE in serve_dual_model.sh.
# Expected result: FAIL (same UUID).
# ============================================================
run_approach_a() {
    echo ""
    echo ">>> Running approach (a): --runtime=nvidia --gpus all -e CUDA_VISIBLE_DEVICES=N"
    echo "    (Expected: FAIL — CUDA_VISIBLE_DEVICES=0 in BOTH containers maps to host GPU 0)"

    out0=$(run_uuid_check wsl2_test_gpu0 \
        --runtime=nvidia --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e CUDA_VISIBLE_DEVICES=0)

    out1=$(run_uuid_check wsl2_test_gpu1 \
        --runtime=nvidia --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e CUDA_VISIBLE_DEVICES=0)

    collect_results "a (BASELINE REPLICATION — expect FAIL)" "${out0}" "${out1}"
}

# ============================================================
# Approach (b): NVIDIA_VISIBLE_DEVICES env only, no --gpus flag
# Only works if accept-nvidia-visible-devices-envvar-when-unprivileged=true in config.toml.
# ============================================================
run_approach_b() {
    echo ""
    echo ">>> Running approach (b): -e NVIDIA_VISIBLE_DEVICES=N only (no --gpus)"
    echo "    (Works only if config.toml: accept-nvidia-visible-devices-envvar-when-unprivileged=true)"

    out0=$(run_uuid_check wsl2_test_gpu0 \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e CUDA_VISIBLE_DEVICES=0)

    out1=$(run_uuid_check wsl2_test_gpu1 \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        -e CUDA_VISIBLE_DEVICES=0)

    collect_results "b (env-only)" "${out0}" "${out1}"
}

# ============================================================
# Approach (c): Manual device file mounts, no --gpus flag
# Bypasses nvidia-container-runtime isolation entirely.
# Container gets ONLY its GPU's /dev/nvidia* device files.
# Expected: PASS if /dev/nvidia0 and /dev/nvidia1 exist in WSL2.
# ============================================================
run_approach_c() {
    echo ""
    echo ">>> Running approach (c): --device=/dev/nvidiaN -- manual device file mounts"
    echo "    (Bypasses nvidia-container-runtime; requires /dev/nvidia0,1 in WSL2 namespace)"

    if [ ! -e /dev/nvidia0 ]; then
        echo "    SKIP: /dev/nvidia0 does not exist in this WSL2 environment"
        echo "    This means WSL2 is using /dev/dxg only (no per-GPU device files)"
        echo "    Approach (c) is not viable on this WSL2 build."
        return
    fi

    out0=$(run_uuid_check wsl2_test_gpu0 \
        --device=/dev/nvidia0:/dev/nvidia0 \
        --device=/dev/nvidiactl:/dev/nvidiactl \
        --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
        -e CUDA_VISIBLE_DEVICES=0)

    out1=$(run_uuid_check wsl2_test_gpu1 \
        --device=/dev/nvidia1:/dev/nvidia1 \
        --device=/dev/nvidiactl:/dev/nvidiactl \
        --device=/dev/nvidia-uvm:/dev/nvidia-uvm \
        -e CUDA_VISIBLE_DEVICES=0)

    collect_results "c (device file mounts)" "${out0}" "${out1}"
}

# ============================================================
# Approach (d): --privileged + explicit device files (nuclear option)
# Escalates container privileges; not suitable for production but tests
# whether the isolation CAN work if device files are properly set.
# ============================================================
run_approach_d() {
    echo ""
    echo ">>> Running approach (d): --privileged + CUDA_VISIBLE_DEVICES (nuclear option)"
    echo "    WARNING: --privileged gives full host access — test only, never production"

    if [ ! -e /dev/nvidia0 ]; then
        echo "    SKIP: /dev/nvidia0 does not exist; skipping privileged device test"
        return
    fi

    out0=$(run_uuid_check wsl2_test_gpu0 \
        --privileged \
        -e CUDA_VISIBLE_DEVICES=0)

    out1=$(run_uuid_check wsl2_test_gpu1 \
        --privileged \
        -e CUDA_VISIBLE_DEVICES=1)

    collect_results "d (privileged + CUDA_VISIBLE_DEVICES)" "${out0}" "${out1}"
}

# ============================================================
# Approach (e): --gpus 'device=N' + CUDA_VISIBLE_DEVICES=N (HOST INDEX, not 0)
# KEY INSIGHT: When no-cgroups=true and cgroup isolation is absent,
# the container sees all GPUs. CUDA_VISIBLE_DEVICES=0 maps to host GPU 0
# for BOTH containers. CUDA_VISIBLE_DEVICES=1 in container 2 maps to
# host GPU 1. This uses CUDA's own device selection as the isolation mechanism.
#
# This is the RECOMMENDED FIX for serve_dual_model.sh.
# Expected: PASS — distinct UUIDs.
# ============================================================
run_approach_e() {
    echo ""
    echo ">>> Running approach (e): --gpus 'device=N' + CUDA_VISIBLE_DEVICES=N (HOST INDEX)"
    echo "    (THE PROPOSED FIX: container 2 uses CUDA_VISIBLE_DEVICES=1, not 0)"
    echo "    Rationale: when no-cgroups=true, all GPU device files visible in container;"
    echo "    CUDA_VISIBLE_DEVICES=N selects by host PCIe enumeration index."

    out0=$(run_uuid_check wsl2_test_gpu0 \
        --gpus '"device=0"' \
        -e NVIDIA_VISIBLE_DEVICES=0 \
        -e CUDA_VISIBLE_DEVICES=0)

    out1=$(run_uuid_check wsl2_test_gpu1 \
        --gpus '"device=1"' \
        -e NVIDIA_VISIBLE_DEVICES=1 \
        -e CUDA_VISIBLE_DEVICES=1)

    collect_results "e (RECOMMENDED FIX — host-index CUDA_VISIBLE_DEVICES)" "${out0}" "${out1}"
}

# ============================================================
# Main
# ============================================================

echo "========================================"
echo "WSL2 GPU Isolation Test Suite"
echo "Tag: W6_wsl2_isolation_rootcause"
echo "Image: ${IMAGE}"
echo "Approach(es): ${APPROACH}"
echo "========================================"

# First: print current system info
echo ""
echo "=== System Info ==="
echo -n "Docker: "; docker version --format '{{.Server.Version}}' 2>/dev/null || echo "unknown"
echo -n "WSL2 kernel: "; uname -r
echo -n "nvidia-smi GPU count: "
nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "unknown"
echo "GPU list:"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader 2>/dev/null || echo "  (nvidia-smi not available)"
echo ""
echo -n "/dev/nvidia0 exists: "; [ -e /dev/nvidia0 ] && echo "YES" || echo "NO"
echo -n "/dev/nvidia1 exists: "; [ -e /dev/nvidia1 ] && echo "YES" || echo "NO"
echo -n "/dev/dxg exists: "; [ -e /dev/dxg ] && echo "YES (WSL2 path confirmed)" || echo "NO"
echo ""

# Dump config.toml key settings
echo "=== nvidia-container-runtime config ==="
if [ -f /etc/nvidia-container-runtime/config.toml ]; then
    grep -E 'no-cgroups|accept-nvidia-visible-devices' /etc/nvidia-container-runtime/config.toml || echo "  (key settings not found in config.toml)"
else
    echo "  /etc/nvidia-container-runtime/config.toml NOT FOUND"
fi
echo ""

case "${APPROACH}" in
    a) run_approach_a ;;
    b) run_approach_b ;;
    c) run_approach_c ;;
    d) run_approach_d ;;
    e) run_approach_e ;;
    all)
        run_approach_a
        run_approach_b
        run_approach_c
        run_approach_d
        run_approach_e
        ;;
    *)
        echo "Unknown approach '${APPROACH}'. Valid: a|b|c|d|e|all"
        exit 1
        ;;
esac

echo ""
echo "=== SUMMARY ==="
echo "If approach (e) PASS: update serve_dual_model.sh to use CUDA_VISIBLE_DEVICES=1 for GPU 1 container."
echo "If approach (e) FAIL: apply config.toml Patch A + docker restart, then re-test approach (b)."
echo "If approach (c) PASS: use manual --device= mounts as fallback; no config change needed."
echo "If ALL fail: WSL2 cgroup isolation is architecturally absent. Use serial containers only."
echo ""
echo "Success criterion: both containers show visible=1 with DISTINCT uuids in GPU-ISOLATION-CHECK line."
