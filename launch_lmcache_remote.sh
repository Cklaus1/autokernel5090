#!/bin/bash
# Launch LMCache Tier 3 — standalone remote shared KV cache server
# Tag: W3_P3_lmcache_hierarchy_staged
#
# Prerequisites:
#   - lmcache==0.4.3 installed in the active Python env
#   - /home/cklaus/projects/autokernel/configs/lmcache_t3_remote.yaml present
#
# CPU pinning strategy (PRO 6000 / Ryzen Threadripper / desktop Zen 4 topology):
#   - On a 3D V-Cache CPU (e.g. Ryzen 9 7950X3D), the 3D V-Cache CCD (CCD0) has
#     boosted L3 but can throttle on sustained memory bandwidth.
#   - Pin T3 server to CCD1 (non-V-Cache) cores for sustained DRAM bandwidth
#     needed to service 60 GB of LMCache traffic.
#   - Default assumes a 16-core Zen4 with 2 CCDs of 8 cores each (cores 8-15 = CCD1).
#   - Adjust LMCACHE_T3_CPUSET to match your actual topology
#     (use `lscpu --extended` or `lstopo` to inspect).
#   - On a non-3D-V-Cache system (uniform L3), CPUSET still provides NUMA isolation.
#
# Usage:
#   ./launch_lmcache_remote.sh            # start T3 server (background)
#   ./launch_lmcache_remote.sh --foreground  # foreground (for debugging)
#   LMCACHE_T3_CPUSET=0-7 ./launch_lmcache_remote.sh   # override core set

set -euo pipefail

AUTOKERNEL_DIR="/home/cklaus/projects/autokernel"
T3_CONFIG="${AUTOKERNEL_DIR}/configs/lmcache_t3_remote.yaml"
T3_HOST="127.0.0.1"
T3_PORT=8200

# Default: CCD1 cores on a 16-core Zen4 (cores 8-15).
# Override with env var if your topology differs.
LMCACHE_T3_CPUSET="${LMCACHE_T3_CPUSET:-8-15}"

FOREGROUND=0
if [[ "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
fi

# ── Validation ─────────────────────────────────────────────────────────────────
if [[ ! -f "${T3_CONFIG}" ]]; then
  echo "[T3] ERROR: config not found at ${T3_CONFIG}" >&2
  exit 1
fi

if ! python3 -c "import lmcache" 2>/dev/null; then
  echo "[T3] ERROR: lmcache package not importable. Run:" >&2
  echo "       pip install --break-system-packages lmcache==0.4.3" >&2
  exit 1
fi

if ! command -v taskset &>/dev/null; then
  echo "[T3] WARNING: taskset not found; launching without CPU pinning." >&2
  TASKSET_PREFIX=""
else
  TASKSET_PREFIX="taskset -c ${LMCACHE_T3_CPUSET}"
fi

# ── Port check ────────────────────────────────────────────────────────────────
if ss -tlnp 2>/dev/null | grep -q ":${T3_PORT}"; then
  echo "[T3] WARNING: port ${T3_PORT} already in use. T3 server may already be running."
  echo "[T3] Check with: ss -tlnp | grep ${T3_PORT}"
  exit 0
fi

# ── Launch ────────────────────────────────────────────────────────────────────
LOG_FILE="/tmp/lmcache_t3.log"

LAUNCH_CMD="${TASKSET_PREFIX} python3 -m lmcache.server \
  --config ${T3_CONFIG} \
  --host ${T3_HOST} \
  --port ${T3_PORT}"

echo "[T3] Launching LMCache remote tier (standalone server)"
echo "[T3] Config   : ${T3_CONFIG}"
echo "[T3] Endpoint : tcp://${T3_HOST}:${T3_PORT}"
echo "[T3] CPU set  : ${LMCACHE_T3_CPUSET}"
echo "[T3] Pool     : 60 GB, eviction=LFU, TTL=86400s"
echo ""

if [[ "${FOREGROUND}" -eq 1 ]]; then
  echo "[T3] Running in foreground (Ctrl-C to stop)..."
  exec ${LAUNCH_CMD}
else
  ${LAUNCH_CMD} > "${LOG_FILE}" 2>&1 &
  T3_PID=$!
  echo "[T3] PID: ${T3_PID}  (log: ${LOG_FILE})"

  # ── Health check: wait up to 30 s for the server to bind ─────────────────
  echo "[T3] Waiting for server to bind on port ${T3_PORT}..."
  WAITED=0
  while [[ ${WAITED} -lt 30 ]]; do
    if ss -tlnp 2>/dev/null | grep -q ":${T3_PORT}"; then
      echo "[T3] Server listening on tcp://${T3_HOST}:${T3_PORT} after ${WAITED}s"
      break
    fi
    # Also accept HEARTBEAT log line as readiness signal
    if grep -q "HEARTBEAT\|Listening\|Server started\|ready" "${LOG_FILE}" 2>/dev/null; then
      echo "[T3] Server log signals ready after ${WAITED}s"
      break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
  done

  if [[ ${WAITED} -ge 30 ]]; then
    echo "[T3] WARNING: server did not appear ready after 30 s. Check ${LOG_FILE}" >&2
    echo "[T3] Last 10 log lines:"
    tail -10 "${LOG_FILE}" 2>/dev/null || true
    exit 1
  fi

  echo ""
  echo "[T3] T3 remote tier READY."
  echo "[T3] Verify with: ss -tlnp | grep ${T3_PORT}"
  echo "[T3] Tail log  : tail -f ${LOG_FILE}"
  echo "[T3] Stop      : kill ${T3_PID}  (or pkill -f 'lmcache.server')"
fi
