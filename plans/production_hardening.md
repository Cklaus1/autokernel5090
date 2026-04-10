# Production Hardening Checklist: Gemma4 NVFP4 + FusenCache Coding Assistant Service

**Stack:** Gemma4-26B-A4B (NVFP4 weights) + FusenCache k4v4b64 KV compression + vLLM v1
**Target hardware:** RTX 5090 (32GB VRAM)
**Date:** 2026-04-09

---

## 1. Error Handling

### GPU OOM

**Root causes specific to this stack:**
- FusenCache page-size calculations depend on `plugin.py`'s `_patch_kv_cache_spec()` giving vLLM the correct slot bytes. If the patch silently fails, vLLM will allocate KV pages at BF16 size (4x larger), exhausting VRAM on first batch.
- NVFP4 CUTLASS kernel pre-allocates workspace buffers at model load time. These are not released between requests.

**Mitigations:**
- Add a startup assertion that verifies the page-size patch is active:
  ```python
  # In launch_vllm.py after plugin registration
  from fusen_kv.spec_resolver import resolve_spec
  spec = resolve_spec(kv_dtype)
  expected_slot_bytes = spec.slot_bytes(head_size=256)  # sliding layers
  assert expected_slot_bytes < 256, f"Patch failed: slot_bytes={expected_slot_bytes}, expected <256"
  ```
- Set `--gpu-memory-utilization 0.85` (not the default 0.90). The gap between 0.85 and 1.0 is the OOM buffer for Triton JIT workspace allocations during the first few requests.
- Catch `torch.cuda.OutOfMemoryError` in the vLLM engine wrapper and trigger a graceful partial drain before restarting (see Section 4).
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` to reduce fragmentation. Do not set `expandable_segments:True` — it conflicts with CUDA graph pre-allocation.

**Recovery path:**
1. Log the OOM with VRAM state (`torch.cuda.memory_summary()`).
2. Reject new requests with HTTP 503.
3. Wait for in-flight requests to complete (up to 30s timeout).
4. Call `torch.cuda.empty_cache()`.
5. If VRAM still over 90%, restart the process (systemd/Docker will bring it back up).

### CUDA Errors

**Sources specific to this stack:**
- FusenCache decode kernel: out-of-bounds block table access (fixed by `FUSEN_DEBUG=1` bounds checking, too slow for production).
- CUDA graph replay with wrong sequence lengths (metadata desync after a failed request).
- Triton JIT compilation failure on first request after cold start (kernel not pre-warmed).

**Handling:**
- Catch `RuntimeError` with `"CUDA error"` in message at the vLLM engine level.
- On any CUDA error: abort all pending requests in the current batch with HTTP 500. Do not try to continue — CUDA errors leave device state undefined.
- Log `torch.cuda.current_device()`, driver version, and the full Python traceback.
- Enable `FUSEN_DEBUG=1` in staging only (adds bounds checks, ~5% overhead).
- Pre-warm Triton kernels on startup before accepting traffic (see Section 9).

### Request Timeouts

**Timeout budget for Gemma4-26B coding assistant:**

| Phase | Budget |
|-------|--------|
| Queue wait | 10s max |
| First token (TTFT) | 30s max (prefill of long context) |
| Per-token (decode) | 500ms max (≈ 2 tok/s floor, warn if slower) |
| Total request | 120s hard cap |

**Implementation:**
- Use vLLM's `--request-timeout 120` CLI flag.
- Add a separate application-layer timeout in the API gateway (nginx/envoy) of 125s — slightly longer than vLLM's internal timeout so vLLM can log the abort cleanly.
- For streaming responses: send an SSE keepalive comment (`": keepalive\n\n"`) every 15s during long prefills so reverse proxies do not close idle connections.
- Return HTTP 408 on timeout with a machine-readable body: `{"error": "timeout", "phase": "prefill|decode"}`.

---

## 2. Monitoring

### Core Metrics to Track

**Throughput and latency (export via vLLM's `/metrics` Prometheus endpoint):**

| Metric | Alert threshold |
|--------|----------------|
| `vllm:request_success_total` rate | Drop >10% vs 5m avg |
| `vllm:time_to_first_token_seconds` p50/p99 | p99 > 10s |
| `vllm:time_per_output_token_seconds` p50/p99 | p99 > 300ms |
| `vllm:e2e_request_latency_seconds` p50/p99 | p99 > 90s |
| `vllm:request_queue_time_seconds` | p99 > 5s |
| HTTP 5xx rate | > 0.1% of requests |

**KV cache utilization (custom metrics, add to FusenKV backend):**
- `fusencache_kv_blocks_used` / `fusencache_kv_blocks_total` — alert if >90% for >60s
- `fusencache_kv_evictions_total` rate — high eviction rate means concurrency is too high for current context lengths
- `fusencache_prefill_ms_p99` — prefill latency broken out (NVFP4 GEMM + KV store)
- `fusencache_decode_ms_p99` — decode latency (FusenCache attention kernel)
- Page fault rate: blocks evicted that were later requested again (cache thrashing indicator)

**GPU hardware:**
- `nvidia_smi_memory_used_bytes` / `nvidia_smi_memory_total_bytes` — alert if >88%
- `nvidia_smi_utilization_gpu_ratio` — alert if <30% for >120s (engine stalled)
- `nvidia_smi_temperature_celsius` — alert if >83°C
- `nvidia_smi_power_draw_watts` — alert if >550W sustained (RTX 5090 TDP ~575W)

**System:**
- CPU utilization of vLLM process — alert if >80% (Python scheduling overhead)
- Resident set size (RSS) — alert if growing >100MB/hour (memory leak)
- File descriptor count — alert if >80% of `ulimit -n`

**Recommended stack:** Prometheus scraping vLLM's `/metrics` + node_exporter for GPU via dcgm-exporter + Grafana for dashboards. Add custom gauges via `prometheus_client` in the FusenKV plugin for the KV-specific metrics above.

---

## 3. Health Checks

### Readiness Probe

The readiness probe answers: "Is this instance ready to serve traffic?"

**Conditions that must all be true before marking ready:**
1. vLLM engine is initialized (model weights loaded, KV cache allocated).
2. FusenKV plugin registered successfully (check `fusen_kv.plugin` import + registration log).
3. Triton kernels pre-compiled (run a synthetic decode step at startup — see Section 9).
4. CUDA graphs captured (vLLM logs "CUDA graph capturing done").
5. VRAM utilization is below 90%.

**Endpoint:** `GET /health/ready` — return HTTP 200 when all conditions true, 503 otherwise.

**vLLM provides:** `GET /health` (liveness) and `GET /v1/models` (service up). Extend with a custom readiness wrapper:
```python
@app.get("/health/ready")
async def readiness():
    if not _engine_initialized:
        return JSONResponse({"status": "initializing"}, status_code=503)
    if not _cuda_graphs_captured:
        return JSONResponse({"status": "warming_up"}, status_code=503)
    vram_used = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
    if vram_used > 0.90:
        return JSONResponse({"status": "oom_risk", "vram_pct": vram_used}, status_code=503)
    return {"status": "ready"}
```

**Probe configuration (Kubernetes):**
```yaml
readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 120   # Model load + CUDA graph capture takes ~90s
  periodSeconds: 10
  failureThreshold: 3
  successThreshold: 1
```

### Liveness Probe

The liveness probe answers: "Is this process alive and not deadlocked?"

**Simpler than readiness:** just check that the HTTP server responds.

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 180   # Give it longer than readiness to avoid killing during warm-up
  periodSeconds: 30
  failureThreshold: 3
```

**Key distinction:** A failed liveness probe kills and restarts the container. A failed readiness probe only removes it from load balancer rotation. Set liveness thresholds conservatively — false positives cause unnecessary restarts that take 90s to recover from.

**Deep liveness check (optional, run less frequently):**
- Send a minimal completion request (`max_tokens=1`) and verify it returns within 10s.
- If this fails, the engine is likely deadlocked (CUDA stream hang, Python GIL deadlock).
- Run this at 5-minute intervals, not 30s — it consumes real GPU resources.

---

## 4. Graceful Shutdown

### Shutdown Sequence

The vLLM process must handle `SIGTERM` (systemd/Docker stop) gracefully:

1. **Stop accepting new requests** — close the listening socket or return HTTP 503 immediately.
2. **Signal in-flight requests** — set a flag so streaming responses send a final `{"finish_reason": "server_shutdown"}` event.
3. **Drain active requests** — wait up to 30s for in-flight requests to complete. After 30s, abort remaining with HTTP 503.
4. **Flush logs** — ensure all structured log lines are flushed before exit (Python's `logging` buffers by default on non-TTY).
5. **Clean up CUDA state** — call `torch.cuda.synchronize()`, then `torch.cuda.empty_cache()`. This ensures no CUDA kernels are running when the process exits (avoids driver-level errors in logs).
6. **Exit cleanly** — `sys.exit(0)`.

**Implementation in `launch_vllm.py`:**
```python
import signal, asyncio

_shutdown_event = asyncio.Event()

def _handle_sigterm(signum, frame):
    logger.info("SIGTERM received, initiating graceful shutdown")
    _shutdown_event.set()

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)
```

**Timeout configuration:**
- `STOPSIGNAL SIGTERM` in Dockerfile (default).
- `TimeoutStopSec=45` in systemd unit — 30s drain + 15s cleanup buffer.
- Docker: `--stop-timeout 45`.

### State to Save

**FusenCache has no persistent state** — the KV cache is in-GPU memory and intentionally volatile. No checkpoint needed on shutdown.

**State worth saving on shutdown:**
- Current request queue depth and active session IDs to a Redis/SQLite queue so a new instance can log that sessions were interrupted (for debugging, not for resumption — KV cache is not transferable).
- Prometheus metrics snapshot (optional, prevents gap in time-series on restart).

---

## 5. Auto-Restart

### Systemd Unit

```ini
# /etc/systemd/system/gemma4-fusen.service
[Unit]
Description=Gemma4 NVFP4 + FusenCache vLLM Server
After=network.target nvidia-persistenced.service
Requires=nvidia-persistenced.service

[Service]
Type=simple
User=vllm
WorkingDirectory=/opt/gemma4-fusen
Environment=FUSEN_PATH=/opt/gemma4-fusen
Environment=CUDA_VISIBLE_DEVICES=0
Environment=PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
Environment=VLLM_LOGGING_LEVEL=INFO
ExecStart=/opt/gemma4-fusen/.venv/bin/python3 /opt/gemma4-fusen/fusen_kv/launch_vllm.py \
    --model google/gemma-4-26b-a4b \
    --quantization nvfp4 \
    --kv-cache-dtype k4v4b64 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 65536 \
    --host 0.0.0.0 \
    --port 8000
ExecStop=/bin/kill -TERM $MAINPID
TimeoutStartSec=180
TimeoutStopSec=45
Restart=on-failure
RestartSec=10
RestartPreventExitCode=0
# Rate-limit restarts: max 5 in 5 minutes, then wait
StartLimitBurst=5
StartLimitIntervalSec=300

[Install]
WantedBy=multi-user.target
```

### Restart Policy Rationale

- `Restart=on-failure` — restarts on nonzero exit (CUDA error, OOM kill) but NOT on clean `sys.exit(0)` (planned maintenance).
- `RestartSec=10` — gives the GPU driver 10s to release resources before re-initialization.
- `StartLimitBurst=5` + `StartLimitIntervalSec=300` — if the process crashes 5 times in 5 minutes, systemd stops trying and pages on-call. Prevents infinite crash loops from consuming GPU resources.
- After hitting the limit: `systemctl reset-failed gemma4-fusen && systemctl start gemma4-fusen` to manually restart after fixing the issue.

### Docker Restart Policy

For Docker deployments (without the Docker-avoidance constraint from CLAUDE.md):
```
restart: unless-stopped
```
This is equivalent to `Restart=on-failure` but restarts even on clean exits, which is appropriate if you want the service always running.

---

## 6. Logging

### What to Log

**At every request (structured JSON, one line per request):**
```json
{
  "ts": "2026-04-09T12:34:56.789Z",
  "event": "request_complete",
  "request_id": "req-abc123",
  "model": "gemma4-26b-nvfp4",
  "prompt_tokens": 4096,
  "completion_tokens": 512,
  "ttft_ms": 892,
  "decode_ms_per_token": 14.3,
  "finish_reason": "stop",
  "kv_cache_pct": 73.2,
  "vram_mb": 26800
}
```

**At engine events (INFO level):**
- Model loaded: weights path, quantization method, elapsed seconds
- KV cache allocated: total blocks, bytes per block, total VRAM consumed
- CUDA graphs captured: batch sizes captured, capture duration
- FusenKV plugin registered: dtype patches applied, kernel loaded (Triton or C++)
- Batch processed: batch size, total tokens, decode throughput tok/s

**At errors (ERROR level, always include full traceback):**
- CUDA errors: device state, memory summary, batch state at time of error
- OOM events: memory_summary() output, current batch size, recent 5 request sizes
- Timeout events: request ID, phase (prefill/decode), elapsed time
- Plugin patch failures: which patch failed, fallback behavior

**Never log:**
- Request content (prompt text, completion text) — privacy/compliance
- API keys or authentication tokens
- Full tensor values (too large, no value)

### Structured Logging Setup

```python
import logging, json, sys
from datetime import datetime, timezone

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            log["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields passed via extra={}
        for k, v in record.__dict__.items():
            if k not in ("msg", "args", "levelname", "name", "exc_info",
                         "exc_text", "stack_info", "lineno", "funcName",
                         "pathname", "filename", "module", "created",
                         "msecs", "relativeCreated", "thread", "threadName",
                         "processName", "process", "taskName"):
                log[k] = v
        return json.dumps(log)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JSONFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
```

### Log Rotation

- Use systemd's journal (automatic, no extra config needed when running as a service).
- Journal size limit: set `SystemMaxUse=2G` in `/etc/systemd/journald.conf`.
- For file-based logging: use `logrotate` with `daily`, `rotate 14`, `compress`, `postrotate systemctl kill -s USR1 gemma4-fusen`.
- Ship logs to a log aggregation service (Loki, CloudWatch, Datadog) — local storage is not a substitute for searchable log retention across restarts.

---

## 7. Security

### API Authentication

**Minimum viable auth:** bearer token validation in the API gateway, not inside vLLM.

vLLM's `--api-key` flag validates a single static token. For multi-tenant production:
- Place nginx or envoy in front of vLLM.
- Validate JWT or API key in the gateway.
- Never expose vLLM's port 8000 directly; bind to `127.0.0.1` and let the gateway proxy.

```nginx
# nginx upstream config
server {
    listen 443 ssl;
    location /v1/ {
        auth_request /auth;
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 130s;
        proxy_send_timeout 130s;
    }
    location /auth {
        internal;
        proxy_pass http://127.0.0.1:9000/verify;  # Auth microservice
    }
}
```

### Rate Limiting

**Two levels:**

1. **Per-IP rate limiting** in nginx:
   ```nginx
   limit_req_zone $binary_remote_addr zone=api:10m rate=10r/m;
   limit_req zone=api burst=5 nodelay;
   ```

2. **Per-token rate limiting** at the application layer: track requests per API key per minute in Redis. For coding assistant use: 60 requests/minute/key, 100K tokens/minute/key.

**Reject with HTTP 429 and include `Retry-After` header.**

### Input Validation

**Before sending to vLLM:**
- Enforce `max_tokens` cap: coding assistant should cap at 8192 output tokens. Reject requests asking for more.
- Enforce `prompt_tokens` cap: reject prompts over 60K tokens (leave headroom for the FusenCache KV budget).
- Sanitize `stop` sequences: reject lists with more than 4 stop sequences or sequences over 50 characters.
- Validate `temperature` is in [0.0, 2.0] and `top_p` in (0.0, 1.0].
- Block known prompt injection patterns for system prompt leakage (if serving a system prompt).
- Content length check: reject requests where `Content-Length` header exceeds 1MB before parsing JSON.

**Specific to this stack:**
- The FusenKV `--kv-cache-dtype` is a server-side config, not a per-request parameter. Never expose it as user-configurable input.

### Network Security

- TLS 1.3 only at the gateway.
- Bind vLLM to `127.0.0.1:8000`, not `0.0.0.0`.
- Firewall: allow only the gateway to connect to port 8000.
- Do not expose the Prometheus `/metrics` endpoint publicly — proxy it through an internal-only path.

---

## 8. Memory Management

### VRAM Budget for Gemma4-26B NVFP4 + FusenCache k4v4b64

| Component | VRAM estimate |
|-----------|--------------|
| NVFP4 model weights (26B params at FP4) | ~13 GB |
| Activation workspace (NVFP4 CUTLASS buffers) | ~1 GB |
| CUDA graphs (batch sizes 1–256) | ~2 GB |
| Triton JIT workspace | ~0.5 GB |
| FusenCache KV blocks (at 0.85 utilization) | ~13 GB |
| OS + driver overhead | ~0.5 GB |
| **Total target** | **~30 GB / 32 GB** |

### VRAM Monitoring

- Poll `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()` every 30s.
- Alert thresholds:
  - >85% reserved: warn, consider reducing `max_num_seqs`
  - >90% allocated: stop accepting new prefill requests
  - >95%: trigger graceful drain (Section 4)

- Monitor fragmentation: `reserved - allocated`. If this exceeds 2GB, schedule a maintenance restart (fragmented allocator hurts future large allocations).

### OOM Prevention

- Set `--max-num-seqs 128` as initial concurrency limit. Tune based on load testing results (Section 9). Each additional sequence consumes ~100MB in FusenCache KV blocks for 4K context.
- Enable `--block-manager-sliding-window` if not already set — this limits sliding-window layer KV memory to the window size (1024 tokens) rather than full context length.
- Watchdog thread: monitor VRAM every 5s; if >90%, set a flag that causes the scheduler to reject new sequence starts until VRAM drops below 80%.
- Set Linux `vm.overcommit_memory=0` on the host to prevent CPU RAM overcommit that can indirectly cause GPU driver instability.

### Swap Configuration

- **Do not use GPU swap** (NVLink/NVMe offload). FusenCache's entire value is keeping KV on-GPU. Swap defeats the purpose and would tank latency.
- CPU RAM swap: disable it or set `vm.swappiness=10`. vLLM's Python process should be resident in RAM. If the system is swapping, add RAM or reduce `--max-num-seqs`.
- NUMA: pin the vLLM process to the NUMA node closest to the GPU (`numactl --cpunodebind=0 --membind=0`). On RTX 5090 workstations with single-socket CPUs, this is typically a no-op but adds no harm.

---

## 9. Load Testing

### Pre-Deployment Test Protocol

Run all tests in a staging environment with identical hardware before deploying to production.

**Step 1: Cold-start warmup test**
```bash
# Time from process start to first successful response
time python3 fusen_kv/launch_vllm.py [args] &
SERVER_PID=$!
until curl -sf http://localhost:8000/health/ready; do sleep 2; done
echo "Ready. PID=$SERVER_PID"
```
Expected: ready within 90-120s. If >150s, investigate CUDA graph capture time.

**Step 2: Triton kernel pre-warm test**
Send a synthetic request with representative dimensions (4096 prompt, 512 output) before opening to real traffic. Verify no JIT compilation happens after the first request by checking that the second identical request is faster.

**Step 3: Throughput sweep**
Using a load generator (recommend `vllm-bench` or `locust`):
```
Concurrency: 1, 2, 4, 8, 16, 32, 64, 128
Prompt length: 1024 tokens (typical coding assistant prompt)
Output length: 512 tokens
Duration: 5 minutes per concurrency level
```
Target metrics: >4,000 tok/s aggregate at C=64 (based on FusenCache projections).
Stop increasing concurrency when p99 TTFT exceeds 10s or VRAM utilization exceeds 88%.

**Step 4: Long context test**
Send 10 concurrent requests with 32K token prompts. Verify:
- No OOM
- FusenCache block utilization does not exceed the page budget
- TTFT for long prompts under 30s

**Step 5: Spike test**
Baseline at C=32. Spike to C=128 for 60s. Verify:
- Excess requests queue (not crash)
- Queue drains within 2x the request timeout after spike ends
- No CUDA errors during the spike

**Step 6: Sustained load test**
Run at 80% of peak throughput for 4 hours. Check:
- No memory growth (RSS and VRAM should be stable after first 15 minutes)
- Error rate stays below 0.01%
- No Triton kernel re-compilation events (would indicate cache eviction)

**Step 7: Failure injection**
- Kill the process mid-request with `kill -9`. Verify systemd restarts within 15s.
- Send malformed JSON. Verify HTTP 400, no crash.
- Send a 70K token prompt (over limit). Verify HTTP 400 rejection.
- Corrupt one request's `stop` field. Verify only that request fails, not others.

### Load Test Tools

- `locust` — Python-native, easy to customize for streaming SSE
- `k6` — for sustained HTTP load
- `vllm-bench` — purpose-built, models realistic LLM request distributions
- For streaming: use `httpx` with `stream=True` to measure TTFT vs total latency separately

---

## 10. Backup and Recovery

### What Needs Backup

**FusenCache has no persistent state.** The KV cache is in GPU VRAM and is intentionally not persisted. On restart, all KV state is rebuilt from prefill.

**Items that do need versioning:**

| Artifact | Where | Backup strategy |
|----------|-------|-----------------|
| Model weights (NVFP4 checkpoint) | Local disk or model hub | Pinned HuggingFace commit SHA in deployment manifest |
| FusenKV plugin code | Git | Tag each production deployment |
| vLLM version | pip | Pin in `pyproject.toml`, freeze with `pip freeze > requirements.lock` |
| Triton kernel source | `kv_cache_gen/kernel.py` | Git-tracked, version-tagged |
| Service configuration | systemd unit / env file | Store in a config repo, apply with Ansible/Salt |
| Launch script | `fusen_kv/launch_vllm.py` | Git-tracked |

### Rollback Procedure

**To roll back a bad deployment:**

1. Identify the last known-good git tag: `git log --oneline --tags`
2. Check out the tag: `git checkout v1.2.3`
3. Reinstall dependencies: `uv sync`
4. Restart service: `systemctl restart gemma4-fusen`
5. Verify readiness: `curl http://localhost:8000/health/ready`
6. Run smoke test: send one completion request, verify response

**Expected rollback time:** 2-3 minutes (no model re-download needed, weights are cached).

### Model Weight Versioning

- Pin the HuggingFace model commit in the launch script:
  ```python
  MODEL_COMMIT = "abc123def456"  # pin exact commit
  model = f"google/gemma-4-26b-a4b@{MODEL_COMMIT}"
  ```
- Keep the last 2 model versions on disk (the current and one previous). Disk space: ~26GB per version.
- On model update: download new version first, validate with smoke test, then switch the symlink. Keep old version for 7 days before deleting.

### FusenCache Kernel Checkpointing

The Triton kernel in `kv_cache_gen/kernel.py` is the most likely source of regressions. The kernel version is Git-tracked. Additional steps:

- Tag kernel versions that pass full benchmarks: `git tag kernel-v12-63toks`
- Before any kernel edit, note the current throughput in `results.tsv`.
- The experiment loop's REVERT path (`git reset --hard HEAD~1`) is the rollback mechanism for kernel changes.

---

## 11. Multi-Model Serving on Same GPU

**Summary verdict:** Not recommended for this stack. Reasons:

1. **NVFP4 model weights are 13GB.** A second NVFP4 model of similar size would require 26GB just for weights, leaving 6GB for both models' KV caches — far too little for meaningful concurrency.
2. **FusenCache's KV block pool is allocated at startup** and cannot be dynamically split between two models without a significant refactor of vLLM's block manager.
3. **CUDA graphs are per-model.** Two models with CUDA graphs require 2x the graph capture memory and time.

**If multi-model is required, options ranked by viability:**

### Option A: Time-Multiplexing (Swap Models)

Run one model at a time. When a request for Model B arrives while Model A is loaded:
1. Drain Model A's in-flight requests (up to 30s).
2. Unload Model A: `del engine; torch.cuda.empty_cache()`.
3. Load Model B: ~45s cold start.
4. Serve Model B.

**Viable for:** low-traffic scenarios where each model serves at most a few requests per minute. Not viable for interactive coding assistant with <5s latency SLA.

### Option B: Smaller Second Model

Pair Gemma4-26B-NVFP4 (~15GB with KV) with a very small model (e.g., Gemma3-2B at ~4GB). The small model handles short/simple requests and does not need FusenCache. Use a request router to dispatch based on prompt complexity.

**Allocation:**
- Gemma4-26B-NVFP4 + FusenCache: 30GB
- Gemma3-2B BF16 (no KV compression needed): 2GB
- Total: 32GB — fits on RTX 5090 if Gemma4 uses `--gpu-memory-utilization 0.70`

**Implementation:** `nginx` upstream routing based on a complexity classifier (response time of a first-pass model, or a simple rule: prompt length < 512 tokens → small model).

### Option C: Multi-GPU

The correct solution. RTX 5090 SLI or a second GPU for the second model. Each model gets a dedicated GPU with full VRAM.

### Option D: vLLM Multi-LoRA (same base model)

If both "models" are fine-tunes of the same Gemma4-26B base with LoRA adapters:
- vLLM's `--enable-lora` serves multiple adapters with one base model in VRAM.
- NVFP4 + LoRA compatibility: check vLLM release notes; as of v1 this requires the base model weights to be the unquantized checkpoint, with NVFP4 applied at load time.
- FusenCache is compatible with multi-LoRA (KV cache is independent of LoRA adapters).

---

## 12. Zero-Downtime Updates

### Updating vLLM

vLLM updates frequently and often require coordinating with FusenKV's monkey-patches (`plugin.py`, `launch_vllm.py`). The patching surface includes:

- `CacheDType` Literal expansion
- `CacheConfig.__init__` wrapping
- `CudaPlatform.get_attn_backend_cls` monkey-patch
- `AttentionSpec.real_page_size_bytes` override

**Any vLLM update must be validated against all five patches before deployment.**

**Update procedure:**
1. Create a staging environment with the new vLLM version.
2. Run the FusenKV test suite: `uv run pytest fusen_kv/tests/ -v`
3. Run a full integration test: load Gemma4, send 100 requests, verify throughput within 5% of baseline.
4. If tests pass: deploy to one production instance (canary).
5. Monitor canary for 30 minutes (check error rate, latency, VRAM).
6. Roll out to remaining instances in batches.

**Blue-green deployment** (if you have two instances):
- Bring up new instance (green) with new vLLM version.
- Wait for readiness probe to pass.
- Shift traffic 10% to green, monitor 5 minutes.
- Shift 50%, monitor 10 minutes.
- Shift 100% to green.
- Keep blue alive for 30 minutes as rollback target.
- Shut down blue.

**For systemd single-instance deployments** (no second GPU for blue-green):
- Accept brief downtime (~90s restart time).
- Schedule during low-traffic window.
- Use `systemctl restart` not `stop` + `start` to minimize gap.

### Updating Model Weights

Fine-tune checkpoint updates are lower risk than vLLM updates because they do not touch the plugin code.

1. Download new weights to a staging path.
2. Run perplexity eval: `uv run fusen_kv/eval_perplexity.py --model /path/to/new/weights`
3. Verify NVFP4 scale tensors loaded correctly (`fix_nvfp4_scales.py`).
4. Compare output quality on a held-out coding prompt set.
5. If quality regression <1% perplexity increase: proceed.
6. Update the symlink: `ln -sfn /models/gemma4-new /models/gemma4-current`
7. Restart service.

### Updating FusenCache Kernel

Kernel updates (`kv_cache_gen/kernel.py`) are high-risk — they directly affect KV quantization quality and decode correctness.

1. Follow the full AutoKernel experiment loop: commit → bench → verify → keep/revert.
2. Run `uv run verify.py` to confirm numerical correctness.
3. Run the serving benchmark at sustained load for 30 minutes before promoting.
4. Tag the kernel version in Git before deploying.
5. Keep the previous kernel version tagged for immediate rollback.

**Rollback trigger:** any of these after a kernel update warrants immediate rollback:
- p99 TTFT increases >20%
- Correctness check fails (cosine similarity < 0.99 vs BF16 reference)
- Any CUDA error in production logs
- Decode throughput drops >5% vs baseline

---

## Deployment Readiness Checklist

Before marking the service production-ready, verify all of the following:

**Functionality:**
- [ ] FusenKV plugin registers successfully on startup (check logs)
- [ ] KV page size patch is active (verify `slot_bytes` assertion passes)
- [ ] CUDA graphs captured for batch sizes 1, 2, 4, 8, 16, 32, 64, 128, 256
- [ ] First request after startup completes within TTFT SLA (<30s for 4K prompt)
- [ ] Sustained load test (4 hours at 80% peak) passes with no errors

**Reliability:**
- [ ] OOM handling tested: process survives a request that would OOM
- [ ] CUDA error handling tested: process recovers or restarts cleanly
- [ ] SIGTERM graceful shutdown tested: in-flight requests complete
- [ ] Systemd restart tested: process comes back after `kill -9`
- [ ] Rollback tested: previous version restores correctly in <5 minutes

**Observability:**
- [ ] Prometheus metrics scraping `/metrics` endpoint
- [ ] Alerts configured for p99 TTFT, error rate, VRAM utilization
- [ ] Structured logs shipping to aggregation system
- [ ] VRAM watchdog running and alerting

**Security:**
- [ ] API authentication in place (gateway or `--api-key`)
- [ ] Rate limiting configured (per-IP and per-token)
- [ ] Input validation rejecting oversized prompts
- [ ] vLLM bound to `127.0.0.1`, not `0.0.0.0`
- [ ] TLS at gateway

**Capacity:**
- [ ] `--max-num-seqs` tuned based on load test results
- [ ] `--gpu-memory-utilization 0.85` set
- [ ] VRAM budget verified: <30GB total at steady state
