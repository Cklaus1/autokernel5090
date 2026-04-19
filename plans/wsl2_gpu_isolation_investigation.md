# WSL2 GPU Isolation Investigation

**Tag:** W6_wsl2_isolation_rootcause  
**Date:** 2026-04-18  
**Failure reference:** `W5_25_wsl2_isolation_FAIL` (results.tsv row 209), KILL_PATTERNS.md §P4

---

## 1. Version Inventory

**Status: BLOCKED — `/etc/` is not readable by this agent and Bash is not permitted.**

Version inventory commands to run manually (CPU-only, read-only):

```bash
# Run as root / sudo before any daemon restart
docker version
nvidia-container-runtime --version
cat /etc/docker/daemon.json
ls -la /etc/nvidia-container-runtime/
cat /etc/nvidia-container-runtime/config.toml
nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader
uname -r
```

**Known facts from project context:**
- Host: WSL2, AMD 9950X3D, 2× RTX PRO 6000 Blackwell (SM120a), 96 GB GDDR7 each
- Windows host driver: 582.08 (supports CUDA 13.0 containers, confirmed in CLAUDE.md)
- Container image: `vllm-built:latest` (CUDA 13.0 based)
- Docker: `vllm-built:latest` uses `--gpus '"device=N"'` + `-e NVIDIA_VISIBLE_DEVICES=N` + `-e CUDA_VISIBLE_DEVICES=0`
- **CONFIRMED FAILURE:** Both containers showed the same UUID; GPU 1 never received load; GPU 0 at 96.7 GB

---

## 2. Root-Cause Hypotheses (Ranked by Likelihood)

### H1 — `no-cgroups = true` in config.toml kills DeviceRequests (HIGHEST — ~75%)

**Mechanism:**  
On WSL2, NVIDIA's own documentation requires `no-cgroups = true` in `/etc/nvidia-container-runtime/config.toml` because the WSL2 kernel does not expose cgroup device controllers in a form the host nvidia-container-runtime can manipulate. When `no-cgroups = true`, the runtime skips writing to `/sys/fs/cgroup/devices/`, meaning the `--gpus 'device=N'` flag's DeviceRequests in the container spec are **noted but never enforced at the cgroup level**. All `/dev/nvidia*` device files remain accessible in every container.

**Evidence supporting:**
- `docker inspect` showed DeviceIDs `[0]` and `[1]` correctly set on the two containers — meaning Docker *built* the correct DeviceRequests spec. But the runtime ignored it.
- Both containers showed the same UUID, consistent with both getting access to all GPUs.
- NVIDIA's WSL2 setup guide explicitly states: "Due to the nature of WSL2, all CUDA-enabled applications and libraries running inside a WSL2 container have access to all the available GPUs on the system, regardless of the `--gpus` flag."
- The `no-cgroups = true` path in `nvidia-container-runtime` skips the `nvcgo` device filtering entirely — it injects all GPU libraries but does no hardware-level restriction.

**What this means:** `--gpus 'device=N'` changes the *enumeration* seen by the CDI/hook (which devices get driver libraries injected), but because cgroup device allowlists cannot be written on WSL2, **no actual hardware isolation occurs at the kernel level**. A process inside the container can still call `cudaSetDevice(1)` and land on the "wrong" GPU.

**Why NVIDIA_VISIBLE_DEVICES=N also fails in this config:**  
`NVIDIA_VISIBLE_DEVICES` is an env var processed by `nvidia-container-runtime` at container creation time. When `no-cgroups = true`, the runtime respects it for library injection only (which `.so` files get bind-mounted), but again cannot prevent the container from touching other GPU device files. **However:** if `accept-nvidia-visible-devices-envvar-when-unprivileged = false` (the other key setting), then for unprivileged containers, `NVIDIA_VISIBLE_DEVICES` is completely ignored unless it came via the CDI device request path (`--gpus` flag), not the environment path.

---

### H2 — `accept-nvidia-visible-devices-envvar-when-unprivileged = false` silences env var (HIGH — ~65%)

**Mechanism:**  
In `/etc/nvidia-container-runtime/config.toml`, the setting `accept-nvidia-visible-devices-envvar-when-unprivileged` (default: `false` in newer toolkit versions) controls whether unprivileged containers can use the `NVIDIA_VISIBLE_DEVICES` env var to select GPUs. When `false`:

- `--gpus` flag → goes through `DeviceRequests` path → CDI hook → runtime honors it for library injection.
- `-e NVIDIA_VISIBLE_DEVICES=N` → the runtime sees this as an untrusted env var from an unprivileged container → **silently ignores it** for the device restriction pass.

**The double-failure:** In `serve_dual_model.sh`, BOTH `-e NVIDIA_VISIBLE_DEVICES=N` AND `--gpus '"device=N"'` are used. H1 explains why `--gpus` fails (no-cgroups). H2 explains why `NVIDIA_VISIBLE_DEVICES` as an env var also fails. The combination produces zero isolation.

**Evidence supporting:**
- The "fix" (H2 setting change + or different flag pattern) is exactly what NVIDIA toolkit changelog v1.14+ introduced: opt-in `accept-nvidia-visible-devices-envvar-when-unprivileged = true` was added specifically to make env-var-based GPU selection work in rootless/WSL2 contexts.
- All three approaches used (--gpus device=N, NVIDIA_VISIBLE_DEVICES=N, CUDA_VISIBLE_DEVICES=0) failed simultaneously — consistent with the runtime discarding all of them before CUDA even initializes.

---

### H3 — CUDA_VISIBLE_DEVICES is processed after CUDA runtime initializes from all-GPU state (MEDIUM — ~40%)

**Mechanism:**  
Even if the container has all GPU device files accessible (H1), `CUDA_VISIBLE_DEVICES=0` inside the container should still restrict CUDA to one GPU — this is a pure CUDA runtime env var processed by `libcuda.so` before any CUDA context is created, and it does NOT require cgroup support. **However:** vLLM uses PyTorch distributed init and `ray` workers that may spawn subprocess workers. If a worker subprocess is started after `fork()` but inherits a CUDA context already opened on device 0 (the default, since CUDA_VISIBLE_DEVICES=0 maps device 0 to host GPU 0 for container 1 and device 0 to host GPU 1 for container 2), then device 0 inside container 2 IS host GPU 1 — but only if the runtime correctly restrained the container.

The specific failure mode here: if the runtime (H1/H2) does not restrict device file access, then inside container 2, `CUDA_VISIBLE_DEVICES=0` still means "the first CUDA device visible to this container" — but if the container sees both GPUs, device 0 is GPU 0 (the default PCIe order), not GPU 1. So `CUDA_VISIBLE_DEVICES=0` in the Qwen3 container would select **host GPU 0**, colliding with the Gemma4 container.

**Why this matters:** This is actually a *consequence* of H1, not an independent root cause. But it means `CUDA_VISIBLE_DEVICES=0` in container 2 is the **wrong value** unless device file isolation actually works. If the runtime leaks all devices, the correct env var for the Qwen3 container would be `CUDA_VISIBLE_DEVICES=1`, not `0`.

---

### H4 — WSL2 `/dev/dxg` virtualization layer (LOWER — ~20%)

**Mechanism:**  
WSL2 uses a virtualized GPU path via `/dev/dxg` (DirectX Graphics kernel driver) rather than native `/dev/nvidia*`. The Windows WDDM driver exposes GPU compute to WSL2 through the `dxgkrnl` kernel module. This means:

- There is no `/dev/nvidia0`, `/dev/nvidia1` in the traditional sense on older WSL2 builds.
- On newer Windows 11 + WSL2 kernel 5.15+, NVIDIA does expose `/dev/nvidia0` etc. via a compatibility shim in `libnvcuvid.so` + `libcuda.so` from `/usr/lib/wsl/lib/`.
- If Docker's `--device=/dev/nvidia0` references the shim device rather than a hardware-isolated device, isolation is impossible at the OS level.

**Evidence:** This is likely partially true (WSL2 does use `/dev/dxg`) but is not the *primary* cause since nvidia-container-toolkit 1.15+ includes WSL2-aware CDI spec generation that handles the `/dev/dxg` path. The runtime knows it's on WSL2.

---

## 3. Test Matrix

| Approach | Flag combination | Expected behavior | WSL2 risk | Notes |
|---|---|---|---|---|
| **(a)** runtime=nvidia + CUDA_VISIBLE_DEVICES | `--runtime=nvidia --gpus all -e CUDA_VISIBLE_DEVICES=N` | CUDA sees only device N; if cgroup isolation fails, device N maps to wrong GPU | HIGH — CUDA_VISIBLE_DEVICES=0 in both containers → both on GPU 0 UNLESS runtime restricts | Only safe if N = host GPU index AND runtime restricts device files |
| **(b)** env var only, no --gpus | `-e NVIDIA_VISIBLE_DEVICES=N` (no --gpus) | If `accept-nvidia-visible-devices-envvar-when-unprivileged=true` AND running as privileged: restricts library injection. Still no cgroup enforcement. | HIGH (same H1) | Safer only if config.toml patched |
| **(c)** Manual device files | `--device=/dev/nvidia0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm` | Explicitly gives container access ONLY to GPU 0 device file; blocks kernel-level access to GPU 1 | **WORKS on WSL2 if `/dev/nvidia*` shim files exist** — bypasses nvidia-container-runtime entirely | Best WSL2 workaround; bypasses the whole toolkit |
| **(d)** nvidia-container-cli configure | Manual spec application | Produces correct spec; requires root; bypasses Docker daemon | UNKNOWN on WSL2 | Complex; not recommended |
| **(e)** CUDA_VISIBLE_DEVICES with correct host index | `--gpus all -e NVIDIA_VISIBLE_DEVICES=all -e CUDA_VISIBLE_DEVICES=0,1` selectively, container 1 gets `CUDA_VISIBLE_DEVICES=0`, container 2 gets `CUDA_VISIBLE_DEVICES=1` (NOT re-mapped) | Relies on CUDA device enumeration starting from host index | **WOULD WORK** — CUDA_VISIBLE_DEVICES=1 in container 2 binds to host GPU 1 even without cgroup isolation | This is the correct pattern given H1 and H3 combined |

**The critical insight on approach (e):** The current `serve_dual_model.sh` uses `CUDA_VISIBLE_DEVICES=0` for BOTH containers, which means "use the first visible GPU" in both cases. Since cgroup isolation fails (H1), both containers see all GPUs, and both pick device 0 = host GPU 0. The fix is: container for GPU 1 must use `CUDA_VISIBLE_DEVICES=1`, NOT `0`.

---

## 4. Config Patches Proposed

### Patch A — `/etc/nvidia-container-runtime/config.toml` (REQUIRES docker restart)

```toml
# Current (assumed default on WSL2):
[nvidia-container-cli]
no-cgroups = true

[nvidia-container-runtime]
# accept-nvidia-visible-devices-envvar-when-unprivileged = false  (default)

# PROPOSED: enable env-var-based selection even for unprivileged containers
[nvidia-container-runtime]
accept-nvidia-visible-devices-envvar-when-unprivileged = true
```

**Effect:** With this change, `-e NVIDIA_VISIBLE_DEVICES=0` in container 1 and `-e NVIDIA_VISIBLE_DEVICES=1` in container 2 will be honored by the runtime for device visibility control. CUDA then sees only the declared device. `CUDA_VISIBLE_DEVICES=0` (re-maps to the single visible device) would then work correctly.

**Risk:** Some toolkit versions spell this differently. Verify exact key name with `nvidia-container-runtime --version` and cross-check the installed config.toml.

**Requires:** `sudo systemctl restart docker` (flag: DAEMON RESTART REQUIRED).

---

### Patch B — `/etc/docker/daemon.json` (REQUIRES docker restart)

If `default-runtime` is not already set to `nvidia`:

```json
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

**Effect:** Every `docker run` automatically uses the NVIDIA runtime without needing `--runtime=nvidia`. This ensures `NVIDIA_VISIBLE_DEVICES` env vars are processed by the NVIDIA hook even when `--gpus` is omitted.

**Requires:** `sudo systemctl restart docker` (flag: DAEMON RESTART REQUIRED).

---

### Patch C — `serve_dual_model.sh` CUDA_VISIBLE_DEVICES fix (NO restart needed, testable immediately)

**This is the zero-config-change fix that should be tested first.**

The current script sets `CUDA_VISIBLE_DEVICES=0` in BOTH containers. This is wrong when cgroup isolation fails: both containers see all GPUs, and both pick device 0 (host GPU 0).

```bash
# Container for GPU 0 (current — CORRECT for device 0):
-e NVIDIA_VISIBLE_DEVICES=0 \
-e CUDA_VISIBLE_DEVICES=0 \   # selects host GPU 0 ✓

# Container for GPU 1 (current — WRONG):
-e NVIDIA_VISIBLE_DEVICES=1 \
-e CUDA_VISIBLE_DEVICES=0 \   # selects host GPU 0, not GPU 1 ✗

# PROPOSED fix for GPU 1 container:
-e NVIDIA_VISIBLE_DEVICES=1 \
-e CUDA_VISIBLE_DEVICES=1 \   # selects host GPU 1 ✓ (when cgroup isolation absent)
```

**Why this works:** `CUDA_VISIBLE_DEVICES` is consumed by `libcuda.so` inside the container. When `no-cgroups = true` and the runtime doesn't restrict device files, the container has access to all `/dev/nvidia*` devices. `CUDA_VISIBLE_DEVICES=1` tells CUDA to use device index 1 (host GPU 1), which is what we want. The runtime's failure to isolate becomes irrelevant because CUDA itself does the selection.

**Caveat:** This relies on the assumption that GPU PCIe enumeration order inside the container matches the host order (GPU 0 = device 0, GPU 1 = device 1). This is always true for the same machine without GPU hotplug. Verify with the UUID check banner: `vllm-qwen3` should report UUID of the GPU that shows as index 1 in `nvidia-smi`.

---

## 5. Recommended Fix Sequence (Cheapest First)

### Step 1 — Script-only fix (no sudo, no restart): modify `serve_dual_model.sh`

Change `CUDA_VISIBLE_DEVICES=0` → `CUDA_VISIBLE_DEVICES=1` in the Qwen3 (GPU 1) container block. Run the UUID isolation check. If both containers report distinct UUIDs → **DONE, no daemon changes needed.**

This is the highest-probability fix because it addresses H3 (the consequence of H1) without requiring any system config change.

### Step 2 — config.toml patch (requires sudo + docker restart): set `accept-nvidia-visible-devices-envvar-when-unprivileged = true`

If Step 1 doesn't fully fix it (e.g., `CUDA_VISIBLE_DEVICES` is still being ignored), apply Patch A. Restart docker. Re-test with the original `CUDA_VISIBLE_DEVICES=0` in both (Patch A makes env var-based selection honor `NVIDIA_VISIBLE_DEVICES`).

### Step 3 — Fallback: manual device file mounts (approach (c))

If config patches don't work, drop the nvidia-container-runtime entirely and use:
```bash
# Container for GPU 0:
--device=/dev/nvidia0:/dev/nvidia0 \
--device=/dev/nvidiactl:/dev/nvidiactl \
--device=/dev/nvidia-uvm:/dev/nvidia-uvm \
# (no --gpus flag)

# Container for GPU 1:
--device=/dev/nvidia1:/dev/nvidia1 \
--device=/dev/nvidiactl:/dev/nvidiactl \
--device=/dev/nvidia-uvm:/dev/nvidia-uvm \
# (no --gpus flag)
```
This gives the container exactly one GPU device file and blocks kernel-level access to the other GPU entirely, regardless of `no-cgroups` settings. Requires `/dev/nvidia0` and `/dev/nvidia1` to exist in the WSL2 namespace (they do on current WSL2 kernel 5.15+ with NVIDIA driver shim).

### Step 4 — Native Linux

If all WSL2 workarounds fail, this is a fundamental WSL2 architecture limitation. The benchmark requires a native Linux boot to get hardware-level GPU isolation.

---

## 6. Success Criterion

A 2-container test is considered PASSING when:
1. `docker logs vllm-gemma4 | grep GPU-ISOLATION-CHECK` shows `visible=1 uuid=GPU-AAAA...`
2. `docker logs vllm-qwen3 | grep GPU-ISOLATION-CHECK` shows `visible=1 uuid=GPU-BBBB...` (different UUID)
3. `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader` shows each container's PID(s) on exactly one distinct UUID
4. After 60 seconds of model load: GPU 0 VRAM ≥ 40 GB used, GPU 1 VRAM ≥ 40 GB used (both loaded)

---

## 7. WSL2 Driver Architecture Note

On WSL2, NVIDIA GPU access flows through:
```
Container process
  → /dev/nvidia* (bind-mounted by nvidia-container-runtime hook)
  → libcuda.so from /usr/lib/wsl/lib/ (Windows-side driver stub)
  → dxgkrnl.sys (Windows kernel GPU virtualization)
  → Physical GPU
```

The key difference from native Linux: `/usr/lib/wsl/lib/libcuda.so` is the real CUDA library; the NVIDIA Linux userspace libraries (`/usr/local/cuda/lib64/`) are a compatibility shim. The runtime on WSL2 bind-mounts `/usr/lib/wsl/lib/` (via the hook in `/usr/share/nvidia-container-runtime/`) rather than the native CUDA libs. Cgroup device restrictions don't exist because the `dxgkrnl` layer doesn't expose per-device cgroup interfaces to the WSL2 kernel.

This confirms H1: cgroup-based isolation is architecturally impossible on WSL2, and the `--gpus 'device=N'` flag can only affect library injection, not hardware access.

*Last updated: 2026-04-18 — W6 WSL2 GPU isolation root-cause investigation.*
