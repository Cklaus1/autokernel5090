#!/usr/bin/env python3
"""T2-N stack ablation harness.

Enumerates env gate combinations (additive sweep + leave-one-out),
launches vLLM containers with each combination, verifies banners, runs
bench, and reports per-gate throughput contribution.

Usage
-----
  # Full additive sweep (baseline → all-ON, one gate added at a time):
  python3 bench_t2n_ablation.py

  # All-ON then leave-one-out sweep only:
  python3 bench_t2n_ablation.py --mode leave_one_out

  # Both sweeps:
  python3 bench_t2n_ablation.py --mode both

  # Specific gate subset only (additive, order matters):
  python3 bench_t2n_ablation.py --gates T2N_CORE,FUSED_NORM_V2,RANK2_QKNORM_ROPE

  # Faster iteration (fewer concurrency points, fewer requests):
  python3 bench_t2n_ablation.py --concurrencies 512,1024 --max-tokens 64 --n-requests 128

Tag: W8_t2n_ablation_harness
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Env gate registry
# ---------------------------------------------------------------------------

# Ordered from lowest rank to highest.  Additive sweep adds in this order.
GATE_ORDER = [
    "T2N_CORE",
    "FUSED_NORM_V2",
    "RANK2_QKNORM_ROPE",
    "RANK3_UNSHUF_WSUM",
    "RANK4_KV_CACHE",
    "RANK5_POST_ATTN_ROUTER",
    "RANK6_SHARED_MLP",
    "PERSISTENT_BUF",
]

ENV_GATES: dict[str, str] = {
    "T2N_CORE":               "AUTOKERNEL_FUSED_SHUFFLE_QUANT",
    "FUSED_NORM_V2":          "AUTOKERNEL_FUSED_NORM_FP4_QWEN3",
    "RANK2_QKNORM_ROPE":      "AUTOKERNEL_FUSED_QKNORM_ROPE",
    "RANK3_UNSHUF_WSUM":      "AUTOKERNEL_FUSED_UNSHUFFLE_WEIGHTEDSUM",
    "RANK4_KV_CACHE":         "AUTOKERNEL_FUSED_KV_CACHE_UPDATE",
    "RANK5_POST_ATTN_ROUTER": "AUTOKERNEL_FUSED_POST_ATTN_ROUTER",
    "RANK6_SHARED_MLP":       "AUTOKERNEL_FUSED_SHARED_EXPERT_MLP",
    "PERSISTENT_BUF":         "AUTOKERNEL_FUSED_PERSISTENT_BUF",
}

# Expected banner substring in docker logs for each gate when it fires.
# A gate with no entry here is "banner-exempt" (no banner expected;
# contribution is still measured by throughput delta).
BANNERS: dict[str, str] = {
    "T2N_CORE":               "[T2-N] Patched run_cutlass_moe_fp4",
    "FUSED_NORM_V2":          "[fused_norm_fp4_qwen3] Patched Qwen3MoeDecoderLayer",
    "RANK2_QKNORM_ROPE":      "[fused_qknorm_rope] active",
    "RANK3_UNSHUF_WSUM":      "[T2N-UNSHUF-WSUM] active",
    "RANK4_KV_CACHE":         "[T2N-KVCACHE-ROPE] active",
    "RANK5_POST_ATTN_ROUTER": "[fused_post_attn_router] active",
    "RANK6_SHARED_MLP":       "[fused_shared_expert_mlp] active",
}

# Static env vars that must always be set inside every container, regardless
# of which gates are toggled.  Mirrors launch_qwen3_fused_norm_fp4.sh.
ALWAYS_ENV: dict[str, str] = {
    "PYTHONPATH":                      "/autokernel:/autokernel/patches",
    "AUTOKERNEL_FUSED_SHUFFLE_QUANT_SO": "/autokernel/workspace/fused_shuffle_quant_sm120a.so",
    "AUTOKERNEL_FUSED_NORM_FP4_SO":    "/autokernel/workspace/fused_rms_norm_fp4_cu13.so",
    "VLLM_USE_FLASHINFER_MOE_FP4":     "0",
    # MIN_TOKENS threshold — kept separate from ablation loop but set always.
    "AUTOKERNEL_FUSED_MIN_TOKENS":     "1",
}

# Plugin entry points registered for each gate (subset that need dist-info).
PLUGIN_ENTRY_POINTS: dict[str, tuple[str, str]] = {
    # gate_key -> (dist_name, "ep_name = module:register")
    "T2N_CORE":      ("fused_shuffle_quant_plugin-0.1", "fused_shuffle_quant = fused_shuffle_quant_plugin:register"),
    "FUSED_NORM_V2": ("fused_norm_fp4_qwen3-0.1",       "fused_norm_fp4_qwen3 = wire_fused_norm_fp4_qwen3_v2:register"),
}

CONTAINER_IMAGE = "vllm-fusencache:latest"
CONTAINER_PORT  = 8003
CONTAINER_NAME_PREFIX = "vllm-t2n-ablation"

VLLM_ARGS = [
    "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/models/Qwen3-30B-A3B-NVFP4",
    "--quantization", "modelopt",
    "--max-model-len", "4096",
    "--max-num-seqs", "512",
    "--trust-remote-code",
    "--port", str(CONTAINER_PORT),
    "--served-model-name", "Qwen3-30B-A3B-NVFP4",
    "--kv-cache-dtype", "fp8",
    "--gpu-memory-utilization", "0.92",
    "-cc.mode", "none",
    "-cc.cudagraph_mode", "full",
]

REPO_ROOT       = "/home/cklaus/projects/autokernel"
RESULT_FILE     = os.path.join(REPO_ROOT, "bench_t2n_ablation_results.json")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _run(cmd: list[str], check: bool = True, capture: bool = False, timeout: int = 60) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        timeout=timeout,
    )


def _make_dist_info_block(dist_name: str, ep_line: str) -> str:
    """Return a shell snippet that installs a minimal dist-info stub."""
    sname = dist_name.replace("-", "_").replace(".", "_")
    dir_var = f"_DIR_{sname}"
    return f"""
# --- dist-info: {dist_name} ---
{dir_var}=/usr/local/lib/python3.12/dist-packages/{dist_name}.dist-info
mkdir -p "${{{dir_var}}}"
cat > "${{{dir_var}}}/METADATA" <<_META
Metadata-Version: 2.1
Name: {dist_name}
Version: 0.1
_META
cat > "${{{dir_var}}}/entry_points.txt" <<_EP
[vllm.general_plugins]
{ep_line}
_EP
cat > "${{{dir_var}}}/RECORD" <<_REC
{dist_name}.dist-info/METADATA,,
{dist_name}.dist-info/entry_points.txt,,
{dist_name}.dist-info/RECORD,,
_REC
"""


def _build_inner_script(gate_values: dict[str, str]) -> str:
    """Build the bash -c inner script for a given gate combination."""
    lines: list[str] = ["set -e", "unset NAME"]

    # GPU isolation check
    lines.append(r"""python3 -c "
import torch
uuid = torch.cuda.get_device_properties(0).uuid
n = torch.cuda.device_count()
print(f'[GPU-ISOLATION-CHECK] visible={n} uuid={uuid}', flush=True)
" 2>&1 || true""")

    # Dist-info stubs — only for gates that are ON and have a plugin entry
    active_plugins: list[str] = []
    for gate_key, ep_value in gate_values.items():
        if ep_value == "1" and gate_key in PLUGIN_ENTRY_POINTS:
            dist_name, ep_line = PLUGIN_ENTRY_POINTS[gate_key]
            ep_name = ep_line.split("=")[0].strip()
            active_plugins.append(ep_name)
            lines.append(_make_dist_info_block(dist_name, ep_line))

    # Static env vars
    for k, v in ALWAYS_ENV.items():
        lines.append(f"export {k}={v}")

    # Gate-specific env vars
    for gate_key, ep_value in gate_values.items():
        env_var = ENV_GATES[gate_key]
        lines.append(f"export {env_var}={ep_value}")

    # VLLM_PLUGINS
    if active_plugins:
        lines.append(f"export VLLM_PLUGINS={','.join(active_plugins)}")
    else:
        lines.append("unset VLLM_PLUGINS")

    lines.append("exec python3 \"$@\"")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Container lifecycle
# ---------------------------------------------------------------------------

def launch_container(combo_id: str, gate_values: dict[str, str], port: int) -> str:
    """Launch a container with the given gate values.  Returns container name."""
    name = f"{CONTAINER_NAME_PREFIX}-{combo_id}"

    # Kill any existing container with same name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)

    inner_script = _build_inner_script(gate_values)

    docker_args = [
        "docker", "run", "-d",
        "--name", name,
        "--gpus", "device=1",
        "--shm-size=8g", "--ipc=host",
        "-e", "CUDA_VISIBLE_DEVICES=0",
        "-v", "/root/models:/models:ro",
        "-v", f"{REPO_ROOT}:/autokernel",
        "-p", f"{port}:{CONTAINER_PORT}",
        "--entrypoint", "bash",
        CONTAINER_IMAGE,
        "-c", inner_script, "bash",
    ] + VLLM_ARGS

    _log(f"Launching container {name} on port {port}")
    _run(docker_args, timeout=30)
    return name


def wait_health(port: int, name: str, timeout_s: int = 480) -> bool:
    """Poll /health until 200.  Returns True on success."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            result = subprocess.run(
                ["curl", "-sf", "-o", "/dev/null", "-w", "%{http_code}", url],
                capture_output=True, text=True, timeout=10,
            )
            code = result.stdout.strip()
            if code == "200":
                elapsed = int(time.time() - (deadline - timeout_s))
                _log(f"  /health OK after {elapsed}s ({attempt} polls) [{name}]")
                return True
        except Exception:
            pass
        time.sleep(5)
        if attempt % 12 == 0:
            log_lines = subprocess.run(
                ["docker", "logs", "--tail", "3", name],
                capture_output=True, text=True,
            ).stderr or ""
            _log(f"  still waiting ({attempt * 5}s)... last log: {log_lines.strip()!r}")
    _log(f"  TIMEOUT waiting for /health on port {port} [{name}]")
    return False


def check_banner(container_name: str, banner_text: str, timeout_s: int = 30) -> bool:
    """Grep docker logs for banner_text.  Returns True if found."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            ["docker", "logs", container_name],
            capture_output=True, text=True, timeout=15,
        )
        combined = result.stdout + result.stderr
        if banner_text in combined:
            return True
        time.sleep(3)
    return False


def verify_banners(container_name: str, gate_values: dict[str, str]) -> dict[str, bool]:
    """Verify all enabled gates that have expected banners.  Returns gate->found."""
    results: dict[str, bool] = {}
    for gate_key, ep_value in gate_values.items():
        if ep_value != "1":
            continue
        if gate_key not in BANNERS:
            continue
        banner = BANNERS[gate_key]
        found = check_banner(container_name, banner, timeout_s=45)
        results[gate_key] = found
        status = "OK" if found else "MISSING (silent-not-firing!)"
        _log(f"    banner [{gate_key}]: {status}")
    return results


def stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
    _log(f"  Removed container {name}")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------

def _get_model(api_base: str, timeout: int = 30) -> str:
    import urllib.request, urllib.error
    url = f"{api_base}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data["data"][0]["id"]
    except Exception as e:
        raise RuntimeError(f"Cannot get model from {url}: {e}")


def _call_one(api_base: str, model: str, prompt: str, max_tokens: int) -> tuple[int, int, float, Optional[str]]:
    import urllib.request, urllib.error
    import json as _json
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            f"{api_base}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            r = _json.loads(resp.read())
        elapsed = time.time() - t0
        if "error" in r:
            return (0, 0, elapsed, str(r["error"])[:120])
        u = r["usage"]
        return (u["prompt_tokens"], u["completion_tokens"], elapsed, None)
    except Exception as e:
        return (0, 0, time.time() - t0, str(e)[:120])


def run_bench(port: int, concurrencies: list[int], max_tokens: int, n_requests: int) -> dict:
    """Run concurrency sweep, return summary dict."""
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    PROMPT = (
        "You are a helpful assistant. Write a detailed 200-word summary of "
        "the history of the Roman Empire, focusing on key emperors, "
        "military campaigns, and cultural achievements from 27 BC to 476 AD."
    )
    api_base = f"http://localhost:{port}/v1"

    # Obtain model id
    model = _get_model(api_base)

    # Warmup
    _call_one(api_base, model, "Hello", 10)
    _log(f"    Warmup OK (model={model})")

    results = []
    _log(f"    {'C':>5} {'N':>6} {'wall_s':>8} {'gen_tok/s':>12} {'p50_s':>8} {'err':>4}")
    _log(f"    {'-'*52}")
    for c in concurrencies:
        n = max(c, n_requests)
        prompts = [PROMPT] * n
        t0 = time.time()
        total_p = total_g = errors = 0
        latencies: list[float] = []
        with ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(_call_one, api_base, model, p, max_tokens) for p in prompts]
            for f in _as_completed(futs):
                pt, gt, lat, err = f.result()
                total_p += pt
                total_g += gt
                latencies.append(lat)
                if err:
                    errors += 1
        wall = time.time() - t0
        latencies.sort()
        p50 = latencies[len(latencies) // 2] if latencies else 0
        r = {
            "concurrency": c,
            "n_requests": n,
            "max_tokens": max_tokens,
            "wall_s": round(wall, 2),
            "gen_tok_s": round(total_g / wall, 1) if wall > 0 else 0,
            "total_tok_s": round((total_p + total_g) / wall, 1) if wall > 0 else 0,
            "p50_latency_s": round(p50, 2),
            "errors": errors,
            "prompt_tokens": total_p,
            "gen_tokens": total_g,
        }
        results.append(r)
        _log(f"    {c:>5} {n:>6} {r['wall_s']:>8.1f} {r['gen_tok_s']:>12.1f} "
             f"{r['p50_latency_s']:>8.2f} {errors:>4}")

    peak = max(r["gen_tok_s"] for r in results) if results else 0
    return {"model": model, "results": results, "peak_gen_tok_s": peak}


# ---------------------------------------------------------------------------
# Single combo runner
# ---------------------------------------------------------------------------

def run_combo(
    combo_id: str,
    gate_values: dict[str, str],
    port: int,
    concurrencies: list[int],
    max_tokens: int,
    n_requests: int,
    skip_bench: bool = False,
) -> dict:
    """Full lifecycle: launch → health → banner-check → bench → teardown."""
    _log(f"=== Combo [{combo_id}] ===")
    active = [k for k, v in gate_values.items() if v == "1"]
    _log(f"  Active gates: {active or ['(none — baseline)']}")

    container_name = launch_container(combo_id, gate_values, port)

    try:
        healthy = wait_health(port, container_name, timeout_s=480)
        if not healthy:
            _log(f"  SKIP bench: container did not become healthy")
            return {
                "combo_id": combo_id,
                "gate_values": gate_values,
                "active_gates": active,
                "healthy": False,
                "banner_results": {},
                "bench": None,
                "error": "health timeout",
            }

        # Send one warmup request to trigger lazy banner emission
        # (some banners only fire on first forward pass)
        try:
            api_base = f"http://localhost:{port}/v1"
            model = _get_model(api_base, timeout=20)
            _call_one(api_base, model, "Hello", 5)
            time.sleep(2)  # Let log lines flush
        except Exception as e:
            _log(f"  warmup-for-banner failed: {e}")

        banner_results = verify_banners(container_name, gate_values)

        silent_not_firing = [k for k, ok in banner_results.items() if not ok]
        if silent_not_firing:
            _log(f"  WARNING: silent-not-firing gates: {silent_not_firing}")

        bench_result: Optional[dict] = None
        if not skip_bench:
            _log("  Running bench sweep...")
            try:
                bench_result = run_bench(port, concurrencies, max_tokens, n_requests)
            except Exception as e:
                _log(f"  bench FAILED: {e}")
                bench_result = {"error": str(e)}

        return {
            "combo_id": combo_id,
            "gate_values": gate_values,
            "active_gates": active,
            "healthy": True,
            "banner_results": banner_results,
            "silent_not_firing": silent_not_firing,
            "bench": bench_result,
        }
    finally:
        stop_container(container_name)


# ---------------------------------------------------------------------------
# Sweep strategies
# ---------------------------------------------------------------------------

def additive_sweep(
    gates: list[str],
    port: int,
    concurrencies: list[int],
    max_tokens: int,
    n_requests: int,
) -> list[dict]:
    """Start from all-OFF, add one gate at a time in GATE_ORDER."""
    combos: list[dict] = []

    # Baseline: all OFF
    base_values = {g: "0" for g in gates}
    _log("\n--- Baseline (all OFF) ---")
    result = run_combo("baseline", base_values, port, concurrencies, max_tokens, n_requests)
    combos.append(result)

    # Add gates incrementally
    current_values = dict(base_values)
    for i, gate in enumerate(gates):
        current_values = dict(current_values)
        current_values[gate] = "1"
        combo_id = f"add_{i+1:02d}_{gate}"
        _log(f"\n--- Additive step {i+1}: +{gate} ---")
        result = run_combo(combo_id, current_values, port, concurrencies, max_tokens, n_requests)
        combos.append(result)

    # All ON
    all_on_values = {g: "1" for g in gates}
    _log("\n--- All ON ---")
    result = run_combo("all_on", all_on_values, port, concurrencies, max_tokens, n_requests)
    combos.append(result)

    return combos


def leave_one_out_sweep(
    gates: list[str],
    port: int,
    concurrencies: list[int],
    max_tokens: int,
    n_requests: int,
) -> list[dict]:
    """Start from all-ON, disable one gate at a time."""
    combos: list[dict] = []

    # All ON first
    all_on_values = {g: "1" for g in gates}
    _log("\n--- All ON (leave-one-out baseline) ---")
    result = run_combo("loo_all_on", all_on_values, port, concurrencies, max_tokens, n_requests)
    combos.append(result)

    for i, gate in enumerate(gates):
        current_values = dict(all_on_values)
        current_values[gate] = "0"
        combo_id = f"loo_{i+1:02d}_minus_{gate}"
        _log(f"\n--- Leave-one-out step {i+1}: -{gate} ---")
        result = run_combo(combo_id, current_values, port, concurrencies, max_tokens, n_requests)
        combos.append(result)

    return combos


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _peak_tok_s(combo: dict) -> Optional[float]:
    bench = combo.get("bench")
    if bench is None:
        return None
    return bench.get("peak_gen_tok_s")


def compute_incremental_contributions(combos: list[dict]) -> list[dict]:
    """For an additive sweep, compute delta per gate vs previous combo."""
    contributions: list[dict] = []
    for i, combo in enumerate(combos):
        cur = _peak_tok_s(combo)
        if i == 0:
            prev = None
            delta = None
        else:
            prev = _peak_tok_s(combos[i - 1])
            delta = round(cur - prev, 1) if (cur is not None and prev is not None) else None

        pct = None
        if delta is not None and prev and prev > 0:
            pct = round(delta / prev * 100, 2)

        # Flag silent-not-firing gates that are newly enabled at this step
        snf = combo.get("silent_not_firing", [])
        active = combo.get("active_gates", [])
        prev_active = combos[i - 1].get("active_gates", []) if i > 0 else []
        newly_added = [g for g in active if g not in prev_active]

        contributions.append({
            "combo_id": combo["combo_id"],
            "newly_added_gate": newly_added[0] if len(newly_added) == 1 else (newly_added or None),
            "peak_gen_tok_s": cur,
            "delta_tok_s": delta,
            "delta_pct": pct,
            "silent_not_firing": snf,
            "banner_verified": all(combo.get("banner_results", {k: True for k in newly_added}).values()),
        })
    return contributions


def print_report(additive_contribs: list[dict], loo_contribs: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  T2-N ABLATION REPORT — ADDITIVE SWEEP")
    print("=" * 72)
    print(f"{'Combo':<30} {'Gate':<26} {'Peak tok/s':>12} {'Delta':>10} {'%':>7} {'Banner':>8}")
    print("-" * 72)
    for c in additive_contribs:
        gate = c["newly_added_gate"] or ""
        if isinstance(gate, list):
            gate = ",".join(gate)
        tok_s = f"{c['peak_gen_tok_s']:.1f}" if c["peak_gen_tok_s"] is not None else "N/A"
        delta = f"+{c['delta_tok_s']:.1f}" if (c["delta_tok_s"] is not None and c["delta_tok_s"] >= 0) \
                else (f"{c['delta_tok_s']:.1f}" if c["delta_tok_s"] is not None else "")
        pct = f"{c['delta_pct']:+.2f}%" if c["delta_pct"] is not None else ""
        snf = "MISSING" if c["silent_not_firing"] else ("OK" if gate else "-")
        print(f"  {c['combo_id']:<28} {gate:<26} {tok_s:>12} {delta:>10} {pct:>7} {snf:>8}")

    if loo_contribs:
        print("\n" + "=" * 72)
        print("  T2-N ABLATION REPORT — LEAVE-ONE-OUT SWEEP")
        print("=" * 72)
        print(f"{'Combo':<34} {'Removed Gate':<26} {'Peak tok/s':>12} {'Delta':>10} {'%':>7}")
        print("-" * 72)
        baseline_tok_s = _peak_tok_s({"bench": loo_contribs[0].get("bench") or {}})  # type: ignore
        loo_all_on = loo_contribs[0]["peak_gen_tok_s"] if loo_contribs else None
        for c in loo_contribs[1:]:
            gate = c["newly_added_gate"] or ""
            if isinstance(gate, list):
                gate = ",".join(gate)
            tok_s = f"{c['peak_gen_tok_s']:.1f}" if c["peak_gen_tok_s"] is not None else "N/A"
            delta_v = (c["peak_gen_tok_s"] - loo_all_on) if (c["peak_gen_tok_s"] is not None and loo_all_on) else None
            delta = f"{delta_v:.1f}" if delta_v is not None else ""
            pct = f"{delta_v / loo_all_on * 100:.2f}%" if (delta_v is not None and loo_all_on) else ""
            print(f"  {c['combo_id']:<34} {gate:<26} {tok_s:>12} {delta:>10} {pct:>7}")

    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["additive", "leave_one_out", "both"], default="additive",
                    help="Sweep strategy (default: additive)")
    ap.add_argument("--gates", default=",".join(GATE_ORDER),
                    help="Comma-separated ordered gate subset to sweep (default: all, in rank order)")
    ap.add_argument("--port", type=int, default=8003,
                    help="Host port to forward into the container (default: 8003)")
    ap.add_argument("--concurrencies", default="512,768,1024",
                    help="Comma-separated concurrency levels (default: 512,768,1024)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="max_tokens per request (default: 128)")
    ap.add_argument("--n-requests", type=int, default=256,
                    help="Min requests per concurrency level (default: 256)")
    ap.add_argument("--out", default=RESULT_FILE,
                    help=f"Output JSON path (default: {RESULT_FILE})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print inner scripts without launching containers")
    args = ap.parse_args()

    gates = [g.strip() for g in args.gates.split(",") if g.strip() in GATE_ORDER]
    if not gates:
        print("ERROR: no valid gates specified", file=sys.stderr)
        sys.exit(1)
    unknown = [g.strip() for g in args.gates.split(",") if g.strip() not in GATE_ORDER and g.strip()]
    if unknown:
        print(f"WARNING: unknown gates ignored: {unknown}", file=sys.stderr)

    concurrencies = [int(x) for x in args.concurrencies.split(",")]

    if args.dry_run:
        print("=== DRY RUN: inner scripts only ===\n")
        for i, gate in enumerate(gates):
            values = {g: ("1" if j <= i else "0") for j, g in enumerate(gates)}
            print(f"--- Combo add_{i+1:02d}_{gate} ---")
            print(_build_inner_script(values))
            print()
        return

    _log(f"T2-N Ablation Harness starting")
    _log(f"  mode={args.mode}  gates={gates}")
    _log(f"  concurrencies={concurrencies}  max_tokens={args.max_tokens}  n_requests={args.n_requests}")
    _log(f"  port={args.port}  out={args.out}")

    n_combos = len(gates) + 2  # baseline + N additive steps + all_on
    est_min = n_combos * 8
    _log(f"  Estimated runtime: ~{n_combos} combos × ~8 min = ~{est_min} min")

    additive_combos: list[dict] = []
    loo_combos: list[dict] = []

    if args.mode in ("additive", "both"):
        _log("\n[Phase 1] Additive sweep")
        additive_combos = additive_sweep(gates, args.port, concurrencies, args.max_tokens, args.n_requests)

    if args.mode in ("leave_one_out", "both"):
        _log("\n[Phase 2] Leave-one-out sweep")
        loo_combos = leave_one_out_sweep(gates, args.port, concurrencies, args.max_tokens, args.n_requests)

    additive_contribs = compute_incremental_contributions(additive_combos)
    loo_contribs = compute_incremental_contributions(loo_combos)

    print_report(additive_contribs, loo_contribs)

    output = {
        "harness_version": "W8_t2n_ablation_harness",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "mode": args.mode,
            "gates": gates,
            "concurrencies": concurrencies,
            "max_tokens": args.max_tokens,
            "n_requests": args.n_requests,
        },
        "additive_sweep": additive_combos,
        "leave_one_out_sweep": loo_combos,
        "additive_contributions": additive_contribs,
        "leave_one_out_contributions": loo_contribs,
    }

    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    _log(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
