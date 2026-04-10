#!/usr/bin/env python3
"""FusenAI Adaptive Server Monitor.

Monitors a running vLLM server via its Prometheus /metrics endpoint,
tracks rolling-window statistics, issues recommendations based on observed
load, and renders a live terminal dashboard.

Usage:
    python tools/adaptive_server.py                          # defaults
    python tools/adaptive_server.py --url http://host:8000   # custom URL
    python tools/adaptive_server.py --interval 5             # poll every 5s
    python tools/adaptive_server.py --log metrics.jsonl      # persist metrics
    python tools/adaptive_server.py --json                   # single JSON snapshot
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Metric snapshot
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    timestamp: float
    num_running: int = 0
    num_waiting: int = 0
    num_swapped: int = 0
    gpu_cache_usage: float = 0.0
    cpu_cache_usage: float = 0.0
    gen_throughput: float = 0.0
    prompt_throughput: float = 0.0
    # Extras that vLLM may expose
    num_preemptions: int = 0
    best_of_total: int = 0
    avg_seq_group_len: float = 0.0

    @property
    def total_active(self) -> int:
        return self.num_running + self.num_waiting


# ---------------------------------------------------------------------------
# Prometheus text-format parser (avoids external dependency)
# ---------------------------------------------------------------------------

def parse_prometheus_text(text: str) -> Dict[str, float]:
    """Parse Prometheus text exposition format into {metric_name: value}."""
    metrics: Dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Lines look like: metric_name{labels} value  or  metric_name value
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        # Strip labels from name for simple lookup
        if "{" in name:
            name = name[: name.index("{")]
        try:
            metrics[name] = float(parts[-1])
        except ValueError:
            continue
    return metrics


def fetch_metrics(base_url: str, timeout: float = 5.0) -> Optional[MetricSnapshot]:
    """Fetch /metrics from vLLM and return a MetricSnapshot."""
    url = base_url.rstrip("/") + "/metrics"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None

    m = parse_prometheus_text(text)
    return MetricSnapshot(
        timestamp=time.time(),
        num_running=int(m.get("vllm:num_requests_running", 0)),
        num_waiting=int(m.get("vllm:num_requests_waiting", 0)),
        num_swapped=int(m.get("vllm:num_requests_swapped", 0)),
        gpu_cache_usage=m.get("vllm:gpu_cache_usage_perc", 0.0),
        cpu_cache_usage=m.get("vllm:cpu_cache_usage_perc", 0.0),
        gen_throughput=m.get("vllm:avg_generation_throughput_toks_per_s", 0.0),
        prompt_throughput=m.get("vllm:avg_prompt_throughput_toks_per_s", 0.0),
        num_preemptions=int(m.get("vllm:num_preemptions_total", 0)),
    )


# ---------------------------------------------------------------------------
# Rolling window statistics
# ---------------------------------------------------------------------------

@dataclass
class RollingStats:
    """Maintains rolling-window stats for 1m, 5m, 15m."""
    history: Deque[MetricSnapshot] = field(default_factory=collections.deque)
    max_history_sec: float = 900.0  # 15 minutes

    def add(self, snap: MetricSnapshot) -> None:
        self.history.append(snap)
        cutoff = time.time() - self.max_history_sec
        while self.history and self.history[0].timestamp < cutoff:
            self.history.popleft()

    def _window(self, seconds: float) -> List[MetricSnapshot]:
        cutoff = time.time() - seconds
        return [s for s in self.history if s.timestamp >= cutoff]

    def avg_throughput(self, seconds: float) -> float:
        w = self._window(seconds)
        if not w:
            return 0.0
        return sum(s.gen_throughput for s in w) / len(w)

    def avg_gpu_cache(self, seconds: float) -> float:
        w = self._window(seconds)
        if not w:
            return 0.0
        return sum(s.gpu_cache_usage for s in w) / len(w)

    def avg_running(self, seconds: float) -> float:
        w = self._window(seconds)
        if not w:
            return 0.0
        return sum(s.num_running for s in w) / len(w)

    def avg_waiting(self, seconds: float) -> float:
        w = self._window(seconds)
        if not w:
            return 0.0
        return sum(s.num_waiting for s in w) / len(w)

    def peak_throughput(self, seconds: float) -> float:
        w = self._window(seconds)
        if not w:
            return 0.0
        return max(s.gen_throughput for s in w)

    def throughput_trend(self, seconds: float = 60.0) -> str:
        """Return arrow indicating trend over the window."""
        w = self._window(seconds)
        if len(w) < 3:
            return "-"
        first_half = w[: len(w) // 2]
        second_half = w[len(w) // 2 :]
        avg1 = sum(s.gen_throughput for s in first_half) / len(first_half)
        avg2 = sum(s.gen_throughput for s in second_half) / len(second_half)
        if avg1 == 0:
            return "-"
        pct = (avg2 - avg1) / max(avg1, 1e-6) * 100
        if pct > 5:
            return "^ trending up"
        elif pct < -5:
            return "v trending down"
        else:
            return "~ stable"

    def waiting_duration(self) -> float:
        """Seconds that num_waiting > 0 continuously (most recent streak)."""
        dur = 0.0
        for snap in reversed(list(self.history)):
            if snap.num_waiting > 0:
                dur = time.time() - snap.timestamp
            else:
                break
        return dur


# ---------------------------------------------------------------------------
# Adaptive policy engine
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    level: str   # "info", "warn", "alert", "action"
    message: str
    detail: str = ""


class AdaptivePolicy:
    """Evaluate metrics and produce recommendations."""

    def __init__(self, baseline_throughput: float = 0.0):
        self.baseline_throughput = baseline_throughput
        self._baseline_set = baseline_throughput > 0
        self._high_cache_since: Optional[float] = None

    def evaluate(self, snap: MetricSnapshot, stats: RollingStats) -> List[Decision]:
        decisions: List[Decision] = []

        # Auto-set baseline from peak observed throughput
        peak_15m = stats.peak_throughput(900)
        if peak_15m > self.baseline_throughput:
            self.baseline_throughput = peak_15m
            self._baseline_set = True

        # --- KV cache pressure ---
        if snap.gpu_cache_usage > 0.90:
            decisions.append(Decision(
                "alert",
                "KV cache critically high (>90%)",
                f"GPU cache at {snap.gpu_cache_usage:.0%}. "
                "Enable FusenCache k4v4b64 for 4x capacity, or reduce max_num_seqs.",
            ))
        elif snap.gpu_cache_usage > 0.80:
            decisions.append(Decision(
                "warn",
                "KV cache pressure high (>80%)",
                f"GPU cache at {snap.gpu_cache_usage:.0%}. "
                "Consider enabling KV compression.",
            ))

        # --- Throughput degradation ---
        if self._baseline_set and self.baseline_throughput > 0:
            avg_1m = stats.avg_throughput(60)
            if avg_1m > 0 and avg_1m < self.baseline_throughput * 0.7:
                decisions.append(Decision(
                    "alert",
                    f"Throughput degraded >30% vs peak",
                    f"Current 1m avg: {avg_1m:,.0f} tok/s, peak: {self.baseline_throughput:,.0f} tok/s.",
                ))
            elif avg_1m > 0 and avg_1m < self.baseline_throughput * 0.8:
                decisions.append(Decision(
                    "warn",
                    f"Throughput degraded >20% vs peak",
                    f"Current 1m avg: {avg_1m:,.0f} tok/s, peak: {self.baseline_throughput:,.0f} tok/s.",
                ))

        # --- Request queueing ---
        waiting_dur = stats.waiting_duration()
        if snap.num_waiting > 0 and waiting_dur > 30:
            decisions.append(Decision(
                "alert",
                f"Requests queueing for {waiting_dur:.0f}s",
                f"{snap.num_waiting} waiting requests. Consider scaling or increasing max_num_seqs.",
            ))
        elif snap.num_waiting > 0 and waiting_dur > 10:
            decisions.append(Decision(
                "warn",
                f"Requests queueing for {waiting_dur:.0f}s",
                f"{snap.num_waiting} waiting requests.",
            ))

        # --- Config recommendations ---
        concurrency = snap.num_running
        if concurrency <= 4 and snap.gen_throughput > 0:
            decisions.append(Decision(
                "info",
                "Low concurrency detected (<=4)",
                "Optimal config: inductor + CUDA graphs for best per-request latency.",
            ))
        elif concurrency > 32:
            decisions.append(Decision(
                "info",
                f"High concurrency ({concurrency} running)",
                "Optimal config: no inductor + CUDA graphs for peak aggregate throughput.",
            ))

        # --- Preemption check ---
        if snap.num_preemptions > 0:
            decisions.append(Decision(
                "warn",
                f"Preemptions detected ({snap.num_preemptions} total)",
                "Requests being evicted from KV cache. Consider larger gpu_memory_utilization or fewer max_num_seqs.",
            ))

        # --- System healthy ---
        if not decisions:
            decisions.append(Decision("info", "System healthy", "No issues detected."))

        return decisions


# ---------------------------------------------------------------------------
# Terminal dashboard
# ---------------------------------------------------------------------------

def format_number(n: float) -> str:
    if n >= 10000:
        return f"{n:,.0f}"
    elif n >= 100:
        return f"{n:,.0f}"
    elif n >= 1:
        return f"{n:,.1f}"
    else:
        return f"{n:.3f}"


def render_dashboard(
    snap: MetricSnapshot,
    stats: RollingStats,
    decisions: List[Decision],
    config: dict,
    clear: bool = True,
) -> str:
    """Render terminal dashboard as a string."""
    W = 60
    lines: List[str] = []

    if clear:
        lines.append("\033[2J\033[H")  # clear screen, cursor to top

    sep_double = "=" * W
    sep_single = "-" * W

    # Header
    lines.append(sep_double)
    lines.append("  FusenAI Adaptive Server Monitor".center(W))
    lines.append(sep_double)

    model = config.get("model", "unknown")
    gpu = config.get("gpu", "unknown")
    backend = config.get("backend", "unknown")
    kv_mode = config.get("kv_mode", "BF16")
    lines.append(f"  Model: {model:<28s} GPU: {gpu}")
    lines.append(f"  Config: {backend:<27s} KV: {kv_mode}")
    lines.append(sep_single)

    # Throughput
    avg_1m = stats.avg_throughput(60)
    trend = stats.throughput_trend(60)
    lines.append(f"  Throughput:  {format_number(avg_1m):>8s} tok/s (1min avg)  {trend}")

    avg_5m = stats.avg_throughput(300)
    avg_15m = stats.avg_throughput(900)
    lines.append(f"  Rolling:     {format_number(avg_5m):>8s} (5m)  {format_number(avg_15m):>8s} (15m)")

    lines.append(sep_single)

    # Concurrency / KV
    lines.append(
        f"  Active: {snap.num_running} reqs  "
        f"Waiting: {snap.num_waiting}  "
        f"Swapped: {snap.num_swapped}  "
        f"KV: {snap.gpu_cache_usage:.0%}"
    )

    # KV detail
    avg_cache_1m = stats.avg_gpu_cache(60)
    lines.append(f"  KV avg (1m): {avg_cache_1m:.1%}   Preemptions: {snap.num_preemptions}")

    lines.append(sep_single)

    # Recommendations
    alerts = [d for d in decisions if d.level == "alert"]
    warns = [d for d in decisions if d.level == "warn"]
    infos = [d for d in decisions if d.level == "info"]

    if alerts:
        lines.append("  [ALERT]")
        for d in alerts:
            lines.append(f"    ! {d.message}")
            if d.detail:
                lines.append(f"      {d.detail}")
    if warns:
        lines.append("  [WARN]")
        for d in warns:
            lines.append(f"    * {d.message}")
            if d.detail:
                lines.append(f"      {d.detail}")
    if not alerts and not warns:
        lines.append("  Recommendations: none (system healthy)")

    lines.append(sep_double)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"  Last updated: {ts}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_snapshot(path: str, snap: MetricSnapshot, decisions: List[Decision]) -> None:
    """Append a JSONL line with snapshot + decisions."""
    record = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "running": snap.num_running,
        "waiting": snap.num_waiting,
        "swapped": snap.num_swapped,
        "gpu_cache": round(snap.gpu_cache_usage, 4),
        "gen_tps": round(snap.gen_throughput, 1),
        "prompt_tps": round(snap.prompt_throughput, 1),
        "preemptions": snap.num_preemptions,
        "decisions": [
            {"level": d.level, "msg": d.message} for d in decisions if d.level != "info"
        ],
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def detect_config(base_url: str) -> dict:
    """Try to detect model/config from vLLM /v1/models endpoint."""
    config = {
        "model": "unknown",
        "gpu": os.environ.get("GPU_NAME", "unknown"),
        "backend": "unknown",
        "kv_mode": "BF16",
    }
    try:
        url = base_url.rstrip("/") + "/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models:
                model_id = models[0].get("id", "unknown")
                # Shorten long paths
                if "/" in model_id:
                    model_id = model_id.split("/")[-1]
                config["model"] = model_id
    except Exception:
        pass
    return config


def run_monitor(
    base_url: str,
    interval: float,
    log_path: Optional[str],
    json_mode: bool,
    baseline_throughput: float,
) -> None:
    stats = RollingStats()
    policy = AdaptivePolicy(baseline_throughput=baseline_throughput)
    config = detect_config(base_url)

    # Graceful shutdown
    stop = False

    def _handle_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if json_mode:
        # Single snapshot mode
        snap = fetch_metrics(base_url)
        if snap is None:
            print(json.dumps({"error": "Cannot reach vLLM at " + base_url}))
            sys.exit(1)
        stats.add(snap)
        decisions = policy.evaluate(snap, stats)
        out = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "running": snap.num_running,
            "waiting": snap.num_waiting,
            "gpu_cache": snap.gpu_cache_usage,
            "gen_throughput": snap.gen_throughput,
            "prompt_throughput": snap.prompt_throughput,
            "decisions": [{"level": d.level, "msg": d.message, "detail": d.detail} for d in decisions],
        }
        print(json.dumps(out, indent=2))
        return

    print(f"Connecting to vLLM at {base_url} (interval={interval}s)...")
    failures = 0
    while not stop:
        snap = fetch_metrics(base_url)
        if snap is None:
            failures += 1
            if failures <= 3:
                print(f"[{datetime.datetime.now():%H:%M:%S}] Cannot reach vLLM (attempt {failures})...")
            elif failures == 4:
                print("Suppressing further connection errors. Retrying silently...")
            time.sleep(interval)
            continue

        if failures > 0:
            print(f"[{datetime.datetime.now():%H:%M:%S}] Reconnected to vLLM.")
            failures = 0
            config = detect_config(base_url)

        stats.add(snap)
        decisions = policy.evaluate(snap, stats)

        dashboard = render_dashboard(snap, stats, decisions, config)
        print(dashboard)

        if log_path:
            log_snapshot(log_path, snap, decisions)

        time.sleep(interval)

    print("\nMonitor stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FusenAI Adaptive Server Monitor for vLLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s                                  # monitor localhost:8000
  %(prog)s --url http://gpu-server:8000     # remote server
  %(prog)s --interval 5 --log metrics.jsonl # 5s poll, persist to file
  %(prog)s --json                           # single JSON snapshot, exit
""",
    )
    parser.add_argument(
        "--url", default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--interval", type=float, default=10.0,
        help="Polling interval in seconds (default: 10)",
    )
    parser.add_argument(
        "--log", dest="log_path", default=None,
        help="Path to JSONL log file for persistent metrics",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Fetch a single snapshot, output as JSON, and exit",
    )
    parser.add_argument(
        "--baseline-throughput", type=float, default=0.0,
        help="Known baseline throughput (tok/s) for regression detection",
    )
    args = parser.parse_args()

    run_monitor(
        base_url=args.url,
        interval=args.interval,
        log_path=args.log_path,
        json_mode=args.json,
        baseline_throughput=args.baseline_throughput,
    )


if __name__ == "__main__":
    main()
