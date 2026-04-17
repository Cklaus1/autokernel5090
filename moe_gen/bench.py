"""MoE execution strategy benchmark.

Simulates Gemma4's MoE layer with real tensor shapes and measures
throughput across different execution strategies.

Usage:
    python3 moe_gen/bench.py
"""

import torch
import torch.nn.functional as F
import time
import sys
sys.path.insert(0, "/root/projects/autokernel")

from moe_gen.spec import MoESpec, GEMMA4_MOE_SPECS

WARMUP = 3
TRIALS = 20


def make_expert_weights(spec: MoESpec, device="cuda"):
    """Allocate expert weights matching the spec."""
    H, I, E = spec.hidden_size, spec.intermediate_size, spec.num_experts
    dtype = torch.float16  # runtime dtype (even for NVFP4 — dequant happens before matmul)

    if spec.has_gate_up_fused:
        # gate_up: [E, 2*I, H], down: [E, H, I]
        gate_up = torch.randn(E, 2 * I, H, device=device, dtype=dtype) * 0.01
        down = torch.randn(E, H, I, device=device, dtype=dtype) * 0.01
        return {"gate_up": gate_up, "down": down}
    else:
        gate = torch.randn(E, I, H, device=device, dtype=dtype) * 0.01
        up = torch.randn(E, I, H, device=device, dtype=dtype) * 0.01
        down = torch.randn(E, H, I, device=device, dtype=dtype) * 0.01
        return {"gate": gate, "up": up, "down": down}


def run_serial(hidden, weights, expert_ids, expert_weights, spec):
    """Baseline: serial expert execution (one expert at a time)."""
    B = hidden.shape[0]
    H, I = spec.hidden_size, spec.intermediate_size
    output = torch.zeros_like(hidden)

    for k in range(spec.top_k):
        eid = expert_ids[:, k]  # [B] — which expert for this slot
        ew = expert_weights[:, k:k+1]  # [B, 1] — routing weight

        # For each unique expert, batch its tokens
        unique_experts = eid.unique()
        for e in unique_experts:
            mask = (eid == e)
            if not mask.any():
                continue
            tokens = hidden[mask]  # [n, H]

            if spec.has_gate_up_fused:
                gate_up_out = tokens @ weights["gate_up"][e].T  # [n, 2*I]
                gate_out = gate_up_out[:, :I]
                up_out = gate_up_out[:, I:]
            else:
                gate_out = tokens @ weights["gate"][e].T  # [n, I]
                up_out = tokens @ weights["up"][e].T

            expert_out = F.gelu(gate_out) * up_out
            expert_out = expert_out @ weights["down"][e].T  # [n, H]
            output[mask] += ew[mask] * expert_out

    return output


def run_stream_parallel(hidden, weights, expert_ids, expert_weights, spec):
    """Multi-stream: partition experts across CUDA streams."""
    B = hidden.shape[0]
    H, I = spec.hidden_size, spec.intermediate_size
    num_streams = spec.num_streams
    output = torch.zeros_like(hidden)

    # Create streams
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    # Collect all (expert_id, mask, routing_weight) pairs
    work_items = []
    for k in range(spec.top_k):
        eid = expert_ids[:, k]
        ew = expert_weights[:, k:k+1]
        unique_experts = eid.unique()
        for e in unique_experts:
            mask = (eid == e)
            if mask.any():
                work_items.append((e.item(), mask, ew))

    # Partition work across streams (round-robin)
    stream_work = [[] for _ in range(num_streams)]
    for i, item in enumerate(work_items):
        stream_work[i % num_streams].append(item)

    # Record event on default stream so worker streams wait for hidden to be ready
    event = torch.cuda.current_stream().record_event()

    # Launch work on each stream
    partial_outputs = [torch.zeros_like(hidden) for _ in range(num_streams)]
    for s_idx, (stream, items) in enumerate(zip(streams, stream_work)):
        with torch.cuda.stream(stream):
            stream.wait_event(event)
            for e, mask, ew in items:
                tokens = hidden[mask]

                if spec.has_gate_up_fused:
                    gate_up_out = tokens @ weights["gate_up"][e].T
                    gate_out = gate_up_out[:, :I]
                    up_out = gate_up_out[:, I:]
                else:
                    gate_out = tokens @ weights["gate"][e].T
                    up_out = tokens @ weights["up"][e].T

                expert_out = F.gelu(gate_out) * up_out
                expert_out = expert_out @ weights["down"][e].T
                partial_outputs[s_idx][mask] += ew[mask] * expert_out

    # Sync all streams and accumulate
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for po in partial_outputs:
        output += po

    return output


def run_grouped_gemm(hidden, weights, expert_ids, expert_weights, spec):
    """Grouped GEMM: sort tokens by expert, batched matmul per expert."""
    B = hidden.shape[0]
    H, I = spec.hidden_size, spec.intermediate_size
    output = torch.zeros_like(hidden)

    # Flatten expert assignments: [B * top_k] pairs of (token_idx, expert_id)
    flat_token_idx = torch.arange(B, device=hidden.device).unsqueeze(1).expand(-1, spec.top_k).reshape(-1)
    flat_expert_id = expert_ids.reshape(-1)
    flat_weight = expert_weights.reshape(-1)

    # Sort by expert for coalesced access
    sort_idx = flat_expert_id.argsort()
    sorted_expert = flat_expert_id[sort_idx]
    sorted_token = flat_token_idx[sort_idx]
    sorted_weight = flat_weight[sort_idx]

    # Find boundaries between experts
    expert_boundaries = torch.where(
        torch.cat([torch.tensor([True], device=hidden.device),
                   sorted_expert[1:] != sorted_expert[:-1]])
    )[0]
    expert_boundaries = torch.cat([expert_boundaries,
                                   torch.tensor([len(sorted_expert)], device=hidden.device)])

    # Process each expert group
    for i in range(len(expert_boundaries) - 1):
        start, end = expert_boundaries[i].item(), expert_boundaries[i+1].item()
        if start >= end:
            continue

        e = sorted_expert[start].item()
        token_ids = sorted_token[start:end]
        w = sorted_weight[start:end].unsqueeze(1)

        tokens = hidden[token_ids]  # [n, H]

        if spec.has_gate_up_fused:
            gate_up_out = tokens @ weights["gate_up"][e].T
            gate_out = gate_up_out[:, :I]
            up_out = gate_up_out[:, I:]
        else:
            gate_out = tokens @ weights["gate"][e].T
            up_out = tokens @ weights["up"][e].T

        expert_out = F.gelu(gate_out) * up_out
        expert_out = expert_out @ weights["down"][e].T  # [n, H]

        # Scatter back with routing weights
        output.index_add_(0, token_ids, w * expert_out)

    return output


STRATEGIES = {
    "serial": run_serial,
    "grouped_gemm": run_grouped_gemm,
    "stream_parallel": run_stream_parallel,
}


def benchmark_spec(spec: MoESpec, batch_sizes, device="cuda"):
    """Benchmark one MoE spec across batch sizes."""
    weights = make_expert_weights(spec, device)
    results = []

    for B in batch_sizes:
        hidden = torch.randn(B, spec.hidden_size, device=device, dtype=torch.float16) * 0.01

        # Router
        router_w = torch.randn(spec.hidden_size, spec.num_experts, device=device, dtype=torch.float16) * 0.01
        logits = hidden @ router_w
        topk_vals, expert_ids = torch.topk(logits, spec.top_k, dim=-1)
        expert_weights = F.softmax(topk_vals.float(), dim=-1).half()

        strategy_fn = STRATEGIES.get(spec.strategy)
        if strategy_fn is None:
            results.append((B, None, None))
            continue

        try:
            # Warmup
            for _ in range(WARMUP):
                strategy_fn(hidden, weights, expert_ids, expert_weights, spec)
            torch.cuda.synchronize()

            # Bench
            t0 = time.perf_counter()
            for _ in range(TRIALS):
                strategy_fn(hidden, weights, expert_ids, expert_weights, spec)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0

            ms = (elapsed / TRIALS) * 1000
            # Tokens per second (each token goes through top_k experts)
            tps = B / (elapsed / TRIALS)
            # FLOPS utilization
            total_flops = B * spec.total_flops_per_token
            flops_achieved = total_flops / (elapsed / TRIALS)
            tflops = flops_achieved / 1e12

            results.append((B, tps, ms, tflops))

        except Exception as e:
            results.append((B, None, str(e)[:50]))

        del hidden
        torch.cuda.empty_cache()

    del weights
    torch.cuda.empty_cache()
    return results


def main():
    BATCH_SIZES = [1, 8, 32, 64, 128, 240]

    print("=" * 80)
    print("MoE Execution Strategy Benchmark — Gemma4 26B-A4B")
    print(f"128 experts, top-8, H={2816}, I={704}")
    print("=" * 80)

    all_results = {}
    for spec_name, spec in GEMMA4_MOE_SPECS.items():
        if spec.strategy not in STRATEGIES:
            continue
        print(f"\n--- {spec_name} (strategy={spec.strategy}, streams={spec.num_streams}) ---")
        results = benchmark_spec(spec, BATCH_SIZES)
        all_results[spec_name] = results
        for r in results:
            if len(r) == 4:
                B, tps, ms, tflops = r
                print(f"  B={B:>3}: {tps:>8.0f} tok/s  {ms:>7.1f}ms  {tflops:>5.1f} TFLOPS")
            else:
                B = r[0]
                print(f"  B={B:>3}: ERROR {r[-1] if len(r) > 2 else 'unknown'}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary: tok/s by strategy × batch")
    print("=" * 80)
    header = f"{'Strategy':<25}"
    for B in BATCH_SIZES:
        header += f" {'B='+str(B):>8}"
    print(header)
    print("-" * (25 + 9 * len(BATCH_SIZES)))

    for name, results in all_results.items():
        row = f"{name:<25}"
        for r in results:
            if len(r) == 4:
                row += f" {r[1]:>8.0f}"
            else:
                row += f" {'ERR':>8}"
        print(row)


if __name__ == "__main__":
    main()
