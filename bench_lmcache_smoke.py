#!/usr/bin/env python3
"""
LMCache KV Connector smoke bench — W1_4e_lmcache_smoke

Workload: 20 concurrent users, each sends:
  - identical 500-token system prompt (shared prefix)
  - distinct 50-token user query
  - 32-token generation

Measures P50/P95/P99 TTFT via streaming (first chunk arrival time).
"""

import argparse
import asyncio
import json
import statistics
import time
import aiohttp

PORT = 8400
BASE_URL = f"http://localhost:{PORT}"
MODEL = "Qwen/Qwen3-8B"
CONCURRENCY = 20
NUM_REQUESTS = 20
MAX_TOKENS = 32

# ~500-token system prompt (approx 2000 chars ≈ 500 tokens at 4 chars/token)
SYSTEM_PROMPT = (
    "You are an expert AI assistant embedded in a multi-agent pipeline called FusenSolver. "
    "Your role is to analyze complex technical problems, decompose them into sub-tasks, "
    "and coordinate with specialist sub-agents to produce high-quality solutions. "
    "You have deep expertise in machine learning, GPU programming, distributed systems, "
    "and software engineering. When given a task, you first reason about the problem domain, "
    "identify key constraints and requirements, then produce a structured plan with clear "
    "action items. You always cite your reasoning and provide working code examples when relevant. "
    "You track dependencies between sub-tasks and ensure that all agents receive the correct "
    "context. You communicate in clear, precise technical language. You never fabricate facts. "
    "If you are uncertain, you say so and propose a verification step. "
    "You understand that latency matters: your responses must be concise and actionable. "
    "You have access to tool_call primitives: search, execute_code, invoke_agent, write_file, "
    "read_file, spawn_subagent. When using tools, emit a JSON block with tool_name and args. "
    "You operate within a 4096-token context window and manage it carefully. "
    "Agent persona: FusenSolver-Coordinator v2.1. Session ID: bench-smoke-prefix-test. "
    "Priority: latency-sensitive. SLA: P99 TTFT < 200ms. Output schema: JSON with fields "
    "'reasoning', 'plan', 'tool_calls', 'response'. Always validate JSON before emitting. "
    "System constraints: GPU memory budget 20GB, CPU RAM budget 64GB, max sub-agents 8. "
    "You are running on a single-node WSL2 environment with an NVIDIA PRO 6000 GPU. "
    "LMCache KV connector is active: shared prefix tokens will be restored from CPU RAM "
    "on subsequent requests, bypassing re-computation. Chunk size: 256 tokens. "
    "This message is exactly the shared prefix used across all 20 concurrent requests "
    "in this smoke bench to measure LMCache hit rate and TTFT reduction. End of system prompt."
)

# 20 distinct user queries (~50 tokens each)
USER_QUERIES = [
    "Analyze the following GPU kernel bottleneck and propose three optimization strategies "
    "targeting memory bandwidth, compute utilization, and register pressure respectively.",
    "Given a transformer model with 8B parameters running on a single GPU, estimate the "
    "optimal batch size for throughput maximization and explain your reasoning step by step.",
    "Describe how LMCache's local CPU backend handles KV cache eviction under memory pressure "
    "and what guarantees it provides about cache coherency across concurrent requests.",
    "Propose a fused CUDA kernel that combines RMSNorm and FP4 quantization into a single "
    "pass, minimizing global memory reads and writes for a hidden dimension of 4096.",
    "Explain how the vLLM paged attention mechanism allocates and reclaims KV cache blocks "
    "under mixed prefill and decode workloads with dynamic sequence lengths.",
    "Design a monitoring strategy for tracking LMCache hit rates, TTFT p99, and GPU memory "
    "utilization in a production multi-agent serving environment with 100 concurrent users.",
    "Compare chunked prefill vs continuous batching for a workload with 20 users sharing a "
    "500-token system prompt followed by distinct 50-token queries and 32-token responses.",
    "What are the trade-offs between using FP8 KV cache compression and CPU offloading via "
    "LMCache for a model with 32 layers, 8 heads, and head dimension 128?",
    "Describe the rolling hash algorithm used by LMCache to identify cacheable prefix tokens "
    "and explain how it handles hash collisions in high-throughput serving scenarios.",
    "Propose a benchmark methodology for measuring the effective prefill reduction ratio "
    "of LMCache across different system prompt lengths from 128 to 4096 tokens.",
    "Explain how CUDA pinned memory allocation works and why it is required for efficient "
    "host-to-device transfer in LMCache's cudaMemcpyAsync-based restore path.",
    "Given a 20-user concurrent workload where 19 of 20 requests share an identical system "
    "prompt, calculate the theoretical speedup from LMCache versus vLLM's built-in prefix cache.",
    "Design a fallback strategy for LMCache when the CPU RAM pool is exhausted and new "
    "prefix tokens cannot be cached, ensuring graceful degradation without request failures.",
    "Analyze the interaction between vLLM's token budget enforcement and LMCache's chunked "
    "KV restore: what happens when a restored prefix exceeds the available KV cache slots?",
    "Describe how to configure LMCache for a multi-replica vLLM deployment where all "
    "replicas share a common CPU-backed KV store via the standalone ZMQ process mode.",
    "Propose metrics and alerting thresholds for detecting LMCache misconfiguration in "
    "production, including symptoms of P2P accidentally enabled on a WSL2 host.",
    "Explain the impact of chunk_size parameter on LMCache hit rate for workloads with "
    "system prompts of exactly 500 tokens: compare chunk_size=128 vs 256 vs 512.",
    "Design an A/B testing framework for comparing baseline vLLM prefix cache versus "
    "LMCache augmentation on a shared-prefix multi-agent serving workload.",
    "What is the expected memory footprint of caching KV tensors for a 500-token prefix "
    "with Qwen3-8B (28 layers, 4 heads GQA, head_dim=128, BF16) in LMCache CPU pool?",
    "Summarize the key architectural differences between LMCache's local CPU backend, "
    "remote Redis backend, and NIXL backend, and recommend one for a single-node WSL2 setup.",
]


async def send_request_streaming(session, url, query_idx, label=""):
    """Send one streaming request and return TTFT (time to first token)."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_QUERIES[query_idx % len(USER_QUERIES)]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": True,
    }
    t_send = time.perf_counter()
    ttft = None
    total_tokens = 0
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120, connect=10),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "idx": query_idx, "status": "ERROR", "code": resp.status,
                    "error": body[:200], "ttft": None, "elapsed": time.perf_counter() - t_send,
                }
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content and ttft is None:
                    ttft = time.perf_counter() - t_send
                if content:
                    total_tokens += 1
        elapsed = time.perf_counter() - t_send
        return {
            "idx": query_idx, "status": "OK", "ttft": ttft,
            "elapsed": elapsed, "tokens_generated": total_tokens,
        }
    except asyncio.TimeoutError:
        return {"idx": query_idx, "status": "TIMEOUT", "ttft": None,
                "elapsed": time.perf_counter() - t_send}
    except Exception as e:
        return {"idx": query_idx, "status": "CRASH", "ttft": None,
                "elapsed": time.perf_counter() - t_send, "error": str(e)[:200]}


async def run_bench(url, label, num_requests=NUM_REQUESTS, concurrency=CONCURRENCY):
    print(f"\n{'='*70}")
    print(f"  RUN: {label}  |  C={concurrency}  |  N={num_requests}")
    print(f"  Prompt: ~500-tok shared prefix + ~50-tok unique query + {MAX_TOKENS}-tok gen")
    print(f"{'='*70}")

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def bounded(idx):
            async with sem:
                return await send_request_streaming(session, url, idx, label)

        t_wall_start = time.perf_counter()
        tasks = [bounded(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - t_wall_start

    ok = [r for r in results if r["status"] == "OK" and r["ttft"] is not None]
    errors = [r for r in results if r["status"] != "OK"]

    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:3]:
            print(f"    {e}")

    if not ok:
        print("  NO SUCCESSFUL REQUESTS — cannot compute TTFT stats")
        return None

    ttfts = sorted(r["ttft"] for r in ok)
    p50 = statistics.median(ttfts)
    p95 = ttfts[int(len(ttfts) * 0.95)]
    p99 = ttfts[int(len(ttfts) * 0.99)] if len(ttfts) >= 100 else ttfts[-1]

    total_gen_tokens = sum(r.get("tokens_generated", 0) for r in ok)
    agg_throughput = total_gen_tokens / wall_time if wall_time > 0 else 0

    print(f"  Successful: {len(ok)}/{num_requests}")
    print(f"  TTFT  P50: {p50*1000:.1f} ms")
    print(f"  TTFT  P95: {p95*1000:.1f} ms")
    print(f"  TTFT  P99: {p99*1000:.1f} ms  (last={ttfts[-1]*1000:.1f} ms)")
    print(f"  Wall time: {wall_time:.2f}s")
    print(f"  Agg gen throughput: {agg_throughput:.1f} tok/s")
    print(f"  All TTFTs (ms): {[f'{t*1000:.0f}' for t in ttfts]}")

    return {"p50": p50, "p95": p95, "p99": p99,
            "wall_time": wall_time, "agg_throughput": agg_throughput,
            "n_ok": len(ok)}


async def wait_for_health(url, timeout=600):
    print(f"\nWaiting for {url}/health ...")
    deadline = time.perf_counter() + timeout
    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        while time.perf_counter() < deadline:
            try:
                async with session.get(f"{url}/health",
                                       timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status == 200:
                        print(f"  Server healthy after {time.perf_counter():.0f}s")
                        return True
            except Exception:
                pass
            await asyncio.sleep(5)
    print("  TIMEOUT waiting for health")
    return False


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--mode", choices=["baseline", "lmcache", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup requests before measurement run")
    parser.add_argument("--requests", type=int, default=NUM_REQUESTS)
    parser.add_argument("--no-health-check", action="store_true")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"

    if not args.no_health_check:
        ok = await wait_for_health(url)
        if not ok:
            print("Server not healthy — aborting")
            return

    # Warmup (populate prefix cache on first call)
    if args.warmup > 0:
        print(f"\nWarmup: sending {args.warmup} requests to populate prefix cache...")
        connector = aiohttp.TCPConnector(limit=args.warmup + 2)
        async with aiohttp.ClientSession(connector=connector) as session:
            warmup_tasks = [send_request_streaming(session, url, i, "warmup")
                            for i in range(args.warmup)]
            warmup_results = await asyncio.gather(*warmup_tasks)
            ok_warmup = sum(1 for r in warmup_results if r["status"] == "OK")
            print(f"  Warmup: {ok_warmup}/{args.warmup} OK")

    result = await run_bench(url, args.mode, num_requests=args.requests)

    if result:
        print(f"\n{'='*70}")
        print(f"  TAG: W1_4e_lmcache_smoke | MODE: {args.mode}")
        print(f"  P50={result['p50']*1000:.1f}ms  P95={result['p95']*1000:.1f}ms  "
              f"P99={result['p99']*1000:.1f}ms")
        print(f"  Agg throughput: {result['agg_throughput']:.1f} tok/s")
        print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
