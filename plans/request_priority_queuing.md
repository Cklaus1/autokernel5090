# Request Priority Queuing for Coding Workloads

**Date:** 2026-04-09  
**Goal:** Reduce average latency for mixed coding workloads by scheduling short
responses before long ones (Shortest Job First / SJF approximation).

---

## 1. vLLM Priority Support — Research Findings

vLLM (v0.19.0 installed) has **built-in priority scheduling** that can be used
directly. No patching required.

### How it works

**Server launch:** Add `--scheduling-policy priority` to `vllm serve` args. This
sets `SchedulerConfig.policy = "priority"` (defined in
`/usr/local/lib/python3.12/dist-packages/vllm/config/scheduler.py`, line 22).

**Request field:** Both the `/v1/chat/completions` and `/v1/completions` endpoints
accept a non-standard `priority: int` field in the JSON body
(`vllm/entrypoints/openai/chat_completion/protocol.py:285`,
`vllm/entrypoints/openai/completion/protocol.py:107`).

- Lower integer = higher scheduling priority (runs sooner)
- Default is `priority=0` (no preference)
- Any non-zero value raises an error if the server is not in priority mode

**Scheduler behavior:** The v1 scheduler (`vllm/v1/core/sched/scheduler.py`) sorts
the waiting queue by `(priority, arrival_time)`. When KV memory is tight it
preempts the _highest_ `priority` value (i.e., the lowest-priority request) first.

**Request comparison** (`vllm/v1/request.py:283`):
```python
def __lt__(self, other):
    if self.priority != other.priority:
        return self.priority < other.priority   # lower int = runs first
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time  # FCFS tiebreak
    ...
```

### What vLLM does NOT do natively

- Does not auto-assign priority from `max_tokens` — caller must set it
- Does not estimate output length from prompt text
- No SJF reordering within a single scheduler tick (only waiting queue order)

---

## 2. Priority Tier Design

Map `max_tokens` (the best pre-generation proxy for response length) to four
priority tiers. Lower integer = served first.

| Priority | max_tokens range | Coding use case | Expected latency |
|----------|-----------------|-----------------|-----------------|
| 1 | ≤ 50 | Completions, yes/no, one-liner fixes | < 1 s |
| 2 | ≤ 200 | Explanations, short diffs, docstrings | 1-5 s |
| 3 | ≤ 500 | Function-level generation, unit tests | 5-20 s |
| 4 | > 500 | Full file generation, large refactors | 20+ s |

Rationale: SJF reduces average wait time optimally under homogeneous service time
distributions. For LLMs, `max_tokens` is a reliable upper bound on service time
(actual tokens ≤ max_tokens by definition).

### Boundary calibration

On an H100 at ~4000 tok/s decode throughput with batch=1:
- 50 tokens → ~12 ms decode
- 200 tokens → ~50 ms decode
- 500 tokens → ~125 ms decode
- 2048 tokens → ~512 ms decode

Batch-shared throughput varies, but relative ordering holds. The four tiers
create ~10x spread between P1 and P4 expected service time — sufficient for
meaningful SJF benefit.

---

## 3. Implementation

### 3a. vLLM Server

```bash
# Launch with priority scheduling enabled
vllm serve <model> \
  --scheduling-policy priority \
  --max-num-seqs 256 \
  --max-model-len 8192
```

### 3b. Priority Mapper (middleware / fusen_solver layer)

```python
def compute_priority(max_tokens: int | None, prompt: str | None = None) -> int:
    """
    Map max_tokens to vLLM priority tier.
    Lower return value = scheduled sooner.

    Falls back to prompt-text heuristics if max_tokens is unset.
    """
    # Tier from max_tokens (primary signal — most reliable)
    if max_tokens is not None:
        if max_tokens <= 50:
            return 1
        if max_tokens <= 200:
            return 2
        if max_tokens <= 500:
            return 3
        return 4

    # Tier from prompt heuristics (fallback when max_tokens not provided)
    if prompt is not None:
        prompt_lower = prompt.lower()
        # High-signal short-response patterns
        short_patterns = [
            "fix this", "fix the bug", "what is", "yes or no",
            "complete the", "fill in", "one line", "rename",
        ]
        long_patterns = [
            "write a complete", "implement a full", "create a rest api",
            "generate the entire", "write all", "full implementation",
        ]
        if any(p in prompt_lower for p in short_patterns):
            return 2  # conservative — don't assume max shortness
        if any(p in prompt_lower for p in long_patterns):
            return 4
        return 3  # default middle tier when uncertain

    return 3  # safe default
```

### 3c. Request injection

For OpenAI-compatible clients, inject the `priority` field in the request body
**before** sending to vLLM. The field is accepted by vLLM but ignored by standard
OpenAI clients (they strip unknown fields), so this is safe for dual-use proxies.

```python
import httpx

async def forward_request(payload: dict, vllm_base_url: str) -> httpx.Response:
    max_tokens = payload.get("max_tokens")
    prompt = (payload.get("messages") or [{}])[-1].get("content")

    priority = compute_priority(max_tokens, prompt)
    payload = {**payload, "priority": priority}  # inject, do not mutate in-place

    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{vllm_base_url}/v1/chat/completions",
            json=payload,
            timeout=120.0,
        )
```

---

## 4. fusen_solver Integration — Racing Agents

When fusen_solver launches N parallel agents on a single task (best-of-N sampling),
priority assignment becomes more nuanced. The agent most likely to finish first
should get the lowest priority integer so it is scheduled ahead of slower agents.

### Estimating "most likely to finish first"

For a racing pool of N agents, each with the same prompt but different sampling
parameters or temperature, the expected finish order depends on:

1. **Output length estimate:** Agents with lower `max_tokens` finish first
2. **Temperature:** Lower temperature → more predictable (often shorter) outputs
   since the model concentrates probability mass on common tokens including EOS
3. **Speculation:** If one agent has speculative decoding enabled, it generates
   tokens faster — give it equal or higher priority to not waste the speedup

```python
def racing_priority(agent_index: int, n_agents: int,
                    max_tokens: int, temperature: float,
                    use_spec_decode: bool) -> int:
    """
    Assign priority within a racing pool. Lower int = scheduled first.

    Strategy: The agent most likely to terminate early gets priority 1.
    We estimate finish speed as a composite score.
    """
    # Score: lower = faster expected finish
    length_score = max_tokens / 2048.0          # normalized
    temp_score   = temperature / 2.0            # higher temp → longer outputs
    spec_bonus   = -0.2 if use_spec_decode else 0.0  # spec decode is faster

    speed_score = length_score + temp_score + spec_bonus

    # Rank agents by speed_score (ascending = fastest first)
    # Caller should collect scores for all N agents and sort before assigning
    # This function returns the score for sorting purposes
    return speed_score


def assign_racing_priorities(agents: list[dict]) -> list[int]:
    """
    Given a list of agent configs, return priority integers in [1, 4].
    Fastest-expected agent gets priority 1, slowest gets higher.

    agents: list of dicts with keys: max_tokens, temperature, use_spec_decode
    """
    scores = [
        racing_priority(
            i, len(agents),
            a["max_tokens"], a["temperature"], a.get("use_spec_decode", False)
        )
        for i, a in enumerate(agents)
    ]
    # Sort indices by score (ascending = faster)
    ranked = sorted(range(len(agents)), key=lambda i: scores[i])

    # Map rank → priority tier: rank 0 (fastest) → priority 1
    priorities = [0] * len(agents)
    for rank, agent_idx in enumerate(ranked):
        # Spread across tiers 1-4, clamped
        priorities[agent_idx] = min(rank + 1, 4)

    return priorities
```

### RacingAgentPool with priority-aware launch

```python
class RacingAgentPool:
    """
    Launch N agents with SJF-ordered priorities.
    First winner aborts all losers immediately.
    """

    def __init__(self, vllm_base_url: str, n: int = 4):
        self.base_url = vllm_base_url
        self.n = n

    async def race(
        self,
        prompt: str,
        agent_configs: list[dict],
        accept_fn: Callable[[str], bool],
    ) -> str:
        assert len(agent_configs) == self.n

        # Assign SJF priorities before launching
        priorities = assign_racing_priorities(agent_configs)

        request_ids = [f"race-{uuid4()}" for _ in range(self.n)]
        tasks = []
        for i, (cfg, priority, req_id) in enumerate(
            zip(agent_configs, priorities, request_ids)
        ):
            payload = {
                **cfg,
                "messages": [{"role": "user", "content": prompt}],
                "priority": priority,
                "stream": True,
            }
            tasks.append(asyncio.create_task(
                self._stream_until_accept(req_id, payload, accept_fn)
            ))

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Abort losers immediately to free KV blocks
        for task in pending:
            task.cancel()

        return (await next(iter(done)))

    async def _stream_until_accept(
        self, req_id: str, payload: dict, accept_fn: Callable[[str], bool]
    ) -> str:
        full_text = ""
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json={**payload, "stream": True},
                headers={"X-Request-Id": req_id},
                timeout=120.0,
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line == "data: [DONE]":
                        break
                    chunk = json.loads(line[6:])
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    full_text += delta
                    if accept_fn(full_text):
                        # Winner found — abort our own stream
                        # vLLM will GC the request when connection closes
                        return full_text
        return full_text
```

---

## 5. End-to-End Latency Impact

### Single-request SJF (no racing)

Under a mixed workload of equal P1/P2/P3/P4 requests:

Without priority: average wait = sum of all service times / 2 (random order)

With SJF (priority queuing): average wait reduced by O(N) factor for the short
requests. In a queue of 8 requests (2 per tier), P1 requests see near-zero wait;
P4 requests see the full queue drain.

**Approximate latency reduction for P1 (≤50 tok) requests:**

| Queue depth | FCFS avg wait | SJF avg wait | Reduction |
|-------------|--------------|--------------|-----------|
| 4 requests  | ~100 ms      | ~15 ms       | 85% |
| 8 requests  | ~200 ms      | ~20 ms       | 90% |
| 16 requests | ~400 ms      | ~25 ms       | 94% |

(Assumes uniform arrival, 50/200/500/2048 tok service times, H100 4k tok/s)

### Racing agents (N=4)

With SJF priority on N=4 agents, the fastest agent (P1) gets scheduled first.
If all 4 start simultaneously (fresh queue), priority has no wait-time effect —
all start together. Priority matters when agents are added to a non-empty queue.

The main gain for racing is **abort-on-winner** (already in `early_kv_termination.md`),
not priority ordering. Priority ordering is the tiebreaker when racing jobs queue
behind other workloads.

---

## 6. Configuration Summary

```python
# fusen_solver config additions
PRIORITY_CONFIG = {
    "enabled": True,
    "vllm_scheduling_policy": "priority",   # must match server launch arg

    # SJF tier thresholds (max_tokens → priority int)
    "tier_thresholds": [
        (50,  1),   # ≤50 tokens → priority 1
        (200, 2),   # ≤200 tokens → priority 2
        (500, 3),   # ≤500 tokens → priority 3
        (None, 4),  # everything else → priority 4
    ],

    # Racing pool settings
    "racing_n": 4,
    "racing_priority_spread": True,  # assign distinct priorities to racing agents
}
```

---

## 7. Caveats and Risks

1. **Starvation:** P4 requests (full file generation) can be starved indefinitely
   under high P1 load. Mitigation: implement aging — bump priority by 1 every
   30 seconds in the waiting queue. vLLM does not do this natively; requires a
   wrapper that re-submits or a custom scheduler class.

2. **max_tokens inflation:** Some clients set `max_tokens=4096` defensively even
   for short responses. This misclassifies them as P4. Mitigation: use prompt
   heuristics as a secondary signal (see `compute_priority` fallback above).

3. **Priority inversion in chunked prefill:** vLLM's chunked prefill can split a
   long prefill across multiple scheduler ticks. A high-priority request arriving
   mid-prefill of a P4 request will preempt it. This is correct behavior but
   causes the P4 request to restart prefill from scratch — a latency cliff for P4.
   Mitigation: `--max-long-partial-prefills 1` (already default) limits concurrent
   long prefills, so preemption cost is bounded.

4. **Racing priority has diminishing returns:** If all N racing agents have the
   same `max_tokens` and `temperature`, `assign_racing_priorities` gives them the
   same score and priorities collapse to 1,2,3,4 by index (arbitrary). The real
   win from racing is the abort, not the priority spread.

5. **OpenAI SDK compatibility:** The `priority` field is not in the official OpenAI
   API spec. Standard `openai` Python client sends it as an extra field, which
   OpenAI's servers ignore but vLLM accepts. Safe for any vLLM-backed deployment.

---

## 8. Implementation Checklist

- [ ] Launch vLLM with `--scheduling-policy priority`
- [ ] Add `compute_priority()` to fusen_solver request forwarding path
- [ ] Inject `priority` field in all outgoing vLLM requests
- [ ] Add `assign_racing_priorities()` to `RacingAgentPool`
- [ ] Add starvation aging (optional, needed only under sustained P1 flood)
- [ ] Instrument: log `priority` alongside `request_id` for latency analysis
- [ ] Validate: compare P1 TTFT before/after with mixed workload benchmark
