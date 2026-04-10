# Parallel Problem-Solving

Use batch GPU capacity to solve ONE coding problem faster, instead of serving isolated users.

## The Idea

Standard LLM serving (vLLM) optimizes for many users with independent requests. This system flips that: ONE user gets ALL GPU capacity to solve a single problem faster through parallel exploration.

With 2x PRO 6000 (1.28M KV tokens via FusenCache), we can run 10+ parallel agents each with 128K context, all sharing the same codebase prefix via vLLM's prefix caching.

```
                     ┌──────────────────────┐
                     │   Problem Orchestrator │
                     │   (decomposes the     │
                     │    problem)            │
                     └──────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
     ┌────────▼───────┐ ┌──────▼────────┐ ┌─────▼─────────┐
     │  Strategy A     │ │  Strategy B    │ │  Strategy C    │
     │  direct fix     │ │  test-first    │ │  rewrite       │
     └────────┬───────┘ └──────┬────────┘ └─────┬─────────┘
              │                 │                 │
     ┌────────▼─────────────────▼─────────────────▼────────┐
     │              vLLM with Prefix Caching                │
     │  Shared: codebase (50K tokens) — cached ONCE         │
     │  Per-agent: unique strategy (10-30K each)             │
     │  Total: 50K + N*30K  (fits in 1.28M KV cache)        │
     │                                                      │
     │  All N agents run in ONE GPU batch = near-linear      │
     │  throughput scaling with almost no per-agent slowdown  │
     └─────────────────────────────────────────────────────┘
```

## Components

| File | Role |
|------|------|
| `orchestrator.py` | Decomposes problems, launches parallel agents, collects and ranks results |
| `prefix_manager.py` | Builds shared-prefix messages for efficient KV cache reuse |
| `streaming.py` | Concurrent async streaming from N agents via aiohttp |
| `solution_scorer.py` | Scores solutions via tests and/or LLM review, merges best insights |
| `cli.py` | CLI for solve, interactive, and benchmark modes |
| `test_solver.py` | Unit and integration tests (mocked HTTP, no GPU needed) |

## Strategies

Eight built-in strategies, each exploring the problem from a different angle:

- **direct** -- minimal targeted fix
- **alternative** -- different algorithm or data structure
- **test_first** -- write tests first, then fix
- **decompose** -- break into sub-problems, solve each
- **review** -- identify all issues, fix the most critical
- **research** -- root cause analysis with 3 fix proposals
- **rewrite** -- clean rewrite from scratch
- **adversarial** -- find every way the code can break, defend against all

Strategy presets group these by problem type: `bug_fix`, `feature`, `refactor`, `architecture`, `optimization`, `explore`.

## Usage

### One-shot solve

```bash
python3 -m parallel_solver solve \
  --problem "Fix the race condition in server.py" \
  --codebase ./src/ \
  --agents 8 \
  --strategies "direct,review,test_first,adversarial" \
  --stream
```

### Interactive mode

```bash
python3 -m parallel_solver interactive \
  --codebase ./src/ \
  --agents 4 \
  --preset bug_fix
```

In interactive mode, prefix `!preset_name` to override the strategy preset:
```
Problem> !refactor Clean up the payment module
```

### Benchmark

```bash
python3 -m parallel_solver benchmark \
  --problems problems.json \
  --agents 1,2,4,8 \
  --output bench_results.json
```

### Programmatic

```python
import asyncio
from parallel_solver import ProblemOrchestrator

async def main():
    orch = ProblemOrchestrator("http://localhost:8000", "my-model")
    result = await orch.solve(
        problem="Fix the off-by-one error in pagination",
        codebase=open("app.py").read(),
        strategies=["direct", "test_first", "review", "adversarial"],
    )
    print(result.best_solution.content)
    print(f"Score: {result.best_solution.overall:.2f}")
    print(f"Aggregate throughput: {result.aggregate_tps:.0f} tok/s")

asyncio.run(main())
```

## Performance Projections

On 2x PRO 6000 with FusenCache (1.28M tokens):

| Config | Throughput | Time for 5K tokens | Quality |
|--------|-----------|-------------------|---------|
| 1 agent | 127 tok/s | 39s | baseline |
| 8 agents (best-of-8) | ~1,000 tok/s aggregate | 39s, 8 attempts | ~8x quality |
| 8 agents (decomposed) | ~1,000 tok/s aggregate | 39s, 8 modules | ~8x speed |

The key insight: batching on GPUs means N agents cost almost the same wall-clock time as 1 agent, because the GPU processes them in parallel. You pay in tokens, not in time.

## Tests

```bash
python3 -m pytest parallel_solver/test_solver.py -v
```

Tests run without a GPU or vLLM instance -- all HTTP calls are mocked.

## Requirements

- `aiohttp` -- async HTTP client for streaming from vLLM
- `pytest`, `pytest-asyncio` -- for tests
- A running vLLM instance with an OpenAI-compatible API
