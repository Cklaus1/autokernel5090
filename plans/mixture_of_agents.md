# Mixture of Agents for fusen_solver

## Hardware Layout

```
RTX PRO 6000 DP=2 (2x 96GB VRAM, PCIe — no NCCL)

GPU 0 (port 8000): Gemma4 26B NVFP4 + FusenCache
  - 121 tok/s single-user (FusenCache measured)
  - ~500K token KV budget per GPU
  - Role: STRONG — deep reasoning, architecture, complex code

GPU 1 (port 8001): Gemma4 E2B (2B) or Qwen3.5-9B
  - E2B:  ~170-350 tok/s estimated (3x+ faster)
  - Qwen: ~200-280 tok/s estimated (medium speed, better quality than E2B)
  - Role: FAST — formatting, translation, simple Q&A, test scaffolding
```

The two GPUs are fully independent (DP=2, no AllReduce). The MoA router runs
entirely on the CPU and adds zero GPU overhead.

---

## Task Difficulty Classification

### Signal Sources (in priority order)

1. **Strategy tag** — fusen_solver already tags strategies. "fast", "minimal" →
   fast model; "analytical", "thorough" → strong model. Zero latency cost.
2. **Problem type** — `Problem.problem_type` field: `bug_fix`, `feature`,
   `refactor`, `architecture`, `optimize`, `test`, `review`.
3. **Keyword classifier** — regex on `Problem.description`. Free, deterministic.
4. **Context size** — if `len(context_tokens) > FAST_MAX_CONTEXT` (e.g. 16K),
   prefer strong model (more robust at long context).
5. **Cascade fallback** — try fast model; if output fails a quick parse check
   (missing code fences, empty diff, truncation), retry on strong model.

### Difficulty Tiers

| Tier | Criteria | Backend |
|------|----------|---------|
| EASY | Formatting, translation, summarization, list generation, simple Q&A, test stub generation | fast (E2B / Qwen) |
| MEDIUM | Code generation < 100 LOC, debugging single function, knowledge lookup, incremental edits | fast (with cascade option) |
| HARD | Architecture design, multi-file refactor, complex algorithm, multi-step math, security audit, long-context reasoning | strong (26B) |

### Keyword Patterns

```
EASY triggers (→ fast):
  format, translate, summarize, list, convert, rename, reorder, sort,
  what is, define, explain briefly, stub, scaffold, boilerplate

HARD triggers (→ strong):
  architect, design, refactor.*large, complex.*algorithm, multi.?step,
  security, audit, optimize.*system, performance.*analysis, root cause,
  concurrency, race condition, distributed, trade.?off

MEDIUM (default → fast with cascade):
  everything else
```

---

## Routing Rules for fusen_solver

### Strategy-to-Backend Mapping

Built on fusen_solver's existing `MultiBackend` + `Strategy.tags` system.
No new abstractions needed — the router reads `strategy.tags`.

| Strategy | Tags | Backend | Rationale |
|----------|------|---------|-----------|
| `direct` | fast, minimal, safe | fast | Targeted 1-line fix |
| `incremental` | safe, minimal, incremental | fast | Small diff, low complexity |
| `test_first` | thorough, safe, testable | fast | Writing tests is structured/template-driven |
| `prototype_then_refine` | iterative, practical | fast (prototype) → strong (refine) | Split: prototype on fast, refinement pass on strong |
| `alternative` | creative, exploratory | strong | Needs architectural breadth |
| `decompose` | structured, thorough | strong | Multi-part reasoning |
| `review` | thorough, quality | strong | Needs depth and judgment |
| `research` | analytical, thorough | strong | Root-cause reasoning |
| `rewrite` | creative, quality | strong | Full redesign |
| `adversarial` | thorough, safe, defensive | strong | Edge-case enumeration |
| `profile_first` | analytical, performance | strong | Perf analysis needs depth |
| `security` | security, thorough | strong | Security audit is high-stakes |

**Tag routing logic:**
- If tags intersect `{"analytical", "thorough", "creative", "security"}` → strong
- Else if tags intersect `{"fast", "minimal", "safe", "incremental"}` → fast
- Else → strong (default safe)

### Cascade Pattern

```python
async def cascade_generate(prompt, fast_backend, strong_backend, timeout_fast=30):
    """Try fast model first; fall back to strong if output is low quality."""
    try:
        result = await asyncio.wait_for(
            fast_backend.generate(prompt), timeout=timeout_fast
        )
        if _is_acceptable(result):
            return result, "fast"
        # Low confidence: retry with strong
        logger.info("Fast model output rejected, retrying with strong backend")
    except asyncio.TimeoutError:
        logger.warning("Fast backend timed out, falling back to strong")

    result = await strong_backend.generate(prompt)
    return result, "strong"


def _is_acceptable(text: str) -> bool:
    """Quick heuristic checks for output quality."""
    if len(text) < 50:                    # suspiciously short
        return False
    if "```" not in text and "def " not in text and "class " not in text:
        # No code block in a code task — likely refused or truncated
        return False
    if text.endswith("...") or text.endswith("etc."):
        return False                       # truncated
    return True
```

### Parallel Solver Configuration

For a `default_n=6` run split across backends:

```
HARD problem (architecture/research):
  - 4 agents on GPU 0 (strong): review, research, decompose, alternative
  - 2 agents on GPU 1 (fast): direct, test_first (as sanity check)

MEDIUM problem (bug fix/feature):
  - 2 agents on GPU 0 (strong): research, adversarial
  - 4 agents on GPU 1 (fast): direct, test_first, incremental, prototype_then_refine

EASY problem (formatting/simple Q&A):
  - 0 agents on GPU 0 (skip strong entirely)
  - 4 agents on GPU 1 (fast): direct × 2, incremental, test_first
```

---

## Qwen3.5-9B vs Gemma4 E2B Trade-offs

| Model | Speed | Quality | Best For |
|-------|-------|---------|----------|
| Gemma4 E2B | ~300 tok/s | Baseline | Trivial formatting, boilerplate |
| Qwen3.5-9B | ~240 tok/s | +20% | Code generation, debugging, medium tasks |
| Gemma4 26B | 121 tok/s | Best | Architecture, reasoning, hard tasks |

**Recommendation:** Run Qwen3.5-9B on GPU 1 (not E2B) for the MEDIUM tier.
It handles code generation well enough that cascade fallback rarely triggers,
which keeps GPU 0 free for truly hard tasks. E2B is faster but more likely
to produce low-quality code that requires a strong-model retry — net worse
latency due to double-inference cost.

**Switch to E2B on GPU 1 when:**
- Workload is >70% EASY tasks (formatting, translation)
- Latency matters more than quality (interactive assistant mode)
- Cascade retries are rare (<5% of requests)

---

## Integration with fusen_solver

### How MultiBackend already supports this

`fusen_solver/backends/multi_backend.py` has `generate_with_strategy(strategy_name)`.
The solver calls this automatically when running strategies. Adding MoA is:

1. Instantiate two `VLLMBackend` objects (one per GPU)
2. Build a `MultiBackend` with `routes=` dict mapping strategy names to backends
3. Add keyword-based pre-routing as a thin wrapper before strategy dispatch

### Wiring into solver.py

```python
# In core/solver.py, before strategy dispatch:
backend = moa_router.select(
    strategy=strategy,
    problem=problem,
    fast=fast_backend,
    strong=strong_backend,
)
result = await backend.generate_with_strategy(messages, strategy.name)
```

No changes to bench.py, orchestrate.py, or any fixed files.

---

## Operational Notes

- **Health checks:** Both backends should be health-checked every 10s. If fast
  backend is down, route everything to strong. If strong is down, route to fast
  with a quality warning.
- **Queue pressure:** If strong backend queue > 8 pending, route MEDIUM tasks
  to fast to avoid starvation.
- **Prefix cache:** For the same problem, all agents share the same system+context
  prefix. Both vLLM instances maintain independent prefix caches. No cross-GPU
  sharing needed (DP=2).
- **Logging:** Log which backend handled each strategy + outcome. Feed into
  fusen_solver's learning engine to tune routing thresholds over time.
- **Future work:** Replace keyword classifier with a tiny embedding model
  (e.g. BGE-small) for semantic routing. Use learning engine's historical
  accept/reject data to train the router.
