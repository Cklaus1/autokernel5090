# Fusen Solver Code Review

Reviewer: ASI Code Reviewer
Date: 2026-04-09
Scope: All source files in `fusen_solver/`

---

## 1. Bugs (Will Cause Incorrect Behavior)

### BUG-01: `_run_agent` passes `priority` kwarg to backends that don't accept it
- **File:** `core/solver.py` line ~1139
- **Severity:** Critical
- **Description:** `_run_agent` calls `self.backend.generate(..., priority=priority)`. The abstract `LLMBackend.generate()` signature (interfaces.py line 75) does NOT include `priority`. Only `VLLMBackend` and `OpenAIBackend` accept `**kwargs` to silently absorb it. `AnthropicBackend` and `OllamaBackend` do NOT have `**kwargs` -- they will raise `TypeError: generate() got an unexpected keyword argument 'priority'` at runtime.
- **Fix:** Either add `**kwargs` to `AnthropicBackend.generate()` and `OllamaBackend.generate()`, or add `priority` as an optional param to the abstract interface, or remove it from the `_run_agent` call and handle priority at the backend router level.
- **Effort:** 5 minutes

### BUG-02: `_weighted_score` division produces wrong result when tests are missing
- **File:** `scoring/engine.py` lines 111-129
- **Severity:** High
- **Description:** When `has_tests=False`, the test weight is redistributed to review. But the final division by `total_weight` includes ALL weights (including the zeroed-out test weight). Since `weights["tests"] = 0.0`, `total_weight` is `0.7` (not `1.0`), and the score is divided by `0.7`. This means the score is inflated by ~1.43x when tests are absent. If `subscores["tests"]` is the sentinel `-1.0`, the `subscores[signal] >= 0` check skips it, but the denominator is still wrong.
- **Fix:** Compute `total_weight` as the sum of only the weights whose signals are actually used (weight > 0 AND subscore >= 0). Or normalize weights to sum to 1.0 after redistribution.
- **Effort:** 10 minutes

### BUG-03: `MultiBackend.stream()` ignores session/strategy routing
- **File:** `backends/multi_backend.py` line 157-169
- **Severity:** High
- **Description:** `stream()` always uses `self._default` regardless of any routing configuration. It doesn't accept `session_id` or `strategy_name` parameters, so streaming always goes to the default backend, violating the routing contract.
- **Fix:** Add `session_id` param, call `self.route()`, and delegate to the resolved backend's `stream()`. Also add fallback handling.
- **Effort:** 10 minutes

### BUG-04: `MultiBackend.generate()` ignores strategy routing
- **File:** `backends/multi_backend.py` line 106-127
- **Severity:** Medium
- **Description:** `generate()` only routes by `session_id`, not by strategy name. The `generate_with_strategy()` method exists but the solver calls `generate()` (via the `LLMBackend` interface), so strategy-based routing is never triggered by the solver.
- **Fix:** Either have the solver call `generate_with_strategy()` when using a `MultiBackend`, or merge the strategy routing into `generate()`.
- **Effort:** 15 minutes

### BUG-05: `LearningEngine._load` double-counts stats
- **File:** `learning/tracker.py` lines 368-393
- **Severity:** Medium
- **Description:** `_load()` is called in `__init__`. It rebuilds `_stats` from `_history`. But `_stats` is initialized as a `defaultdict` in `__init__` before `_load()` is called. If `_load` is ever called again (no current call path, but the method is public), stats are accumulated on top of existing stats. More importantly: after `_load`, any subsequent `record()` call appends to `_history` AND updates `_stats`. On the next process restart, `_load` replays ALL history entries including the ones from `record()`, so the counts are correct. But if `_load` were called twice in one session, everything would double-count.
- **Fix:** Clear `_stats` at the start of `_load()`.
- **Effort:** 2 minutes

### BUG-06: `AgentMemory.recall()` mutates and saves on every call even with no results
- **File:** `learning/tracker.py` lines 449-474
- **Severity:** Low
- **Description:** `recall()` increments `used_count` on recalled memories and saves to disk. This means every call to `_run_agent` (which calls `recall`) triggers a disk write. More critically, the `used_count` increase changes the ranking for subsequent calls within the same solve -- later agents get different memory rankings than earlier agents. This introduces non-determinism that depends on agent execution order.
- **Fix:** Defer the `used_count` increment to after the solve completes, or batch the saves.
- **Effort:** 15 minutes

### BUG-07: `_extract_code_blocks` regex is greedy across blocks
- **File:** `core/solver.py` lines 1240-1261
- **Severity:** Medium
- **Description:** The regex `r"```(?:(\S+)\n)?(.*?)```"` with `re.DOTALL` uses `.*?` which is non-greedy, but the outer `findall` may still have issues when code blocks contain triple backticks in comments or strings. More importantly: if the label is a language name like `python`, the block gets named `block_0.txt` instead of being associated with the actual filename. When the LLM returns `\`\`\`python\n...\`\`\`` (which is the most common format), the filename is lost.
- **Fix:** Also check for a filename comment/header pattern before the code block (e.g., `# filename.py` on the line before the fence). The current heuristic of requiring `.` or `/` in the label to treat it as a filename will misclassify `python`, `javascript`, etc. as non-filenames, which is correct, but then the extracted code has no meaningful filename association.
- **Effort:** 20 minutes

### BUG-08: Session affinity in MultiBackend is not thread-safe
- **File:** `backends/multi_backend.py` lines 47-58
- **Severity:** Medium
- **Description:** `_session_map` is a plain dict mutated by `_evict_expired_sessions()`, `route()`, and potentially concurrent async tasks. In an async context with multiple concurrent `generate()` calls (which is the whole point of parallel agents), dict mutation during iteration in `_evict_expired_sessions` can raise `RuntimeError: dictionary changed size during iteration` if two tasks evict concurrently.
- **Fix:** Use `asyncio.Lock` around session map mutations, or copy the dict before iterating.
- **Effort:** 10 minutes

---

## 2. Design Issues (Works But Fragile)

### DESIGN-01: aiohttp session created per-request in all backends
- **File:** `backends/vllm_backend.py` lines 64-66, `backends/openai_backend.py` lines 70-72, etc.
- **Severity:** High
- **Description:** Every `generate()` and `stream()` call creates a new `aiohttp.ClientSession` and destroys it after the request. This means a new TCP connection (and TLS handshake for HTTPS backends) for every single LLM call. With N=8 parallel agents, that's 8 connection setups per solve. This is a significant latency penalty.
- **Fix:** Create the session once in `__init__` (or lazily) and reuse it. Implement `async def close()` for cleanup. Consider using `__aenter__`/`__aexit__` pattern.
- **Effort:** 20 minutes per backend

### DESIGN-02: Scoring engine `_run_tests` is synchronous/blocking in async context
- **File:** `scoring/engine.py` line 157, `scoring/sandbox.py` lines 134-189
- **Severity:** High
- **Description:** `_run_tests` is a `@staticmethod` (not async) that calls `subprocess.run()` synchronously. It's called from `score_all()` which is async. This blocks the event loop for the entire duration of test execution (up to 30 seconds per command). During this time, no other async work (like concurrent agent scoring) can proceed.
- **Fix:** Use `asyncio.create_subprocess_exec()` or run `subprocess.run` in an executor via `asyncio.get_event_loop().run_in_executor()`.
- **Effort:** 30 minutes

### DESIGN-03: `_format_codebase` and `_build_messages` are static methods on FusenSolver
- **File:** `core/solver.py` lines 1196-1261
- **Severity:** Low
- **Description:** These utility methods don't use `self` and are marked `@staticmethod`, but they're coupled to the solver class. They duplicate functionality that `PrefixManager` and `CodebaseIndex` already provide (building context strings, composing messages). The solver should use `PrefixManager` instead of reimplementing context formatting.
- **Fix:** Inject a `PrefixManager` into the solver and delegate context building to it.
- **Effort:** 30 minutes

### DESIGN-04: No retry logic for LLM API calls
- **File:** All backends
- **Severity:** High
- **Description:** A transient HTTP 500 or 429 (rate limit) from any backend causes the entire agent to fail. With parallel agents, a single rate-limit response can cascade to lose that entire strategy's contribution. There is no retry with exponential backoff.
- **Fix:** Add a retry decorator (3 retries, exponential backoff, jitter) around generate/stream calls in each backend. Or implement it once as a wrapper/mixin.
- **Effort:** 30 minutes

### DESIGN-05: Config values hardcoded instead of sourced from Config
- **File:** `core/solver.py`, `integrations/cli.py`
- **Severity:** Medium
- **Description:** The solver reads `strategy`, `learning`, and `scoring` config sections in `load_config()`, but then `_make_backend` in cli.py doesn't pass scoring weights from config to `ScoringEngine`. The solver always uses default scoring weights. Similarly, learning engine paths from config are never passed through.
- **Fix:** Wire config values through to `ScoringEngine`, `LearningEngine`, etc. in cli.py `cmd_solve`.
- **Effort:** 15 minutes

### DESIGN-06: `solve_decomposed` has no timeout
- **File:** `core/solver.py` line 589
- **Severity:** Medium
- **Description:** Decomposed mode makes multiple sequential LLM calls (decomposition + N levels of generation + integration verification). There is no overall timeout. A slow LLM or large decomposition could run indefinitely.
- **Fix:** Add an overall timeout (e.g., `asyncio.wait_for` wrapping the whole method).
- **Effort:** 10 minutes

---

## 3. Security Issues

### SEC-01: Arbitrary command execution in unsandboxed test fallback
- **File:** `scoring/sandbox.py` lines 166-188
- **Severity:** Critical
- **Description:** When Docker is unavailable, test commands are run via `subprocess.run(cmd, shell=True)` directly on the host. The test commands come from `Problem.tests`, which in the API server path (`integrations/api.py` line 49) comes directly from the HTTP request body with no validation. An attacker can POST `{"tests": ["rm -rf /"]}` and it will execute on the host.
- **Fix:** (1) Validate/sanitize test commands in the API handler. (2) Refuse to run tests without Docker in production/API mode. (3) At minimum, add a whitelist of allowed command prefixes (e.g., `python`, `pytest`, `npm test`).
- **Effort:** 30 minutes

### SEC-02: Path traversal in sandbox `_write_code`
- **File:** `scoring/sandbox.py` lines 126-132
- **Severity:** High
- **Description:** `_write_code` joins user-supplied file paths with `tmpdir` using `os.path.join`. A malicious path like `../../etc/cron.d/evil` in the `code` dict would write outside the tmpdir. `os.path.join(tmpdir, "../../etc/foo")` resolves to `/etc/foo`.
- **Fix:** Validate that the resolved path starts with `tmpdir`: `assert os.path.realpath(full_path).startswith(os.path.realpath(tmpdir))`.
- **Effort:** 5 minutes

### SEC-03: Path traversal in `CodebaseIndex` and `build_context`
- **File:** `core/codebase_index.py` line 400
- **Severity:** Medium
- **Description:** `build_context` joins `self.root` with `rel_path` from `self.files`. Since `self.files` is populated from `os.walk`, the paths are safe. But `build_context` is a public method that accepts arbitrary path strings. A caller passing `"../../etc/passwd"` would read files outside the codebase root.
- **Fix:** Validate that `os.path.realpath(abs_path).startswith(os.path.realpath(self.root))`.
- **Effort:** 5 minutes

### SEC-04: API key could leak in error messages
- **File:** `backends/openai_backend.py` line 80, `backends/anthropic_backend.py` line 82
- **Severity:** Medium
- **Description:** On HTTP errors, the full response text is included in the RuntimeError. If the API returns an error that echoes the request (including Authorization headers), the API key could leak into logs or error handlers. This is unlikely with OpenAI/Anthropic but possible with proxies or custom endpoints.
- **Fix:** Truncate error messages and never log request headers. Add a `_safe_error` method that strips sensitive data.
- **Effort:** 10 minutes

### SEC-05: API server binds to 0.0.0.0 with no authentication
- **File:** `integrations/api.py` line 143
- **Severity:** High
- **Description:** The API server listens on all interfaces with no authentication. Any machine on the network can submit arbitrary problems (and test commands -- see SEC-01). This is a compound vulnerability: network-accessible + arbitrary code execution.
- **Fix:** Default to `127.0.0.1`. Add API key authentication middleware. Document that production deployments must use a reverse proxy with auth.
- **Effort:** 20 minutes

---

## 4. Performance Issues

### PERF-01: New aiohttp session per request (repeated from DESIGN-01)
- **File:** All backends
- **Severity:** High
- **Description:** TCP/TLS connection overhead on every LLM call. For vLLM (local, HTTP), this adds ~1-5ms per request. For OpenAI/Anthropic (HTTPS, remote), this adds ~50-200ms per request due to TLS handshake.
- **Fix:** Reuse sessions.
- **Effort:** 20 minutes per backend

### PERF-02: `CodebaseIndex.index()` reads every file synchronously on init
- **File:** `core/codebase_index.py` line 183
- **Severity:** Medium
- **Description:** `index()` is called in `__init__`, reads up to 512KB of every file synchronously. For a large repo with 10K files, this could take several seconds and blocks the calling thread.
- **Fix:** Make `index()` async or run in an executor. Or make it lazy (index on first `select_relevant` call).
- **Effort:** 20 minutes

### PERF-03: `build_context` re-reads files from disk
- **File:** `core/codebase_index.py` lines 398-411
- **Severity:** Medium
- **Description:** `build_context` opens and reads each file from disk again, even though the file was already read during indexing (content_preview has the first 500 chars, but the full content is discarded). The `FileInfo._content_cache` field exists but is never populated.
- **Fix:** Populate `_content_cache` during `_extract_metadata` for files under the token budget. Use it in `build_context` instead of re-reading.
- **Effort:** 10 minutes

### PERF-04: LLM review scoring is sequential per solution
- **File:** `scoring/engine.py` lines 63-109
- **Severity:** Medium
- **Description:** `score_all` iterates over solutions sequentially, calling `await self._llm_review()` for each one. With N=8 solutions, this means 8 sequential LLM calls for review alone. These could be parallelized with `asyncio.gather`.
- **Fix:** Collect all review tasks and run them in parallel.
- **Effort:** 15 minutes

### PERF-05: Scoring in racing mode makes extra LLM call per solution
- **File:** `core/solver.py` lines 480-486
- **Severity:** Medium
- **Description:** In racing mode, each solution is scored individually as it arrives (calling `score_all` with `[sol]`). If `backend` is passed to `score_all`, this triggers an LLM review call for EVERY solution. In racing mode where speed matters, spending time on LLM review before deciding to accept/reject defeats the purpose.
- **Fix:** Use a lightweight scoring mode for racing (syntax + diff only, skip LLM review) and only do full scoring on the final winner.
- **Effort:** 15 minutes

### PERF-06: `AgentMemory` writes to disk on every `remember()` and `recall()`
- **File:** `learning/tracker.py` lines 445, 473
- **Severity:** Low
- **Description:** Every `recall()` call (one per agent per solve) triggers JSON serialization and a disk write. With N=8 agents, that's 8 unnecessary disk writes during a single solve.
- **Fix:** Batch saves. Write once after the solve completes, or use a dirty flag.
- **Effort:** 10 minutes

---

## 5. Missing Features (Quick Wins)

### FEAT-01: Retry with exponential backoff for LLM calls
- **Severity:** High
- **Description:** A single 429/500/timeout should not kill an agent. Add 3 retries with exponential backoff (1s, 2s, 4s) and jitter.
- **Effort:** 20 minutes

### FEAT-02: Structured logging
- **Severity:** Medium
- **Description:** All logging uses `logging.getLogger` with plain text messages. For production monitoring, structured JSON logging (with fields like `agent_idx`, `strategy`, `score`, `latency_ms`) would enable much better observability.
- **Effort:** 30 minutes

### FEAT-03: Rate limiting for API server
- **Severity:** Medium
- **Description:** The API server has no rate limiting. A single client can submit unlimited solve requests, each consuming N LLM calls.
- **Fix:** Add `aiohttp-ratelimiter` middleware or a simple token-bucket per IP.
- **Effort:** 20 minutes

### FEAT-04: Health checks for backends
- **Severity:** Medium
- **Description:** No way to verify a backend is reachable before starting a solve. A health check endpoint probe (e.g., `GET /v1/models` for vLLM/OpenAI) would give fast-fail on misconfiguration.
- **Effort:** 15 minutes

### FEAT-05: Timeout configuration per backend
- **Severity:** Low
- **Description:** Backend timeouts are constructor params but not exposed in config.yaml. Users can't tune timeouts without code changes.
- **Fix:** Wire `timeout` through from the config YAML to backend constructors in `_make_backend`.
- **Effort:** 5 minutes

### FEAT-06: `--dry-run` mode for CLI
- **Severity:** Low
- **Description:** No way to see which strategies would be selected and how many agents would run without actually executing. Useful for debugging strategy selection.
- **Effort:** 15 minutes

---

## 6. Test Coverage Gaps

### TEST-01: No tests for `solve_racing` mode
- **Severity:** High
- **Description:** Racing mode is a complex async flow with cancellation, timeouts, and score thresholds. There are zero tests for it. The `RacingCoordinator`, `CancellableRequest`, and `RacingStats` classes in `streaming.py` are completely untested.
- **Effort:** 45 minutes

### TEST-02: No tests for `solve_decomposed` mode
- **Severity:** High
- **Description:** Decomposed mode has complex dependency-level sorting, per-file generation, merging, and integration verification. No tests exist for any of these code paths.
- **Effort:** 30 minutes

### TEST-03: No tests for `MultiBackend`
- **Severity:** High
- **Description:** The multi-backend router (strategy routing, session affinity, fallback, TTL eviction) is completely untested.
- **Effort:** 30 minutes

### TEST-04: No tests for API server endpoints
- **Severity:** Medium
- **Description:** `integrations/api.py` has zero tests. The `/solve`, `/feedback`, `/health`, and `/stats` endpoints are untested.
- **Effort:** 30 minutes

### TEST-05: No tests for CLI
- **Severity:** Medium
- **Description:** `integrations/cli.py` has zero tests. `_make_backend`, `_load_codebase`, `_print_result` are untested.
- **Effort:** 20 minutes

### TEST-06: No tests for `PrefixManager.load_codebase_auto` with large codebases
- **Severity:** Low
- **Description:** The auto-selection path (smart loading) is tested with small repos only. The boundary condition where `estimated_total > budget` is never tested.
- **Effort:** 10 minutes

### TEST-07: `_build_dependency_levels` circular dependency handling untested
- **Severity:** Medium
- **Description:** The circular dependency fallback in `_build_dependency_levels` (line 807-809) is untested. This is a critical correctness path -- if the LLM returns circular deps, the solver must handle it gracefully.
- **Effort:** 10 minutes

### TEST-08: Scoring engine `_weighted_score` redistribution untested
- **Severity:** Medium
- **Description:** The weight redistribution logic when tests are missing (BUG-02) has no direct unit test. The bug described above would have been caught by a targeted test.
- **Effort:** 10 minutes

---

## Summary

| Category | Critical | High | Medium | Low | Total |
|----------|----------|------|--------|-----|-------|
| Bugs | 1 | 2 | 4 | 1 | 8 |
| Design | 0 | 3 | 2 | 1 | 6 |
| Security | 1 | 2 | 2 | 0 | 5 |
| Performance | 0 | 1 | 4 | 1 | 6 |
| Missing Features | 0 | 1 | 2 | 3 | 6 |
| Test Gaps | 0 | 3 | 3 | 2 | 8 |
| **Total** | **2** | **12** | **17** | **8** | **39** |

### Top 5 Highest Priority Fixes

1. **SEC-01 + SEC-05:** Arbitrary command execution via API (unauthenticated, network-exposed). This is a remote code execution vulnerability. Fix both together.
2. **BUG-01:** `priority` kwarg crashes Anthropic and Ollama backends. Immediate runtime failure for 2 of 4 backends.
3. **SEC-02:** Path traversal in sandbox `_write_code`. Combined with SEC-01, this allows writing arbitrary files on the host.
4. **DESIGN-01 / PERF-01:** Session-per-request. Significant latency penalty on every LLM call.
5. **DESIGN-04 / FEAT-01:** No retry logic. Transient failures kill agents unnecessarily.
