# BCode Swarm Analysis: Pipeline Complexity, Cruft Inventory, and fusen_solver Integration Plan

**Date:** 2026-04-09
**Analyst:** ASI pipeline analysis
**Scope:** `/root/projects/BTask/packages/bcode` (b-harness, b-swarm, b-lab, b-meta) vs `/root/projects/autokernel/fusen_solver`

---

## 1. Pipeline Map

### 1.1 BCode Architecture (4 packages, 32,452 total lines)

```
b-meta (5,971 LOC) --> b-swarm (14,840 LOC) --> b-harness (2,911 LOC)
                              |
                              v
                         b-lab (8,730 LOC)
```

#### b-harness (2,911 LOC) -- LLM Output Reliability Layer

| File | Lines | Role | Complexity |
|------|-------|------|------------|
| lenient-parse.ts | 535 | Recover broken JSON from LLMs (unquoted keys, trailing commas, unclosed strings) | HIGH -- 15+ recovery heuristics |
| coerce.ts | 333 | Schema-based type forcing (string->number, etc.) | MEDIUM -- handles discriminated unions |
| adapter.ts | 306 | Pi-mono integration hooks | MEDIUM -- bridges to external agent framework |
| feedback.ts | 243 | Structured error messages for LLM self-correction | LOW |
| types.ts | 242 | Type definitions | LOW |
| learned-rules.ts | 231 | Coercion rule learning from history | MEDIUM -- tracks which coercions work |
| schema-preprocess.ts | 179 | Schema normalization | MEDIUM |
| discriminator.ts | 182 | Union type resolution | HIGH -- JSON schema edge cases |
| dedup.ts | 174 | Error deduplication | LOW |
| classify.ts | 133 | Error classification | LOW |
| sanitize.ts | 121 | Input sanitization | LOW |
| enforce.ts | 34 | Function-call enforcement | LOW |
| retry.ts | 82 | Backoff computation | LOW |

**Verdict:** Clean, well-factored, zero-dependency. The crown jewel of the stack. `lenient-parse.ts` handles real-world LLM output failures that no other system addresses this thoroughly.

#### b-swarm (14,840 LOC) -- Multi-Agent Execution Engine

| File | Lines | Role | Complexity |
|------|-------|------|------------|
| modes/pipeline.ts | 1,930 | **Main orchestrator** -- decompose, parallel chunks, wave execution, merge, heal, verify | EXTREME -- single function ~800 lines, 20+ interleaved concerns |
| cli.ts | 2,246 | CLI entry point + config parsing | HIGH -- 36 feature flags, model routing, provider config |
| providers.ts | 891 | Multi-provider LLM client (Anthropic, OpenAI, NVIDIA, Ollama, vLLM, SGLang) | HIGH -- 7 providers, rate limiting, key pools, quirk handling |
| decompose.ts | 791 | PRD -> chunk DAG splitting | MEDIUM -- LLM-based decomposition with heuristic fallbacks |
| pipeline-heal.ts | 759 | LLM-based pipeline code self-repair | HIGH -- diagnoses its own infrastructure bugs |
| planner.ts | 633 | Codebase analysis for brownfield projects | MEDIUM -- AST + regex symbol extraction, LSP integration |
| qa-gates.ts | 550 | 7-level progressive QA (compile -> unit -> integration -> e2e -> acceptance -> security -> fuzz) | MEDIUM |
| dense-spec.ts | 414 | PRD compression for token budget | MEDIUM |
| tools/ast-universal.ts | 622 | 11-language AST extraction | HIGH -- regex-based cross-language |
| tools/lsp.ts | 724 | In-process LSP diagnostics | HIGH -- TypeScript LSP integration |
| tools/compile-check.ts | 438 | Multi-language compile checking | MEDIUM |
| memory.ts | 386 | CMS cross-run learning integration | MEDIUM |
| tool-validator.ts | 378 | Runtime tool call validation + repair | MEDIUM |
| self-heal.ts | 357 | Outer self-healing loop (run -> detect -> fix -> rebuild -> retry) | MEDIUM |
| merge.ts | 309 | Chunk output merging with conflict resolution | MEDIUM |
| diagnostics.ts | 260 | Pre/post-wave diagnostic signals | LOW |
| tools/web-*.ts | ~686 | Web search + fetch tools | LOW |
| tools/grep,find,ls,diff | ~781 | Filesystem tools | LOW |
| chunk-context.ts | 194 | Context budget assembly per chunk | LOW |
| workspace.ts | 176 | Git-based workspace management | LOW |
| normalizer.ts | 134 | Import sorting, dedup, style enforcement | LOW |
| safe-exec.ts | 110 | Sandboxed subprocess execution | LOW |

**Verdict:** `pipeline.ts` is the critical file -- a 1,930-line function that has accumulated wave execution, checkpoint/resume, type inference, AST surgical fixes, brownfield support, convention extraction, diagnostic signals, and 20+ feature-flag-gated code paths. It works but is the primary source of accumulated complexity.

#### b-lab (8,730 LOC) -- Inner Pipeline + Benchmarking

| File | Lines | Role | Complexity |
|------|-------|------|------------|
| pipeline-v3.ts | 2,590 | **Inner pipeline**: spec -> generate -> compile -> test -> heal -> score | HIGH -- the actual LLM code generation |
| benchmark.ts | 1,052 | Multi-PRD benchmarking with scoring | MEDIUM |
| prompt-modules.ts | 767 | Modular prompt assembly system | MEDIUM |
| architect-pipeline.ts | 655 | Architecture planning phase | MEDIUM |
| cli.ts | 564 | Lab CLI | LOW |
| language-adapter.ts | 511 | 29-language adapter system (reads JSON configs) | MEDIUM |
| judge-panel.ts | 469 | Multi-judge scoring panel | MEDIUM |
| runner.ts | 359 | Experiment runner | LOW |
| introspect.ts | 355 | Self-introspection engine | MEDIUM |
| pipeline-memory.ts | 324 | Per-pipeline CMS memory | LOW |
| ast.ts | 231 | AST parsing utilities | LOW |
| experiments.ts | 231 | Experiment definitions | LOW |

**Verdict:** Well-structured. The 29-language adapter system (JSON configs, not hardcoded) is excellent engineering. `pipeline-v3.ts` is large but handles a legitimately complex flow.

#### b-meta (5,971 LOC) -- Self-Improving Meta-Orchestration

| File | Lines | Role | Complexity |
|------|-------|------|------------|
| meta-loop.ts | 894 | Main meta-loop: clarify -> decompose -> execute -> measure -> analyze -> evolve -> repeat | HIGH |
| introspect.ts | 848 | Blind spot detection across 8 categories | HIGH |
| cli.ts | 828 | Meta CLI | MEDIUM |
| strategy.ts | 525 | Strategy representation with 6 dimensions | MEDIUM |
| pipeline-memory.ts | 324 | Strategy memory | LOW |
| reporting.ts | 316 | Multi-format report generation | LOW |
| failure-taxonomy.ts | 294 | Failure classification learning | MEDIUM |
| measure.ts | 287 | Multi-metric measurement | LOW |
| evolve-prompts.ts | 273 | LLM prompt evolution | MEDIUM |
| role-generator.ts | 259 | Dynamic role creation from failure patterns | MEDIUM |
| analyze.ts | 239 | Root cause analysis | LOW |
| strategy-memory.ts | 222 | Cross-project strategy index | LOW |
| clarify.ts | 170 | Goal clarification via LLM | LOW |
| loop-mutation.ts | 139 | Meta-loop structure variation | LOW |
| evolve.ts | 135 | Strategy mutation with Bayesian exploration | LOW |
| integration.ts | 116 | Cross-module integration checking | LOW |
| questions.ts | 111 | Question value tracking | LOW |
| teams.ts | 102 | Team formation | LOW |
| goal-spec.ts | 121 | Goal specification schema | LOW |

**Verdict:** Ambitious and largely clean. The introspection system (8 blind-spot categories, convergence detection) is sophisticated. Strategy evolution uses proper exploration/exploitation tradeoffs.

### 1.2 fusen_solver Architecture (5,429 LOC)

| File | Lines | Role | Complexity |
|------|-------|------|------------|
| core/solver.py | 1,261 | Main orchestrator: isolated, collaborative, decomposed, racing modes | MEDIUM |
| core/codebase_index.py | 515 | Multi-signal file relevance scoring | LOW |
| learning/tracker.py | 501 | Bayesian strategy weight adaptation + AgentMemory | MEDIUM |
| integrations/cli.py | 357 | CLI with streaming | LOW |
| prefix_manager.py | 284 | KV cache prefix management for vLLM | LOW |
| scoring/engine.py | 269 | 5-signal weighted scoring (syntax, tests, review, diff, confidence) | LOW |
| strategies/presets.py | 208 | 12 named strategies + problem-type presets | LOW |
| scoring/sandbox.py | 189 | Docker-based test sandbox | LOW |
| backends/*.py | 756 | 5 LLM backends (OpenAI, Anthropic, Ollama, vLLM, multi) | LOW |
| streaming.py | 178 | Token-by-token streaming | LOW |
| config.py | 170 | Configuration management | LOW |
| core/interfaces.py | 156 | Clean abstract interfaces (Problem, Solution, Strategy, LLMBackend, PlatformPlugin) | LOW |
| core/incremental_context.py | 147 | Incremental context for follow-up solves | LOW |
| strategies/engine.py | 139 | Strategy selection with weight-based sampling | LOW |
| core/priority.py | 41 | Priority computation | LOW |

**Verdict:** Clean, minimal, well-abstracted. 3x less code than bcode for comparable problem scope. The interface design (Problem/Solution/Strategy/LLMBackend/PlatformPlugin) is textbook-quality. But it lacks bcode's battle-tested production complexity (29 languages, checkpoint/resume, wave execution, AST surgical fixes, etc.).

---

## 2. Cruft Inventory

### 2.1 Accumulated Complexity in pipeline.ts

The `executePipeline` function in `b-swarm/src/modes/pipeline.ts` is 1,930 lines and exhibits classic "successful system syndrome" -- it works, but has accumulated:

1. **Inline timing instrumentation** (lines 114-128) -- manual timing accumulator with `timed()` wrapper interleaved with every operation.

2. **Dynamic import gymnastics** (lines 26-38, 85-104, 151-173) -- Three separate `try { import() } catch { try { import(alternative) } catch {} }` blocks because b-lab can be loaded via npm link OR relative path. The `getBlab()` function caches the import but the detection logic is duplicated.

3. **Language detection duplication** (lines 147-173) -- Detects language from PRD in pipeline.ts, again in decompose.ts, again in chunk-context.ts. Each has its own fallback chain.

4. **Checkpoint/resume state management** (lines 56-82, 136-140, 378-384, 936-943) -- Checkpoint saving/loading scattered across 4+ locations in the file instead of a unified checkpoint manager.

5. **Feature-flag spaghetti** -- At least 8 features gated by `(config as Record<string, unknown>).featureName === true/false`: denseSpecs, normalize, noHeal, astFix, skipTests, denseOutput, qaLevel. Each adds a conditional branch.

6. **Convention extraction runs twice** (lines 348-363 and 957-982) -- Before first wave and after first wave, with nearly identical code.

7. **copyScaffoldFiles hardcoded fallback** (lines 1878-1923) -- Lists 30+ manifest files across 20 languages as a "broad fallback" because the function is synchronous and cannot async-import the adapter. Comment at line 1883 acknowledges this.

8. **Merge threshold magic numbers** -- Line 700: `const mergeThreshold = t.waves.length <= 3 ? 50 : 70;` -- adaptive threshold with no derivation. Line 753: inconsistent `< 30` check (compare `< 30` vs threshold `50/70`).

9. **Three separate AST fix flows** -- Wave-level AST fix (lines 851-914), final-verify AST fix (lines 1247-1301), and chunk-level healChunk() AST fix. All three do the same pattern (parse -> find node -> fix -> compile check) but with slightly different error handling.

### 2.2 TypeScript-Specific Workarounds

1. **`as Record<string, unknown>` casts** -- Used 15+ times in pipeline.ts to access config properties that should be typed. The SwarmConfig interface has `[key: string]: unknown` escape hatch (types.ts line 33).

2. **Dynamic import type assertions** -- `await import("@btask/b-lab" as string) as typeof _blab` -- the `as string` silences TypeScript's module resolution, and the type cast is manual.

3. **Synchronous constraint workarounds** -- `copyScaffoldFiles` cannot be async (called from sync context), so it cannot read adapter JSON files and must use a hardcoded fallback list.

### 2.3 1-Off Fixes That Became Permanent

1. **Test file stripping** (lines 570-586, 1106-1128) -- Two separate blocks that delete test files the model generated "despite being told not to." The comment at line 1106: "This is infrastructure defense -- don't blame the model, remove the files."

2. **Brownfield file deduplication** (lines 700-722) -- Reads every merged file, compares content byte-by-byte against workspace original, deletes if identical. Needed because the merge system copies everything.

3. **Auto-install missing npm packages** (lines 811-828) -- Parses `Cannot find module` from compile errors, extracts package names, runs `npm install --save-dev`. This is a workaround for incomplete scaffold setup.

4. **Unclosed brace repair** (line 237) -- After LLM generates type stubs, counts `{` vs `}` and appends missing `}`. One line, but reveals the fragility of LLM output parsing.

5. **`void 0;` placeholder** (line 417) -- Literal placeholder comment `void 0; // placeholder` in the contract chunk handling logic.

### 2.4 Self-Healing Patterns Known to Be Fragile

From `self-heal.ts`, the KNOWN_PATTERNS array (lines 52-133) includes:
- `test-files-in-skipTests-chunk` -- uses `find | xargs rm -f` shell commands
- `index-ts-barrel-errors` -- deletes `src/index.ts` when it has TS1131 errors
- `ENOENT-copyfile-crash` -- detected but fix says "pipeline code needs fix"
- `brownfield-stale-files` -- deletes specific hardcoded files (`src/index.ts`, `src/orchestrator.ts`)
- `missing-scaffold` -- runs `npm install` as a fix

These are pattern-matched against raw console output strings, not structured error objects.

---

## 3. Comparison Matrix: BCode vs fusen_solver

| Capability | BCode | fusen_solver | Winner |
|-----------|-------|-------------|--------|
| **LOC** | 32,452 | 5,429 | fusen_solver (6x smaller) |
| **Languages supported** | 29 (JSON adapter system) | 10+ (regex-based) | BCode |
| **LLM providers** | 7 (Anthropic, OpenAI, NVIDIA, OpenRouter, Ollama, vLLM, SGLang) | 5 (OpenAI, Anthropic, Ollama, vLLM, multi) | BCode |
| **Solve modes** | 1 (pipeline, formerly 5) | 4 (isolated, collaborative, decomposed, racing) | fusen_solver |
| **Parallel execution** | Wave-based DAG with worktrees | asyncio.gather per round | Tie (different granularity) |
| **Strategy selection** | Fixed pipeline (empirically tuned) | Bayesian weight adaptation from history | fusen_solver |
| **Scoring** | Per-chunk score (compile + test binary) | 5-signal weighted (syntax, tests, review, diff, confidence) | fusen_solver |
| **Learning** | CMS memory (episodes, dead-ends, playbooks) | JSON-based win/loss tracking + AgentMemory | Tie (BCode richer, fusen_solver cleaner) |
| **Self-healing** | 3-level (chunk heal, wave AST fix, outer self-heal loop) | None built-in | BCode |
| **LLM output recovery** | b-harness: 15+ lenient parse heuristics, type coercion, discriminated unions | Basic JSON extraction | BCode (dramatically) |
| **Codebase analysis** | AST + regex + LSP diagnostics + keyword scoring | Multi-signal relevance scoring (keyword, import graph, recency, type hints) | Tie |
| **Meta-optimization** | b-meta: introspection, strategy evolution, failure taxonomy, prompt evolution, loop mutation | LearningEngine: Bayesian weights, mode suggestion | BCode |
| **Factorial experimentation** | 40-run Plackett-Burman with 37 features | None | BCode |
| **Checkpoint/resume** | Yes (per-wave checkpoint) | No | BCode |
| **Interface design** | Implicit contracts via TypeScript types | Explicit ABC interfaces (Problem, Solution, Strategy, LLMBackend, PlatformPlugin) | fusen_solver |
| **Testability** | Good (9 + 7 + 8 test files) | Good (3 test files but clean interfaces) | BCode |
| **Docker sandbox** | Per-chunk workspace isolation | Optional Docker test sandbox | Tie |
| **Dense spec compression** | Yes (PRD -> compressed spec with edge cases + dependency DAG) | No | BCode |
| **Brownfield support** | Full (plan, modify, preserve, already-implemented detection) | Context-based (file selection) | BCode |
| **Configuration** | 36 feature flags + per-role model routing | Minimal (n, max_tokens, strategies) | BCode (more, possibly too much) |

---

## 4. Refactoring Recommendation

### 4.1 What to KEEP from BCode

1. **b-harness entire package** -- It is perfect. Zero dependencies, battle-tested LLM output recovery. fusen_solver has nothing comparable.

2. **29-language adapter system** (b-lab/adapters/*.json) -- Data-driven language support is the correct architecture. Each language is a JSON config (compileCmd, testCmd, sourceExt, scaffoldInit, etc.), not hardcoded logic.

3. **Wave-based execution with checkpoint/resume** -- For large PRDs, decomposing into chunks and running in dependency-ordered waves is essential. The checkpoint/resume system saves enormous time (110s per resumed chunk).

4. **Self-healing hierarchy** (chunk-level -> wave-level AST fix -> outer heal loop) -- The concept is right even if the implementation has accumulated cruft. The data shows it works: scores improve by 20-40 points after healing.

5. **Factorial experimentation infrastructure** -- The Plackett-Burman design with 37 features and 40 runs is real science. The findings (e.g., `continuousCompile` hurts by -6.4%, `earlyTermination` helps by +5.5%) are valuable and replicable.

6. **CMS cross-run learning** -- Episodes, dead-ends, playbooks, and package API caching across runs is the right approach for long-lived projects.

7. **Dense spec extraction** -- Compressing large PRDs while preserving contracts, edge cases, and dependency DAGs is a good solution to context-window limits.

### 4.2 What to REPLACE with fusen_solver Concepts

1. **Replace implicit strategy with explicit Strategy objects** -- BCode's pipeline has one hardcoded flow. fusen_solver's Strategy dataclass (name, prompt, weight, temperature, tags) enables mixing approaches. The BCode pipeline should select strategies per-chunk, not use the same approach for everything.

2. **Replace binary scoring with multi-signal scoring** -- BCode scores chunks as "compile pass + test pass = 100." fusen_solver's 5-signal weighted scoring (syntax 10%, tests 40%, review 30%, diff 15%, confidence 5%) gives gradient information. A chunk that compiles and has clean code but fails one test shouldn't score the same as one that crashes.

3. **Replace fixed solve mode with data-driven mode selection** -- BCode collapsed from 5 modes to 1 (pipeline). fusen_solver's `suggest_mode()` uses historical acceptance rates to route between isolated/collaborative/decomposed. BCode should bring back modes but let data decide.

4. **Replace scattered timing with fusen_solver's clean SolveResult** -- fusen_solver's `SolveResult` dataclass has `total_time_s`, `generation_time_s`, `scoring_time_s` as first-class fields. BCode's `t = { llmMs: 0, decomposeMs: 0, ...}` accumulator is fragile.

5. **Replace hardcoded KNOWN_PATTERNS with LearningEngine** -- The self-heal patterns in self-heal.ts are regex-matched against console output. fusen_solver's LearningEngine tracks success/failure patterns with Bayesian confidence. Heal patterns should be learned, not hardcoded.

6. **Replace `as Record<string, unknown>` casts with proper interfaces** -- fusen_solver's Problem/Solution/Strategy interfaces are explicitly typed. BCode's SwarmConfig escape hatch `[key: string]: unknown` should be replaced with proper discriminated config types.

### 4.3 What to ADD (neither has)

1. **Structured error taxonomy with remediation playbooks** -- Both systems detect errors but lack a structured taxonomy that maps error patterns to remediation strategies with success probabilities. Example: `TS2305 "Module has no exported member"` -> `remediation: add export to upstream file, P(success)=0.85`.

2. **Token budget optimizer** -- Neither system optimally allocates tokens across chunks. Given a total budget of N tokens, what is the optimal allocation per chunk considering chunk difficulty, dependency criticality, and historical token usage? This is a constrained optimization problem.

3. **A/B testing framework for pipeline changes** -- BCode has factorial experiments but they are manual. An automated system that forks the pipeline, runs both versions on the same PRD, and records which wins would accelerate improvement.

4. **Incremental compilation cache** -- Both systems recompile from scratch on every change. A cache that tracks which files changed and only recompiles affected modules would save 30-50% of compile time.

5. **LLM output confidence calibration** -- Neither system calibrates the model's self-reported confidence against actual outcomes. A calibration layer that learns "when this model says confidence=0.8, it actually succeeds 0.6 of the time" would improve scoring accuracy.

6. **Deterministic replay** -- Neither system can replay a failed run with the same random seeds, model states, and inputs. This makes debugging intermittent failures extremely hard.

---

## 5. Integration Plan

### Phase 1: Extract + Clean (1-2 days)

1. Extract `pipeline.ts` into 5 focused modules:
   - `pipeline-orchestrator.ts` -- wave execution loop (200 lines)
   - `pipeline-checkpoint.ts` -- checkpoint/resume management (150 lines)
   - `pipeline-scaffold.ts` -- language scaffold + dependency setup (200 lines)
   - `pipeline-heal.ts` -- unified AST fix + chunk heal (300 lines, merge 3 current flows)
   - `pipeline-verify.ts` -- final compilation, test, QA (200 lines)

2. Type the config properly -- replace `[key: string]: unknown` with discriminated union of feature configs.

3. Deduplicate language detection -- single `detectLanguage(prd, workDir)` call at pipeline start, passed down.

### Phase 2: Integrate fusen_solver Scoring (1 day)

1. Port fusen_solver's `ScoringEngine` to TypeScript.
2. Replace binary chunk scoring with 5-signal weighted scoring.
3. Add gradient-based healing decisions: score 0.7+ = targeted fix, 0.4-0.7 = retry, <0.4 = skip.

### Phase 3: Integrate fusen_solver Strategy Selection (1 day)

1. Port fusen_solver's `StrategyEngine` and `LearningEngine` to TypeScript.
2. Create BCode-specific strategies: `generate_and_compile`, `type_first`, `test_first`, `decompose_further`, `copy_pattern`.
3. Per-chunk strategy selection based on chunk characteristics (has tests? has types? brownfield? greenfield?).
4. Record wins/losses to CMS, feed back into weight adaptation.

### Phase 4: Restore Multi-Mode with Data-Driven Selection (1 day)

1. Implement `suggest_mode()` in TypeScript using CMS history.
2. Modes: `pipeline` (current), `shotgun` (N parallel full-solution), `racing` (all approaches, first to pass wins).
3. Auto-select based on PRD size, historical success rates, available compute.

### Phase 5: Unified LLM Backend Abstraction (0.5 days)

1. Port fusen_solver's `LLMBackend` ABC to TypeScript interface.
2. Wrap existing providers.ts behind it.
3. Add `supports_batch`, `max_context` properties for smarter resource allocation.

---

## 6. The Ideal ASI Coding Platform

Combining the best of both, the ideal system would have:

### Layer 0: LLM Output Reliability (b-harness)
- Lenient JSON parsing with 15+ recovery heuristics
- Schema-based type coercion with discriminated union support
- Structured error feedback for self-correction
- Learned coercion rules from history

### Layer 1: Problem Representation (fusen_solver interfaces)
- `Problem(description, context, problem_type, constraints, tests, language, solve_mode, priority)`
- `Solution(code, explanation, strategy_used, score, subscores)`
- `Strategy(name, prompt, weight, temperature, tags)`
- All data-driven, no hardcoded behavior

### Layer 2: Codebase Intelligence (hybrid)
- fusen_solver's multi-signal file relevance scoring (keyword, import graph, recency, type hints)
- BCode's AST + LSP diagnostic integration for TypeScript
- BCode's 29-language adapter system for compile/test/scaffold
- Incremental context for follow-up solves (fusen_solver's IncrementalContext)

### Layer 3: Strategy Engine (fusen_solver + BCode strategies)
- 12 base strategies (direct, alternative, test_first, decompose, review, research, rewrite, adversarial, prototype_then_refine, incremental, profile_first, security)
- Problem-type presets with Bayesian weight adaptation
- BCode's collaborative roles (analyst, researcher, test_writer, coder_a, coder_b, reviewer, merger)
- Racing mode: all strategies in parallel, first to pass wins

### Layer 4: Execution Engine (BCode pipeline, cleaned)
- Wave-based DAG execution for large PRDs
- Checkpoint/resume for crash recovery
- Per-chunk workspace isolation with brownfield support
- Dense spec compression for context-window management
- Convention extraction from generated code

### Layer 5: Scoring Engine (fusen_solver, extended)
- 5-signal weighted scoring (syntax, tests, review, diff, confidence)
- Calibrated confidence (learned mapping from model confidence to actual success)
- Gradient-based healing decisions (not binary pass/fail)

### Layer 6: Self-Healing (BCode, unified)
- Single AST surgical fix flow (not 3 duplicated versions)
- Learned remediation playbooks (not hardcoded KNOWN_PATTERNS)
- Escalation: targeted fix -> retry -> re-decompose -> skip

### Layer 7: Meta-Optimization (b-meta + fusen_solver learning)
- Strategy evolution with exploration/exploitation (b-meta)
- Failure taxonomy learning (b-meta)
- Prompt evolution for underperforming roles (b-meta)
- Bayesian win/loss tracking for strategy weights (fusen_solver)
- Cross-project strategy warm-starting (b-meta)
- Loop structure mutation (b-meta)

### Layer 8: Measurement + Science (BCode)
- Factorial experiment infrastructure
- Feature policy derivation from data
- Timing breakdown for bottleneck diagnosis
- Quality regression detection (dense mode vs baseline)

---

## 7. Effort Estimate (ASI-calibrated)

| Phase | Description | Effort | Risk |
|-------|-------------|--------|------|
| Phase 1 | Extract pipeline.ts into 5 modules | 2 days | LOW -- purely structural, no behavior change |
| Phase 2 | Integrate fusen_solver scoring | 1 day | LOW -- additive, existing scoring still works |
| Phase 3 | Integrate fusen_solver strategies | 1 day | MEDIUM -- strategy selection changes behavior |
| Phase 4 | Restore multi-mode with data selection | 1 day | MEDIUM -- modes interact with pipeline |
| Phase 5 | Unified LLM backend abstraction | 0.5 days | LOW -- wrapper around existing code |
| **Total** | | **5.5 days** | |

"ASI-calibrated" means: an agent system (Claude Code, Cursor, etc.) doing the work with human review. A human developer would be 3-4x slower. A junior would be 10x slower.

---

## 8. Risk Assessment

### HIGH Risk

1. **pipeline.ts extraction breaks subtle interactions** -- The 1,930-line function has implicit state sharing (e.g., `conventionsContext` is mutated after wave 0 and read by subsequent waves). Extracting to separate modules requires explicit state passing, which may miss edge cases.

2. **Multi-signal scoring changes heal decisions** -- Current healing triggers on binary compile pass/fail. Switching to gradient scoring (0.0-1.0) changes WHEN healing kicks in. If the threshold is wrong, either too much healing (wasted tokens) or too little (broken output shipped).

3. **Strategy diversity introduces variance** -- BCode's single pipeline mode is deterministic for a given PRD + model. Adding strategy selection introduces variance in outputs. Users who rely on consistent results may be surprised.

### MEDIUM Risk

4. **CMS schema migration** -- fusen_solver's history.json format differs from BCode's CMS JSONL format. Merging learning systems requires schema alignment or a translation layer.

5. **TypeScript port of Python abstractions** -- fusen_solver's Strategy/Problem/Solution dataclasses map cleanly to TypeScript interfaces, but Python's `@dataclass` field defaults and TypeScript's class semantics differ in subtle ways.

6. **Regression in factorial-validated features** -- The 40-run Plackett-Burman experiment validated the current pipeline. Changes to scoring, strategies, or modes invalidate those results. A new factorial run is needed after integration.

### LOW Risk

7. **b-harness compatibility** -- b-harness is a standalone package with no dependencies on b-swarm internals. It will not be affected by any refactoring.

8. **Language adapter compatibility** -- The 29 JSON adapter files are data, not code. They are read by whatever system replaces the current pipeline.

---

## Appendix: Key File Paths

### BCode
- Pipeline orchestrator: `/root/projects/BTask/packages/bcode/b-swarm/src/modes/pipeline.ts`
- LLM output harness: `/root/projects/BTask/packages/bcode/b-harness/src/`
- Language adapters: `/root/projects/BTask/packages/bcode/b-lab/adapters/`
- Inner pipeline: `/root/projects/BTask/packages/bcode/b-lab/src/pipeline-v3.ts`
- Meta-loop: `/root/projects/BTask/packages/bcode/b-meta/src/meta-loop.ts`
- Strategy evolution: `/root/projects/BTask/packages/bcode/b-meta/src/evolve.ts`
- Factorial results: `/root/projects/BTask/packages/bcode/docs/factorial-results-2026-04-01.md`
- Lessons learned: `/root/projects/BTask/packages/bcode/b-swarm/LESSONS.md`
- Design decisions: `/root/projects/BTask/packages/bcode/docs/design-decisions.md`

### fusen_solver
- Main solver: `/root/projects/autokernel/fusen_solver/core/solver.py`
- Interfaces: `/root/projects/autokernel/fusen_solver/core/interfaces.py`
- Strategy presets: `/root/projects/autokernel/fusen_solver/strategies/presets.py`
- Learning engine: `/root/projects/autokernel/fusen_solver/learning/tracker.py`
- Scoring engine: `/root/projects/autokernel/fusen_solver/scoring/engine.py`
- Codebase index: `/root/projects/autokernel/fusen_solver/core/codebase_index.py`
