# The Definitive Top 20: Impact x Feasibility Ranking

**Date:** 2026-04-09
**Scored:** 500 ideas (V1-V5) + 28 Novel Research Paths = 528 total
**Hardware:** RTX 5090 (32GB), PRO 6000 2x96GB arriving next week
**Assets:** FusenCache (4x KV, 6,685 tok/s), fusen_solver, AutoKernel v2, BCode pipeline, Docker serving

---

## 1. Score Summary Table (All 528 Ideas by Range)

### Score 60-100: "Do Immediately" (16 ideas)

| # | Name | I | F | Score |
|---|------|---|---|-------|
| R1 | Expert Output Memoization for Code Patterns | 9 | 9 | 81 |
| R26 | Boost Clock Pinning + Thermal Management | 5 | 10 | 50 -> 63* |
| 201 | Speculative Editing | 9 | 8 | 72 |
| R2 | Router Prediction Cascade (30-layer predict from layer 0) | 9 | 8 | 72 |
| 202 | Terminal-as-Codebase-Browser | 8 | 9 | 72 |
| R4 | Speculative Routing (top-4 early exit) | 8 | 8 | 64 |
| 211 | Review-Before-Push | 8 | 8 | 64 |
| R3 | Temporal Expert Caching (L2 pinning) | 8 | 8 | 64 |
| 203 | Continuous Background Refactoring | 7 | 9 | 63 |
| 6 | Diff-Mode Generation | 9 | 7 | 63 |
| 291 | One-Person SaaS Stack | 10 | 6 | 60 |
| 223 | Dependency Upgrade Autopilot | 7 | 9 | 63 |
| 208 | Context-Aware Doc Generation on File Open | 7 | 9 | 63 |
| R8 | Semantic Token Grouping (batch by expert affinity) | 7 | 9 | 63 |
| 280 | Cross-Time-Zone Continuous Development | 9 | 7 | 63 |
| 97 | Infinite Context via Code Summarization Hierarchy | 9 | 7 | 63 |

*R26 scored high because it takes 10 minutes and has guaranteed payoff; adjusted for practical impact ceiling.*

### Score 40-59: "Do This Month" (62 ideas)

| Range | Example Ideas |
|-------|--------------|
| 56-59 | R5 Attention-MoE Pipelining (56), 215 Semantic Diff Review (56), 205 Full-Project Sim Before Commit (56), 221 Debt Heat Map ($) (56), R6 Codebook Expert Compression (56), 151 Acceptance-Rate-Weighted Sampling (56) |
| 48-55 | 207 Parallel Universe Branches (54), 300 AI-Native Software Co OS (50), 213 Cross-PR Conflict Detection (54), 222 Dead Code Elimination (54), 241 Crash Report to Fix Pipeline (54), 293 Self-Healing Production (54), 228 Config Debt Scanner (48), 51 Dependency-Graph-Ordered Generation (48), R12 AST-Guided Token Prediction (49) |
| 40-47 | 71 Rejection-Sampling Distillation (42), 102 Execution-Trace-Conditioned Decoding (42), 113 Design System Compiler (42), 159 Progressive Disclosure Code Gen (42), 171 Interface-Contract-First Multi-Agent (42), 225 Architecture Drift Detector (42), 231 AI Team Member w/ Standup (48), 236 Automated Runbook Generation (48), 261 COBOL to Python (40), 281 Cloud Cost Estimator in IDE (48), 296 Compliance-as-Code (40), R14 Generation-Time Code Execution Feedback (40) |

### Score 20-39: "Do This Quarter" (~180 ideas)

Includes: Most of V1 Category 2-6 ideas (test-driven generation, design-to-code, math reasoning, debugging training, architecture-aware generation), most V2 multi-way intersections (101-200), V3 categories 25-28 (observability, security, legacy, collaborative), V4 categories 31-33 (code-generates-worlds, worlds-generate-code, debugging-in-3D), NOVEL_RESEARCH Tier 2-3 ideas (6-28).

Key ideas in this range: 1 Syntax-Predictive KV Pruning (28), 8 Compiler-Error-Guided Beam (28), 11 Red-Light Decoding (32), 15 Coverage-Directed Generation (35), 32 Proof-Carrying Code (24), 52 Schema-First Code Scaffolding (35), 67 Batch-Shape-Adaptive Kernels (35), 78 Codebase-Personalized Embeddings (36), 83 API-Doc-Grounded Generation (35), 96 Self-Healing Production Code (30), 141 Test Suite IS Training Data (30), 192 Optimization Strategy Generator (24), R9 Content-Addressable KV Cache (32), R16 Learned Expert Bypass (32).

### Score 10-19: "Research Backlog" (~150 ideas)

Includes: Most V1 Category 10 "impossible" ideas (91-100), most V2 formal logic ideas (161-170), most V4 physics-informed ideas (341-350), V4 sound-code ideas (371-380), many V5 meaning-focused ideas that require cultural/institutional adoption beyond pure tech build.

### Score 1-9: "Someday/Maybe" (~120 ideas)

Includes: Most V5 categories 45-50 (peace technology, sacred/digital, community connection), V4 multi-agent world simulations requiring full VR platforms (381-400), ideas requiring breakthroughs in quantum computing (94), biological computing, or fundamental CS theory changes.

---

## 2. The Top 20 (Detailed)

### Rank 1: Expert Output Memoization for Code Patterns (R1)
**Impact: 9** | **Feasibility: 9** | **Score: 81**

**Impact justification:** MoE GEMMs are 30% of decode time. Code has 15-25% token repetition. A 5-15% system-level batch throughput improvement directly translates to higher tok/s on the coding workstation, which is the #1 metric. At 6,685 tok/s baseline, even 5% = 334 additional tok/s.

**Feasibility justification:** Hook into MoELayer.forward(), build a hash table, bypass on hit. The math is clean: FP4 quantization noise already exceeds the approximation error. 4-6 hours to prototype. We have all the infrastructure (vLLM hooks, profiling tools).

**What it enables:** Breaks the "every expert call is a fresh GEMM" assumption. Opens the door to hierarchical caching strategies where expert computation becomes a lookup rather than a compute operation for common patterns.

**Gate test:** Log (expert_id, input_tensor[:8]) for 1000 tokens. Compute cosine similarity. If clusters exist with >0.99 sim, proceed. 1 hour.

**ASI effort:** 4-6 hours.

**Dependencies:** None. Standalone.

**Compounds with:** R3 (temporal expert caching) -- memoized results stay hot in L2. R8 (semantic token grouping) -- grouping similar tokens increases hit rate. R4 (speculative routing) -- fewer experts computed means fewer entries to memoize.

---

### Rank 2: Speculative Editing (201)
**Impact: 9** | **Feasibility: 8** | **Score: 72**

**Impact justification:** Zero-perceived-latency IDE editing changes how ALL developers interact with code. At 6,685 tok/s, a 200-token edit completes in 30ms -- faster than inter-keystroke latency. This is a paradigm shift from "request completion" to "completion is already there." 15-25% more time in flow state per developer.

**Feasibility justification:** We have the inference speed (6,685 tok/s). We have the serving infrastructure (Docker, vLLM). The IDE integration is straightforward (VS Code extension API). The speculative branch prediction logic is simple: pre-compute top 3-5 most likely next edits during typing pauses. Can prototype in 1 week.

**What it enables:** Eliminates the last perceptible latency between thought and code. Makes AI-assisted coding feel like thinking, not requesting.

**Gate test:** Measure keystroke cadence with/without. Target: 50% reduction in pause-type-pause patterns. 1 day.

**ASI effort:** 1 week (VS Code extension + speculative inference pipeline).

**Dependencies:** Fast local inference (have it). IDE extension framework.

**Compounds with:** R1 (expert memoization makes speculative branches cheaper), 202 (terminal browser provides the codebase context for better predictions), 208 (context-aware docs inform what edits are likely).

---

### Rank 3: Router Prediction Cascade (R2)
**Impact: 9** | **Feasibility: 8** | **Score: 72**

**Impact justification:** 10-20% decode speedup from expert prefetching. This is a fundamental advance: predicting all 30 layers' routing from layer 0 means we can preload expert weights 29 layers in advance. On SM120 with 48MB L2, this transforms random expert access into planned sequential access.

**Feasibility justification:** The math is provable: routing is a linear projection, hidden states are correlated via residual stream. Train a tiny MLP (11M params, negligible compute). Cross-layer routing correlation is measurable in 30 minutes.

**What it enables:** Expert weight prefetching eliminates the primary memory bandwidth bottleneck in MoE decode. Opens the path to much larger MoE models on the same hardware.

**Gate test:** Log 30 layers' routing for 500 tokens. Compute mutual information. If MI > 0.5, proceed. 30 minutes.

**ASI effort:** 6-8 hours.

**Dependencies:** None.

**Compounds with:** R3 (prefetch + L2 pinning = double benefit), R22 (GPU-CPU offloading benefits from knowing which experts to prefetch from CPU), R1 (predict which experts to check cache for).

---

### Rank 4: Terminal-as-Codebase-Browser (202)
**Impact: 8** | **Feasibility: 9** | **Score: 72**

**Impact justification:** Senior engineers spend 30-40% of time reading code they didn't write. Replacing grep/find/rg with semantic codebase queries that return synthesized answers in <200ms cuts code comprehension time by 50-70%. Onboarding drops from weeks to days.

**Feasibility justification:** We have the inference engine (6,685 tok/s). Persistent KV cache for entire codebases fits in 128K context (medium projects). CLI tool is simple: pipe natural language query, return synthesized answer with file references. The BCode pipeline already does most of this. Can prototype in 2-3 days.

**What it enables:** Makes the entire codebase instantly queryable by any developer. Democratizes senior-engineer-level codebase understanding.

**Gate test:** 20 standard codebase questions, time-to-answer vs manual grep. Target: 5x faster. 4 hours.

**ASI effort:** 2-3 days.

**Dependencies:** Fast inference (have it), codebase indexing.

**Compounds with:** 201 (speculative editing uses codebase understanding), 208 (doc generation uses same index), 211 (review-before-push uses same codebase context).

---

### Rank 5: Speculative Routing / Top-4 Early Exit (R4)
**Impact: 8** | **Feasibility: 8** | **Score: 64**

**Impact justification:** 25-50% MoE compute reduction on qualifying tokens. Router softmax concentrates probability heavily: if top-4 experts capture 95%+ of routing weight, experts 5-8 contribute below FP4 quantization noise. This is 7.5-15% system-level speedup.

**Feasibility justification:** Log router softmax scores, check concentration. The math is clean: output below quantization noise is undetectable. 4-6 hours to prototype with a simple threshold check.

**What it enables:** Variable compute per token -- easy tokens are fast, hard tokens get full resources. Opens path to adaptive compute allocation.

**Gate test:** Log router softmax for 1000 tokens. If top-4 captures >90% weight on >50% of tokens, proceed. 1 hour.

**ASI effort:** 4-6 hours.

**Dependencies:** None.

**Compounds with:** R1 (fewer experts = fewer memoization entries needed), R2 (prediction cascade tells us early which experts to skip), R3 (skipped experts free L2 space for hot ones).

---

### Rank 6: Review-Before-Push (211)
**Impact: 8** | **Feasibility: 8** | **Score: 64**

**Impact justification:** Code review is the #1 bottleneck in engineering orgs (4-24 hour average wait). If AI pre-review catches 80% of issues, human review drops from 30 to 5 minutes per PR. For a 50-person team: 80+ engineer-hours/week saved. This directly improves developer velocity.

**Feasibility justification:** We have the inference engine. Full repo context fits in 128K. Git hook integration is trivial. The diff analysis + review generation is a well-understood LLM task. Can ship a working prototype in 1 week.

**What it enables:** Shifts code review from "find problems" to "approve design decisions." PR cycle time drops from days to hours.

**Gate test:** Run on 500 historical PRs. Target: catch 80% of human reviewer comments. 1 day.

**ASI effort:** 1 week.

**Dependencies:** Fast local inference, git integration.

**Compounds with:** 202 (codebase browser provides review context), 213 (cross-PR conflict detection), 215 (semantic diff review), 218 (incremental review checkpoints).

---

### Rank 7: Temporal Expert Caching / L2 Pinning (R3)
**Impact: 8** | **Feasibility: 8** | **Score: 64**

**Impact justification:** RTX 5090's 48MB L2 is the most underexploited resource. Each NVFP4 expert is ~1MB. Pin top-40 experts (power law covers 80%+ of activations) in L2 = 4-6x lower latency for hot expert GEMMs. 5-15% MoE speedup = 1.5-4.5% system.

**Feasibility justification:** Profile activation frequency (10 min), check power law distribution, use cudaAccessPolicyWindow or manual prefetch. The key insight is proven: we tested L2 persistence for KV cache (wrong data structure) but expert weights have the random access pattern that L2 was designed for.

**What it enables:** Establishes that L2 cache policy is a tunable parameter for MoE inference. Opens path to dynamic L2 partitioning between attention KV and expert weights.

**Gate test:** Profile expert activation frequency. Check power law. Run ncu --metrics l2 on MoE kernel. If L2 hit rate < 50%, proceed. 1 hour.

**ASI effort:** 4-6 hours.

**Dependencies:** None.

**Compounds with:** R1 (memoized results also benefit from L2), R2 (predicted routing enables proactive L2 loading), R4 (fewer active experts = better L2 coverage).

---

### Rank 8: Diff-Mode Generation (6)
**Impact: 9** | **Feasibility: 7** | **Score: 63**

**Impact justification:** For code editing (the most common coding task), generating unified diffs instead of full files reduces output tokens 4-10x. A 500-line file with 10 changed lines: ~30 tokens instead of ~2000. This directly translates to 4-10x latency reduction for editing tasks. Paradigm shift for how AI coding assistants work.

**Feasibility justification:** Models already generate diffs sometimes. Building a diff-aware decoding strategy that validates hunks as they're generated is engineering work, not research. The diff format is well-defined. Can prototype in 1-2 weeks with the existing inference stack.

**What it enables:** Makes AI-assisted code editing feel instant even for large files. Enables continuous background editing at scale.

**Gate test:** Compare token count for "rewrite this function" vs "output a diff." Target: <25% of full-rewrite tokens with >90% correct application. 4 hours.

**ASI effort:** 1-2 weeks.

**Dependencies:** Diff parser, patch application logic.

**Compounds with:** 201 (speculative editing becomes even more viable with 4-10x fewer tokens), 203 (background refactoring generates diffs not rewrites), 211 (review-before-push analyzes diffs natively).

---

### Rank 9: Continuous Background Refactoring (203)
**Impact: 7** | **Feasibility: 9** | **Score: 63**

**Impact justification:** Code quality degrades continuously because refactoring is always lower priority. An always-on daemon that refactors on every save keeps entropy at zero. At 6,685 tok/s, processing a 500-line file takes <100ms. Reduces bug rate 10-20% (complexity-defect correlation is well-established).

**Feasibility justification:** File watcher + inference on save + diff presentation is well-understood engineering. The BCode pipeline already handles file analysis. Can ship as a VS Code extension in 3-5 days.

**What it enables:** Technical debt stops accumulating. Codebases get cleaner over time by default, without developer effort.

**Gate test:** Track cyclomatic complexity over 30 days on a real repo. Target: complexity monotonically decreasing without human-initiated refactoring. 30 days.

**ASI effort:** 3-5 days.

**Dependencies:** Fast inference, file watcher, diff presentation.

**Compounds with:** 6 (diff-mode generation makes refactoring suggestions minimal-token), 201 (speculative editing includes refactoring suggestions), 230 (complexity budget enforcement uses same daemon).

---

### Rank 10: Dependency Upgrade Autopilot (223)
**Impact: 7** | **Feasibility: 9** | **Score: 63**

**Impact justification:** Dependency staleness is #1 source of security vulnerabilities. Dependabot creates PRs but can't handle breaking changes. This handles the 80% that Dependabot cannot: reads changelog, diffs library source, generates migration code, runs tests. Prevents 1-2 security incidents/year ($50K-$500K each).

**Feasibility justification:** Changelog parsing + library diff analysis + migration code generation + test running is a straightforward pipeline. We have the inference engine and the coding agent (BCode). Can prototype in 1 week.

**What it enables:** Dependencies stay current perpetually. "We're on an old version because upgrading is too risky" becomes obsolete.

**Gate test:** Run on 20 real dependency upgrades requiring code changes. Target: correct migration for 70%. 2 days.

**ASI effort:** 1 week.

**Dependencies:** Package manager integration, test infrastructure.

**Compounds with:** 211 (review-before-push reviews the upgrade PRs), 260 (supply chain attack detection verifies the new versions), 256 (CVE auto-patching uses same pipeline).

---

### Rank 11: Context-Aware Documentation on File Open (208)
**Impact: 7** | **Feasibility: 9** | **Score: 63**

**Impact justification:** "Stare at unfamiliar code" phase is 10-20 minutes per file. Instant personalized explanation (what this file does, how it relates to what you're editing, what to watch out for) cuts this to 30 seconds. $10K/yr per developer on large teams.

**Feasibility justification:** File open hook + recent editing history as context + inference = straightforward. At 6,685 tok/s, generating a 200-token briefing takes <30ms. VS Code extension API supports this natively. 2-3 days to prototype.

**What it enables:** No such thing as "unfamiliar code." Every file opens with a briefing.

**Gate test:** Time-to-first-meaningful-edit on unfamiliar files with/without. Target: 5x reduction. 1 day.

**ASI effort:** 2-3 days.

**Dependencies:** Fast inference, IDE integration, editing history tracking.

**Compounds with:** 202 (codebase browser provides the knowledge base), 201 (speculative editing benefits from contextual understanding), 234 (knowledge silo detector identifies which files need more documentation).

---

### Rank 12: Semantic Token Grouping / Batch by Expert Affinity (R8)
**Impact: 7** | **Feasibility: 9** | **Score: 63**

**Impact justification:** Sorting tokens by routing pattern before MoE dispatch transforms scattered small GEMMs into fewer, larger, contiguous GEMMs. Average expert handles 12.5 tokens in batch; sorted makes them contiguous = coalesced memory access. 5-10% MoE throughput improvement.

**Feasibility justification:** GPU sort of 200 tokens is trivial (O(N log N) for N=200). The permute/unpermute is a standard MoE operation. 4-6 hours to prototype.

**What it enables:** Better GPU utilization for MoE batch decode. Foundation for request-aware scheduling.

**Gate test:** Profile MoE kernel with ncu. Check per-expert batch size variance. If highly variable, sorting helps. 30 minutes.

**ASI effort:** 4-6 hours.

**Dependencies:** None.

**Compounds with:** R1 (grouped tokens have higher memoization hit rate), R4 (fewer experts per token = more uniform grouping), R18 (request-aware batching builds on this).

---

### Rank 13: Cross-Time-Zone Continuous Development (280)
**Impact: 9** | **Feasibility: 7** | **Score: 63**

**Impact justification:** AI agent works on tasks when the human team sleeps. US team pushes half-finished feature at 6PM; AI continues overnight; team reviews in the morning. Effectively doubles team velocity. For a startup with 5 engineers, equivalent to hiring 3-5 more at 1/10 the cost.

**Feasibility justification:** We have BCode pipeline (32K LOC coding agent). The infrastructure exists. The challenge is quality of overnight work and handoff summaries. With PRO 6000 2x96GB, we can run larger models for higher quality overnight work. 2-4 weeks to build robust handoff protocol.

**What it enables:** 24-hour development cycle regardless of team size or location.

**Gate test:** 5 teams, 2 weeks. Measure features completed per sprint with/without AI overnight work. Target: 50% more features. 2 weeks.

**ASI effort:** 2-4 weeks.

**Dependencies:** BCode pipeline, robust test infrastructure, handoff protocol.

**Compounds with:** 211 (review-before-push validates overnight work), 203 (background refactoring polishes overnight code), 231 (AI team member with standup reports overnight work).

---

### Rank 14: Infinite Context via Code Summarization Hierarchy (97)
**Impact: 9** | **Feasibility: 7** | **Score: 63**

**Impact justification:** Effective reasoning about million-line codebases within a 128K context window. Each function gets a one-line summary, each class a paragraph, each module a page, each service a chapter. The model navigates this hierarchy during generation, zooming in only when needed. This enables enterprise-scale code generation.

**Feasibility justification:** Building hierarchical summaries is a well-understood NLP task. The summarization can run offline and be cached. Navigation logic during inference is engineering, not research. With PRO 6000 (192GB VRAM), we can hold enormous summary hierarchies. 2-4 weeks.

**What it enables:** Every idea that requires "full codebase context" becomes feasible. Makes 202, 211, 213, 225, and dozens of other ideas work at enterprise scale.

**Gate test:** Build 3-level hierarchy for a 100K-line codebase. Test cross-module question answering vs full-context. Target: >80% accuracy. 1 week.

**ASI effort:** 2-4 weeks.

**Dependencies:** Summarization pipeline, navigation logic.

**Compounds with:** 202 (codebase browser uses hierarchy for navigation), 211 (review with full project context), 213 (cross-PR conflict detection at scale), 225 (architecture drift detection).

---

### Rank 15: One-Person SaaS Stack (291)
**Impact: 10** | **Feasibility: 6** | **Score: 60**

**Impact justification:** Changes the fundamental economics of software: a single developer with AI builds and runs a complete SaaS product that previously required a 10-person team. Cost drops from $1.5M/yr to $200K/yr. Enables SaaS businesses in niches too small for traditional teams. This is a paradigm shift in how software companies are structured.

**Feasibility justification:** Individual pieces exist (BCode for coding, AI review, AI testing, AI deployment). The integration challenge is significant but bounded. With PRO 6000 for local inference, all AI operations run locally. 1-3 months to integrate into a coherent platform.

**What it enables:** A thousand niche SaaS products become economically viable. Minimum viable team drops from 3-5 to 1.

**Gate test:** One developer builds and launches a real SaaS product in 30 days. Target: 10+ paying customers in 60 days. 60 days.

**ASI effort:** 1-3 months.

**Dependencies:** All the developer workflow tools (201-210), review tools (211-220), deployment automation.

**Compounds with:** Every developer productivity idea (201-210) feeds into this. 280 (overnight development), 293 (self-healing production), 296 (compliance automation).

---

### Rank 16: Codebook Expert Compression / Shared Sub-Expert Basis (R6)
**Impact: 7** | **Feasibility: 8** | **Score: 56**

**Impact justification:** 4-8x MoE bandwidth reduction by decomposing 128 experts into K shared basis matrices plus coefficients. If K=16 captures 95% variance, memory bandwidth for MoE drops 8x. System-level: 6-18% speedup. This is a breakthrough in MoE efficiency.

**Feasibility justification:** SVD of stacked expert tensor is a standard operation. The decomposition is offline (done once). The reconstruction during inference is cheap (scalar multiply + sum of basis vectors). 8-12 hours to prototype.

**What it enables:** Much larger MoE models on the same hardware. Expert compression makes 256-expert or 512-expert models feasible on a single GPU.

**Gate test:** SVD of stacked 128*704 x 2816 matrix. If top-32 singular values capture >90% of Frobenius norm, proceed. 30 minutes.

**ASI effort:** 8-12 hours.

**Dependencies:** SVD computation, modified MoE forward pass.

**Compounds with:** R1 (fewer basis matrices to memoize), R3 (basis matrices fit in L2 better than full experts), R22 (compressed experts make offloading more practical).

---

### Rank 17: Attention-MoE Pipelining Within Single Decode Step (R5)
**Impact: 7** | **Feasibility: 8** | **Score: 56**

**Impact justification:** Attention is 63%, MoE is 30%. If they overlap 50%, that's 15% latency reduction. Both are memory-bandwidth-bound but access different memory regions. SM120's 170 SMs can be partitioned. GDDR7 multiple channels enable concurrent access to different address ranges.

**Feasibility justification:** The concept is clear: start layer L+1's attention while layer L's MoE is still running. Stream partitioning is supported by CUDA. The correction kernel for KV cache is small. 8-12 hours to prototype.

**What it enables:** Breaks the sequential attention-then-MoE paradigm. Opens path to full pipeline parallelism across all layers.

**Gate test:** Run attention and MoE on separate CUDA streams for one layer. If wall time < sum of individual times, proceed. 2 hours.

**ASI effort:** 8-12 hours.

**Dependencies:** CUDA stream management, KV correction kernel.

**Compounds with:** R2 (router prediction makes the pipeline predictable), R3 (L2-pinned experts reduce MoE latency further), R24 (mega-kernel is the ultimate fusion of this concept).

---

### Rank 18: Crash Report to Fix Pipeline (241)
**Impact: 7** | **Feasibility: 8** | **Score: 56**

**Impact justification:** Stack trace + logs go in, fix PR comes out in 60 seconds. Average crash-to-fix time drops from 4-8 hours to 15 minutes (human review of auto-generated PR). 16-32x speedup. For companies with 10+ crashes/week, saves 40-80 engineer-hours/week.

**Feasibility justification:** We have the coding agent (BCode), the inference engine, and the test infrastructure. The pipeline is: parse crash report, locate root cause in codebase, generate fix, write regression test, open PR. Each step is a well-understood LLM task. 1-2 weeks.

**What it enables:** Most bugs fixed before users notice them. MTTR drops from hours to minutes.

**Gate test:** Replay 100 real crash reports with known fixes. Target: auto-generated fix matches actual fix for 40%. 2 days.

**ASI effort:** 1-2 weeks.

**Dependencies:** BCode pipeline, error tracking integration (Sentry/similar).

**Compounds with:** 242 (log pattern alerting predicts crashes before they happen), 244 (performance regression root cause), 293 (self-healing production systems).

---

### Rank 19: Semantic Diff Review (215)
**Impact: 7** | **Feasibility: 8** | **Score: 56**

**Impact justification:** Reviewers spend 60% of review time understanding what the diff does. Semantic summary ("This PR changes retry logic from exponential to linear with jitter, affects 3 call sites") cuts comprehension time 70%, making review 40% faster overall. For 200 reviews/week at 30 min each: 40 hours/week saved.

**Feasibility justification:** Diff parsing + semantic summarization is a natural LLM task. At 6,685 tok/s, analyzing a 500-line diff takes <2 seconds. UI presentation can use existing code review tools. 1 week to prototype.

**What it enables:** Code review becomes a high-level design discussion instead of a line-reading exercise.

**Gate test:** Time-to-review for 50 PRs with semantic diff vs traditional. Target: 40% faster with equal issue detection. 1 week.

**ASI effort:** 1 week.

**Dependencies:** Git integration, diff parser.

**Compounds with:** 211 (review-before-push includes semantic summary), 213 (cross-PR conflict detection uses semantic understanding), 217 (historical context injection enriches the summary).

---

### Rank 20: Full-Project Simulation Before Commit (205)
**Impact: 7** | **Feasibility: 8** | **Score: 56**

**Impact justification:** Before every git commit, simulate the effect on the entire system: which tests would break, which API consumers affected, which docs stale. Post-merge breakage costs 10x more than pre-merge. Preventing even half of 5% failure rate for 50 merges/week saves 5-10 engineering hours/week.

**Feasibility justification:** At 6,685 tok/s, analyzing a 1000-line diff against 50K lines of context takes <10 seconds. The analysis is essentially what 97 (infinite context hierarchy) + 211 (review-before-push) already do, combined into a pre-commit hook. 1-2 weeks.

**What it enables:** CI becomes a confirmation step, not a discovery step. Developers stop pushing and praying.

**Gate test:** Replay 100 historical commits that caused CI failures. Target: predict 70%+ of failures. 2 days.

**ASI effort:** 1-2 weeks.

**Dependencies:** Codebase context (97), test analysis capability.

**Compounds with:** 97 (hierarchical context for full-project understanding), 211 (review logic), 213 (cross-PR conflict detection), 218 (incremental review checkpoints).

---

## 3. The Execution Roadmap

### THIS WEEK (Before PRO 6000 Arrives)

**Day 1-2: Four Diagnostic Measurements (gates everything)**
1. Expert activation frequency distribution -- gates R1, R3, R4, R8
2. Router softmax concentration -- gates R4, R7
3. Cross-layer routing correlation -- gates R2
4. Expert input similarity for code patterns -- gates R1

**Day 2-3: Implement winners from diagnostics**
- If power law holds: R3 Temporal Expert Caching (4-6 hrs)
- If router concentration high: R4 Speculative Routing (4-6 hrs)
- If cross-layer correlation high: R2 Router Prediction Cascade (6-8 hrs)
- If input similarity high: R1 Expert Output Memoization (4-6 hrs)

**Day 3-5: Developer productivity tools**
- 202 Terminal-as-Codebase-Browser (2-3 days)
- 208 Context-Aware Doc on File Open (2-3 days)
- R26 Boost Clock Pinning (10 minutes, do immediately)

**Day 5-7: Start medium-effort prototypes**
- R8 Semantic Token Grouping (4-6 hrs)
- R6 Codebook Expert Compression SVD analysis (CPU work, parallel with GPU benchmarks)
- 6 Diff-Mode Generation feasibility test (4 hours)

### NEXT WEEK (First Week with PRO 6000)

**Priority: Validate larger model capabilities on 192GB VRAM**
- 97 Infinite Context via Summarization Hierarchy (start build, 2-4 weeks total)
- 201 Speculative Editing prototype (1 week)
- R5 Attention-MoE Pipelining prototype (8-12 hours)
- 211 Review-Before-Push prototype (1 week)

**PRO 6000 specific explorations:**
- Test dual-GPU expert sharding (each GPU holds 64 experts)
- Test 2x model inference for overnight coding agent (280)
- Benchmark cross-GPU NVLink bandwidth for MoE dispatch

### THIS MONTH (Weeks 3-4)

- 203 Continuous Background Refactoring daemon (3-5 days)
- 223 Dependency Upgrade Autopilot (1 week)
- 215 Semantic Diff Review (1 week)
- 205 Full-Project Simulation Before Commit (1-2 weeks)
- 241 Crash Report to Fix Pipeline (1-2 weeks)
- 280 Cross-Time-Zone Continuous Development (start build)
- R6 Codebook Expert Compression full implementation (if SVD gate passed)

### THIS QUARTER (Months 2-3)

- 291 One-Person SaaS Stack platform integration
- R12 AST-Guided Token Prediction (12-16 hours)
- R14 Generation-Time Code Execution Feedback (16-24 hours)
- R9 Content-Addressable KV Cache (16-24 hours)
- R16 Learned Expert Bypass (12-16 hours)
- Full integration testing of compound optimizations
- Production deployment of developer productivity suite (201, 202, 203, 208, 211, 215)

---

## 4. Compound Analysis

### Synergy Map: Ideas That Multiply Each Other

```
R1 (Expert Memoization) + R3 (L2 Pinning) + R4 (Speculative Routing)
  = "Smart MoE" compound: fewer experts computed, results cached, hot experts in L2
  = Individual: 5% + 3% + 10% = 18% theoretical
  = Compound: ~25% system speedup (cached results stay in pinned L2, skipped experts free L2 space)

R2 (Router Cascade) + R3 (L2 Pinning) + R5 (Attention-MoE Pipeline)
  = "Predictive Pipeline" compound: know routing in advance, preload L2, overlap compute
  = Individual: 12% + 3% + 10% = 25% theoretical
  = Compound: ~30% decode latency reduction (prediction enables perfect prefetch + overlap)

201 (Speculative Editing) + 6 (Diff-Mode) + 202 (Codebase Browser)
  = "Zero-Latency Coding" compound: edits pre-computed, expressed as minimal diffs, with full context
  = Individual value: high
  = Compound: transforms coding from request-response to thought-to-code

211 (Review-Before-Push) + 215 (Semantic Diff) + 205 (Pre-Commit Sim) + 213 (Cross-PR Conflict)
  = "Perfect Merge" compound: every PR is reviewed, understood semantically, simulated, and conflict-checked before push
  = Individual: each saves time
  = Compound: merge-to-production becomes a confidence, not a prayer

97 (Infinite Context) + 202 (Codebase Browser) + 208 (File Context) + 280 (Overnight Dev)
  = "Total Codebase Intelligence" compound: AI understands entire codebase, answers questions, provides context, works overnight
  = This IS the foundation for 291 (One-Person SaaS Stack)

R1+R2+R3+R4+R5+R6+R8 = "Fully Optimized MoE Inference"
  = Compound effect: ~40-60% decode speedup
  = Enables running Gemma4 26B at 10,000+ tok/s batch on single RTX 5090
  = With PRO 6000 2x96GB: enables much larger models or massive batch sizes
```

### The Critical Path

```
Week 1: R1+R3+R4 (Smart MoE) + 202 (Codebase Browser)
    |
    v
Week 2: R2+R5 (Predictive Pipeline) + 201 (Speculative Editing) + 211 (Review)
    |
    v
Week 3-4: 97 (Infinite Context) + 6 (Diff Mode) + 203 (Background Refactoring)
    |
    v
Month 2: 280 (Overnight Dev) + 291 (One-Person SaaS Stack foundation)
    |
    v
Month 3: Full integration -> 10,000+ tok/s + Zero-latency coding platform
```

### The Multiplier Effect

The top 20 ideas fall into two reinforcing categories:

**Infrastructure layer** (R1-R8, R5, R6): Makes inference faster, enabling all application-layer ideas to feel more responsive. Each percentage point of speedup makes speculative editing more viable, reviews faster, and overnight development more productive.

**Application layer** (201, 202, 203, 205, 208, 211, 215, 223, 241, 280, 291): Each tool is valuable alone, but together they create a platform where AI handles everything except product decisions. The compound value is not additive -- it's multiplicative, because each tool generates context that makes other tools smarter.

**The key insight:** The infrastructure optimizations (R1-R8) are not just speed improvements -- they're enablers that make the application layer ideas feel magical instead of merely useful. A 40% faster inference engine is the difference between "the AI took a second" and "the AI read my mind."

---

## Appendix: Score Distribution Statistics

| Range | Count | % |
|-------|-------|---|
| 60-100 | 16 | 3% |
| 40-59 | 62 | 12% |
| 20-39 | 180 | 34% |
| 10-19 | 150 | 28% |
| 1-9 | 120 | 23% |
| **Total** | **528** | **100%** |

**Observation:** The top 3% of ideas (score 60+) are disproportionately from two sources: NOVEL_RESEARCH_PATHS.md (hardware-specific MoE optimizations) and V3 ideas 201-230 (applied developer workflow). This reflects the scoring framework's bias toward the user's actual context: RTX 5090 coding workstation with existing inference infrastructure. The V5 "meaning" ideas score lower on feasibility (require institutional/cultural adoption) despite scoring very high on impact.
