# Documentation Index

This directory holds documentation that doesn't belong at the repo root.
The root keeps only what GitHub displays prominently or what the agent
loop needs (README, WHITEPAPER, program.md, AGENTS.md, CLAUDE.md, LICENSE).

## Layout

### `architecture/`
Stable architecture docs — how things are designed, not what happened on a given day.

- [DEEP_DIVE.md](architecture/DEEP_DIVE.md) — system-wide architectural deep dive
- [FUSENCACHE.md](architecture/FUSENCACHE.md) — FusenCache KV plugin design
- [README_OPEN_SOURCE.md](architecture/README_OPEN_SOURCE.md) — open-source release variant of the README

### `sessions/`
Session-status snapshots — point-in-time progress reports. Newest first.

- [SESSION_FINAL_STATUS_V2.md](sessions/SESSION_FINAL_STATUS_V2.md) — most recent session summary
- [SESSION_FINAL_STATUS.md](sessions/SESSION_FINAL_STATUS.md)
- [GEMMA4_NVFP4_STATUS.md](sessions/GEMMA4_NVFP4_STATUS.md) — Gemma 4 NVFP4 working state
- [FUSENCACHE_VLLM_MAIN_STATUS.md](sessions/FUSENCACHE_VLLM_MAIN_STATUS.md) — FusenCache port to vLLM main
- [DAY1_STATUS.md](sessions/DAY1_STATUS.md) — origin

### `research/`
Experiments, benchmarks, bug reports, lessons.

- [EXPERIMENTS.md](research/EXPERIMENTS.md) — running log
- [EXPERIMENT_DISCOVERIES.md](research/EXPERIMENT_DISCOVERIES.md) — distilled findings
- [GEMMA4_NVFP4_BENCHMARKS.md](research/GEMMA4_NVFP4_BENCHMARKS.md)
- [MOE_PROFILING.md](research/MOE_PROFILING.md)
- [MTP_BATCH_FIX.md](research/MTP_BATCH_FIX.md)
- [DFLASH_PATCHES.md](research/DFLASH_PATCHES.md) — patches to vLLM 0.17.0 + FlashInfer 0.6.4
- [vllm_mtp_batch_bug_report.md](research/vllm_mtp_batch_bug_report.md) — upstream bug
- [fusencache_experiments.md](research/fusencache_experiments.md)
- [fusencache_future_experiments.md](research/fusencache_future_experiments.md)
- [fusencache_nvfp4_results.md](research/fusencache_nvfp4_results.md)
- [expert_caching_experiment.md](research/expert_caching_experiment.md)
- [april7_experiments.md](research/april7_experiments.md)
- [optimization_lessons_part2.md](research/optimization_lessons_part2.md)

For session-by-session planning artifacts, see `plans/` at the repo root.
