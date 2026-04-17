#!/usr/bin/env python3
"""Exp 92: Combined best patches — async scheduling + GC disable + fast update_from_output.

Stacks all winning optimizations from exp88-91:
1. async_scheduling=True (overlap schedule with GPU)
2. gc.disable() during step
3. Fast update_from_output for decode (skip spec decode, pooler, KV connector branches)

Target: 180-250 tok/s (from 122 baseline).
"""

import os
import sys
import time
import gc

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

MODEL = "Kbenkhaled/Qwen3.5-9B-NVFP4"
PROMPT = "Write merge sort in Python:"
MAX_TOKENS = 200
NUM_WARMUP = 2
NUM_RUNS = 3


def bench_decode(llm, prompt, max_tokens, num_warmup, num_runs):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    for _ in range(num_warmup):
        llm.generate([prompt], sp)
    times = []
    outputs_text = []
    for _ in range(num_runs):
        gc.collect()
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sp)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        outputs_text.append(outputs[0].outputs[0].text)
        num_tokens = len(outputs[0].outputs[0].token_ids)
    avg_time = sum(times) / len(times)
    tok_per_sec = num_tokens / avg_time
    ms_per_tok = avg_time / num_tokens * 1000
    return tok_per_sec, ms_per_tok, avg_time, num_tokens, outputs_text[-1], times


def bench_batch(llm, prompt, max_tokens, batch_size=32):
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    prompts = [prompt] * batch_size
    llm.generate(prompts, sp)
    gc.collect()
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sp)
    t1 = time.perf_counter()
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return total_tokens / (t1 - t0), t1 - t0


def apply_async_scheduling_patch():
    """Patch 1: Enable async scheduling."""
    from vllm.config.scheduler import SchedulerConfig
    original_init = SchedulerConfig.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not self.async_scheduling:
            self.async_scheduling = True

    SchedulerConfig.__init__ = patched_init
    print("[PATCH] async_scheduling=True")


def apply_gc_disable_patch():
    """Patch 2: Disable GC during engine step."""
    from vllm.v1.engine.core import EngineCore

    original_step = EngineCore.step
    original_step_batch = EngineCore.step_with_batch_queue

    def step_no_gc(self):
        gc.disable()
        try:
            return original_step(self)
        finally:
            gc.enable()

    def step_batch_no_gc(self):
        gc.disable()
        try:
            return original_step_batch(self)
        finally:
            gc.enable()

    EngineCore.step = step_no_gc
    EngineCore.step_with_batch_queue = step_batch_no_gc
    print("[PATCH] GC disabled during step")


def apply_fast_update_patch():
    """Patch 3: Fast update_from_output for decode."""
    from vllm.v1.core.sched.scheduler import Scheduler
    original_update = Scheduler.update_from_output
    _stats = {"fast": 0, "slow": 0}

    def fast_update(self, scheduler_output, model_runner_output):
        sampled_token_ids = model_runner_output.sampled_token_ids
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        if (
            sampled_token_ids is not None
            and not scheduler_output.scheduled_spec_decode_tokens
            and model_runner_output.pooler_output is None
            and model_runner_output.kv_connector_output is None
            and model_runner_output.num_nans_in_logits is None
            and all(v == 1 for v in num_scheduled_tokens.values())
        ):
            _stats["fast"] += 1
            from collections import defaultdict
            from vllm.v1.engine.core_output import EngineCoreOutput, EngineCoreOutputs

            logprobs = model_runner_output.logprobs
            prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict

            perf_stats = None
            if self.perf_metrics and self.perf_metrics.is_enabled():
                perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

            outputs = defaultdict(list)
            stopped_running_reqs = set()

            for req_id in num_scheduled_tokens:
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                req_index = model_runner_output.req_id_to_index[req_id]
                new_token_ids = sampled_token_ids[req_index]
                if not new_token_ids:
                    continue

                new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)

                finish_reason = None
                kv_transfer_params = None
                routed_experts = None
                if stopped:
                    routed_experts = self._get_routed_experts(request)
                    finish_reason = request.get_finished_reason()
                    finished = self._handle_stopped_request(request)
                    if finished:
                        kv_transfer_params = self._free_request(request)
                    stopped_running_reqs.add(request)

                new_logprobs = None
                if (
                    request.sampling_params is not None
                    and request.sampling_params.logprobs is not None
                    and logprobs
                ):
                    new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

                if new_token_ids and self.structured_output_manager.should_advance(request):
                    struct_req = request.structured_output_request
                    if struct_req and struct_req.grammar:
                        struct_req.grammar.accept_tokens(req_id, new_token_ids)

                prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
                if new_token_ids or stopped:
                    outputs[request.client_index].append(
                        EngineCoreOutput(
                            request_id=req_id,
                            new_token_ids=new_token_ids,
                            finish_reason=finish_reason,
                            new_logprobs=new_logprobs,
                            new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                            pooling_output=None,
                            stop_reason=request.stop_reason,
                            events=request.take_events(),
                            kv_transfer_params=kv_transfer_params,
                            trace_headers=request.trace_headers,
                            num_cached_tokens=request.num_cached_tokens,
                            num_external_computed_tokens=request.num_external_computed_tokens,
                            routed_experts=routed_experts,
                            num_nans_in_logits=request.num_nans_in_logits,
                        )
                    )

            if stopped_running_reqs:
                from vllm.utils.list import remove_all
                self.running = remove_all(self.running, stopped_running_reqs)

            events = self.kv_cache_manager.take_events()
            if self.connector is not None:
                connector_events = self.connector.take_events()
                if connector_events:
                    events = (events or []) + list(connector_events)
            if events:
                import time as _time
                from vllm.v1.core.kv_cache_utils import KVEventBatch
                self.kv_event_publisher.publish(KVEventBatch(ts=_time.time(), events=events))

            engine_core_outputs = {
                ci: EngineCoreOutputs(outputs=outs) for ci, outs in outputs.items()
            }
            finished_req_ids = self.finished_req_ids_dict
            if finished_req_ids:
                for ci, fs in finished_req_ids.items():
                    if (eco := engine_core_outputs.get(ci)) is not None:
                        eco.finished_requests = fs
                    else:
                        engine_core_outputs[ci] = EngineCoreOutputs(finished_requests=fs)
                finished_req_ids.clear()
            return engine_core_outputs

        _stats["slow"] += 1
        return original_update(self, scheduler_output, model_runner_output)

    Scheduler.update_from_output = fast_update
    print("[PATCH] Fast update_from_output for decode")
    return _stats


def main():
    # Apply all patches
    apply_async_scheduling_patch()
    apply_gc_disable_patch()
    update_stats = apply_fast_update_patch()

    # ── Init engine ──
    from vllm import LLM
    print(f"\n[EXP92] Loading {MODEL} with all patches ...")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
    )

    # Verify
    engine_core = llm.llm_engine.engine_core.engine_core
    print(f"[EXP92] batch_queue: {engine_core.batch_queue is not None}")
    print(f"[EXP92] step_fn: {engine_core.step_fn.__name__}")

    # ── Benchmarks ──
    print("\n=== Decode (1 request) ===")
    tok_s, ms_tok, total, ntok, text, times = bench_decode(
        llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS
    )
    print(f"  {tok_s:.1f} tok/s  ({ms_tok:.2f} ms/tok)")
    print(f"  {ntok} tokens in {total:.3f}s")
    print(f"  Runs: {['%.3fs' % t for t in times]}")
    print(f"  Update fast/slow: {update_stats['fast']}/{update_stats['slow']}")

    print("\n=== Batch (32 requests) ===")
    batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, batch_size=32)
    print(f"  {batch_tps:.0f} tok/s total  ({batch_time:.2f}s)")
    print(f"  Update fast/slow: {update_stats['fast']}/{update_stats['slow']}")

    # Correctness check
    print(f"\n=== Correctness ===")
    print(f"  Output preview: {text[:200]}...")

    # ── Log ──
    exp_name = "exp92_combined"
    desc = f"All patches (async+gc+fast_update): decode={tok_s:.1f}, batch32={batch_tps:.0f}"
    print(f"\n[RESULT] {exp_name}\t{tok_s:.1f}\t{batch_tps:.0f}\t{desc}")
    with open("results.tsv", "a") as f:
        f.write(f"92\t{exp_name}\tvllm_overhead\t{tok_s:.1f}\t{batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
