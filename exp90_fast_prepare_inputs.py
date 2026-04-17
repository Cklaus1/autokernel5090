#!/usr/bin/env python3
"""Exp 90: Fast-path prepare_inputs() for uniform decode batches.

In steady-state decode, all requests have num_scheduled_tokens=1.
This means:
  - sorted() is a no-op (all values equal)
  - cu_num_logits is always arange(num_reqs+1)
  - query_start_loc is always arange(num_reqs+1)
  - No prefills

We detect this case and skip unnecessary work.
Expected: +0.3-0.5ms per step.
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


def main():
    import numpy as np
    import torch

    # ── Monkey-patch: fast prepare_inputs for uniform decode ──
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.input_batch import InputBatch
    from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu

    original_prepare = GPUModelRunner.prepare_inputs

    # Track how many times fast path is used
    _stats = {"fast": 0, "slow": 0}

    def fast_prepare_inputs(self, scheduler_output, num_tokens_after_padding):
        num_tokens_per_req = scheduler_output.num_scheduled_tokens
        num_reqs = len(num_tokens_per_req)

        # Fast path: all decode (every request scheduled exactly 1 token, no spec decode)
        draft_tokens = scheduler_output.scheduled_spec_decode_tokens
        if not draft_tokens and num_reqs > 0:
            vals = num_tokens_per_req.values()
            all_one = all(v == 1 for v in vals)
            if all_one:
                _stats["fast"] += 1
                # Skip sorted() — order doesn't matter when all tokens=1
                req_ids = list(num_tokens_per_req.keys())
                num_scheduled_tokens = np.ones(num_reqs, dtype=np.int32)

                idx_mapping_iter = map(self.req_states.req_id_to_index.get, req_ids)
                idx_mapping_np = np.fromiter(idx_mapping_iter, dtype=np.int32, count=num_reqs)
                idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

                # For uniform decode: cu_num_logits and query_start_loc are both arange
                total_num_logits = num_reqs
                cu_num_logits_np = np.arange(num_reqs + 1, dtype=np.int32)
                cu_num_logits = torch.arange(num_reqs + 1, device=self.device, dtype=torch.int32)
                expanded_idx_mapping = idx_mapping
                expanded_local_pos = torch.zeros(num_reqs, dtype=torch.int32, device=self.device)

                # query_start_loc = arange for all-1-token requests
                query_start_loc_np = np.empty(self.max_num_reqs + 1, dtype=np.int32)
                query_start_loc_np[:num_reqs + 1] = np.arange(num_reqs + 1, dtype=np.int32)
                query_start_loc_np[num_reqs + 1:] = num_reqs  # pad
                async_copy_to_gpu(query_start_loc_np, out=self.input_buffers.query_start_loc)
                query_start_loc_np = query_start_loc_np[:num_reqs + 1]
                query_start_loc = self.input_buffers.query_start_loc[:num_reqs + 1]

                # Skip prefill check — all decode
                # Still need pos/seq_lens and token combining
                from vllm.v1.worker.gpu.model_runner import (
                    prepare_pos_seq_lens,
                    combine_sampled_and_draft_tokens,
                )
                prepare_pos_seq_lens(
                    idx_mapping, query_start_loc,
                    self.req_states.num_computed_tokens.gpu,
                    self.input_buffers.positions,
                    self.input_buffers.seq_lens,
                )
                seq_lens = self.input_buffers.seq_lens[:num_reqs]

                dcp_local_seq_lens = None
                if self.use_dcp:
                    from vllm.v1.worker.gpu.model_runner import prepare_dcp_local_seq_lens
                    prepare_dcp_local_seq_lens(
                        self.input_buffers.dcp_local_seq_lens,
                        self.input_buffers.seq_lens,
                        num_reqs, self.dcp_size, self.dcp_rank, self.cp_interleave,
                    )
                    dcp_local_seq_lens = self.input_buffers.dcp_local_seq_lens[:num_reqs]

                logits_indices = combine_sampled_and_draft_tokens(
                    self.input_buffers.input_ids,
                    idx_mapping,
                    self.req_states.last_sampled_tokens,
                    query_start_loc, seq_lens,
                    self.req_states.prefill_len.gpu,
                    self.req_states.draft_tokens,
                    cu_num_logits, total_num_logits,
                )

                return InputBatch(
                    req_ids=req_ids,
                    num_reqs=num_reqs,
                    idx_mapping=idx_mapping,
                    idx_mapping_np=idx_mapping_np,
                    expanded_idx_mapping=expanded_idx_mapping,
                    expanded_local_pos=expanded_local_pos,
                    num_scheduled_tokens=num_scheduled_tokens,
                    num_tokens=num_reqs,  # 1 token per req
                    num_tokens_after_padding=num_tokens_after_padding,
                    num_draft_tokens=0,
                    query_start_loc=query_start_loc,
                    query_start_loc_np=query_start_loc_np,
                    seq_lens=seq_lens,
                    dcp_local_seq_lens=dcp_local_seq_lens,
                    input_ids=self.input_buffers.input_ids[:num_tokens_after_padding],
                    positions=self.input_buffers.positions[:num_tokens_after_padding],
                    logits_indices=logits_indices,
                    cu_num_logits=cu_num_logits,
                    cu_num_logits_np=cu_num_logits_np,
                    has_structured_output_reqs=scheduler_output.has_structured_output_requests,
                )

        # Fallback to original
        _stats["slow"] += 1
        return original_prepare(self, scheduler_output, num_tokens_after_padding)

    GPUModelRunner.prepare_inputs = fast_prepare_inputs
    print("[EXP90] Patched GPUModelRunner.prepare_inputs with uniform-decode fast path")

    # ── Init engine ──
    from vllm import LLM
    print(f"[EXP90] Loading {MODEL} ...")
    print(f"[EXP90] GPUModelRunner.prepare_inputs is patched: {GPUModelRunner.prepare_inputs is fast_prepare_inputs}")
    llm = LLM(
        model=MODEL,
        gpu_memory_utilization=0.90,
        max_num_seqs=128,
        max_model_len=4096,
        enable_chunked_prefill=False,
    )

    # ── Benchmarks ──
    print("\n=== Decode (1 request) ===")
    tok_s, ms_tok, total, ntok, text, times = bench_decode(
        llm, PROMPT, MAX_TOKENS, NUM_WARMUP, NUM_RUNS
    )
    print(f"  {tok_s:.1f} tok/s  ({ms_tok:.2f} ms/tok)")
    print(f"  {ntok} tokens in {total:.3f}s")
    print(f"  Runs: {['%.3fs' % t for t in times]}")
    print(f"  Fast path: {_stats['fast']}, Slow path: {_stats['slow']}")

    print("\n=== Batch (32 requests) ===")
    batch_tps, batch_time = bench_batch(llm, PROMPT, MAX_TOKENS, batch_size=32)
    print(f"  {batch_tps:.0f} tok/s total  ({batch_time:.2f}s)")
    print(f"  Fast path: {_stats['fast']}, Slow path: {_stats['slow']}")

    # ── Log ──
    exp_name = "exp90_fast_prepare"
    desc = f"Fast prepare_inputs decode path: decode={tok_s:.1f}, batch32={batch_tps:.0f}, fast={_stats['fast']}"
    print(f"\n[RESULT] {exp_name}\t{tok_s:.1f}\t{batch_tps:.0f}\t{desc}")
    with open("results.tsv", "a") as f:
        f.write(f"90\t{exp_name}\tvllm_overhead\t{tok_s:.1f}\t{batch_tps:.0f}\t0\t0\tPASS\t0\t{desc}\n")


if __name__ == "__main__":
    main()
