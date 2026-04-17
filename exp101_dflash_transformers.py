#!/usr/bin/env python3
"""
Experiment 101: DFlash speculative decoding via Transformers backend
Target: Qwen3-4B (pure transformer, no Mamba)
Draft: z-lab/Qwen3-4B-DFlash-b16 (5 layers, block_size=16)

Uses the DFlash model's built-in spec_generate() method.
No server needed - direct GPU inference.
"""

import sys
sys.path.insert(0, "/root/projects/dflash")

import time
import torch
import numpy as np

torch.manual_seed(42)

TARGET_MODEL = "Qwen/Qwen3-4B"
DRAFT_MODEL = "z-lab/Qwen3-4B-DFlash-b16"
DEVICE = "cuda:0"

def cuda_time():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    print("=" * 70)
    print("EXP 101: DFlash via Transformers Backend")
    print(f"Target: {TARGET_MODEL}")
    print(f"Draft:  {DRAFT_MODEL}")
    print(f"GPU:    RTX 5090 32GB, SM120")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(DEVICE).eval()

    print("Loading DFlash draft model...")
    from model import DFlashDraftModel
    draft = DFlashDraftModel.from_pretrained(
        DRAFT_MODEL,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    ).to(DEVICE).eval()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
    block_size = draft.block_size
    print(f"Block size: {block_size}")
    print(f"Target layers for context: {draft.target_layer_ids}")

    # GPU memory check
    mem_used = torch.cuda.memory_allocated() / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {mem_used:.1f} / {mem_total:.1f} GB")

    # Test prompts
    prompts = [
        "Write a Python function that implements a binary search tree with insert, delete, and search operations.",
        "Solve step by step: Find all positive integers n such that n^2 + 2n + 2 is divisible by n + 1.",
        "Explain the time complexity of quicksort and when it degrades to O(n^2).",
        "Write a Python class that implements a thread-safe LRU cache.",
    ]

    # ── Phase 1: Baseline (autoregressive, block_size=1) ──
    print("\n[Phase 1] Baseline (autoregressive, no speculation)")
    baseline_results = []

    from model.utils import sample, extract_context_feature
    from transformers import DynamicCache

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
        num_input = input_ids.shape[1]
        max_new = 256

        # Warmup
        if i == 0:
            with torch.no_grad():
                _ = target(input_ids[:, :10], use_cache=False)

        # Autoregressive baseline
        past_kv = DynamicCache()
        t0 = cuda_time()

        with torch.no_grad():
            # Prefill
            out = target(input_ids, past_key_values=past_kv, use_cache=True, logits_to_keep=1)
            next_token = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
            generated = [next_token.item()]

            # Decode
            for _ in range(max_new - 1):
                out = target(next_token, past_key_values=past_kv, use_cache=True)
                next_token = torch.argmax(out.logits[:, -1], dim=-1, keepdim=True)
                generated.append(next_token.item())
                if next_token.item() == tokenizer.eos_token_id:
                    break

        elapsed = cuda_time() - t0
        num_tokens = len(generated)
        tok_s = num_tokens / elapsed
        baseline_results.append(tok_s)
        print(f"  Prompt {i+1}: {tok_s:.1f} tok/s ({num_tokens} tokens in {elapsed:.2f}s)")

    avg_baseline = np.mean(baseline_results)
    print(f"  Average baseline: {avg_baseline:.1f} tok/s")

    # ── Phase 2: DFlash speculative decoding ──
    print(f"\n[Phase 2] DFlash (block_size={block_size})")
    dflash_results = []
    acceptance_lengths_all = []

    for i, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
        input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

        t0 = cuda_time()
        output_ids = draft.spec_generate(
            target=target,
            input_ids=input_ids,
            max_new_tokens=256,
            stop_token_ids=[tokenizer.eos_token_id],
            temperature=0.0,
        )
        elapsed = cuda_time() - t0

        num_tokens = output_ids.shape[1] - input_ids.shape[1]
        tok_s = num_tokens / elapsed
        dflash_results.append(tok_s)

        # Decode output
        out_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
        print(f"  Prompt {i+1}: {tok_s:.1f} tok/s ({num_tokens} tokens in {elapsed:.2f}s)")

    avg_dflash = np.mean(dflash_results)
    print(f"  Average DFlash: {avg_dflash:.1f} tok/s")

    # ── Phase 3: DFlash with detailed acceptance tracking ──
    print(f"\n[Phase 3] DFlash detailed acceptance analysis")
    sys.path.insert(0, "/root/projects/dflash")
    from benchmark import dflash_generate

    prompt = prompts[0]
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(DEVICE)

    # Run with tracking - block_size=1 (baseline) vs block_size=16 (DFlash)
    print("  Running baseline (bs=1)...")
    result_bs1 = dflash_generate(
        model=draft, target=target, input_ids=input_ids,
        mask_token_id=draft.mask_token_id,
        max_new_tokens=256, block_size=1,
        stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
    )

    print("  Running DFlash (bs=16)...")
    result_bs16 = dflash_generate(
        model=draft, target=target, input_ids=input_ids,
        mask_token_id=draft.mask_token_id,
        max_new_tokens=256, block_size=block_size,
        stop_token_ids=[tokenizer.eos_token_id], temperature=0.0,
    )

    # ── Summary ──
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nModel: Qwen3-4B (pure transformer)")
    print(f"DFlash draft: 5 layers, block_size={block_size}")

    print(f"\nSingle-request decode throughput:")
    print(f"  Baseline (AR):     {avg_baseline:.1f} tok/s")
    print(f"  DFlash:            {avg_dflash:.1f} tok/s")
    speedup = avg_dflash / avg_baseline if avg_baseline > 0 else 0
    print(f"  Speedup:           {speedup:.2f}x")

    print(f"\nDFlash acceptance analysis (benchmark.py):")
    print(f"  Baseline time/tok: {result_bs1.time_per_output_token*1000:.2f} ms")
    print(f"  DFlash time/tok:   {result_bs16.time_per_output_token*1000:.2f} ms")
    speedup2 = result_bs1.time_per_output_token / result_bs16.time_per_output_token
    print(f"  Speedup:           {speedup2:.2f}x")

    if result_bs16.acceptance_lengths:
        avg_accept = np.mean(result_bs16.acceptance_lengths)
        print(f"  Avg acceptance:    {avg_accept:.2f} tokens (out of {block_size})")
        print(f"  Acceptance dist:   {result_bs16.acceptance_lengths[:10]}")

    print(f"\nComparison with vLLM autokernel results:")
    print(f"  vLLM 9B baseline:  122.2 tok/s")
    print(f"  vLLM 9B MTP3:      167.3 tok/s")
    print(f"  This experiment uses 4B (smaller), so absolute tok/s not directly comparable")
    print(f"  Key metric: DFlash speedup ratio = {speedup:.2f}x")
    print(f"  If applied to 9B: estimated {122.2 * speedup:.0f} tok/s")


if __name__ == "__main__":
    main()
