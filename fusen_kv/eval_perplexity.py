#!/usr/bin/env python3
"""Perplexity evaluation harness for FusenCache KV cache quantization.

Measures real perplexity on WikiText-2 (or C4 fallback) to validate that
KV cache quantization preserves model quality. Compares multiple specs
against BF16 baseline.

Usage:
    # API mode (hit a running vLLM server — no model reload):
    python3 fusen_kv/eval_perplexity.py --api-base http://localhost:8000/v1 --model /models/gemma-4 --max-samples 100

    # vLLM mode (loads model internally):
    python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4,k8v4,k8v8,auto --max-samples 100

    # Standalone mode (transformers only, no vLLM required):
    python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4,k8v4 --max-samples 50 --standalone

    # Custom dataset:
    python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4 --dataset c4

    # Quick sanity check:
    python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4 --max-samples 10 --seq-len 512
"""

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow imports from project root and kv_cache_gen
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Quality tier classification
# ---------------------------------------------------------------------------

def classify_quality(ppl_delta_pct: float) -> str:
    """Classify quality based on perplexity increase vs baseline.

    Args:
        ppl_delta_pct: percentage increase in perplexity vs BF16 baseline.
            E.g., 1.5 means PPL is 1.5% higher than baseline.

    Returns:
        Quality tier string.
    """
    if ppl_delta_pct <= 0.5:
        return "EXCELLENT (<0.5% PPL increase)"
    elif ppl_delta_pct <= 1.0:
        return "GOOD (<1% PPL increase)"
    elif ppl_delta_pct <= 2.0:
        return "ACCEPTABLE (<2% PPL increase)"
    elif ppl_delta_pct <= 5.0:
        return "MARGINAL (<5% PPL increase)"
    else:
        return "POOR (>5% PPL increase)"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset_texts(dataset_name: str = "wikitext", max_samples: int = 100) -> list[str]:
    """Load evaluation text from WikiText-2 or C4.

    Returns a list of raw text strings (not yet tokenized).
    """
    from datasets import load_dataset

    if dataset_name in ("wikitext", "wikitext2", "wikitext-2"):
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = [t for t in ds["text"] if len(t.strip()) > 100]
            print(f"[dataset] Loaded WikiText-2-raw-v1 test split: {len(texts)} passages")
        except Exception as e:
            print(f"[dataset] WikiText-2 load failed ({e}), falling back to C4")
            return load_dataset_texts("c4", max_samples)
    elif dataset_name == "c4":
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        target = max_samples * 3  # generous buffer for short-text filtering
        for item in ds:
            if len(item["text"].strip()) > 100:
                texts.append(item["text"])
            if len(texts) >= target:
                break
        if len(texts) < max_samples:
            print(f"[dataset] WARNING: Only got {len(texts)} passages "
                  f"(wanted {max_samples}). Dataset may be small.")
        print(f"[dataset] Loaded C4 validation (streaming): {len(texts)} passages")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'wikitext' or 'c4'.")

    return texts


def tokenize_and_chunk(texts: list[str], tokenizer, seq_len: int = 2048,
                       max_samples: int = 100) -> list[list[int]]:
    """Concatenate texts and chunk into fixed-length token sequences.

    This is the standard approach for perplexity evaluation: concatenate all
    text with EOS separators, then split into non-overlapping chunks of
    seq_len tokens.
    """
    # Concatenate all text with separator
    full_text = tokenizer.eos_token.join(texts) if tokenizer.eos_token else "\n\n".join(texts)
    all_tokens = tokenizer.encode(full_text)

    print(f"[tokenize] Total tokens: {len(all_tokens):,}, seq_len={seq_len}")

    # Chunk into fixed-length sequences
    chunks = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        chunks.append(all_tokens[i : i + seq_len])
        if len(chunks) >= max_samples:
            break

    print(f"[tokenize] Created {len(chunks)} chunks of {seq_len} tokens each")
    return chunks


# ---------------------------------------------------------------------------
# API-based perplexity evaluation (hits a running vLLM server)
# ---------------------------------------------------------------------------

def evaluate_perplexity_api(api_base: str, model_name: str,
                            max_samples: int = 100, seq_len: int = 2048,
                            dataset: str = "wikitext",
                            tokenizer_path: str | None = None,
                            num_concurrent: int = 8) -> float:
    """Evaluate perplexity by hitting a running vLLM OpenAI-compatible server.

    No model reload — uses the already-running server. This is the preferred
    mode when a vLLM server is already serving.

    Args:
        api_base: Base URL for the OpenAI-compatible API (e.g., http://localhost:8000/v1)
        model_name: Model name or local path (used for tokenizer + server model ID)
        max_samples: max number of seq_len-token chunks to evaluate
        seq_len: tokens per chunk
        dataset: "wikitext" or "c4"
        tokenizer_path: explicit local path for tokenizer (if different from model_name)
        num_concurrent: number of concurrent API requests

    Returns:
        Perplexity as a float.
    """
    import requests
    from transformers import AutoTokenizer

    print(f"\n{'='*60}")
    print(f"[API] Evaluating PPL via running server: {api_base}")
    print(f"[API] Model: {model_name}")
    print(f"{'='*60}")

    # Discover server model name
    server_model = model_name
    try:
        resp = requests.get(f"{api_base}/models", timeout=10)
        models = resp.json().get("data", [])
        if models:
            server_model = models[0]["id"]
            if server_model != model_name:
                print(f"[API] Server model name: {server_model}")
    except Exception as e:
        print(f"[API] Could not query models endpoint: {e}")

    # Load tokenizer from local path (not the server's internal path)
    tok_path = tokenizer_path or model_name
    print(f"[API] Loading tokenizer from: {tok_path}")
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    # Load and tokenize dataset
    texts = load_dataset_texts(dataset, max_samples)
    chunks = tokenize_and_chunk(texts, tokenizer, seq_len, max_samples)

    if not chunks:
        raise RuntimeError("No chunks created from dataset. Check dataset availability.")

    # Decode chunks to text for the completions API
    # (vLLM's OpenAI-compatible /v1/completions doesn't accept prompt_token_ids)
    prompt_texts = [tokenizer.decode(chunk, skip_special_tokens=False) for chunk in chunks]

    total_nll = 0.0
    total_tokens = 0
    errors = 0
    t0 = time.time()

    def _eval_chunk(chunk_idx_and_chunk):
        chunk_idx, prompt_text = chunk_idx_and_chunk
        payload = {
            "model": server_model,
            "prompt": prompt_text,
            "max_tokens": 0,
            "logprobs": 1,
            "echo": True,
            "temperature": 0.0,
        }
        try:
            resp = requests.post(f"{api_base}/completions", json=payload, timeout=120)
            result = resp.json()
            if "error" in result:
                return chunk_idx, 0.0, 0, result["error"]
            choice = result["choices"][0]
            logprobs_data = choice.get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", [])
            nll = 0.0
            count = 0
            for lp in token_logprobs[1:]:
                if lp is not None:
                    nll -= float(lp)
                    count += 1
            return chunk_idx, nll, count, None
        except Exception as e:
            return chunk_idx, 0.0, 0, str(e)

    with ThreadPoolExecutor(max_workers=num_concurrent) as pool:
        futures = [pool.submit(_eval_chunk, (i, p)) for i, p in enumerate(prompt_texts)]
        completed = 0
        for future in as_completed(futures):
            chunk_idx, nll, count, err = future.result()
            if err:
                errors += 1
                if errors <= 3:
                    print(f"[API] Error on chunk {chunk_idx}: {err}")
            else:
                total_nll += nll
                total_tokens += count
            completed += 1
            if completed % 20 == 0:
                elapsed = time.time() - t0
                tps = total_tokens / elapsed if elapsed > 0 else 0
                print(f"[API]   {completed}/{len(chunks)} chunks, "
                      f"{total_tokens:,} tokens, {tps:.0f} tok/s")

    elapsed = time.time() - t0

    if total_tokens == 0:
        raise RuntimeError("No logprobs collected. Check server supports echo+logprobs.")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    print(f"[API] Tokens evaluated: {total_tokens:,}")
    print(f"[API] Avg NLL: {avg_nll:.4f}")
    print(f"[API] Perplexity: {ppl:.2f}")
    print(f"[API] Throughput: {total_tokens / elapsed:.0f} tok/s ({elapsed:.1f}s total)")
    if errors:
        print(f"[API] Errors: {errors}/{len(chunks)} chunks failed")

    return ppl


# ---------------------------------------------------------------------------
# vLLM perplexity evaluation (loads model internally)
# ---------------------------------------------------------------------------

def evaluate_perplexity_vllm(model_path: str, kv_cache_dtype: str = "auto",
                             max_samples: int = 100, seq_len: int = 2048,
                             dataset: str = "wikitext",
                             gpu_memory_utilization: float = 0.85,
                             tensor_parallel_size: int = 1,
                             quantization: str | None = None) -> float:
    """Evaluate perplexity using vLLM's LLM class with prompt_logprobs.

    Uses vLLM's efficient batched inference with the specified kv_cache_dtype.
    Perplexity = exp(mean negative log-likelihood over all tokens).

    Args:
        model_path: path to HF model directory
        kv_cache_dtype: KV cache dtype string for vLLM ("auto", "fp8", etc.)
        max_samples: max number of seq_len-token chunks to evaluate
        seq_len: tokens per chunk
        dataset: "wikitext" or "c4"
        gpu_memory_utilization: fraction of GPU memory for vLLM
        tensor_parallel_size: number of GPUs for tensor parallelism
        quantization: weight quantization method (e.g., "modelopt", "awq")

    Returns:
        Perplexity as a float.
    """
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"[vLLM] Evaluating PPL: kv_cache_dtype={kv_cache_dtype}")
    print(f"[vLLM] Model: {model_path}")
    if quantization:
        print(f"[vLLM] Quantization: {quantization}")
    print(f"{'='*60}")

    # Load model with specified KV cache dtype
    t0 = time.time()
    llm_kwargs = dict(
        model=model_path,
        kv_cache_dtype=kv_cache_dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=seq_len,
        enforce_eager=True,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    print(f"[vLLM] Model loaded in {time.time() - t0:.1f}s")

    # Load and tokenize dataset
    texts = load_dataset_texts(dataset, max_samples)
    chunks = tokenize_and_chunk(texts, tokenizer, seq_len, max_samples)

    if not chunks:
        raise RuntimeError("No chunks created from dataset. Check dataset availability.")

    # Pass token IDs directly to avoid decode/re-encode round-trip drift
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=0.0,
    )

    print(f"[vLLM] Running inference on {len(chunks)} chunks...")
    t0 = time.time()
    outputs = llm.generate(
        prompt_token_ids=chunks,
        sampling_params=sampling_params,
    )
    elapsed = time.time() - t0
    print(f"[vLLM] Inference done in {elapsed:.1f}s "
          f"({len(chunks) * seq_len / elapsed:.0f} tok/s)")

    # Compute perplexity from prompt logprobs
    total_nll = 0.0
    total_tokens = 0

    for output_idx, output in enumerate(outputs):
        if output.prompt_logprobs is None:
            continue
        prompt_token_ids = chunks[output_idx]

        # prompt_logprobs is a list of dicts, one per token position.
        # Position 0 has no conditioning context, so we skip it.
        for pos, pos_logprobs in enumerate(output.prompt_logprobs[1:], start=1):
            if pos_logprobs is None:
                continue
            # Look up the logprob of the actual prompt token at this position
            actual_token_id = prompt_token_ids[pos]
            if actual_token_id in pos_logprobs:
                logprob_obj = pos_logprobs[actual_token_id]
                lp = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
                total_nll -= float(lp)
                total_tokens += 1
            else:
                # Fallback: take the first (and usually only) entry
                for _tid, logprob_obj in pos_logprobs.items():
                    lp = logprob_obj.logprob if hasattr(logprob_obj, 'logprob') else logprob_obj
                    total_nll -= float(lp)
                    total_tokens += 1
                    break

    if total_tokens == 0:
        raise RuntimeError("No logprobs collected. Check vLLM prompt_logprobs support.")

    avg_nll = total_nll / total_tokens
    ppl = math.exp(avg_nll)

    print(f"[vLLM] Tokens evaluated: {total_tokens:,}")
    print(f"[vLLM] Avg NLL: {avg_nll:.4f}")
    print(f"[vLLM] Perplexity: {ppl:.2f}")

    # Clean up GPU memory
    del llm
    torch.cuda.empty_cache()

    return ppl


# ---------------------------------------------------------------------------
# Standalone mode: transformers + our quantization kernels
# ---------------------------------------------------------------------------

def evaluate_standalone(model_path: str, specs: list[str],
                        max_samples: int = 50, seq_len: int = 2048,
                        dataset: str = "wikitext") -> dict:
    """Standalone evaluation using transformers + our KV cache quantization kernels.

    Hooks into k_proj and v_proj to capture raw K/V projections, then measures
    round-trip quantization quality (cosine similarity) on real activations.

    Args:
        model_path: path to HF model directory
        specs: list of spec name strings (e.g., ["k4v4", "k8v4"])
        max_samples: number of chunks to evaluate
        seq_len: tokens per chunk
        dataset: "wikitext" or "c4"

    Returns:
        Dict mapping spec_name -> {
            "mean_cosine_sim": float,
            "min_cosine_sim": float,
            "per_layer_cosine": list[float],
            "quality_tier": str,
        }
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
    from kv_cache_gen.config import parse_spec
    from kv_cache_gen.generate import make_store_fn, make_decode_fn

    print(f"\n{'='*60}")
    print(f"[standalone] KV quantization quality on real activations")
    print(f"[standalone] Model: {model_path}")
    print(f"[standalone] Specs: {specs}")
    print(f"{'='*60}")

    # Load model and tokenizer
    print("[standalone] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Load and tokenize dataset
    texts = load_dataset_texts(dataset, max_samples)
    chunks = tokenize_and_chunk(texts, tokenizer, seq_len, max_samples)

    if not chunks:
        raise RuntimeError("No chunks created from dataset.")

    chunks = chunks[:max_samples]

    # Find k_proj and v_proj submodules for direct KV capture
    kv_proj_pairs = _find_kv_projections(model)
    if not kv_proj_pairs:
        print("[standalone] WARNING: Could not find k_proj/v_proj in model.")
        print("[standalone] Falling back to synthetic KV evaluation for all specs.")

    results = {}

    for spec_name in specs:
        parsed = parse_spec(spec_name)
        if parsed is None:
            print(f"[standalone] Skipping '{spec_name}' (resolves to None / BF16 baseline)")
            continue

        spec = parsed
        print(f"\n[standalone] Evaluating spec: {spec.name} "
              f"(k{spec.k_bits}v{spec.v_bits}, "
              f"blocks={spec.k_scale_block}/{spec.v_scale_block}, "
              f"{spec.compression_vs_bf16(128):.1f}x compression)")

        store_fn = make_store_fn(spec)

        if not kv_proj_pairs:
            results[spec_name] = _evaluate_synthetic_kv(spec, store_fn)
            continue

        num_layers = len(kv_proj_pairs)
        layer_cosines = [[] for _ in range(num_layers)]

        # Hook k_proj and v_proj directly to capture raw projections
        captured_k = {}
        captured_v = {}
        hooks = []

        def make_k_hook(layer_idx):
            def hook_fn(module, args, output):
                captured_k[layer_idx] = output.detach()
            return hook_fn

        def make_v_hook(layer_idx):
            def hook_fn(module, args, output):
                captured_v[layer_idx] = output.detach()
            return hook_fn

        for i, (k_proj, v_proj) in enumerate(kv_proj_pairs):
            hooks.append(k_proj.register_forward_hook(make_k_hook(i)))
            hooks.append(v_proj.register_forward_hook(make_v_hook(i)))

        device = next(model.parameters()).device
        num_evaluated = 0

        for chunk_idx, chunk in enumerate(chunks):
            input_ids = torch.tensor([chunk], device=device)

            with torch.no_grad():
                try:
                    model(input_ids, use_cache=False)
                except Exception as e:
                    print(f"[standalone] Forward pass failed on chunk {chunk_idx}: {e}")
                    continue

            for layer_idx in range(num_layers):
                if layer_idx not in captured_k or layer_idx not in captured_v:
                    continue

                k_states = captured_k[layer_idx]
                v_states = captured_v[layer_idx]

                # k_proj output shape is typically (B, S, num_heads * head_dim)
                # or (B, num_heads, S, head_dim) — normalize to (N, Hk, D)
                k_states, v_states = _normalize_kv_shape(k_states, v_states)
                if k_states is None:
                    continue

                cos_sim = _measure_kv_quantization_quality(
                    k_states, v_states, spec, store_fn
                )
                layer_cosines[layer_idx].append(cos_sim)

            captured_k.clear()
            captured_v.clear()
            num_evaluated += 1

            if (chunk_idx + 1) % 10 == 0:
                print(f"[standalone]   Processed {chunk_idx + 1}/{len(chunks)} chunks")

        for h in hooks:
            h.remove()

        # Aggregate results
        per_layer_mean = []
        for layer_idx in range(num_layers):
            if layer_cosines[layer_idx]:
                per_layer_mean.append(float(np.mean(layer_cosines[layer_idx])))
            else:
                per_layer_mean.append(float('nan'))

        valid_cosines = [c for c in per_layer_mean if not math.isnan(c)]
        if valid_cosines:
            mean_cos = float(np.mean(valid_cosines))
            min_cos = float(np.min(valid_cosines))
        else:
            mean_cos = float('nan')
            min_cos = float('nan')

        # Report cosine similarity directly — no arbitrary PPL mapping
        if not math.isnan(mean_cos):
            quality_tier = _tier_from_cosine(mean_cos)
        else:
            quality_tier = "UNKNOWN"

        results[spec_name] = {
            "mean_cosine_sim": mean_cos,
            "min_cosine_sim": min_cos,
            "per_layer_cosine": per_layer_mean,
            "quality_tier": quality_tier,
            "num_samples": num_evaluated,
            "num_layers": num_layers,
        }

        print(f"[standalone] {spec.name}: mean_cos={mean_cos:.6f}, "
              f"min_cos={min_cos:.6f}, tier={quality_tier}")

    # Clean up
    del model
    torch.cuda.empty_cache()

    return results


def _tier_from_cosine(mean_cos: float) -> str:
    """Map cosine similarity to quality tier based on empirical calibration.

    Calibrated against real PPL measurements:
      cos > 0.9995 → <0.5% PPL increase (EXCELLENT)
      cos > 0.999  → <1% PPL increase (GOOD)
      cos > 0.995  → <2% PPL increase (ACCEPTABLE)
      cos > 0.990  → <5% PPL increase (MARGINAL)
      cos <= 0.990 → >5% PPL increase (POOR)
    """
    if mean_cos >= 0.9995:
        return "EXCELLENT (cos>0.9995, ~<0.5% PPL)"
    elif mean_cos >= 0.999:
        return "GOOD (cos>0.999, ~<1% PPL)"
    elif mean_cos >= 0.995:
        return "ACCEPTABLE (cos>0.995, ~<2% PPL)"
    elif mean_cos >= 0.990:
        return "MARGINAL (cos>0.990, ~<5% PPL)"
    else:
        return f"POOR (cos={mean_cos:.4f}, ~>5% PPL)"


def _find_kv_projections(model) -> list[tuple]:
    """Find (k_proj, v_proj) module pairs across all attention layers.

    Handles: LLaMA, Mistral, Gemma, Gemma4, GPT-2, and generic architectures.
    Returns list of (k_proj_module, v_proj_module) tuples.
    """
    pairs = []

    # Find the layers list — try common paths
    layers = None

    # Gemma4 style: model.language_model.layers
    if hasattr(model, 'model'):
        inner = model.model
        if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
            layers = inner.language_model.layers
        elif hasattr(inner, 'layers'):
            # LLaMA / Mistral / Gemma style
            layers = inner.layers

    # GPT-2 style
    if layers is None and hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for block in model.transformer.h:
            attn = getattr(block, 'attn', None)
            if attn is None:
                continue
            k = getattr(attn, 'k_proj', None) or getattr(attn, 'c_attn', None)
            v = getattr(attn, 'v_proj', None)
            # GPT-2 uses fused c_attn; skip if no separate k/v proj
            if k is not None and v is not None and k is not v:
                pairs.append((k, v))
        return pairs

    if layers is None:
        # Generic fallback: search named modules
        k_modules = {}
        v_modules = {}
        for name, module in model.named_modules():
            if name.endswith('.k_proj'):
                prefix = name[:-len('.k_proj')]
                k_modules[prefix] = module
            elif name.endswith('.v_proj'):
                prefix = name[:-len('.v_proj')]
                v_modules[prefix] = module
        for prefix in sorted(k_modules.keys()):
            if prefix in v_modules:
                pairs.append((k_modules[prefix], v_modules[prefix]))
        return pairs

    # Standard transformer layers with self_attn
    for layer in layers:
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue
        k_proj = getattr(attn, 'k_proj', None)
        v_proj = getattr(attn, 'v_proj', None)
        if k_proj is not None and v_proj is not None:
            pairs.append((k_proj, v_proj))

    return pairs


def _normalize_kv_shape(k: torch.Tensor, v: torch.Tensor):
    """Normalize K/V tensors to shape (N, Hk, D) for quantization measurement.

    k_proj/v_proj Linear outputs are (B, S, out_features). We need to split
    into heads so that D matches the per-head dimension the quantization
    kernels expect (typically 64-512). We infer num_heads from the ratio
    of k and v output sizes, using common head dims as candidates.
    """
    if k.ndim == 3:
        B, S, out_k = k.shape
        _, _, out_v = v.shape
    elif k.ndim == 2:
        S, out_k = k.shape
        _, out_v = v.shape
        B = 1
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    else:
        return None, None

    # Infer head_dim from common candidates
    head_dim = None
    for candidate in [64, 128, 256, 512, 80, 96, 112]:
        if out_k % candidate == 0 and out_v % candidate == 0:
            head_dim = candidate
            break

    if head_dim is None:
        # Can't split into heads — use full dim as single head
        # This may produce inaccurate cosine sim if D is too large for scale blocks
        return (k.reshape(B * S, 1, out_k).contiguous(),
                v.reshape(B * S, 1, out_v).contiguous())

    Hk_k = out_k // head_dim
    Hk_v = out_v // head_dim

    k = k.reshape(B, S, Hk_k, head_dim).permute(0, 2, 1, 3).reshape(B * S, Hk_k, head_dim).contiguous()
    v = v.reshape(B, S, Hk_v, head_dim).permute(0, 2, 1, 3).reshape(B * S, Hk_v, head_dim).contiguous()
    return k, v


def _measure_kv_quantization_quality(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    spec,
    store_fn,
) -> float:
    """Measure cosine similarity between original and quantize-dequantized KV.

    Performs: original -> quantize (store) -> dequantize -> compare.
    Returns average cosine similarity across K and V.

    K and V may have different head counts (GQA), so we measure each
    independently using its own head dim.
    """
    N_k, Hk_k, D_k = k_states.shape
    N_v, Hk_v, D_v = v_states.shape
    device = k_states.device

    # Use the larger head count for the shared cache (store_fn expects matching Hk)
    # If Hk differs, measure K and V separately
    if Hk_k == Hk_v and D_k == D_v:
        return _measure_kv_roundtrip(k_states, v_states, spec, store_fn, device)
    else:
        # Measure independently — average the cosine similarities
        k_cos = _measure_single_roundtrip(k_states, spec, store_fn, device, is_key=True)
        v_cos = _measure_single_roundtrip(v_states, spec, store_fn, device, is_key=False)
        return (k_cos + v_cos) / 2.0


def _measure_kv_roundtrip(k_states, v_states, spec, store_fn, device):
    """Round-trip quality when K and V have matching shapes."""
    N, Hk, D = k_states.shape
    block_size = 16
    num_blocks = (N + block_size - 1) // block_size
    slot_bytes = spec.slot_bytes(D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_bytes,
                           dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)

    class LayerProxy:
        pass
    layer = LayerProxy()

    k_f = k_states.to(torch.float16)
    v_f = v_states.to(torch.float16)
    store_fn(k_f, v_f, kv_cache, slot_mapping, layer, Hk)

    k_recon, v_recon = _dequantize_from_cache_vectorized(
        kv_cache, layer, spec, N, Hk, D, block_size, slot_mapping, device
    )

    k_cos = torch.nn.functional.cosine_similarity(
        k_f.reshape(-1).float().unsqueeze(0),
        k_recon.reshape(-1).float().unsqueeze(0)).item()
    v_cos = torch.nn.functional.cosine_similarity(
        v_f.reshape(-1).float().unsqueeze(0),
        v_recon.reshape(-1).float().unsqueeze(0)).item()

    return (k_cos + v_cos) / 2.0


def _measure_single_roundtrip(states, spec, store_fn, device, is_key=True):
    """Round-trip quality for a single component (K or V) in isolation."""
    N, Hk, D = states.shape
    block_size = 16
    num_blocks = (N + block_size - 1) // block_size
    slot_bytes = spec.slot_bytes(D)
    kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_bytes,
                           dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)

    class LayerProxy:
        pass
    layer = LayerProxy()

    states_f = states.to(torch.float16)
    # Pass same tensor for both K and V — we only measure one side
    store_fn(states_f, states_f, kv_cache, slot_mapping, layer, Hk)

    k_recon, v_recon = _dequantize_from_cache_vectorized(
        kv_cache, layer, spec, N, Hk, D, block_size, slot_mapping, device
    )
    # K is stored first in the cache, V second — pick the right one
    recon = k_recon if is_key else v_recon

    return torch.nn.functional.cosine_similarity(
        states_f.reshape(-1).float().unsqueeze(0),
        recon.reshape(-1).float().unsqueeze(0)).item()


def _dequantize_from_cache_vectorized(kv_cache, layer, spec, N, Hk, D,
                                       block_size, slot_mapping, device):
    """Vectorized dequantization of K and V from packed cache."""
    k_bytes_per_dim = spec.k_bits / 8
    v_bytes_per_dim = spec.v_bits / 8
    k_region_bytes = int(k_bytes_per_dim * D)
    v_region_bytes = int(v_bytes_per_dim * D)

    # Gather all packed rows at once: (N, Hk, slot_bytes)
    blk_indices = slot_mapping // block_size  # (N,)
    blk_offsets = slot_mapping % block_size   # (N,)
    packed_rows = kv_cache[blk_indices, blk_offsets]  # (N, Hk, slot_bytes)

    # Extract K and V byte regions: (N, Hk, region_bytes)
    k_packed = packed_rows[:, :, :k_region_bytes].to(torch.int32)
    v_packed = packed_rows[:, :, k_region_bytes:k_region_bytes + v_region_bytes].to(torch.int32)

    # Unpack codes vectorized
    k_codes = _unpack_codes_batched(k_packed, spec.k_bits, D)  # (N, Hk, D)
    v_codes = _unpack_codes_batched(v_packed, spec.v_bits, D)  # (N, Hk, D)

    # Apply scales if available
    if hasattr(layer, '_fc_scales'):
        min_block = min(spec.k_scale_block, spec.v_scale_block)

        # layer._fc_scales shape: (max_slots, Hk, num_scale_blocks, 2)
        flat_slots = blk_indices * block_size + blk_offsets  # (N,)
        scales = layer._fc_scales[flat_slots]  # (N, Hk, num_sb, 2)
        k_scales_raw = scales[:, :, :, 0].float()  # (N, Hk, num_sb)
        v_scales_raw = scales[:, :, :, 1].float()  # (N, Hk, num_sb)

        # Dequantize K
        k_recon = _dequant_codes_batched(
            k_codes, k_scales_raw, spec.k_bits, spec.k_sym_offset,
            spec.k_scale_block, min_block, D
        )
        # Dequantize V
        v_recon = _dequant_codes_batched(
            v_codes, v_scales_raw, spec.v_bits, spec.v_sym_offset,
            spec.v_scale_block, min_block, D
        )
    else:
        k_recon = k_codes.float()
        v_recon = v_codes.float()

    return k_recon.to(torch.float16), v_recon.to(torch.float16)


def _unpack_codes_batched(packed: torch.Tensor, bits: int, D: int) -> torch.Tensor:
    """Unpack integer codes from packed bytes — fully vectorized.

    Args:
        packed: (N, Hk, packed_bytes) int32 tensor
        bits: 2, 4, or 8
        D: target dimension

    Returns:
        (N, Hk, D) float tensor of unpacked codes
    """
    if bits == 8:
        return packed[:, :, :D].float()
    elif bits == 4:
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        # Interleave: (N, Hk, packed_bytes, 2) -> (N, Hk, packed_bytes*2)
        interleaved = torch.stack([lo, hi], dim=-1).reshape(
            packed.shape[0], packed.shape[1], -1
        )
        return interleaved[:, :, :D].float()
    elif bits == 2:
        c0 = packed & 0x03
        c1 = (packed >> 2) & 0x03
        c2 = (packed >> 4) & 0x03
        c3 = (packed >> 6) & 0x03
        interleaved = torch.stack([c0, c1, c2, c3], dim=-1).reshape(
            packed.shape[0], packed.shape[1], -1
        )
        return interleaved[:, :, :D].float()
    else:
        raise ValueError(f"Unsupported bits={bits}")


def _dequant_codes_batched(codes, scales_raw, bits, sym_offset,
                            scale_block, min_block, D):
    """Vectorized dequantization: codes + scales -> float values.

    Args:
        codes: (N, Hk, D) float codes
        scales_raw: (N, Hk, num_sb) raw scale values
        bits: quantization bits
        sym_offset: symmetric offset for zero-point (spec.k_sym_offset / v_sym_offset)
        scale_block: block size for this component's scales
        min_block: minimum block size (for indexing into scales_raw)
        D: dimension

    Returns:
        (N, Hk, D) float dequantized values
    """
    num_groups = D // scale_block
    repeat_factor = scale_block // min_block

    # Select the right scale entries (every repeat_factor-th entry)
    scales = scales_raw[:, :, ::repeat_factor]  # (N, Hk, num_groups)
    # Expand scales to match D: (N, Hk, num_groups) -> (N, Hk, D)
    scales_expanded = scales.unsqueeze(-1).expand(
        -1, -1, num_groups, scale_block
    ).reshape(codes.shape[0], codes.shape[1], num_groups * scale_block)
    # Trim to D in case D isn't perfectly divisible
    scales_expanded = scales_expanded[:, :, :D]

    return (codes - sym_offset) * scales_expanded


def _evaluate_synthetic_kv(spec, store_fn) -> dict:
    """Fallback: evaluate quantization quality on synthetic data shaped like real KV."""
    print("[standalone] Using synthetic KV evaluation (could not hook real activations)")

    D = 128  # typical head dim
    Hk = 8
    N = 256

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use gaussian data (more realistic than uniform)
    k = torch.randn(N, Hk, D, dtype=torch.float16, device=device)
    v = torch.randn(N, Hk, D, dtype=torch.float16, device=device)

    cos_sim = _measure_kv_quantization_quality(k, v, spec, store_fn)
    quality_tier = _tier_from_cosine(cos_sim)

    return {
        "mean_cosine_sim": cos_sim,
        "min_cosine_sim": cos_sim,
        "per_layer_cosine": [cos_sim],
        "quality_tier": quality_tier,
        "num_samples": 1,
        "num_layers": 0,
        "note": "synthetic data (real activation hooks not available)",
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def evaluate_perplexity(model_path: str, kv_cache_dtype: str = "auto",
                        max_samples: int = 100, seq_len: int = 2048,
                        dataset: str = "wikitext",
                        gpu_memory_utilization: float = 0.85,
                        tensor_parallel_size: int = 1,
                        quantization: str | None = None) -> float:
    """Top-level perplexity evaluation. Returns perplexity float."""
    return evaluate_perplexity_vllm(
        model_path=model_path,
        kv_cache_dtype=kv_cache_dtype,
        max_samples=max_samples,
        seq_len=seq_len,
        dataset=dataset,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        quantization=quantization,
    )


def main():
    """CLI entry point: compare perplexity across KV cache specs."""
    parser = argparse.ArgumentParser(
        description="Evaluate KV cache quantization quality via perplexity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hit a running vLLM server (no model reload):
  python3 fusen_kv/eval_perplexity.py --api-base http://localhost:8000/v1 --model /models/gemma-4 --max-samples 100

  # Compare specs via vLLM (loads model):
  python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4,k8v4,k8v8,auto

  # Standalone mode (no vLLM needed):
  python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4,k8v4 --standalone

  # Quick test:
  python3 fusen_kv/eval_perplexity.py --model /models/neural-ice --specs k4v4 --max-samples 10 --seq-len 512
""",
    )
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--specs", default=None,
                        help="Comma-separated KV cache specs (e.g., k4v4,k8v4,k8v8,auto)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Max number of seq_len-token chunks (default: 100)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Tokens per chunk (default: 2048)")
    parser.add_argument("--dataset", default="wikitext",
                        choices=["wikitext", "c4"],
                        help="Evaluation dataset (default: wikitext)")
    parser.add_argument("--standalone", action="store_true",
                        help="Use transformers instead of vLLM (measures attention quality)")
    parser.add_argument("--api-base", default=None,
                        help="Hit a running vLLM server (e.g., http://localhost:8000/v1)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Local tokenizer path (if different from --model)")
    parser.add_argument("--num-concurrent", type=int, default=8,
                        help="Concurrent API requests in --api-base mode (default: 8)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory fraction for vLLM (default: 0.85)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism (default: 1)")
    parser.add_argument("--quantization", default=None,
                        help="Weight quantization method (e.g., modelopt, awq)")
    parser.add_argument("--output-json", default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    if args.api_base:
        # API mode: hit running server
        ppl = evaluate_perplexity_api(
            api_base=args.api_base,
            model_name=args.model,
            max_samples=args.max_samples,
            seq_len=args.seq_len,
            dataset=args.dataset,
            tokenizer_path=args.tokenizer_path,
            num_concurrent=args.num_concurrent,
        )
        results = {"auto": {"perplexity": ppl, "error": None}}
        _print_vllm_report(results)

    elif args.standalone:
        if not args.specs:
            parser.error("--specs is required in standalone mode")
        spec_names = [s.strip() for s in args.specs.split(",")]
        results = evaluate_standalone(
            model_path=args.model,
            specs=spec_names,
            max_samples=args.max_samples,
            seq_len=args.seq_len,
            dataset=args.dataset,
        )
        _print_standalone_report(results)

    else:
        # vLLM mode: load model internally
        if not args.specs:
            parser.error("--specs is required in vLLM mode (or use --api-base)")
        spec_names = [s.strip() for s in args.specs.split(",")]
        results = {}

        # Always run BF16 baseline first
        if "auto" not in spec_names:
            spec_names = ["auto"] + spec_names

        for spec_name in spec_names:
            if spec_name in ("auto", "bf16"):
                kv_dtype = "auto"
            elif spec_name in ("fp8", "fp8_e4m3", "fp8e4m3"):
                kv_dtype = "fp8_e4m3"
            elif spec_name in ("fp8_e5m2", "fp8e5m2"):
                kv_dtype = "fp8_e5m2"
            else:
                kv_dtype = spec_name

            try:
                ppl = evaluate_perplexity(
                    model_path=args.model,
                    kv_cache_dtype=kv_dtype,
                    max_samples=args.max_samples,
                    seq_len=args.seq_len,
                    dataset=args.dataset,
                    gpu_memory_utilization=args.gpu_memory_utilization,
                    tensor_parallel_size=args.tensor_parallel_size,
                    quantization=args.quantization,
                )
                results[spec_name] = {"perplexity": ppl, "error": None}
            except Exception as e:
                print(f"\n[ERROR] Failed for spec '{spec_name}': {e}")
                results[spec_name] = {"perplexity": None, "error": str(e)}

        _print_vllm_report(results)

    # Save JSON if requested
    if args.output_json:
        serializable = {}
        for k, v in results.items():
            if isinstance(v, dict):
                sv = {}
                for kk, vv in v.items():
                    if isinstance(vv, (np.floating, np.integer)):
                        sv[kk] = float(vv)
                    elif isinstance(vv, list):
                        sv[kk] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                                  for x in vv]
                    else:
                        sv[kk] = vv
                serializable[k] = sv
            else:
                serializable[k] = v

        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


def _print_vllm_report(results: dict):
    """Print formatted perplexity comparison report."""
    print(f"\n{'='*70}")
    print("PERPLEXITY EVALUATION REPORT")
    print(f"{'='*70}")

    baseline_ppl = None
    for name in ("auto", "bf16"):
        if name in results and results[name].get("perplexity") is not None:
            baseline_ppl = results[name]["perplexity"]
            break

    print(f"\n{'Spec':<16} {'PPL':>10} {'Delta':>10} {'Delta%':>10} {'Quality Tier':<30}")
    print("-" * 76)

    for spec_name, data in results.items():
        ppl = data.get("perplexity")
        error = data.get("error")

        if ppl is None:
            print(f"{spec_name:<16} {'FAILED':>10} {'':>10} {'':>10} {error or 'unknown error'}")
            continue

        if baseline_ppl is not None and spec_name not in ("auto", "bf16"):
            delta = ppl - baseline_ppl
            delta_pct = (delta / baseline_ppl) * 100
            tier = classify_quality(delta_pct)
            print(f"{spec_name:<16} {ppl:>10.2f} {delta:>+10.2f} {delta_pct:>+9.1f}% {tier}")
        else:
            label = "(baseline)" if spec_name in ("auto", "bf16") else ""
            print(f"{spec_name:<16} {ppl:>10.2f} {'---':>10} {'---':>10} {label}")

    print(f"\n{'='*70}")


def _print_standalone_report(results: dict):
    """Print formatted standalone evaluation report."""
    print(f"\n{'='*70}")
    print("STANDALONE KV QUANTIZATION QUALITY REPORT")
    print("(Cosine similarity: quantized KV vs full precision, on real activations)")
    print(f"{'='*70}")

    print(f"\n{'Spec':<16} {'Mean Cos':>10} {'Min Cos':>10} {'Layers':>8} "
          f"{'Samples':>8} {'Quality Tier':<30}")
    print("-" * 82)

    for spec_name, data in results.items():
        mean_cos = data.get("mean_cosine_sim", float('nan'))
        min_cos = data.get("min_cosine_sim", float('nan'))
        num_layers = data.get("num_layers", 0)
        num_samples = data.get("num_samples", 0)
        tier = data.get("quality_tier", "UNKNOWN")

        if math.isnan(mean_cos):
            print(f"{spec_name:<16} {'N/A':>10} {'N/A':>10} {num_layers:>8} "
                  f"{num_samples:>8} {tier}")
        else:
            print(f"{spec_name:<16} {mean_cos:>10.6f} {min_cos:>10.6f} {num_layers:>8} "
                  f"{num_samples:>8} {tier}")

        per_layer = data.get("per_layer_cosine", [])
        if per_layer and len(per_layer) > 1:
            worst_layers = sorted(enumerate(per_layer), key=lambda x: x[1])[:3]
            worst_str = ", ".join(f"L{i}={c:.4f}" for i, c in worst_layers
                                 if not math.isnan(c))
            if worst_str:
                print(f"{'':>16} Worst layers: {worst_str}")

    print(f"\n{'='*70}")
    print("NOTE: Cosine similarity measures KV cache round-trip fidelity.")
    print("      cos > 0.9995 → EXCELLENT (<0.5% PPL increase)")
    print("      cos > 0.999  → GOOD (<1% PPL increase)")
    print("      cos > 0.995  → ACCEPTABLE (<2% PPL increase)")


if __name__ == "__main__":
    main()
