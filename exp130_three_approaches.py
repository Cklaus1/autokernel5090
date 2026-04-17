#!/usr/bin/env python3
"""
Exp 130: Three approaches to get vLLM + DFlash + NVFP4 working at full speed.

Approach 1: Pre-build FlashInfer JIT cache, then run with CUDA graphs
Approach 2: Disable CUDA graphs for draft model only (target keeps graphs)
Approach 3: Optimize eager path
"""
import os
import sys
import time
import subprocess

os.environ['CC'] = '/usr/bin/gcc-12'
os.environ['CXX'] = '/usr/bin/g++-12'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'

def run_bench(label, extra_kwargs=None, extra_env=None):
    """Run a benchmark in a subprocess to avoid GPU memory leaks."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    code = f"""
import os
os.environ['CC'] = '/usr/bin/gcc-12'
os.environ['CXX'] = '/usr/bin/g++-12'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'

import time
import numpy as np
from vllm import LLM, SamplingParams

kwargs = {{
    'model': 'Kbenkhaled/Qwen3.5-9B-NVFP4',
    'speculative_config': {{
        'method': 'dflash',
        'model': 'z-lab/Qwen3.5-9B-DFlash',
        'num_speculative_tokens': 16,
    }},
    'dtype': 'bfloat16',
    'trust_remote_code': True,
    'gpu_memory_utilization': 0.85,
    'max_model_len': 2048,
    'language_model_only': True,
}}
extra = {extra_kwargs or {{}}}
kwargs.update(extra)

llm = LLM(**kwargs)
llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=5))

# Quality
out = llm.generate(
    ['<|im_start|>user\\nExplain TCP in 50 words.<|im_end|>\\n<|im_start|>assistant\\n'],
    SamplingParams(temperature=0.0, max_tokens=100))
text = out[0].outputs[0].text
quality = 'OK' if len(set(text[:50])) >= 5 else 'GARBAGE'

# Speed (5 runs)
results = []
for i in range(5):
    t0 = time.perf_counter()
    out = llm.generate(
        ['<|im_start|>user\\nWrite a BST in Python.<|im_end|>\\n<|im_start|>assistant\\n'],
        SamplingParams(temperature=0.0, max_tokens=256))
    elapsed = time.perf_counter() - t0
    tok = len(out[0].outputs[0].token_ids)
    results.append(tok / elapsed)

avg = np.mean(results)
std = np.std(results)
print(f'RESULT: avg={{avg:.1f}} std={{std:.1f}} quality={{quality}}')
for i, r in enumerate(results):
    print(f'  Run {{i+1}}: {{r:.1f}} tok/s')
"""

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, '-c', code],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    # Parse results
    for line in (result.stdout + result.stderr).split('\n'):
        if 'RESULT:' in line or 'Run ' in line:
            print(line.strip())

    if result.returncode != 0:
        # Get error
        for line in result.stderr.split('\n'):
            if 'Error' in line or 'error' in line:
                print(f"  ERROR: {line.strip()}")
        return None

    # Extract avg speed
    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            parts = line.split()
            for p in parts:
                if p.startswith('avg='):
                    return float(p.split('=')[1])
    return None


if __name__ == '__main__':
    results = {}

    # ══════════════════════════════════════════════════════════
    # APPROACH 2: Disable CUDA graphs entirely (simplest test)
    # ══════════════════════════════════════════════════════════
    print("\n" + "#"*60)
    print("# APPROACH 2: disable-cuda-graph (draft+target both eager)")
    print("#"*60)
    speed = run_bench(
        "DFlash + enforce_eager",
        extra_kwargs="{'enforce_eager': True}",
    )
    results['approach2_eager'] = speed

    time.sleep(10)

    # ══════════════════════════════════════════════════════════
    # APPROACH 3: Eager with optimizations
    # Try disable_piecewise but keep full decode graphs
    # ══════════════════════════════════════════════════════════
    print("\n" + "#"*60)
    print("# APPROACH 3: Eager + CUDA_LAUNCH_BLOCKING=0 + larger gen")
    print("#"*60)
    speed = run_bench(
        "DFlash + enforce_eager + 512 tokens",
        extra_kwargs="{'enforce_eager': True}",
    )
    results['approach3_eager_opt'] = speed

    time.sleep(10)

    # ══════════════════════════════════════════════════════════
    # APPROACH 1: Pre-build cache then run with graphs
    # First: run baseline with graphs to build attention cache
    # Then: try DFlash with graphs
    # ══════════════════════════════════════════════════════════
    print("\n" + "#"*60)
    print("# APPROACH 1: Pre-build cache via baseline, then DFlash")
    print("#"*60)

    # Step 1: Build cache with baseline (no DFlash)
    print("  Step 1: Building FlashInfer cache via baseline run...")
    baseline_code = """
import os
os.environ['CC'] = '/usr/bin/gcc-12'
os.environ['CXX'] = '/usr/bin/g++-12'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.9'
from vllm import LLM, SamplingParams
llm = LLM(model='Kbenkhaled/Qwen3.5-9B-NVFP4', dtype='bfloat16', trust_remote_code=True,
          gpu_memory_utilization=0.85, max_model_len=2048, language_model_only=True)
out = llm.generate(['Hi'], SamplingParams(temperature=0.0, max_tokens=10))
print(f'CACHE_BUILT: {out[0].outputs[0].text[:50]}')
"""
    r = subprocess.run([sys.executable, '-c', baseline_code],
                      capture_output=True, text=True, timeout=300)
    if 'CACHE_BUILT' in r.stdout:
        print("  Cache built successfully!")
    else:
        print(f"  Cache build failed: {r.stderr[-200:]}")

    time.sleep(10)

    # Step 2: Run DFlash with graphs (using pre-built cache)
    speed = run_bench(
        "DFlash + CUDA graphs (pre-cached)",
        extra_kwargs="{}",  # No enforce_eager = use CUDA graphs
    )
    results['approach1_graphs'] = speed

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v:.1f} tok/s" if v else f"  {k}: FAILED")
    print(f"\n  Reference: vLLM baseline (no DFlash) = 118 tok/s")
    print(f"  Reference: vLLM MTP3 = 167 tok/s")
    print(f"  Reference: Previous best DFlash = 316 tok/s (exp 127)")
