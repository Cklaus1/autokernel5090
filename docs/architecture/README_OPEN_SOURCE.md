# AutoKernel + FusenKV + FusenSolver

Three open-source packages from the AutoKernel GPU kernel optimization project:

| Package | What it does | Key result |
|---|---|---|
| **autokernel** | Autonomous Triton kernel optimizer | 2.95x fused kernel speedup over PyTorch |
| **fusen-kv** | 4x KV cache compression for vLLM | 6,685 tok/s on Gemma 4 31B (H100) |
| **fusen-solver** | Parallel AI problem solver | N agents × prefix-cached LLM inference |

---

## autokernel

AutoKernel profiles a PyTorch model, extracts the top-N bottleneck operations, and iteratively rewrites their Triton kernels. Each candidate is benchmarked for correctness and throughput; improvements are kept, regressions reverted. The loop runs autonomously for 300+ experiments per kernel.

### Quick start

```bash
git clone https://github.com/autokernel/autokernel
cd autokernel
uv sync
uv run prepare.py           # one-time: create test data + baselines

# Profile a model and extract bottlenecks
uv run profiler.py --model models/llama_7b.py --class-name LlamaModel \
                   --input-shape 1,2048 --dtype float16
uv run extract.py --top 5

# Autonomous optimization loop
uv run orchestrate.py next   # which kernel to work on
# edit kernel.py ...
uv run bench.py > run.log 2>&1
uv run orchestrate.py record kernel.py 142.3 keep "wider BLOCK_SIZE_M"

# Final report
uv run orchestrate.py report
uv run analysis.py           # generates progress.png + report.md
```

### Key results

- **W4A16 matmul:** 328 TFLOPS (157% of dense FP16 peak via cuBLAS dequant split)
- **NVFP4 matmul:** 1,270 TFLOPS via `torch._scaled_mm_v2` (3.9x best Triton kernel)
- **Fused MLP kernel:** 2.95x speedup over unfused PyTorch baseline

### Supported kernel types

`matmul`, `softmax`, `layernorm`, `rmsnorm`, `flash_attention`, `fused_mlp`, `cross_entropy`, `rotary_embedding`, `reduce`, `quantized_matmul_w4a16`, `dequantize_fused_gemm`, `nvfp4_matmul`

---

## fusen-kv

FusenKV plugs into vLLM's general plugin system to replace the default FP16 KV cache with a quantized layout. K and V tensors are stored in 2–8 bit with per-block FP16 scales, reducing KV cache VRAM by up to 4x. No modifications to vLLM source required.

### Install

```bash
pip install fusen-kv
# or:
pip install "fusen-kv[vllm]"
```

### Quick start

```bash
# Auto-select best compression spec
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype fusen

# Explicit: 4-bit K, 4-bit V, block size 32
vllm serve meta-llama/Llama-3-8B-Instruct --kv-cache-dtype k4v4b32

# Debug mode (bounds-checking assertions, zero cost when disabled)
FUSEN_DEBUG=1 vllm serve ... --kv-cache-dtype k4v4b32
```

### Supported formats

| dtype | K bits | V bits | Block | Compression |
|---|---|---|---|---|
| `k4v4b64` | 4 | 4 | 64 | ~4x |
| `k8v4b32` | 8 | 4 | 32 | ~3x |
| `k8v8b32` | 8 | 8 | 32 | ~2x |
| `k4v2b16` | 4 | 2 | 16 | ~6x |

### Key results

- **6,685 tok/s** decode throughput on Gemma 4 31B (H100 80 GB SXM) with `k4v4b64`
- **4x KV cache compression** — fits 4x longer contexts or 4x more concurrent sequences
- **Compatible with NVFP4 weight quantization** (orthogonal systems)
- Triton decode kernel + optional pre-compiled CUDA C++ kernel for CUDA graph compatibility

### HuggingFace model

The experiments used `gemma-4-27B-it` quantized to NVFP4 weights:
`google/gemma-4-27b-it` + `modelopt` NVFP4 quantization

---

## fusen-solver

FusenSolver dispatches N independent LLM agents in parallel on the same coding problem, scores every solution, and returns the best one. Works with any OpenAI-compatible backend. When paired with a local vLLM server, all N agents share a prefix-cached encoding of the codebase context — one GPU pass for the context, N independent decode streams.

### Install

```bash
pip install fusen-solver
```

### Quick start — Python API

```python
import asyncio
from fusen_solver import FusenSolver, Problem
from fusen_solver.backends import OpenAIBackend

solver = FusenSolver(
    backend=OpenAIBackend(api_key="sk-..."),
    default_n=4,
)

result = asyncio.run(solver.solve(Problem(
    description="Fix the race condition in the request handler",
    context={"server.py": open("server.py").read()},
    problem_type="bug_fix",
)))

print(result.best.explanation)
```

### Quick start — local vLLM backend

```python
from fusen_solver import FusenSolver, Problem
from fusen_solver.backends import VLLMBackend

solver = FusenSolver(
    backend=VLLMBackend(
        base_url="http://localhost:8000/v1",
        model="gemma-4-27B-it",
    ),
    default_n=8,  # 8 agents, batched on the same GPU
)
```

### Quick start — CLI

```bash
fusen-solver solve --problem "Add input validation to the login endpoint" \
                   --codebase ./src/ \
                   --agents 4
```

### Quick start — REST API

```bash
python -m fusen_solver.integrations.api --port 8080

curl -X POST http://localhost:8080/solve \
  -H 'Content-Type: application/json' \
  -d '{"problem": "Refactor authenticate() to async", "context": {"auth.py": "..."}}'
```

### How it works

1. **Strategy selection** — picks prompt strategies appropriate for the problem type (bug fix, feature, refactor, test generation)
2. **Parallel generation** — dispatches N async LLM calls with different strategies and temperatures
3. **Scoring** — ranks solutions by test pass rate, code review heuristics, diff size, and syntax validity
4. **Learning** — tracks which strategies worked for each problem type and adapts future runs

### Supported backends

`VLLMBackend` (local GPU, fastest), `OpenAIBackend`, `AnthropicBackend`, `OllamaBackend` (fully local), `MultiBackend` (fan-out)

---

## Repository structure

```
autokernel/
├── autokernel/          # kernel optimizer (profiler, extractor, bench harness)
│   ├── kernels/         # 12 Triton kernel templates
│   └── models/          # self-contained PyTorch model definitions
├── fusen_kv/            # KV cache compression plugin for vLLM
│   ├── backend.py       # FusenKV attention backend (Triton + optional CUDA C++)
│   ├── plugin.py        # vLLM plugin entry point
│   └── spec_resolver.py # dtype string → compression spec mapping
└── fusen_solver/        # parallel AI problem solver
    ├── core/            # solver, interfaces
    ├── backends/        # vLLM, OpenAI, Anthropic, Ollama adapters
    ├── strategies/      # prompt strategy engine
    ├── scoring/         # solution ranking
    └── integrations/    # CLI + REST API
```

## License

- `fusen-kv`: Apache 2.0
- `fusen-solver`: MIT
- `autokernel`: MIT
