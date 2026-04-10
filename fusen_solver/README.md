# fusen-solver

Universal parallel AI problem solver. Given a coding problem and a codebase, fusen-solver dispatches N independent LLM agents in parallel, scores every solution, and returns the best one. Runs with any OpenAI-compatible backend — local vLLM, Anthropic, or OpenAI.

## Install

```bash
pip install fusen-solver
# or with PyYAML config support:
pip install "fusen-solver[yaml]"
```

## Quick start — Python API

```python
import asyncio
from fusen_solver import FusenSolver, Problem
from fusen_solver.backends import OpenAIBackend

solver = FusenSolver(
    backend=OpenAIBackend(api_key="sk-..."),
    default_n=4,   # 4 parallel agents
)

result = asyncio.run(solver.solve(Problem(
    description="Fix the race condition in the request handler",
    context={"server.py": open("server.py").read()},
    problem_type="bug_fix",
)))

print(result.best.explanation)
print(result.best.code_changes)
```

## Quick start — vLLM backend (local GPU)

vLLM batches all N agent requests in a single GPU call and uses prefix caching so the shared codebase context is encoded only once.

```python
from fusen_solver import FusenSolver, Problem
from fusen_solver.backends import VLLMBackend

solver = FusenSolver(
    backend=VLLMBackend(
        base_url="http://localhost:8000/v1",
        model="gemma-4-27B-it",
    ),
    default_n=8,
)
```

## Quick start — CLI

```bash
# Solve a problem described as a string
fusen-solver solve --problem "Add input validation to login endpoint" \
                   --codebase ./src/ \
                   --agents 4

# Interactive mode — describe the problem interactively
fusen-solver interactive --codebase ./src/

# Show solve statistics / learning history
fusen-solver stats
```

## Quick start — REST API server

```bash
python -m fusen_solver.integrations.api --port 8080
```

```bash
curl -X POST http://localhost:8080/solve \
  -H 'Content-Type: application/json' \
  -d '{"problem": "Refactor authenticate() to be async", "context": {"auth.py": "..."}}'
```

## Configuration

Create `~/.fusen_solver/config.yaml`:

```yaml
backends:
  primary:
    type: vllm
    url: http://localhost:8000/v1
    model: gemma-4-27B-it
    max_context: 131072

strategy:
  default_n: 4      # parallel agents per solve
  auto_n: true      # auto-scale N based on problem complexity

scoring:
  test_weight: 0.4
  review_weight: 0.3
  diff_weight: 0.15
  syntax_weight: 0.1
  confidence_weight: 0.05

learning:
  enabled: true
  db_path: ~/.fusen_solver/history.json
```

Supported backend types: `vllm`, `openai`, `anthropic`, `ollama`.

## Supported backends

| Backend | Class | Notes |
|---|---|---|
| vLLM (local) | `VLLMBackend` | Fastest; batches all agents; prefix cache |
| OpenAI | `OpenAIBackend` | GPT-4o, o1, etc. |
| Anthropic | `AnthropicBackend` | Claude 3.5 / Claude 4 |
| Ollama | `OllamaBackend` | Fully local, no API key |
| Multi | `MultiBackend` | Fan-out across multiple backends |

## How it works

1. **Strategy selection** — picks prompt strategies for the problem type (bug fix, feature, refactor, test generation, …)
2. **Parallel generation** — dispatches N async LLM calls with different strategies/temperatures
3. **Scoring** — ranks solutions by test pass rate, code review heuristics, diff size, syntax validity
4. **Learning** — remembers which strategies worked best for each problem type and adapts future runs

## License

MIT
