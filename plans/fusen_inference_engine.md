# Fusen Inference Engine

## Vision

One API endpoint. One config line. The system handles everything else.

```yaml
# What the user sees
endpoint: https://api.fusen.world/v1/chat/completions
quality: high
```

```
# What the user sends
curl https://api.fusen.world/v1/chat/completions \
  -H "Authorization: Bearer $KEY" \
  -d '{"messages": [{"role": "user", "content": "..."}]}'
```

OpenAI-compatible API. Drop-in replacement for any existing integration. Behind it, the engine makes every decision automatically:

- **Which model** — routes simple tasks to a fast small model, complex tasks to a powerful large model
- **Which GPU config** — KV cache format, CUDA graphs, batch size, concurrency tuned per-request
- **Which kernel** — generated from spec, autotuned for the hardware, fused for maximum throughput
- **When to improve** — captures every request as training data, distills large→small, fine-tunes on user feedback, retrains the router

The user never picks a model, never configures vLLM, never tunes concurrency, never writes a kernel. They set a quality level and a budget. The system optimizes everything else and gets better over time.

### User Experience — Simpler Than Ollama

Ollama's breakthrough was eliminating config. The user runs `ollama run gemma3` and it works. But the user still picks a model — and has no idea if it's the best one for their GPU, their task, or their quality needs.

Fusen eliminates even that choice:

```bash
# Install
curl -fsSL https://fusen.world/install.sh | sh

# Run — one command, zero choices
fusen
```

What happens:
```
$ fusen
Detecting hardware... RTX 5090 (32GB)
Selecting optimal model... gemma-4-26B-A4B-AWQ (128K context)
Downloading... ████████████████ 13 GB
Configuring... FP8 KV cache, C=39, CUDA graphs ✓
Starting... http://localhost:8000

Ready. OpenAI-compatible API.
  Chat:    fusen chat
  API:     curl http://localhost:8000/v1/chat/completions
  Status:  fusen status
```

The system detected the GPU, picked the best model that fits, selected the optimal KV cache format, tuned concurrency to the measured sweet spot, captured CUDA graphs, and started serving. Two GPUs → larger model. 8GB VRAM → smaller model. No GPU → CPU mode.

**The full CLI:**

```bash
# Zero-config (auto-detects everything)
fusen                               # start serving

# Chat
fusen chat                          # interactive terminal chat
fusen chat "what is 2+2"            # one-shot

# Quality tradeoff (the only choice a user might make)
fusen --quality high                # best model, max accuracy
fusen --quality fast                # smallest model, lowest latency
fusen --quality balanced            # default

# Hardware constraints
fusen --budget 8gb                  # fit in 8GB VRAM
fusen --budget cpu                  # CPU-only

# Serve
fusen serve                         # OpenAI-compatible, localhost:8000
fusen serve --port 3000             # custom port

# Monitor
fusen status                        # model, throughput, KV usage, uptime
fusen bench                         # run performance benchmark

# Self-improvement (opt-in)
fusen --learn                       # capture requests as training data
fusen train                         # run distillation/fine-tuning cycle
```

Everything else is automatic and invisible: model selection, quantization, KV cache format, concurrency tuning, CUDA graphs, prefix caching, multi-model routing, model updates.

**Comparison:**

| | Ollama | Fusen |
|---|--------|-------|
| Install | `curl \| sh` | `curl \| sh` |
| Run | `ollama run gemma3` | `fusen` |
| User picks model | Yes | No — auto-selected for GPU |
| User tunes serving | N/A | No — auto-profiled |
| Multi-model routing | No | Automatic if GPU fits 2 models |
| Learns from usage | No | Opt-in (`--learn`) |
| OpenAI-compatible API | Partial | Full drop-in |
| Optimal for hardware | Basic | Per-GPU profiled and tuned |

Ollama eliminated config. Fusen eliminates decisions. The user expresses **intent** (quality level) and the system handles **implementation** (model, config, optimization, improvement).

### What's Below the API

```
┌──────────────────────────────────────────────────────────────┐
│                    User: One API Call                         │
├──────────────────────────────────────────────────────────────┤
│              Intelligent Router (< 1ms)                      │
│    Routes by: task type, complexity, tool needs, cost        │
│    Learns from: every request outcome                        │
├──────────┬──────────┬──────────┬─────────────────────────────┤
│  Small   │  Medium  │  Large   │  Specialist Models          │
│  Model   │  Model   │  Model   │  (code, math, vision, ...)  │
├──────────┴──────────┴──────────┴─────────────────────────────┤
│         Data-Driven Serving Engine (this plan)               │
│  Adaptive config, per-model KV optimization, CUDA graphs     │
├──────────────────────────────────────────────────────────────┤
│         Kernel Generator + AutoKernel Optimizer              │
│  Universal Triton kernel, spec-driven, autotuned             │
├──────────────────────────────────────────────────────────────┤
│         Data Pipeline + Training Loop                        │
│  Capture traces → distill → fine-tune → improve router       │
├──────────────────────────────────────────────────────────────┤
│         Hardware (GPU / Multi-GPU / Multi-Node)              │
└──────────────────────────────────────────────────────────────┘
```

Every layer is data-driven. Every layer improves itself. The user sees none of it.

### The Complexity It Hides

What we manually did in 2 days (April 7-8, 2025):
- Patched 3 vLLM bugs to enable 128K context
- Tested 2 models (31B dense, 26B MoE) to find the right one
- Tried 3 KV formats (BF16, FP8, K4V4) and measured each
- Swept concurrency from 1-300 to find C=39 sweet spot
- Wrote 2 Triton kernels for custom KV compression
- Fixed NVFP4 weight loading for community quants
- Tried speculative decoding, prefix caching, CUDA graphs
- Ran ~20 benchmark sweeps with ~3,000 test requests

The engine does all of this automatically. A new user sends their first request and gets the optimal config without knowing any of it happened.

---

## Technical Foundation

### Problem Statement

Today's open-source inference stack (vLLM, SGLang, TensorRT-LLM) has:
- **~150 model files** averaging ~1500 lines each, 60% is weight loading boilerplate
- **Per-dtype if/elif chains** in attention, cache allocation, memory planning
- **isinstance checks** for feature compatibility (FP8 + AWQ bug)
- **Per-model weight iterators** that break when checkpoint naming changes
- **Manual backend registration** for each new attention variant

Every new model on HuggingFace, every new quant format, every new KV cache scheme requires code changes in 3-8 files. We fixed 3 bugs in one day that were all "the data was there but the code didn't read it."

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   Model Descriptor                   │
│  (from HF config.json + quantization_config)         │
├──────────┬──────────┬───────────┬───────────────────┤
│  Layers  │ Attention│   MoE     │  Weight Mapping    │
│  types,  │ heads,   │  experts, │  checkpoint →      │
│  counts  │ dims,    │  layout,  │  model params      │
│          │ windows  │  routing  │                     │
└────┬─────┴────┬─────┴─────┬─────┴─────────┬─────────┘
     │          │           │               │
     ▼          ▼           ▼               ▼
┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────────┐
│ Memory  │ │  KV    │ │ Kernel  │ │    Weight     │
│ Planner │ │ Cache  │ │ Selector│ │    Loader     │
│         │ │ Spec   │ │         │ │              │
└─────────┘ └────────┘ └─────────┘ └──────────────┘
     │          │           │               │
     ▼          ▼           ▼               ▼
┌─────────────────────────────────────────────────────┐
│              Generic Engine Runtime                  │
│  (no model-specific code in hot path)                │
└─────────────────────────────────────────────────────┘
```

---

## Layer 1: Model Descriptor

### 1.1 What exists today

HuggingFace `config.json` already contains most of what we need:
```json
{
  "num_hidden_layers": 30,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 256,
  "sliding_window": 1024,
  "layer_types": ["sliding_attention", ..., "full_attention", ...],
  "num_experts": 128,
  "quantization_config": { "quant_method": "compressed-tensors", ... }
}
```

### 1.2 What's missing

The config doesn't describe:
- How checkpoint weight names map to model parameter names
- Which projections are fused (gate+up) vs separate
- What scale/zero-point tensors exist per quantization format
- KV cache compatibility constraints

### 1.3 Proposed Model Descriptor

Extends `config.json` with a `vllm_descriptor` section (or separate file):

```yaml
vllm_descriptor:
  version: 1
  
  attention:
    layers:
      - type: sliding_window
        count: 25
        window: 1024
        heads: {q: 16, kv: 8, dim: 256}
      - type: full_attention
        count: 5
        heads: {q: 16, kv: 8, dim: 256}
    # Memory rule: sliding layers need min(window, max_model_len) tokens
    # Full layers need max_model_len tokens
    # This replaces the entire max_memory_usage_bytes() method
  
  moe:
    num_experts: 128
    experts_per_token: 4
    projections:
      gate_up:
        type: fused          # gate + up in one tensor
        shard_map: {gate_proj: w1, up_proj: w3}
        fused_param: w13_weight
      down:
        type: single
        shard_map: {down_proj: w2}
        param: w2_weight
  
  weight_mapping:
    prefix_strip: "model.language_model."
    renames:
      - from: ".router.per_expert_scale"
        to: ".moe.per_expert_scale"
    expert_format: per_expert   # vs "fused"
    # If per_expert: checkpoint has experts.{id}.{proj}.{suffix}
    # If fused: checkpoint has experts.{proj}_packed (3D stacked)
    expert_suffixes:             # which suffixes exist per expert
      - weight                   # always
      - weight_packed            # nvfp4
      - weight_scale             # nvfp4
      - weight_global_scale      # nvfp4
      - input_global_scale       # nvfp4
```

### 1.4 Auto-detection

For models without a descriptor, auto-detect from checkpoint:

```python
def infer_descriptor(checkpoint_path):
    """Infer model descriptor from checkpoint weight names."""
    weight_names = list_safetensor_keys(checkpoint_path)
    
    descriptor = {}
    
    # Detect expert format
    if any(re.match(r".*experts\.\d+\.", n) for n in weight_names):
        descriptor["expert_format"] = "per_expert"
        # Detect suffixes
        suffixes = set()
        for n in weight_names:
            m = re.match(r".*experts\.\d+\.\w+\.(\w+)$", n)
            if m:
                suffixes.add(m.group(1))
        descriptor["expert_suffixes"] = list(suffixes)
    elif any("gate_up_proj" in n for n in weight_names):
        descriptor["expert_format"] = "fused"
    
    # Detect prefix
    if any(n.startswith("model.language_model.") for n in weight_names):
        descriptor["prefix_strip"] = "model.language_model."
    
    return descriptor
```

---

## Layer 2: Weight Loader

### 2.1 Current Problem

Each model file has a `load_weights()` method with:
- String matching for weight names (~50 lines)
- Expert params mapping (~30 lines)
- Special cases for k_eq_v, per-layer norms, etc. (~30 lines)
- `_weight_iterator()` with name remapping (~80 lines)

Total: ~200 lines per model, 90% is boilerplate that could be generic.

### 2.2 Generic Weight Loader

```python
class DeclarativeWeightLoader:
    def __init__(self, model, descriptor):
        self.model = model
        self.desc = descriptor
        self.params_dict = dict(model.named_parameters())
        self.params_dict.update(dict(model.named_buffers()))
    
    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            # Step 1: Apply prefix strip
            name = name.replace(self.desc.prefix_strip, "", 1)
            
            # Step 2: Apply renames
            for rename in self.desc.renames:
                name = name.replace(rename["from"], rename["to"])
            
            # Step 3: Route to handler
            if self._is_expert_weight(name):
                loaded.update(self._load_expert_weight(name, tensor))
            elif self._is_stacked_param(name):
                loaded.update(self._load_stacked_param(name, tensor))
            elif name in self.params_dict:
                self._load_direct(name, tensor)
                loaded.add(name)
        
        return loaded
    
    def _is_expert_weight(self, name):
        return bool(re.match(r".*\.(experts|moe)\.", name))
    
    def _load_expert_weight(self, name, tensor):
        """Handle both per-expert and fused expert formats."""
        if self.desc.expert_format == "per_expert":
            return self._load_per_expert(name, tensor)
        else:
            return self._load_fused_expert(name, tensor)
    
    def _load_per_expert(self, name, tensor):
        """Map experts.{id}.{proj}.{suffix} → fused FusedMoE params."""
        m = re.match(
            r"(.*)\.moe\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.?(.*)",
            name
        )
        if not m:
            return set()
        
        prefix, expert_id, proj, suffix = m.groups()
        expert_id = int(expert_id)
        
        # Look up the shard mapping from descriptor
        shard_info = self.desc.moe.shard_map[proj]
        param_name = f"{prefix}.moe.{shard_info.param}"
        if suffix:
            param_name = param_name.replace("_weight", f"_{suffix}")
        
        if param_name not in self.params_dict:
            return set()
        
        param = self.params_dict[param_name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        
        wn = f"{proj}.{suffix}" if suffix else f"{proj}.weight"
        weight_loader(param, tensor, wn, shard_id=shard_info.shard, expert_id=expert_id)
        
        return {param_name}
```

### 2.3 Impact

- Gemma 4 `load_weights()`: 200 lines → 0 lines (descriptor handles it)
- New model support: write a 20-line YAML descriptor instead of a 1500-line Python file
- Community NVFP4 quants: auto-detected from checkpoint weight names, no code changes

---

## Layer 3: KV Cache Planner

### 3.1 Current Problem

`get_kv_cache_spec()` in `attention.py` has a growing if/elif chain:
```python
if self.kv_cache_dtype == "fusen": ...
elif self.kv_cache_dtype.startswith("tq"): ...
elif self.kv_cache_dtype == "fp8_e5m2": ...
else: ...  # default
```

`max_memory_usage_bytes()` has the sliding window bug because it doesn't read its own field.

### 3.2 Data-Driven Planner

```python
# Registry of KV cache formats
KV_CACHE_REGISTRY = {
    "auto": {
        "dtype": "from_model",     # inherit from model dtype
        "bytes_per_element": 2,    # bf16
        "cache_dtype": torch.bfloat16,
    },
    "fp8_e5m2": {
        "dtype": torch.float8_e5m2,
        "bytes_per_element": 1,
        "cache_dtype": torch.uint8,
    },
    "fp8_e4m3": {
        "dtype": torch.float8_e4m3fn,
        "bytes_per_element": 1,
        "cache_dtype": torch.uint8,
    },
    "k8v4b16": {
        "dtype": torch.uint8,
        "k_bytes": lambda D: D,        # int8
        "v_bytes": lambda D: D // 2,   # int4 packed
        "scale_bytes": lambda D: D // 16 * 2 * 2,  # per-block-16, K+V, fp16
        "cache_dtype": torch.uint8,
        "backend": "FUSENCACHE",
    },
    "k4v4b32": {
        "dtype": torch.uint8,
        "k_bytes": lambda D: D // 2,
        "v_bytes": lambda D: D // 2,
        "scale_bytes": lambda D: D // 32 * 2 * 2,
        "cache_dtype": torch.uint8,
        "backend": "FUSENCACHE",
    },
}

def get_kv_cache_spec(layer, kv_cache_dtype, block_size):
    """Generic spec creation — no if/elif chains."""
    fmt = KV_CACHE_REGISTRY[kv_cache_dtype]
    
    # Calculate effective head size from format
    D = layer.head_size
    if "k_bytes" in fmt:
        slot_bytes = fmt["k_bytes"](D) + fmt["v_bytes"](D)
        effective_head_size = slot_bytes // 2
    else:
        effective_head_size = D
    
    # Determine attention type from layer config
    if layer.sliding_window is not None:
        return SlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=layer.num_kv_heads,
            head_size=effective_head_size,
            dtype=fmt["cache_dtype"],
            sliding_window=layer.sliding_window,
        )
    else:
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=layer.num_kv_heads,
            head_size=effective_head_size,
            dtype=fmt["cache_dtype"],
        )
```

### 3.3 Memory Formula

Replace `max_memory_usage_bytes()` with a universal formula:

```python
def max_memory_for_layer(layer_spec, max_model_len):
    """Universal memory calculation — no subclass overrides needed."""
    
    # How many tokens does this layer need?
    if layer_spec.sliding_window is not None:
        tokens_needed = min(layer_spec.sliding_window, max_model_len)
    elif layer_spec.attention_chunk_size is not None:
        tokens_needed = min(
            layer_spec.attention_chunk_size + max_num_batched_tokens,
            max_model_len
        )
    else:
        tokens_needed = max_model_len
    
    # How many blocks? How big is each block?
    num_blocks = ceil(tokens_needed / layer_spec.block_size)
    return num_blocks * layer_spec.page_size_bytes
```

One function replaces 6 separate `max_memory_usage_bytes()` implementations across FullAttentionSpec, SlidingWindowSpec, ChunkedLocalAttentionSpec, CrossAttentionSpec, MambaSpec, EncoderOnlyAttentionSpec.

---

## Layer 4: Compatibility Matrix

### 4.1 Current Problem

Compatibility is encoded as isinstance chains:
```python
if layer.kv_cache_dtype == "fp8_e5m2":
    raise ValueError("not supported with fp8 checkpoints")
```

This is wrong because it checks the wrong thing (quant method type instead of checkpoint type).

### 4.2 Data-Driven Compatibility

```python
# Explicit compatibility matrix
# (weight_quant, kv_cache_dtype) → allowed
COMPATIBILITY = {
    # Weight quant          KV cache dtype    Allowed  Notes
    ("none",                "auto"):          True,
    ("none",                "fp8_e5m2"):      True,
    ("none",                "fp8_e4m3"):      True,
    ("compressed-tensors",  "auto"):          True,    # AWQ, GPTQ
    ("compressed-tensors",  "fp8_e5m2"):      True,    # <-- this was the bug
    ("compressed-tensors",  "fp8_e4m3"):      True,
    ("compressed-tensors",  "fusen"):         True,
    ("fp8",                 "auto"):          True,
    ("fp8",                 "fp8_e4m3"):      True,
    ("fp8",                 "fp8_e5m2"):      False,   # actual conflict
    ("modelopt_fp4",        "auto"):          True,
    ("modelopt_fp4",        "fp8_e5m2"):      True,
}

def check_kv_cache_compatibility(weight_quant, kv_cache_dtype):
    key = (weight_quant, kv_cache_dtype)
    if key not in COMPATIBILITY:
        # Default: allow unless explicitly blocked
        return True
    return COMPATIBILITY[key]
```

Adding a new quant format or KV cache type = add rows to the matrix. No isinstance checks, no code changes.

---

## Layer 5: Attention Backend Registry

### 5.1 Current Problem

Each attention backend is a manually registered class with hardcoded feature flags:
```python
class FusenCacheAttentionBackend(AttentionBackend):
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["fusen"]
    # ... 10 static methods ...
```

### 5.2 Data-Driven Registry

```python
ATTENTION_BACKENDS = {
    "TRITON_ATTN": {
        "supported_dtypes": [torch.float16, torch.bfloat16],
        "supported_kv_cache_dtypes": ["auto", "fp8_e4m3", "fp8_e5m2"],
        "supports_cuda_graphs": True,
        "supports_gqa": True,
        "impl_class": "vllm.v1.attention.backends.triton_attn.TritonAttentionImpl",
    },
    "FLASH_ATTN": {
        "supported_dtypes": [torch.float16, torch.bfloat16],
        "supported_kv_cache_dtypes": ["auto", "fp8_e4m3", "fp8_e5m2"],
        "supports_cuda_graphs": True,
        "max_head_dim": 256,
        "impl_class": "vllm.v1.attention.backends.flash_attn.FlashAttentionImpl",
    },
    "FUSENCACHE": {
        "supported_dtypes": [torch.float16, torch.bfloat16],
        "supported_kv_cache_dtypes": ["fusen"],
        "supports_cuda_graphs": True,  # after our fix
        "impl_class": "vllm.v1.attention.backends.fusencache_attn.FusenCacheAttentionImpl",
        "kv_cache_spec_override": "fusencache",  # custom cache shape
    },
}

def select_backend(model_dtype, kv_cache_dtype, head_dim, features_needed):
    """Select best backend from registry."""
    candidates = []
    for name, spec in ATTENTION_BACKENDS.items():
        if model_dtype not in spec["supported_dtypes"]:
            continue
        if kv_cache_dtype not in spec["supported_kv_cache_dtypes"]:
            continue
        if head_dim > spec.get("max_head_dim", float("inf")):
            continue
        if not all(spec.get(f, True) for f in features_needed):
            continue
        candidates.append((name, spec))
    
    # Priority: FLASH_ATTN > TRITON_ATTN > others
    return candidates[0] if candidates else None
```

---

## Layer 6: The Self-Modifying Aspect

### 6.1 Auto-Discovery

When a new model is loaded:
1. Read `config.json` — extract layer types, attention config, MoE config
2. Read checkpoint keys — infer weight naming, expert format, scale suffixes
3. Generate model descriptor automatically
4. If descriptor doesn't match any known pattern, log it and attempt generic loading
5. If generic loading succeeds, cache the inferred descriptor for future use

### 6.2 Format Learning

When a new KV cache format is tested:
1. User specifies `{k_bits, v_bits, block_size, dequant_type}`
2. System generates: cache spec, memory formula, attention kernel, store kernel
3. System runs quality + speed benchmark automatically
4. If it passes, adds to the registry for future use
5. The registry persists — next time, just specify `kv_cache_dtype="k4v4b32"`

### 6.3 Compatibility Auto-Testing

Instead of a hardcoded compatibility matrix:
1. When a new (weight_quant, kv_cache_dtype) combination is first attempted, try it
2. Run a quick 5-prompt smoke test
3. If it works, add to compatibility matrix as "allowed"
4. If it fails, add as "blocked" with the error message
5. Matrix grows automatically as users try new combinations

---

## Implementation Phases

### Phase 1: Weight Loader (2 weeks)
- Build `DeclarativeWeightLoader` with descriptor-driven mapping
- Auto-detect expert format from checkpoint
- Validate on Gemma 4 31B AWQ, 26B AWQ, 26B NVFP4
- Eliminate `_weight_iterator()` and `expert_params_mapping` from gemma4.py
- **Impact:** Prevents weight loading bugs for new models/quants

### Phase 2: KV Cache Planner (1 week)
- Build `KV_CACHE_REGISTRY` with format descriptors
- Replace `get_kv_cache_spec()` if/elif chain
- Replace 6 separate `max_memory_usage_bytes()` with one formula
- **Impact:** Prevents memory calculation bugs, enables new KV formats without code

### Phase 3: Compatibility Matrix (3 days)
- Replace isinstance checks with explicit matrix
- Add auto-testing for unknown combinations
- **Impact:** Prevents false-positive blocks like FP8+AWQ

### Phase 4: Kernel Generator (2 weeks)
- Build universal Triton decode kernel with constexpr dispatch
- Build primitive registry (unpack, dequant, scale, dot)
- Validate against hand-written v5/v6 kernels
- **Impact:** New KV formats = YAML spec, zero kernel code

### Phase 5: Full Generic Model Loader (4 weeks)
- Auto-detect model structure from HF config + checkpoint
- Generic attention layer creation from descriptor
- Generic MoE layer creation from descriptor
- Goal: load any transformer model without a model-specific Python file
- **Impact:** New HuggingFace models work on day 1, not after someone writes a model file

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Lines of code to support new model | ~1500 | ~20 (YAML descriptor) |
| Lines of code for new KV cache format | ~500 | ~10 (spec entry) |
| Bugs from hardcoded assumptions | 3 in 1 day | 0 (data validates itself) |
| Time to support new HF model | Days-weeks | Minutes (auto-detect) |
| Time to test new KV quant config | Hours (write kernel) | Seconds (generate + sweep) |
| Compatibility check accuracy | Wrong (FP8+AWQ) | Correct (explicit matrix) |

---

## Relationship to AutoKernel

This plan is AutoKernel's natural evolution:

```
AutoKernel today:  profile → extract op → optimize Triton kernel → benchmark → keep/revert
AutoKernel future: describe format → generate kernel + loader + planner → benchmark → deploy

Today:  human writes kernel, machine optimizes it
Future: machine writes kernel from spec, machine optimizes it
```

The kernel generator (Phase 4) is the bridge. Once kernels are generated from specs, AutoKernel's optimization loop can explore the spec space automatically — varying k_bits, v_bits, block_size, and autotuning the Triton kernel parameters, all in one unified loop.

---

## Phase 6: Maximum Performance Architecture

### 6.1 The Problem With Today's Stack

We measured 1,816 tok/s at C=39 on RTX 5090. The theoretical peak is ~23x higher for single-stream. The gap comes from 6 sources, all of which the data-driven architecture can address because they become parameter changes, not code rewrites.

### 6.2 Performance Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    Performance Layer Stack                       │
├─────────────────────────────────────────────────────────────────┤
│ L0: Weight Matmul Format                                        │
│     AWQ INT4→FP16 Marlin (current) → NVFP4 native tensor core  │
│     Impact: 2-3x on 60% of decode time                          │
│     Config: weight_format: nvfp4_native                          │
├─────────────────────────────────────────────────────────────────┤
│ L1: Fused Attention + KV Dequant                                │
│     Separate dequant+dot (current) → in-register dequant        │
│     Impact: 1.3x (eliminates K/V float materialization)          │
│     Config: kernel.fusion: attention_dequant                     │
├─────────────────────────────────────────────────────────────────┤
│ L2: Async Memory Pipeline                                       │
│     num_stages=1 (current) → autotuned pipelining               │
│     Impact: 1.15x (overlaps KV load with compute)               │
│     Config: kernel.pipeline_stages: auto                         │
├─────────────────────────────────────────────────────────────────┤
│ L3: Per-Layer Adaptive KV Quantization                          │
│     Uniform format (current) → sensitivity-driven per-layer     │
│     Impact: 1.5x KV capacity at same quality                    │
│     Config: kv_cache.per_layer: [format per layer group]         │
├─────────────────────────────────────────────────────────────────┤
│ L4: Kernel Launch Elimination                                   │
│     150 launches/token (current) → 1 graph replay               │
│     Impact: 1.1x (eliminates 750μs launch overhead)             │
│     Config: kernel.cuda_graph: full                              │
├─────────────────────────────────────────────────────────────────┤
│ L5: MoE Expert Prefetch                                         │
│     Serial expert load (current) → speculative prefetch          │
│     Impact: 1.2x (overlaps weight load with compute)            │
│     Config: serving.expert_prefetch: true                        │
├─────────────────────────────────────────────────────────────────┤
│ L6: Zero-Python Hot Path                                        │
│     Python scheduler (current) → CUDA-native scheduling         │
│     Impact: 1.05x at high C (eliminates GIL contention)         │
│     Config: serving.scheduler: cuda_native                       │
└─────────────────────────────────────────────────────────────────┘

Combined: ~3.9x → 1,816 × 3.9 ≈ 7,000 tok/s at C=39
```

### 6.3 Why Data-Driven Enables This

Each optimization layer is blocked today because it requires rewriting code:
- FP4 matmuls → new weight loader + new model file
- Fused dequant → new Triton kernel
- Per-layer quant → new cache planner + 30 kernel variants
- Expert prefetch → new MoE forward pass

With the data-driven architecture, each is a config change:
- FP4 matmuls → `weight_format: nvfp4_native` (weight loader reads format descriptor)
- Fused dequant → `fusion: attention_dequant` (kernel generator adds constexpr flag)
- Per-layer quant → `per_layer: [...]` (planner reads per-layer spec, generator makes per-layer kernels)
- Expert prefetch → `expert_prefetch: true` (MoE layer reads serving config)

### 6.4 The Automated Search

The ultimate version: AutoKernel doesn't just optimize one kernel — it searches the entire config space:

```python
# AutoKernel performance search
search_space = {
    "weight_format": ["awq_int4", "nvfp4_native", "fp8_e4m3"],
    "kv_format": ["fp8", "k8v4b16", "k4v4b32", "k4v2b64"],
    "kv_per_layer": [False, True],
    "fusion": ["separate", "attention_dequant"],
    "pipeline_stages": [1, 2, 3],
    "block_kv": [8, 16, 32],
    "block_h": [4, 8, 16],
    "expert_prefetch": [False, True],
    "cuda_graph": ["piecewise", "full"],
}

# Total configs: 3 × 4 × 2 × 2 × 3 × 3 × 3 × 2 × 2 = 5,184
# Each test: generate kernel + load model + 5-prompt quality + throughput bench
# With kernel caching: ~30s per config
# Full search: ~43 hours → run overnight on one GPU

for config in grid_search(search_space):
    spec = build_spec(config)
    kernel = generate_kernel(spec)          # from kernel generator
    loader = build_weight_loader(spec)      # from weight loader
    planner = build_memory_plan(spec)       # from cache planner
    
    quality = quick_quality_test(kernel, loader)
    if quality < threshold:
        continue
    
    throughput = serving_benchmark(kernel, concurrency=39)
    log_result(config, quality, throughput)

# Output: Pareto frontier of quality vs throughput vs KV capacity
```

### 6.5 Self-Improving System

Once the search finds the best config, it can go deeper:

1. **Autotune the winning kernel** — vary Triton parameters (num_warps, num_stages, block sizes) around the winning spec
2. **Sensitivity analysis** — perturb each layer's KV format and measure quality impact
3. **Profile bottleneck** — identify which layer/operation is now the bottleneck and focus there
4. **Export optimized config** — save as a single YAML that reproduces the result

This is the AutoKernel loop applied to the entire serving stack:
```
describe → generate → benchmark → optimize → describe (refined) → ...
```

Each iteration produces a better config. The system improves itself by modifying its own data, not its code.

### 6.6 Hardware Portability

The same spec-driven approach works across GPUs:

```yaml
# RTX 5090 optimized
hardware: rtx5090
weights: nvfp4_native         # has FP4 tensor cores
kv_cache: k4v4b32             # bandwidth-optimized
kernel:
  fusion: attention_dequant
  pipeline_stages: 2

# A100 optimized (different hardware)
hardware: a100
weights: awq_int4              # no FP4 tensor cores
kv_cache: fp8_e4m3             # FP8 is native on A100
kernel:
  fusion: separate             # less shared memory
  pipeline_stages: 1

# H100 optimized
hardware: h100
weights: fp8_e4m3              # FP8 tensor cores
kv_cache: fp8_e4m3
kernel:
  fusion: attention_dequant
  pipeline_stages: 3           # more shared memory
```

The kernel generator produces different PTX for each hardware target from the same spec. The search finds the optimal config per GPU automatically.

---

## Phase 7: Missing Dimensions

### 7.1 Quality Assurance Layer

The data-driven system needs a quality dimension — not just "does it generate text" but "does quantization degrade real-world task performance."

```python
# Quality registry — paired with every KV cache format
QUALITY_BASELINES = {}

def register_quality(spec_name, model_name, results):
    """Store quality results for a (spec, model) pair."""
    QUALITY_BASELINES[(spec_name, model_name)] = results

def check_quality_regression(spec_name, model_name, new_results):
    """Compare against baseline, flag regressions."""
    baseline = QUALITY_BASELINES.get((spec_name, model_name))
    if baseline is None:
        return True  # no baseline, allow
    for metric, value in new_results.items():
        if value < baseline[metric] * 0.95:  # 5% regression threshold
            return False
    return True
```

Integrated into the data-driven config:
```yaml
quality:
  eval_suite: [mmlu_100, humaneval_50, mt_bench_80, needle_haystack_20]
  baseline: fp16_auto              # compare against FP16 KV
  regression_threshold: 0.05       # max 5% drop
  gate_on: [deploy, format_change] # when to run
```

Every new KV format, every model change, every config tweak gets automatically quality-tested before deployment. The compatibility matrix gains a quality dimension:

```python
# Extended compatibility: (weight_quant, kv_dtype) → (allowed, quality_score)
COMPATIBILITY = {
    ("compressed-tensors", "fp8_e5m2"):    (True, 0.99),   # 1% quality loss
    ("compressed-tensors", "k4v4b32"):     (True, 0.94),   # 6% quality loss
    ("compressed-tensors", "k4v2b64"):     (True, None),   # untested
}
```

### 7.2 Prefill-Decode Split Architecture

The system should treat prefill and decode as separate optimization problems:

```yaml
model:
  prefill:
    attention: flash_attention     # native, no custom kernel needed
    kv_store: quantize_on_write    # quantize K/V as they're stored
    chunked: true
    chunk_size: 8192
    # Prefill is compute-bound → optimize FLOPS utilization
    # No dequant needed (reading fresh FP16 K/V from Q/K/V projections)
    
  decode:
    attention: generated_kernel    # from kernel generator
    kv_read: dequant_on_read       # dequant packed K/V from cache
    fusion: attention_dequant
    # Decode is memory-bound → optimize bandwidth utilization
```

The weight loader, memory planner, and kernel generator all understand this split. The cache planner accounts for prefill's temporary memory spike (chunked_prefill tokens × full precision) while allocating decode's compressed capacity.

### 7.3 Model Selection Search

The model is the largest single variable in the optimization. Different MoE architectures have radically different performance profiles on the same GPU:

```python
# Model search — runs once per GPU type
def search_optimal_model(gpu, quality_target, optimize_for="throughput"):
    results = []
    for model in MODEL_REGISTRY:
        # Quick check: does it fit?
        if model.weights_gb > gpu.memory_gb * 0.6:
            continue
        
        # Estimate KV capacity
        kv_budget = gpu.memory_gb * 0.92 - model.weights_gb - 1.6
        kv_tokens = estimate_kv_tokens(model, kv_budget, kv_format="fp8")
        
        # Quick quality check
        quality = run_quick_eval(model, quality_target.suite)
        if quality < quality_target.minimum:
            continue
        
        # Benchmark
        throughput = serving_benchmark(model, concurrency=40)
        
        results.append({
            "model": model.name,
            "kv_tokens": kv_tokens,
            "throughput": throughput,
            "quality": quality,
            "cost_per_mtok": 1.0 / throughput * 3600 * gpu.cost_per_hour,
        })
    
    # Return pareto frontier
    return pareto_optimal(results, optimize_for)
```

The descriptor system makes this feasible because loading a new model is just reading a new descriptor — no code changes needed.

### 7.4 Request-Aware Serving

Different requests have different optimal treatment. The serving layer should be data-driven too:

```yaml
serving:
  profiles:
    chat:
      max_tokens: 500
      optimize_for: latency
      kv_format: fp8              # fast, good quality
      concurrency_weight: 1.0
      
    document_summary:
      max_tokens: 2000
      optimize_for: throughput
      kv_format: k4v4b32          # max KV for long input
      concurrency_weight: 0.5     # counts as 2 "slots"
      
    code_generation:
      max_tokens: 4000
      optimize_for: quality
      kv_format: fp8              # quality matters most
      concurrency_weight: 1.5

  routing:
    method: header_based          # X-Request-Profile header
    fallback: chat
    
  # Multiple kernel configs loaded simultaneously
  # Router selects per request
  # KV pools managed separately per profile
```

### 7.5 Continuous Improvement Pipeline

The system should improve itself over time:

```yaml
continuous_improvement:
  # 1. Monitor production quality
  sample_rate: 0.01               # evaluate 1% of responses
  quality_window: 1000            # rolling window
  
  # 2. Detect degradation
  alerts:
    quality_drop: 0.03            # 3% drop triggers investigation
    latency_spike: 2.0x           # 2x P50 triggers alert
    kv_fragmentation: 0.20        # 20% wasted blocks
    
  # 3. Auto-heal
  actions:
    quality_drop: rollback_config  # revert to last known good
    latency_spike: reduce_concurrency
    kv_fragmentation: trigger_defrag
    
  # 4. Periodic re-optimization
  schedule:
    weekly: autotune_kernel_params    # re-run autotune grid
    monthly: model_selection_search   # check for better models
    on_vllm_update: revalidate_all   # re-test after vLLM upgrade
```

### 7.6 Prefix Caching as First-Class Feature

Prefix caching gave +52% on 31B. It should be a measured, tracked, optimized dimension:

```yaml
prefix_cache:
  enabled: true
  
  # System prompt optimization
  system_prompts:
    - name: default_assistant
      tokens: ~500
      usage: 0.80                 # 80% of requests use this
    - name: code_assistant  
      tokens: ~800
      usage: 0.15
      
  # Expected impact calculation
  # hit_rate × prefill_savings = throughput_gain
  # 0.80 × 0.40 = 0.32 → +32% minimum from prefix cache
  
  # Monitoring
  metrics:
    hit_rate: track               # actual vs expected
    prefill_savings_ms: track     # time saved per hit
    memory_overhead: track        # blocks pinned for prefixes
```

### 7.7 Speculative Decoding Integration

The data-driven system should evaluate spec decode as another optimization dimension:

```yaml
speculative_decoding:
  search:
    enabled: true
    draft_candidates:
      - gemma-4-E2B-text-only-FP8     # 4.6 GB
      - gemma-4-E2B-text-only-INT4     # ~2.5 GB (if available)
    
    # Auto-check: does draft + target fit in GPU memory?
    # Auto-check: acceptance rate on representative prompts
    # Auto-benchmark: does speculation actually help?
    
    acceptance_threshold: 0.6     # need >60% acceptance to benefit
    max_spec_tokens: 5
    
  # If beneficial, integrate into serving config
  # If not, disable — spec decode hurts short Q&A (we proved this)
  workload_filter:
    enable_for: [long_generation, code, creative]
    disable_for: [short_qa, factual]
```

### 7.8 Updated Implementation Phases

Extend the existing 5 phases:

### Phase 6: Quality Infrastructure (1 week)
- Build eval suite runner (MMLU subset, HumanEval subset, needle-in-haystack)
- Integrate quality gate into kernel generator sweep
- Store baselines per (model, kv_format) pair
- **Impact:** Prevents shipping silently degraded configs

### Phase 7: Prefix Cache Optimization (2 days)
- Benchmark prefix caching on 26B MoE + FP8 KV
- Measure hit rate with representative system prompts
- Add to serving config as tracked metric
- **Impact:** Likely +40-60% free throughput

### Phase 8: Model Selection Search (1 week)
- Build model benchmark harness (quality + throughput + KV capacity)
- Test 3-5 MoE models on RTX 5090
- Output pareto frontier
- **Impact:** Might find a better model than Gemma 4 26B

### Phase 9: Production Monitoring (3 days)
- Add quality sampling to serving endpoint
- Add KV utilization and fragmentation tracking
- Build alert pipeline for regressions
- **Impact:** Catches degradation before users notice

### Phase 10: Request Routing (2 weeks)
- Multiple kernel configs loaded simultaneously
- Per-request profile selection
- Separate KV pools per profile
- **Impact:** Optimal treatment for every request type

### Updated Success Metrics

| Metric | Current | Phase 5 Target | Phase 10 Target |
|--------|---------|----------------|-----------------|
| Lines per new model | ~1500 | ~20 | ~0 (auto-detect) |
| Lines per new KV format | ~500 | ~10 | ~10 |
| Quality regression detection | None | Per-format gate | Per-request sampling |
| Throughput (C=39) | 1,816 tok/s | ~3,000 (with prefix) | ~7,000 (all optimizations) |
| Time to support new model | Days | Minutes | Seconds (auto) |
| Config search space | Manual | 5,184 auto | 50,000+ (with model search) |
| Production monitoring | None | Basic | Full pipeline |

---

## Phase 8: Adaptive Multi-Model Intelligence

### 8.1 The Flywheel

```
Requests → Serve → Record → Analyze → Optimize → Better Serving
    ↑                                                      │
    └──────────── Train/Distill ◄──────────────────────────┘
```

Three capabilities that compound:
1. Each model auto-optimizes its own config based on observed request patterns
2. A router sends requests to the right model based on difficulty/type
3. Request-response pairs become training data that improves models and routing

### 8.2 Per-Model Dynamic Config

The system observes real traffic patterns and adjusts each model's config in real-time.

```yaml
adaptive_config:
  observe:
    window: 5_minutes              # rolling observation window
    metrics:
      - avg_prompt_tokens           # are prompts getting longer?
      - avg_completion_tokens       # are responses getting longer?
      - concurrent_requests         # current load
      - queue_depth                 # are requests waiting?
      - time_of_day                 # usage patterns
      - cache_hit_rate              # prefix cache effectiveness
      
  profiles:
    low_load:                       # C < 10, night time
      trigger: concurrent < 10
      kv_format: fp8               # quality over compression
      max_batch: 8
      priority: latency            # fast single-user responses
      
    normal_load:                   # C = 20-40, business hours
      trigger: 10 <= concurrent < 40
      kv_format: fp8
      max_batch: 39                # our measured sweet spot
      priority: balanced
      
    peak_load:                     # C > 40, high demand
      trigger: concurrent >= 40
      kv_format: k4v4b32           # sacrifice speed for capacity
      max_batch: 100
      priority: throughput         # maximize users served
      
    long_context_surge:            # many long prompts
      trigger: avg_prompt_tokens > 5000
      kv_format: k4v4b32           # need the KV capacity
      max_batch: 20
      priority: capacity
      
  transition:
    cooldown: 30_seconds           # min time between profile switches
    method: drain_and_switch       # finish current requests, then switch
    # No cold restart — pre-compile CUDA graphs for all profiles at startup
```

Implementation: at startup, capture CUDA graphs for each profile. Switching profiles is just selecting which graph set to replay — zero downtime.

### 8.3 Multi-Model Routing

Multiple models loaded on the same GPU (or across GPUs), with an intelligent router that matches requests to the best model.

```yaml
models:
  large:
    model: gemma-4-26B-A4B-AWQ
    gpu: cuda:0
    memory_budget: 0.70            # 70% of GPU for this model
    strengths: [reasoning, long_context, complex_instructions]
    cost_weight: 3.0               # 3x more expensive per token
    
  small:
    model: gemma-4-E2B-FP8         # 4.6 GB — tiny
    gpu: cuda:0
    memory_budget: 0.20            # 20% of GPU, colocated
    strengths: [factual_qa, classification, extraction, short_chat]
    cost_weight: 1.0
    
  # 10% reserved for router model + system overhead

router:
  method: cascade                  # try small first, escalate if needed
  
  # Level 1: Rule-based fast routing (< 1ms)
  rules:
    - condition: prompt_tokens < 100 AND expected_output < 200
      route: small                 # simple Q&A
    - condition: prompt_tokens > 5000
      route: large                 # long context needs big model
    - condition: "code" in system_prompt
      route: large                 # code gen needs reasoning
      
  # Level 2: Confidence-based cascade (when rules don't match)
  cascade:
    try_first: small
    confidence_threshold: 0.85     # if small model's confidence < 85%
    escalate_to: large             # re-generate with large model
    
  # Level 3: Learned router (trained on historical data)
  learned:
    model: router_classifier       # tiny model, ~10M params
    features: [prompt_length, topic_embedding, complexity_score]
    trained_on: historical_routing_outcomes
    retrain_interval: weekly
```

Memory layout for colocated models:
```
GPU Memory (32 GB):
├── Gemma 4 26B AWQ:     13 GB weights + 7 GB KV cache (FP8)
├── Gemma 4 E2B FP8:      4.6 GB weights + 1.5 GB KV cache
├── Router classifier:    0.1 GB
├── CUDA context:         1.6 GB
└── Overhead:             4.2 GB
```

Both models serve simultaneously. The router adds <1ms latency (rule check or tiny classifier forward pass).

### 8.4 Requests as Training Data

Every request-response pair is potential training data. The serving system becomes a data collection pipeline.

```yaml
data_pipeline:
  capture:
    enabled: true
    storage: /data/serving_logs/
    format: jsonl
    
    fields:
      - timestamp
      - request_id
      - model_used                 # which model served this
      - prompt                     # full input
      - response                   # full output
      - prompt_tokens
      - completion_tokens
      - latency_ms
      - kv_format_used
      - route_decision             # why this model was chosen
      - confidence_score           # model's self-assessed confidence
      
    # Privacy controls
    pii_filter: enabled            # strip emails, phones, names
    opt_out: respect_header        # X-No-Record: true
    retention: 90_days
    
  # What the data enables:
  
  uses:
    # 1. Distillation: large model teaches small model
    distillation:
      teacher: large
      student: small
      method: on_policy            # student generates, teacher scores
      schedule: weekly
      # Over time, the small model gets better at tasks it used to
      # escalate to the large model. The escalation rate drops.
      # The system gets faster and cheaper automatically.
      
    # 2. Router training: learn which model handles which queries
    router_training:
      features: prompt_embedding
      labels: [model_used, quality_score, latency]
      method: classification       # predict best model for query
      retrain: weekly
      # Router gets smarter over time. Fewer mis-routes.
      
    # 3. Fine-tuning: adapt models to actual usage patterns
    fine_tuning:
      method: preference_pairs     # (prompt, good_response, bad_response)
      source: user_feedback        # thumbs up/down, regenerate clicks
      schedule: monthly
      validation: held_out_10pct
      # Models get better at the specific tasks users actually need.
      
    # 4. Synthetic data: fill gaps in model knowledge
    gap_detection:
      method: cluster_failures     # find topics where models fail
      action: generate_training_data  # use large model to create examples
      target: small_model          # fine-tune small model on gaps
      
    # 5. Benchmark evolution: real-world eval suite
    benchmark:
      sample: 1000_requests_weekly
      compare: [current_model, candidate_model]
      metrics: [quality, latency, user_satisfaction]
      # Real user requests are better benchmarks than academic datasets
```

### 8.5 The Self-Improving Loop

All three capabilities create a compounding improvement loop:

```
Week 1:
  - Large model handles 100% of requests (no router yet)
  - System records all request-response pairs
  - Measures: 1,816 tok/s, 100% to large model

Week 2:
  - Train router on Week 1 data
  - Router identifies: 40% of requests are simple Q&A
  - Route 40% to small model → effective throughput increases
  - Small model handles simple tasks at 5x speed
  - Measures: ~2,800 tok/s effective, 60% large + 40% small

Week 4:
  - Distill large model's responses into small model
  - Small model quality improves on previously-escalated tasks
  - Router learns: now 55% can go to small model
  - Measures: ~3,500 tok/s effective, 45% large + 55% small

Week 8:
  - Fine-tune both models on real user preferences
  - Gap detection finds topics where small model still fails
  - Generate targeted training data, fine-tune small model
  - Router accuracy: 95%+
  - Measures: ~4,000 tok/s effective, 35% large + 65% small
  - Quality: same or better than Week 1 (models adapted to users)

Month 6:
  - Small model handles 80% of traffic at high quality
  - Large model reserved for genuinely hard problems
  - System is 3-4x faster than Week 1 at same or better quality
  - Training data: 10M+ real conversations
  - The system improved itself. No human intervention.
```

### 8.6 Architecture

```
                    ┌──────────────┐
                    │   Incoming    │
                    │   Request     │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │   Router      │◄──── Learned from data
                    │  (< 1ms)     │
                    └──┬───────┬───┘
                       │       │
              ┌────────▼──┐ ┌──▼────────┐
              │  Small     │ │  Large     │
              │  Model     │ │  Model     │
              │  (fast)    │ │  (smart)   │
              └────┬───────┘ └──┬─────────┘
                   │            │
              ┌────▼────────────▼──┐
              │   Response +       │
              │   Confidence Score │
              └────────┬───────────┘
                       │
              ┌────────▼───────────┐
              │  Data Pipeline     │──── Store everything
              │  (async, non-     │
              │   blocking)        │
              └────────┬───────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    ┌────▼────┐  ┌────▼─────┐  ┌───▼────────┐
    │ Distill  │  │ Router   │  │ Fine-tune   │
    │ large→   │  │ training │  │ on user     │
    │ small    │  │          │  │ preferences │
    └──────────┘  └──────────┘  └─────────────┘
         │             │             │
         └─────────────┼─────────────┘
                       │
                  Weekly/Monthly
                  model updates
```

### 8.7 Integration with Data-Driven Architecture

This entire system is expressible in the same declarative format:

```yaml
# The complete self-improving serving system
system:
  name: autokernel-serve
  version: 1
  
  models:
    large: {path: gemma-4-26B-A4B-AWQ, gpu_budget: 0.70}
    small: {path: gemma-4-E2B-FP8, gpu_budget: 0.20}
    router: {path: router_v1, gpu_budget: 0.02}
  
  serving:
    adaptive_config: {profiles: [low_load, normal, peak]}
    routing: {method: cascade, learned: true}
    prefix_cache: {enabled: true}
  
  kernel:
    large: {kv: fp8, fusion: attention_dequant, graphs: full}
    small: {kv: auto, graphs: full}
  
  data:
    capture: {enabled: true, pii_filter: true}
    distillation: {schedule: weekly, teacher: large, student: small}
    router_training: {schedule: weekly}
    fine_tuning: {schedule: monthly, method: dpo}
  
  monitoring:
    quality_sampling: 0.01
    alerts: {quality_drop: 0.03, latency_spike: 2x}
    auto_heal: {quality_drop: rollback, latency_spike: reduce_concurrency}
```

The system reads this config, sets up everything, and improves itself over time. Adding a new model = one line. Changing the routing strategy = one line. Enabling distillation = one line. No code changes ever.

### 8.8 What This Means

This isn't just a serving system — it's an **inference optimization engine that learns from its own production traffic**. The traditional ML pipeline is:

```
Collect data → Train model → Deploy → Serve → (stop)
```

This system closes the loop:

```
Serve → Collect data → Improve models → Improve routing → Serve better → ...
```

Every request makes the system smarter. Every model update makes serving faster. Every routing improvement reduces cost. The human only sets the initial config and the quality constraints. Everything else is automated.

### Phase 11: Multi-Model + Data Pipeline (4 weeks)
- Colocate large + small model on single GPU
- Build rule-based router with cascade fallback
- Build request logging pipeline with PII filtering
- **Impact:** 2-3x effective throughput from routing alone

### Phase 12: Learned Router (2 weeks)
- Train router classifier on logged request data
- A/B test learned vs rule-based routing
- Measure escalation rate, quality, latency
- **Impact:** Reduces escalation rate from ~60% to ~20% over time

### Phase 13: Distillation Pipeline (3 weeks)
- Implement on-policy distillation (student generates, teacher scores)
- Automated weekly distillation runs
- Quality gate: distilled model must pass eval suite
- **Impact:** Small model handles more traffic each week, compounding savings

### Phase 14: Fine-Tuning Pipeline (4 weeks)
- Capture user preference signals (regenerate, thumbs up/down)
- Build DPO training pipeline from preference pairs
- Monthly fine-tuning with held-out validation
- **Impact:** Models adapt to actual usage, quality improves on real tasks

### Final Success Metrics

| Metric | Week 1 | Month 1 | Month 6 |
|--------|--------|---------|---------|
| Effective throughput | 1,816 tok/s | 2,800 tok/s | 4,000+ tok/s |
| Large model traffic share | 100% | 60% | 20% |
| Router accuracy | N/A (no router) | 80% | 95%+ |
| Training data collected | 0 | 100K requests | 10M+ requests |
| Quality (vs baseline) | 1.00 | 1.00 | 1.02 (improved) |
| Human intervention needed | Daily | Weekly | Monthly |
| Cost per million tokens | $X | $0.5X | $0.2X |

---

## Phase 9: Tool-Call-Aware Data Capture

### 9.1 Why Tool Calls Are the Best Training Data

Raw chat logs are noisy — you can't tell if a response was good without human judgment. Tool calls are different: they're **structured, verifiable, and self-labeling**.

```
Chat log:     "The capital of France is Paris"  → Was this good? Need human to judge.
Tool call:    read_file("/src/main.py")          → Did it return content? Objective.
Tool chain:   search → read → edit → test       → Did tests pass? Ground truth.
```

Every tool-using conversation produces a complete decision trace:
1. What the user wanted (prompt)
2. What the model decided to do (tool selection + arguments)
3. What happened (tool result — success/failure/error)
4. What the model did next (adapt, retry, or proceed)
5. Whether the user was satisfied (accept, reject, correct)

This is a supervised learning dataset with free labels.

### 9.2 The Capture Schema

```yaml
tool_call_capture:
  # Every request through the router gets a trace
  trace_schema:
    request_id: uuid
    timestamp: iso8601
    session_id: uuid              # groups multi-turn conversations
    
    # Routing decision
    routing:
      model_selected: string      # which model handled this
      route_reason: string        # rule match, cascade, learned
      confidence: float           # router's confidence
      
    # The conversation turn
    turn:
      system_prompt: string
      user_message: string
      user_message_tokens: int
      
    # Tool calls (zero or more per turn)
    tool_calls:
      - index: int
        tool_name: string         # read_file, edit, bash, search, etc.
        tool_args: json           # structured arguments
        tool_result: string       # what the tool returned
        tool_success: bool        # did it error?
        tool_latency_ms: int
        
        # Derived quality signals
        was_retry: bool           # did model retry this tool?
        retry_of: int             # index of previous attempt
        args_changed: json        # what changed in retry (learning signal)
        
    # Final response
    response:
      content: string
      tokens: int
      latency_ms: int
      finish_reason: string       # stop, length, tool_use
      
    # Outcome signals (collected async)
    outcome:
      user_accepted: bool         # did user proceed without correction?
      user_corrected: bool        # did user say "no, do X instead"?
      user_regenerated: bool      # did user ask to regenerate?
      follow_up_turns: int        # how many more turns in session?
      task_completed: bool        # did the session end successfully?
```

### 9.3 What Each Signal Teaches

```
┌─────────────────────────┬──────────────────────────────────────────┐
│ Signal                  │ What It Trains                           │
├─────────────────────────┼──────────────────────────────────────────┤
│ tool_name + tool_args   │ Tool selection policy (when to use what) │
│ tool_success: false     │ Negative examples (don't do this)        │
│ was_retry: true         │ Error recovery patterns                  │
│ args_changed in retry   │ Self-correction behavior                 │
│ user_corrected: true    │ Preference pairs for DPO/RLHF            │
│ user_accepted: true     │ Positive examples (do more of this)      │
│ tool chain sequence     │ Multi-step planning / reasoning          │
│ model_selected + outcome│ Router training (which model for what)   │
│ task_completed: true    │ End-to-end success signal                │
│ follow_up_turns: 0      │ First-try success (highest quality)      │
│ follow_up_turns: 5+     │ Struggling — model needs improvement     │
└─────────────────────────┴──────────────────────────────────────────┘
```

### 9.4 Training Data Generation

The captured traces produce multiple training datasets automatically:

```python
# From one captured session, generate multiple training signals:

def process_trace(trace):
    datasets = {}
    
    # 1. Tool selection SFT: (context, correct_tool_call)
    for tc in trace.tool_calls:
        if tc.tool_success and not tc.was_retry:
            datasets["tool_sft"].append({
                "input": trace.turn.user_message + context_so_far,
                "output": {"tool": tc.tool_name, "args": tc.tool_args},
                "label": "positive",
            })
    
    # 2. Error recovery SFT: (failed_attempt, successful_retry)
    for tc in trace.tool_calls:
        if tc.was_retry and tc.tool_success:
            original = trace.tool_calls[tc.retry_of]
            datasets["error_recovery"].append({
                "failed_call": original,
                "error_message": original.tool_result,
                "successful_retry": tc,
                "what_changed": tc.args_changed,
            })
    
    # 3. DPO pairs: (prompt, chosen_response, rejected_response)
    if trace.outcome.user_corrected:
        datasets["dpo_pairs"].append({
            "prompt": trace.turn.user_message,
            "rejected": trace.response.content,     # model's attempt
            "chosen": next_turn_user_message,        # user's correction
        })
    
    # 4. Router training: (features, best_model)
    datasets["router"].append({
        "prompt_embedding": embed(trace.turn.user_message),
        "prompt_length": trace.turn.user_message_tokens,
        "has_tool_use": len(trace.tool_calls) > 0,
        "num_tools": len(trace.tool_calls),
        "task_type": classify_task(trace),
        "best_model": trace.routing.model_selected,
        "outcome_quality": trace.outcome.task_completed,
    })
    
    # 5. Multi-step planning: (task, full_tool_chain)
    if len(trace.tool_calls) >= 3 and trace.outcome.task_completed:
        datasets["planning"].append({
            "task": trace.turn.user_message,
            "plan": [
                {"step": i, "tool": tc.tool_name, "args": tc.tool_args}
                for i, tc in enumerate(trace.tool_calls)
            ],
            "success": True,
        })
    
    return datasets
```

### 9.5 The Tool-Aware Router

The router doesn't just pick a model — it predicts whether the request will need tool use and routes accordingly:

```yaml
router:
  tool_awareness:
    # Predict tool needs from prompt
    predict_tools: true
    
    routing_rules:
      # Code editing tasks → needs file read/edit tools → large model
      - predicted_tools: [read_file, edit, bash]
        route: large
        reason: "multi-step tool use requires strong reasoning"
        
      # Simple factual Q&A → no tools → small model
      - predicted_tools: []
        route: small
        reason: "no tool use, simple generation"
        
      # Search-only tasks → one tool → small model can handle
      - predicted_tools: [search]
        route: small
        reason: "single tool, pattern-matchable"
        
      # Complex multi-tool chains → large model
      - predicted_tools_count: ">3"
        route: large
        reason: "complex planning required"
    
    # Learn from outcomes: if small model fails tool tasks,
    # update routing to send similar requests to large model
    feedback_loop:
      on_tool_failure: increase_large_routing_weight
      on_tool_success_small: decrease_large_routing_weight
      update_interval: daily
```

### 9.6 Distillation from Tool Traces

The most powerful use: teach the small model to use tools by watching the large model.

```yaml
tool_distillation:
  # Phase 1: Imitation learning
  # Small model learns to mimic large model's tool calls
  imitation:
    teacher: large
    student: small
    data: tool_sft_dataset         # from captured traces
    method: supervised_fine_tuning
    filter: task_completed == true  # only learn from successes
    schedule: weekly
    
  # Phase 2: Self-play with verification
  # Small model attempts tool tasks, large model verifies
  self_play:
    enabled: true                  # after Phase 1 converges
    method:
      - small model generates tool calls for a prompt
      - tools execute (sandboxed)
      - large model scores the result
      - if score > threshold: positive example for small model
      - if score < threshold: large model generates correct trace → training pair
    budget: 1000_traces_per_week
    
  # Phase 3: Curriculum learning
  # Start with easy tool tasks, progressively harder
  curriculum:
    stages:
      - single_tool_calls          # read a file, run a search
      - two_step_chains            # read then edit
      - multi_step_with_branching  # search, evaluate, decide, act
      - error_recovery             # handle failures gracefully
    promotion_threshold: 0.90      # 90% success rate to advance
```

### 9.7 Privacy and Safety

```yaml
data_safety:
  # PII detection and scrubbing
  pii:
    detect: [email, phone, ssn, credit_card, api_key, password]
    action: hash_and_replace       # replace with hashed placeholder
    model: presidio                # or custom NER
    
  # Code safety
  code:
    strip_secrets: true            # detect .env values, tokens, keys
    strip_paths: true              # replace /home/user with /home/<user>
    
  # Opt-out
  consent:
    header: "X-No-Training: true"  # per-request opt-out
    session_flag: true             # per-session opt-out
    account_setting: true          # per-account opt-out
    
  # Data retention
  retention:
    raw_traces: 90_days            # delete raw after 90 days
    training_datasets: 1_year      # processed datasets kept longer
    aggregated_metrics: forever    # anonymized stats kept permanently
    
  # Access control
  access:
    raw_traces: [data_team]
    training_data: [ml_team]
    metrics: [everyone]
```

### 9.8 The Complete Data Flywheel

```
                         ┌─────────────────────┐
                         │    User Request      │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │   Tool-Aware Router   │◄─── Learned from traces
                         └────┬────────────┬─────┘
                              │            │
                    ┌─────────▼──┐    ┌───▼──────────┐
                    │ Small Model │    │  Large Model  │
                    │ + Tools     │    │  + Tools      │
                    └──────┬──────┘    └──────┬────────┘
                           │                  │
                    ┌──────▼──────────────────▼────────┐
                    │         Tool Execution            │
                    │   read, edit, bash, search, ...   │
                    └──────────────┬────────────────────┘
                                  │
                    ┌─────────────▼─────────────────────┐
                    │       Trace Capture Layer          │
                    │  (every tool call + args + result  │
                    │   + timing + outcome + feedback)   │
                    └──────────────┬────────────────────┘
                                  │
              ┌───────────────────┼──────────────────────┐
              │                   │                      │
       ┌──────▼──────┐   ┌──────▼───────┐   ┌──────────▼──────────┐
       │  Tool SFT    │   │  DPO Pairs   │   │  Router Training    │
       │  Dataset     │   │  Dataset     │   │  Dataset            │
       └──────┬───────┘   └──────┬───────┘   └──────────┬──────────┘
              │                  │                       │
       ┌──────▼──────┐   ┌──────▼───────┐   ┌──────────▼──────────┐
       │  Distill     │   │  Fine-tune   │   │  Improve Router     │
       │  large→small │   │  both models │   │  accuracy           │
       └──────┬───────┘   └──────┬───────┘   └──────────┬──────────┘
              │                  │                       │
              └──────────────────┼───────────────────────┘
                                 │
                         ┌───────▼────────┐
                         │  Better Models  │
                         │  Better Router  │
                         │  Lower Cost     │
                         └───────┬─────────┘
                                 │
                                 └──────────► Next Request (improved)
```

Every request that flows through the system makes it smarter. Tool calls provide the richest signal because they're structured, verifiable, and contain both the decision (which tool, what args) and the outcome (success, failure, user reaction).

After 6 months of production traffic:
- The small model handles 80% of tool-use tasks (distilled from large)
- The router knows which tasks need the large model (learned from outcomes)
- Both models are fine-tuned on real user preferences (DPO from corrections)
- Error recovery patterns are learned from actual failures and retries
- Multi-step planning is learned from successful tool chains

The system wrote its own training data, trained itself, and deployed improvements — all from the data flowing through the router.

### Phase 15: Tool-Call Capture Layer (1 week)
- Intercept tool calls at router level
- Structured trace logging with outcome tracking
- PII scrubbing pipeline
- **Impact:** Enables all downstream training pipelines

### Phase 16: Tool Distillation Pipeline (3 weeks)
- Generate tool SFT dataset from captured traces
- Imitation learning: small model mimics large model's tool use
- Self-play with verification for curriculum learning
- **Impact:** Small model learns tool use, handles 50%+ of tool tasks

### Final Success Metrics (Updated)

| Metric | Week 1 | Month 1 | Month 6 |
|--------|--------|---------|---------|
| Effective throughput | 1,816 tok/s | 2,800 tok/s | 4,000+ tok/s |
| Large model traffic share | 100% | 60% | 20% |
| Small model tool-use accuracy | 0% | 40% | 80% |
| Router accuracy | N/A | 80% | 95%+ |
| Training traces collected | 0 | 100K | 10M+ |
| Tool call traces | 0 | 30K | 3M+ |
| Quality (vs baseline) | 1.00 | 1.00 | 1.02 |
| Cost per million tokens | $X | $0.5X | $0.15X |
| Human intervention | Daily | Weekly | Monthly |
