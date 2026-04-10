# Kernel Generator Plan

## Vision

A single parameterized Triton kernel template that generates decode attention kernels from a declarative spec. Instead of writing 250-line kernels per KV cache format, describe the format in ~20 lines of YAML and get a correct, optimized kernel.

## Current State

We have 2 hand-written kernels:
- `triton_fusencache_v5.py` — K8V4B16 (K=int8, V=int4, block=16)
- `triton_fusencache_v6.py` — K4V4B32 (K=int4, V=int4, block=32)

They share ~70% of their code. The differences are:
1. How K is unpacked (int8 view vs nibble split)
2. How K is dequantized (k * scale vs (k - 7.5) * scale)
3. How QK^T is computed (direct tl.dot vs split even/odd)
4. Cache layout offsets (D vs D/2 for K region)
5. Scale block size (16 vs 32)

Everything else — page table lookup, online softmax, V accumulation, split reduction — is identical.

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────┐
│  KV Cache Spec   │────▶│  Kernel Generator │────▶│ Triton Kernel │
│  (YAML / dict)   │     │  (Python)         │     │ (compiled)    │
└─────────────────┘     └──────────────────┘     └──────────────┘
                              │
                              ▼
                        ┌──────────────┐
                        │  Primitives   │
                        │  Registry     │
                        └──────────────┘
```

## Phase 1: Primitive Registry

### 1.1 Unpack Primitives

Each unpacker takes raw cache bytes and produces numerical values ready for dequant.

```python
# registry of unpack functions, each a Triton-JIT-compatible snippet
UNPACK = {
    "int8": {
        # int8: 1 value per byte, direct cast
        "values_per_byte": 1,
        "code": lambda raw: "raw.to(tl.int8).to(tl.float32)",
        "output_dims": "full",  # output has same dims as head_dim
    },
    "nibble_pair": {
        # int4: 2 values per byte, low and high nibble
        "values_per_byte": 2,
        "code": lambda raw: ("(raw.to(tl.int32) & 0xF).to(tl.float32)",
                             "((raw.to(tl.int32) >> 4) & 0xF).to(tl.float32)"),
        "output_dims": "half_split",  # produces even/odd half-dim arrays
    },
    "fp8_e5m2": {
        "values_per_byte": 1,
        "code": lambda raw: "raw.to(tl.float8e5m2).to(tl.float32)",
        "output_dims": "full",
    },
    "fp8_e4m3": {
        "values_per_byte": 1,
        "code": lambda raw: "raw.to(tl.float8e4m3fn).to(tl.float32)",
        "output_dims": "full",
    },
    "crumb_quad": {
        # int2: 4 values per byte
        "values_per_byte": 4,
        "code": lambda raw: ("(raw & 0x3)", "((raw >> 2) & 0x3)",
                             "((raw >> 4) & 0x3)", "((raw >> 6) & 0x3)"),
        "output_dims": "quarter_split",
    },
}
```

### 1.2 Dequant Primitives

```python
DEQUANT = {
    "symmetric": {
        # (code - offset) * scale
        "params": ["offset", "scale"],
        "code": "(codes - {offset}) * scale",
    },
    "asymmetric": {
        # code * scale + zero_point
        "params": ["scale", "zero_point"],
        "code": "codes * scale + {zero_point}",
    },
    "identity": {
        # no-op (FP8, FP16)
        "params": [],
        "code": "codes",
    },
}
```

### 1.3 Scale Primitives

```python
SCALE_LOAD = {
    "per_block": {
        # One scale per block_size elements
        "params": ["block_size"],
        "index": "dim_offset // (block_size // values_per_byte)",
        "storage": "side_tensor",  # separate (max_slots, Hk, D/block_size, 2) tensor
    },
    "per_tensor": {
        "params": [],
        "index": "0",
        "storage": "side_tensor",
    },
    "per_channel": {
        "params": [],
        "index": "head_idx",
        "storage": "side_tensor",
    },
    "none": {
        "params": [],
        "storage": "none",
    },
}
```

### 1.4 Dot Product Primitives

```python
DOT_STRATEGY = {
    "direct": {
        # Full-dim dot: tl.dot(q, k^T)
        # Works when K has full-dim output (int8, fp8, fp16)
        "requires": "full",
        "code": "tl.dot(q.to(tl.float32), k_T)",
    },
    "split_even_odd": {
        # Split Q into even/odd dims, two half-dim dots
        # Works when K is nibble-packed (int4)
        "requires": "half_split",
        "code": "tl.dot(q_even, k_lo_T) + tl.dot(q_odd, k_hi_T)",
    },
    "split_4way": {
        # 4 quarter-dim dots for 2-bit packing
        "requires": "quarter_split",
        "code": "tl.dot(q0, k0_T) + tl.dot(q1, k1_T) + tl.dot(q2, k2_T) + tl.dot(q3, k3_T)",
    },
}
```

### 1.5 Fixed Blocks (never change)

```python
# Online softmax — always the same
ONLINE_SOFTMAX = """
n_e_max = tl.maximum(tl.max(qk, 1), e_max)
re_scale = tl.exp(e_max - n_e_max)
p = tl.exp(qk - n_e_max[:, None])
{v_accumulate}
e_sum = e_sum * re_scale + tl.sum(p, 1)
e_max = n_e_max
"""

# Page table lookup — always the same
PAGE_TABLE_LOOKUP = """
block_nums = tl.load(Block_table_ptr + cur_batch * stride_bt_b + kv_offs // PAGE_SIZE,
                     mask=kv_mask, other=0)
page_off = kv_offs % PAGE_SIZE
slot_bases = (block_nums * stride_cache_block
              + page_off * stride_cache_pos
              + cur_kv_head * stride_cache_head)
"""

# Split-KV reduction (stage 2) — always the same
SPLIT_REDUCE = """
for s in range(NUM_KV_SPLITS):
    sl = tl.cdiv(seq_len, NUM_KV_SPLITS)
    if tl.minimum(sl * s + sl, seq_len) > sl * s:
        off = mid_base + s * stride_mid_s
        tv = tl.load(Mid_out_ptr + off + d_offs, mask=d_mask, other=0.0)
        tlogic = tl.load(Mid_out_ptr + off + HEAD_DIM)
        n = tl.maximum(tlogic, e_max)
        r = tl.exp(e_max - n)
        acc = acc * r + tl.exp(tlogic - n) * tv
        e_sum = e_sum * r + tl.exp(tlogic - n)
        e_max = n
"""
```

---

## Phase 2: Spec Format

### 2.1 KV Cache Spec

```python
@dataclass
class KVCacheQuantSpec:
    """Fully describes a KV cache quantization format."""
    name: str
    
    # K config
    k_bits: int                    # 2, 4, 6, 8, 16
    k_unpack: str                  # key into UNPACK registry
    k_dequant: str                 # key into DEQUANT registry
    k_dequant_offset: float        # e.g. 7.5 for int4 symmetric
    k_scale: str                   # key into SCALE_LOAD registry
    k_scale_block: int             # elements per scale block
    
    # V config
    v_bits: int
    v_unpack: str
    v_dequant: str
    v_dequant_offset: float
    v_scale: str
    v_scale_block: int
    
    # Derived
    @property
    def k_bytes_per_element(self): return self.k_bits / 8
    @property
    def v_bytes_per_element(self): return self.v_bits / 8
    @property
    def slot_bytes(self): return int((self.k_bits + self.v_bits) / 8 * D)
    @property
    def compression_vs_bf16(self): return 32.0 / (self.k_bits + self.v_bits)
    @property
    def dot_strategy(self): return "direct" if self.k_bits >= 8 else "split_even_odd" if self.k_bits == 4 else "split_4way"
```

### 2.2 Predefined Specs

```python
SPECS = {
    "fp16":    KVCacheQuantSpec("fp16",    16, "identity", "identity", 0, "none", 0,     16, "identity", "identity", 0, "none", 0),
    "fp8":     KVCacheQuantSpec("fp8",      8, "fp8_e5m2", "identity", 0, "none", 0,      8, "fp8_e5m2", "identity", 0, "none", 0),
    "k8v4b16": KVCacheQuantSpec("k8v4b16",  8, "int8", "symmetric", 0,   "per_block", 16, 4, "nibble_pair", "symmetric", 7.5, "per_block", 16),
    "k4v4b32": KVCacheQuantSpec("k4v4b32",  4, "nibble_pair", "symmetric", 7.5, "per_block", 32, 4, "nibble_pair", "symmetric", 7.5, "per_block", 32),
    "k8v2b16": KVCacheQuantSpec("k8v2b16",  8, "int8", "symmetric", 0,   "per_block", 16, 2, "crumb_quad", "symmetric", 1.5, "per_block", 16),
    "k4v4b16": KVCacheQuantSpec("k4v4b16",  4, "nibble_pair", "symmetric", 7.5, "per_block", 16, 4, "nibble_pair", "symmetric", 7.5, "per_block", 16),
    "k6v4b32": KVCacheQuantSpec("k6v4b32",  6, "int6_pack", "symmetric", 31.5, "per_block", 32, 4, "nibble_pair", "symmetric", 7.5, "per_block", 32),
}
```

Adding a new format = 1 line in this table.

---

## Phase 3: Kernel Generator

### 3.1 Code Generation Approach

Two options for generating the Triton kernel:

**Option A: Template string substitution**
- Generate the full kernel source as a string
- `exec()` or write to file, then import
- Pro: simple, debuggable (can print the generated source)
- Con: string manipulation, hard to get right

**Option B: Composable JIT functions**
- Write each primitive as a `@triton.jit` helper
- Compose them in a master kernel that dispatches based on `tl.constexpr` flags
- Pro: native Triton, compiler optimizes dead branches
- Con: all primitives must be constexpr-selectable

**Recommended: Option B** — constexpr dispatch. Triton's compiler eliminates dead branches at compile time, so a single kernel with `if K_BITS: tl.constexpr == 4` compiles to the same code as a hand-written int4 kernel.

### 3.2 Master Kernel Structure

```python
@triton.jit
def universal_decode_stage1(
    Q_ptr, KV_cache_ptr, Scales_ptr, Block_table_ptr, Seq_lens_ptr, Mid_out_ptr,
    # ... strides ...
    sm_scale,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    Q_HEAD_NUM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    # Data-driven constexprs:
    K_BITS: tl.constexpr,          # 4, 8, 16
    V_BITS: tl.constexpr,          # 4, 8, 16
    K_OFFSET: tl.constexpr,        # 0.0, 7.5, 127.0
    V_OFFSET: tl.constexpr,        # 0.0, 7.5
    SCALE_BLOCK: tl.constexpr,     # 16, 32, 0 (none)
    K_CACHE_OFFSET: tl.constexpr,  # byte offset of K in slot
    V_CACHE_OFFSET: tl.constexpr,  # byte offset of V in slot
    K_PACKED_SIZE: tl.constexpr,   # bytes of K per head
    V_PACKED_SIZE: tl.constexpr,   # bytes of V per head
):
    # === FIXED: setup ===
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    # ... head masking, seq bounds ...
    
    # === FIXED: load Q (split if needed) ===
    if K_BITS >= 8:
        q = load_q_full(Q_ptr, ...)
    else:
        q_even, q_odd = load_q_split(Q_ptr, ...)
    
    # === FIXED: accumulators ===
    e_max, e_sum = init_softmax()
    if V_BITS >= 8:
        acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)
    else:
        acc_even = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)
        acc_odd = tl.zeros([BLOCK_H, BLOCK_D // 2], dtype=tl.float32)
    
    for start_n in range(split_start, split_end, BLOCK_KV):
        # === FIXED: page table lookup ===
        slot_bases, sc_base = page_lookup(...)
        
        # === DATA-DRIVEN: K unpack + dequant ===
        if K_BITS == 8:
            k = unpack_int8(KV_cache_ptr, slot_bases, K_CACHE_OFFSET)
            if K_OFFSET == 0:
                k = k * load_scale(Scales_ptr, sc_base, 0, SCALE_BLOCK)
            else:
                k = (k - K_OFFSET) * load_scale(...)
        elif K_BITS == 4:
            k_lo, k_hi = unpack_nibble(KV_cache_ptr, slot_bases, K_CACHE_OFFSET)
            k_sc = load_scale(Scales_ptr, sc_base, 0, SCALE_BLOCK)
            k_lo = (k_lo - K_OFFSET) * k_sc
            k_hi = (k_hi - K_OFFSET) * k_sc
        
        # === DATA-DRIVEN: QK dot ===
        if K_BITS >= 8:
            qk = tl.dot(q, k) * sm_scale       # direct
        else:
            qk = (tl.dot(q_even, tl.trans(k_lo))
                  + tl.dot(q_odd, tl.trans(k_hi))) * sm_scale
        
        # === DATA-DRIVEN: V unpack + dequant (same pattern as K) ===
        # ... mirrors K logic with V_BITS, V_OFFSET, V_CACHE_OFFSET ...
        
        # === FIXED: online softmax + accumulate ===
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        if V_BITS >= 8:
            acc = acc * re_scale[:, None] + tl.dot(p, v)
        else:
            acc_even = acc_even * re_scale[:, None] + tl.dot(p, v_lo)
            acc_odd = acc_odd * re_scale[:, None] + tl.dot(p, v_hi)
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max
    
    # === FIXED: store results ===
    # ...
```

Because `K_BITS`, `V_BITS`, etc. are `tl.constexpr`, Triton eliminates all the dead branches at compile time. The compiled PTX for `K_BITS=4` is identical to a hand-written int4 kernel.

### 3.3 Python Wrapper Generator

```python
def make_decode_fn(spec: KVCacheQuantSpec):
    """Generate a decode function from spec."""
    
    BLOCK_D = triton.next_power_of_2(HEAD_DIM)
    
    def decode(query, kv_cache, scales, block_table, seq_lens, scale, num_kv_heads):
        B, Hq, D = query.shape
        mid_out = torch.empty(B, Hq, 64, D + 1, dtype=torch.float32, device=query.device)
        output = torch.empty(B, Hq, D, dtype=query.dtype, device=query.device)
        
        grid1 = (B, triton.cdiv(Hq, 8), 64)
        universal_decode_stage1[grid1](
            query, kv_cache, scales, block_table, seq_lens, mid_out,
            # ... strides ...
            scale,
            HEAD_DIM=D, BLOCK_D=BLOCK_D, BLOCK_KV=16, BLOCK_H=8,
            NUM_KV_SPLITS=64, KV_GROUP_SIZE=Hq // num_kv_heads,
            Q_HEAD_NUM=Hq, PAGE_SIZE=kv_cache.shape[1],
            # Data-driven params from spec:
            K_BITS=spec.k_bits, V_BITS=spec.v_bits,
            K_OFFSET=spec.k_dequant_offset, V_OFFSET=spec.v_dequant_offset,
            SCALE_BLOCK=spec.k_scale_block,
            K_CACHE_OFFSET=0,
            V_CACHE_OFFSET=int(spec.k_bits * D / 8),
            K_PACKED_SIZE=int(spec.k_bits * D / 8),
            V_PACKED_SIZE=int(spec.v_bits * D / 8),
            num_warps=4, num_stages=1,
        )
        
        grid2 = (B, Hq)
        universal_decode_stage2[grid2](mid_out, output, seq_lens, ...)
        return output
    
    return decode
```

### 3.4 Store Kernel Generator

Same approach for the `do_kv_cache_update` (quantize + pack + scatter):

```python
def make_store_fn(spec: KVCacheQuantSpec):
    """Generate a store function that quantizes and packs K/V."""
    
    def store(key, value, kv_cache, slot_mapping, scales_tensor):
        # Quantize K
        if spec.k_bits == 8:
            k_packed = quantize_int8_symmetric(key, spec.k_scale_block)
        elif spec.k_bits == 4:
            k_packed = quantize_int4_symmetric(key, spec.k_scale_block, spec.k_dequant_offset)
        
        # Quantize V (same pattern)
        ...
        
        # Pack and scatter (fixed pattern)
        packed = torch.cat([k_packed, v_packed], dim=-1)
        scatter_to_cache(packed, kv_cache, slot_mapping)
        scatter_scales(k_scale, v_scale, scales_tensor, slot_mapping)
    
    return store
```

---

## Phase 4: Autotune Integration

The kernel generator plugs directly into AutoKernel's optimization loop:

```python
# Generate all candidate configs
candidates = [
    KVCacheQuantSpec("k8v4b16", 8, ..., 4, ..., 16),
    KVCacheQuantSpec("k8v4b32", 8, ..., 4, ..., 32),
    KVCacheQuantSpec("k4v4b16", 4, ..., 4, ..., 16),
    KVCacheQuantSpec("k4v4b32", 4, ..., 4, ..., 32),
    KVCacheQuantSpec("k4v4b64", 4, ..., 4, ..., 64),
    # ... 50+ configs ...
]

for spec in candidates:
    decode_fn = make_decode_fn(spec)
    store_fn = make_store_fn(spec)
    
    # Quality test (fast, ~0.01s per config)
    quality = test_attention_accuracy(decode_fn, store_fn, real_kv_data)
    if quality < threshold:
        continue
    
    # Speed test
    throughput = benchmark_decode(decode_fn, batch_size=1, seq_len=1024)
    
    # Log result
    log_to_tsv(spec.name, quality, throughput, spec.compression_vs_bf16)
```

---

## Phase 5: Deliverables

### Files

```
autokernel/
  kv_cache_gen/
    __init__.py
    spec.py              # KVCacheQuantSpec dataclass
    primitives.py        # UNPACK, DEQUANT, SCALE_LOAD, DOT_STRATEGY registries
    kernel_template.py   # universal_decode_stage1/stage2 Triton kernels
    generate.py          # make_decode_fn(), make_store_fn()
    sweep.py             # Automated spec sweep with quality + speed benchmarks
    predefined.py        # SPECS dict with known-good configs
```

### API

```python
from autokernel.kv_cache_gen import KVCacheQuantSpec, make_decode_fn, make_store_fn

# Use a predefined spec
spec = KVCacheQuantSpec.from_name("k4v4b32")

# Or define a new one
spec = KVCacheQuantSpec(
    name="k6v4b32",
    k_bits=6, k_unpack="int6_pack", k_dequant="symmetric",
    k_dequant_offset=31.5, k_scale="per_block", k_scale_block=32,
    v_bits=4, v_unpack="nibble_pair", v_dequant="symmetric",
    v_dequant_offset=7.5, v_scale="per_block", v_scale_block=32,
)

# Generate kernel
decode = make_decode_fn(spec)
store = make_store_fn(spec)

# Use in vLLM attention backend
output = decode(query, kv_cache, scales, block_table, seq_lens, scale, num_kv_heads)
```

---

## Success Criteria

1. Universal kernel matches hand-written v5/v6 output (bitwise or within 1e-4)
2. Universal kernel matches hand-written v5/v6 throughput (within 5%)
3. Adding a new KV format takes <10 lines of spec, zero Triton code
4. Sweep of 50+ configs runs in <60 seconds
5. Generated kernels are CUDA-graph safe

---

## Phase 6: Maximum Performance (ASI-level)

### 6.1 The Performance Gap

Current: 1,816 tok/s at C=39 (26B MoE, FP8 KV, RTX 5090).
Theoretical: ~896 tok/s single-stream (memory bandwidth bound).
We're at ~4.2% of theoretical peak for single-stream, ~25% for aggregate.

The kernel generator enables the following optimizations because they become parameter changes, not rewrites.

### 6.2 Fused Attention + Dequant

Current kernels dequant K/V to float in registers, then do the dot product — two logical steps. The fused version dequants *inside* the dot product accumulation loop, so K/V values exist only in registers, never materialized as full float tensors.

Add to the universal kernel:

```python
# New constexpr
FUSE_DEQUANT: tl.constexpr,  # True/False

# In the KV loop:
if FUSE_DEQUANT:
    # Load raw packed K, dequant and dot in one step
    # K values live only in registers
    for block_d in range(0, HEAD_DIM, DEQUANT_TILE):
        k_tile = load_and_dequant_tile(KV_cache, slot, block_d, K_BITS, K_OFFSET, scale)
        qk_partial += tl.dot(q_tile, k_tile)
else:
    # Current: full dequant then dot
    k = dequant_full(...)
    qk = tl.dot(q, k)
```

**Expected gain:** ~1.3x (eliminates one full K/V read-write per layer).

Spec addition:
```yaml
kernel:
  fusion: attention_dequant    # vs "separate" (current)
  dequant_tile: 32             # tile size for fused dequant loop
```

### 6.3 Async KV Pipelining

The `num_stages` Triton parameter controls software pipelining — loading the next tile of KV cache while computing on the current tile. We use `num_stages=1` (no overlap) because higher values caused shared memory overflow.

The fix: reduce `BLOCK_KV` and increase `num_stages`. The kernel generator can autotune this:

```python
# Autotune grid for the universal kernel
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_KV": 8,  "num_stages": 2}),
        triton.Config({"BLOCK_KV": 16, "num_stages": 1}),
        triton.Config({"BLOCK_KV": 16, "num_stages": 2}),
        triton.Config({"BLOCK_KV": 32, "num_stages": 1}),
    ],
    key=["HEAD_DIM", "K_BITS", "V_BITS"],
)
def universal_decode_stage1(...):
```

**Expected gain:** ~1.15x (overlaps memory and compute).

Spec addition:
```yaml
kernel:
  pipeline_stages: auto        # autotune 1-3
  block_kv: auto               # autotune 8-32
```

### 6.4 Per-Layer Adaptive Quantization

Different layers have different sensitivity to KV quantization. Early layers (close to input) are typically more sensitive than later layers. The kernel generator makes this trivial — each layer can use a different spec:

```python
# Per-layer spec selection
layer_specs = []
for i in range(30):
    sensitivity = measure_layer_sensitivity(model, layer=i)
    if sensitivity > 0.9:
        layer_specs.append(SPECS["k8v4b16"])    # high quality
    elif sensitivity > 0.5:
        layer_specs.append(SPECS["k4v4b32"])    # balanced
    else:
        layer_specs.append(SPECS["k2v2b64"])    # max compression
```

The universal kernel handles this because each layer's `do_kv_cache_update` and `_decode_attention` read from the layer's own spec. No code changes — just different constexpr values per layer.

**Expected gain:** ~1.5x KV capacity at same quality, or better quality at same capacity.

Spec addition:
```yaml
kv_cache:
  per_layer:
    - layers: [0, 5, 11, 17, 23, 29]   # full attention (sensitive)
      format: k8v4b16
    - layers: [1, 2, 3, 4, 6, 7, 8, 9, 10]  # sliding (medium)
      format: k4v4b32
    - layers: [12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28]
      format: k4v2b64                    # sliding (tolerant)
```

### 6.5 FP4 Native Tensor Core Matmuls

The RTX 5090 has FP4 tensor cores (838 TFLOPS vs 209 TFLOPS for FP16). Currently we use AWQ INT4 → FP16 matmul via Marlin kernels. Native FP4 would be ~2-4x faster for the weight matmuls that dominate 60% of decode time.

This isn't a kernel generator change — it's a weight format change. But the data-driven weight loader makes it a config change:

```yaml
model:
  weight_format: nvfp4_native    # use FP4 tensor cores directly
  # vs current: awq_int4         # INT4 dequant → FP16 matmul
```

**Expected gain:** ~2-3x on the 60% of time spent in matmuls = ~1.6-2.2x overall.

### 6.6 Persistent Kernel / Full CUDA Graph

Current: 30 layers × ~5 kernel launches = 150 launches per token (~750μs overhead).
Goal: 1 CUDA graph replay = 1 launch per token.

The kernel generator produces graph-safe kernels by construction (no dynamic allocation, no Python conditionals, no .item() calls). This makes full CUDA graph coverage achievable:

```yaml
kernel:
  cuda_graph: full              # vs "piecewise" (current)
  persistent: false             # future: single megakernel
```

**Expected gain:** ~1.1x (eliminates launch overhead).

### 6.7 Speculative Expert Prefetch (MoE-specific)

For MoE models, only 4/128 experts are active per token. The router is a small linear layer — we can predict the next token's expert selection during the current token's computation and prefetch those weights.

```python
# In the MoE forward:
# 1. Compute current token normally
# 2. Run router on current hidden state to predict next expert selection
# 3. Issue async prefetch for predicted expert weights
# 4. Next token's expert weights are already in L2 cache

prefetch_spec:
  enabled: true
  predictor: router_reuse      # reuse router weights for prediction
  prefetch_depth: 1            # prefetch 1 token ahead
  confidence_threshold: 0.8    # only prefetch if router confidence > 80%
```

**Expected gain:** ~1.2x (eliminates weight loading stalls for MoE).

### 6.8 Combined Projection

| Optimization | Spec Change | Expected Gain | Cumulative |
|---|---|---|---|
| Fused attention+dequant | `fusion: attention_dequant` | 1.3x | 1.3x |
| Async KV pipelining | `pipeline_stages: auto` | 1.15x | 1.5x |
| Per-layer adaptive quant | `per_layer: [...]` | 1.5x KV | — (capacity) |
| FP4 native matmuls | `weight_format: nvfp4` | 2.0x | 3.0x |
| Full CUDA graph | `cuda_graph: full` | 1.1x | 3.3x |
| Expert prefetch | `prefetch: true` | 1.2x | 3.9x |

**Projected: 1,816 × 3.9 ≈ 7,000 tok/s** at C=39 on RTX 5090.

Or: same 1,816 tok/s at C=10 with ~4x lower latency (P50 < 0.6s).

### 6.9 The 10-line Config That Does It All

```yaml
# autokernel peak performance config
model: gemma-4-26B-A4B-it
weights: nvfp4_native
kv_cache:
  default: k4v4b32
  per_layer:
    sensitive: k8v4b16
    tolerant: k4v2b64
kernel:
  fusion: attention_dequant
  pipeline_stages: auto
  cuda_graph: full
serving:
  expert_prefetch: true
  concurrency: 39
```

Every line is a parameter. Zero custom code. The kernel generator, weight loader, memory planner, and serving config all read from this one spec. AutoKernel's optimization loop can search the space of all valid configs automatically.

---

## Phase 7: Blind Spots and Missing Dimensions

### 7.1 Quality Gates

We've validated kernels with 5-prompt smoke tests. This proves "not garbage" but doesn't catch subtle quality regressions from quantization. Every generated kernel needs an automated quality gate before deployment.

```python
QUALITY_SUITE = {
    "fast": {
        # 30 seconds, run on every kernel generation
        "factual_accuracy": 20,      # simple Q&A
        "math": 10,                  # arithmetic, reasoning
        "code": 5,                   # simple Python
    },
    "full": {
        # 10 minutes, run on winning configs
        "mmlu_subset": 100,          # knowledge breadth
        "humaneval_subset": 50,      # code generation
        "mt_bench": 80,              # conversation quality
        "needle_in_haystack": 20,    # long context retrieval
    },
}

# Kernel generator runs fast suite automatically
# Winning configs get full suite before production deployment
# Results stored per-spec so quality is tracked over time
```

Add to spec:
```yaml
quality:
  gate: fast                     # or "full" for production
  min_accuracy: 0.95             # relative to FP16 baseline
  regression_threshold: 0.02     # max allowed quality drop
```

### 7.2 Prefill Optimization

All our kernel work targets decode (1 token at a time, memory-bound). Prefill (processing the input prompt, compute-bound) is a completely different bottleneck. For a 50K token document, prefill dominates wall-clock time.

The kernel generator should also produce prefill kernels:

```yaml
kernel:
  decode:
    fusion: attention_dequant
    pipeline_stages: auto
  prefill:
    method: flash_attention       # use native flash attn for prefill
    chunked: true                 # chunked prefill for long inputs
    chunk_size: 8192
    # Prefill uses FP16 K/V (not quantized) then quantizes on store
    # No custom prefill kernel needed — only the store path matters
```

The insight: prefill doesn't read from quantized KV cache — it produces fresh K/V and stores them. So the kernel generator only needs the store (quantize+pack) path for prefill, not the decode (dequant+attention) path. Flash attention handles the prefill attention natively.

### 7.3 Model as Optimization Variable

The data-driven architecture should treat model selection as a searchable parameter, not a fixed choice:

```python
MODEL_CANDIDATES = {
    "gemma-4-26B-A4B-AWQ": {
        "weights_gb": 13, "experts": 128, "active": 4,
        "layers": 30, "kv_heads": 8, "head_dim": 256,
    },
    "qwen3-30B-A3B-AWQ": {
        "weights_gb": 11, "experts": 128, "active": 8,
        "layers": 48, "kv_heads": 4, "head_dim": 128,
    },
    "llama-4-scout-17B-AWQ": {
        "weights_gb": 9, "experts": 16, "active": 2,
        "layers": 32, "kv_heads": 8, "head_dim": 128,
    },
}

# For each model: estimate KV tokens, generate kernel, benchmark
# Output: pareto frontier of quality × throughput × KV capacity
```

Add to spec:
```yaml
search:
  models: [gemma-4-26B, qwen3-30B, llama-4-scout-17B]
  optimize_for: throughput        # or "latency", "quality", "kv_capacity"
  constraint: quality > 0.95_of_best
```

### 7.4 Request-Aware Routing

Different requests have different optimal configs. The serving layer should route based on request characteristics:

```yaml
routing:
  rules:
    - condition: prompt_tokens < 500
      config: low_latency          # minimal KV, fast decode
    - condition: prompt_tokens > 10000
      config: high_compression     # K4V4, max KV capacity
    - condition: task == "code"
      config: high_quality         # K8V4, better precision
  default: balanced
```

This requires the kernel generator to produce multiple compiled kernels and the serving layer to select per-request. The data-driven cache planner would manage separate KV pools per config.

### 7.5 Production Monitoring

A deployed system needs continuous quality and performance tracking:

```yaml
monitoring:
  metrics:
    - time_to_first_token          # prefill latency
    - inter_token_latency          # decode speed
    - kv_cache_utilization         # % of KV tokens in use
    - cache_hit_rate               # prefix cache effectiveness
    - memory_fragmentation         # KV block fragmentation over time
    - quality_sample_rate: 0.01    # randomly eval 1% of responses
  alerts:
    - kv_utilization > 0.95        # approaching capacity
    - p99_latency > 30s            # latency spike
    - quality_score < 0.90         # quality regression
```

### 7.6 Speculative Decoding on MoE

We got spec decode working on the 31B (experiment 52, 2.1 tok/s — limited by KV). The 26B MoE with FP8 KV has 109K tokens — enough to fit a draft model. The data-driven system would express this as:

```yaml
speculative:
  enabled: true
  draft_model: gemma-4-E2B-text-only-FP8
  num_speculative_tokens: 5
  # System auto-checks: do both models fit in GPU memory?
  # System auto-checks: is the KV overhead acceptable?
  # System auto-benchmarks: does speculation actually help for this workload?
```

### 7.7 Prefix Caching Quantification

On the 31B model, prefix caching gave +52% throughput. We haven't tested it on the 26B MoE with FP8 KV. For production with shared system prompts (which most deployments have), this could be the single largest free gain:

```yaml
prefix_cache:
  enabled: true                    # vLLM default
  system_prompt_length: 500        # tokens
  expected_hit_rate: 0.80          # 80% of requests share prefix
  # Estimated impact: +40-60% throughput at no cost
```

### 7.8 Updated Success Criteria

Add to existing criteria:

6. Quality gate passes on MMLU/HumanEval subset (>95% of FP16 baseline)
7. Prefill speed within 10% of native flash attention
8. Model search identifies optimal model for target GPU in <2 hours
9. Prefix cache hit rate >80% with shared system prompt
10. Production monitoring detects quality regression within 100 requests
