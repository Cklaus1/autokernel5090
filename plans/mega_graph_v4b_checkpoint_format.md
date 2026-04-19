# Mega-Graph v4b: Gemma4 26B NVFP4 Checkpoint Format Audit

**Date:** 2026-04-18  
**Model:** `/root/models/gemma-4-26B-A4B-it-NVFP4-modelopt` (17.0 GB)  
**Purpose:** Characterize on-disk FP4 weight and scale tensors for v4b FP4 inline-weight integration.

---

## 1. Checkpoint Provenance

This checkpoint was produced by `convert_ct_to_modelopt.py` from RedHat's
compressed-tensors format. It follows NVIDIA's Gemma-4-31B-IT-NVFP4
`NVFP4_MLP_ONLY_CFG` pattern:

- **Quantized layers:** All MLP/MoE expert layers — `gate_proj`, `up_proj`,
  `down_proj` across 128 experts × 30 layers = 3,840 modules.
- **BF16 layers:** All `self_attn` (q/k/v/o_proj), `router.proj`, `lm_head`,
  vision/audio towers. Attention was explicitly dequantized during conversion
  to avoid vLLM QKV-fusion scale underflow.

---

## 2. `config.json` Quantization Config

```json
{
  "quantization_config": {
    "quant_method": "modelopt",
    "kv_cache_quant_algo": null,
    "quantization": {
      "quant_algo": "NVFP4",
      "group_size": 16,
      "exclude_modules": ["lm_head", "model.language_model.layers.*.self_attn*"]
    }
  }
}
```

Companion `hf_quant_config.json`:
```json
{
  "producer": {"name": "modelopt", "version": "0.37.0"},
  "quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": null, "group_size": 16}
}
```

---

## 3. Tensor Naming Convention (per vLLM `ModelOptNvFp4LinearMethod`)

For each quantized linear layer (e.g., `model.language_model.layers.0.experts.0.gate_proj`):

| Field | Tensor name suffix | Role |
|---|---|---|
| Packed FP4 weights | `.weight` | uint8, 2 FP4 nibbles per byte |
| Per-block scale | `.weight_scale` | fp8_e4m3fn, one value per 16-element block |
| Global / output scale | `.weight_scale_2` | float32 scalar, `1 / weight_global_scale` (inverted from CT format) |
| Input global scale | `.input_scale` | float32 scalar, `1 / input_global_scale` (inverted from CT format) |

**Note:** The CT format used `.weight_global_scale` (divisor) and
`.input_global_scale` (divisor); the modelopt format stores their reciprocals
as `.weight_scale_2` and `.input_scale`. The multiply-path is:

```
dequant = weight_packed (fp4 nibbles) * weight_scale (fp8 per-block) * weight_scale_2 (fp32 global)
output  = (activation * input_scale) @ dequant_weight   # where input_scale is 1/input_global_scale
```

---

## 4. FP4 Weight Tensor: dtype, shape, stride

**Example tensor:** `model.language_model.layers.0.experts.0.gate_proj.weight`

| Property | Value | Notes |
|---|---|---|
| dtype | `torch.uint8` | 2 × FP4-E2M1 nibbles packed per byte |
| shape | `[N, K/2]` | N = output dim, K = input dim; K/2 because 2 nibbles/byte |
| stride | row-major `[K/2, 1]` | standard contiguous layout |
| on-disk layout | row-major (NOT transposed) | CUTLASS NT convention applied at load time |

For Gemma4 26B expert layers:
- `gate_proj` and `up_proj`: N = hidden_dim_per_expert (varies), K = 4096
- `down_proj`: N = 4096, K = hidden_dim_per_expert
- Typical hidden_dim_per_expert ≈ 512–2048 depending on expert width

**Packing convention (low-nibble first):**
```
byte i contains: nibble_lo = weight[row, 2*i]     (elements 0,2,4,...)
                 nibble_hi = weight[row, 2*i + 1]  (elements 1,3,5,...)
packed_byte = (nibble_hi << 4) | nibble_lo
```

This matches vLLM's `scaled_fp4_quant` output and the dequant in
`convert_ct_to_modelopt.py`:
```python
lo = table[packed & 0xF]     # even columns
hi = table[(packed >> 4) & 0xF]  # odd columns
```

---

## 5. Scale Tensor: dtype, shape, granularity

**Example tensor:** `model.language_model.layers.0.experts.0.gate_proj.weight_scale`

| Property | Value | Notes |
|---|---|---|
| dtype | `torch.float8_e4m3fn` (FP8 E4M3) | NOT float32; stored as hardware FP8 |
| shape | `[N, K/16]` | one scale per block of 16 consecutive K-elements |
| block size | **16 elements** | `FP4_BLOCK_SIZE = 16` (confirmed in both .cu files) |
| stride | row-major before swizzle | `[K/16, 1]` in the original CT checkpoint |

**Key finding:** In the modelopt checkpoint as stored on disk, `weight_scale`
is in **row-major layout** (not CUTLASS swizzled). The swizzle is applied at
kernel load time inside `cutlass_fp4_moe_mm` / `rms_norm_dynamic_fp4_quant.cu`
(see Section 6 below). This is confirmed by:
1. The `is_sf_swizzled` boolean parameter in `rms_norm_dynamic_fp4_quant_kernel`.
2. The `fused_shuffle_quant.cu` kernel writing swizzled output for runtime use,
   whereas the checkpoint was built by modelopt without pre-swizzling.

---

## 6. CUTLASS 128×4 Blockscale Swizzle

The swizzle maps from a logical `(row, block_col)` index to a flat byte
offset for the CUTLASS `SF_LAYOUT_SWIZZLED` tensor. Defined in both
`rms_norm_dynamic_fp4_quant.cu::swizzled_sf_offset` and
`fused_shuffle_quant.cu::compute_cutlass_sf_offset`:

```c
// Given: row (row index in scale buffer), sf_col (block index in K dimension)
// numKTiles = ceil(K / 64)  [since 64 elements = 4 blocks × 16 elem/block]
int mTileIdx  = row >> 7;            // row / 128
int outerMIdx = row & 31;            // row % 32
int innerMIdx = (row >> 5) & 3;      // (row / 32) % 4
int kTileIdx  = sf_col >> 2;         // sf_col / 4
int innerKIdx = sf_col & 3;          // sf_col % 4
byte_offset = ((mTileIdx * numKTiles + kTileIdx) << 9)
              | (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;
```

**Logical layout:** `[numMTiles, numKTiles, 32, 4, 4]` (5D)
- `numMTiles = ceil(M / 128)` (M-tile dimension)
- `numKTiles = ceil(K / 64)` (K-tile dimension; 64 = 4 blocks × 16 elem)
- Innermost `[32, 4, 4]` = `[outerM, innerM, innerK]`

**On-disk:** row-major (NOT swizzled)  
**At runtime:** swizzled during the first quantization call or via the
fused kernel. The `fused_shuffle_quant.cu` MoE path writes directly in swizzled format.

---

## 7. Global Scale Fields (`weight_scale_2` and `input_scale`)

Both are **per-tensor float32 scalars** stored as rank-0 (0-dim) or rank-1
(single-element) tensors:

| Tensor | dtype | shape | value |
|---|---|---|---|
| `.weight_scale_2` | float32 | `[]` or `[1]` | `1 / weight_global_scale` (conversion from CT) |
| `.input_scale` | float32 | `[]` or `[1]` | `1 / input_global_scale` |

These correspond to vLLM `ModelOptNvFp4LinearMethod`:
- `weight_scale_2` → `alpha` (weight dequant global factor, multiply path)
- `input_scale` → `input_global_scale` (activation quant global factor)

The **dequant formula** for a single element is:
```
fp4_val = e2m1_decode[nibble]       # 8-value lookup: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}
block_scale = fp8_to_fp32(weight_scale[row, col//16])
global_scale = weight_scale_2       # fp32 scalar
dequant_weight = fp4_val * block_scale * global_scale
```

---

## 8. FP4-E2M1 Encoding

FP4-E2M1: 1 sign + 2 exponent + 1 mantissa bit. 4-bit nibble, 16 codes:

```
code  magnitude  (sign bit duplicates for negative)
0 → 0.0
1 → 0.5
2 → 1.0
3 → 1.5
4 → 2.0
5 → 3.0
6 → 4.0
7 → 6.0
8 → -0.0  (sign=1, code=0)
9 → -0.5
A → -1.0
B → -1.5
C → -2.0
D → -3.0
E → -4.0
F → -6.0
```

Quantization boundaries: `0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0`  
Max value: `FP4_E2M1_MAX = 6.0`

---

## 9. Safetensors Header Meta Fields

The checkpoint uses a single `model.safetensors` file (17 GB). Safetensors
header contains per-tensor metadata:

```json
{
  "model.language_model.layers.0.experts.0.gate_proj.weight": {
    "__metadata__": {},
    "dtype": "U8",           // uint8 (packed FP4)
    "shape": [N, K_half],    // [out_features, in_features // 2]
    "data_offsets": [start, end]
  },
  "model.language_model.layers.0.experts.0.gate_proj.weight_scale": {
    "dtype": "F8_E4M3",      // fp8_e4m3fn
    "shape": [N, K_blocks],  // [out_features, in_features // 16]
    "data_offsets": [start, end]
  },
  "model.language_model.layers.0.experts.0.gate_proj.weight_scale_2": {
    "dtype": "F32",
    "shape": [],             // scalar
    "data_offsets": [start, end]
  }
}
```

Attention weights (BF16 after dequantization):
```json
{
  "model.language_model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [start, end]
  }
}
```

---

## 10. Checkpoint Format Summary Table

| Property | Value |
|---|---|
| Format | modelopt (`quant_method: modelopt`) |
| FP4 dtype on disk | `uint8` (2 nibbles packed per byte) |
| FP4 packing | low-nibble = even element, high-nibble = odd element |
| Scale dtype | `fp8_e4m3fn` (FP8 E4M3, NOT float32) |
| Scale granularity | **16 elements per block** (K dimension) |
| Scale shape | `[N, K/16]` |
| Scale layout on disk | **Row-major (NOT swizzled)** |
| Swizzle applied | At runtime by CUTLASS quant kernel |
| Global scale (`weight_scale_2`) | float32 scalar, `1/weight_global_scale` |
| Input scale (`input_scale`) | float32 scalar, `1/input_global_scale` |
| Attention layers | BF16 (dequantized during conversion) |
| MoE expert layers | NVFP4 (weight + weight_scale + weight_scale_2) |

---

## 11. v4b Integration Notes

For the v4b inline-FP4-weights mega-graph kernel:

1. **Load weight tiles as uint8** from HBM. 0.5 bytes/element (vs 2 bytes for BF16).
2. **Load scale tiles as fp8_e4m3fn** from HBM. One byte per 16 FP4 elements.
3. **Dequant to BF16 in smem** before WMMA (Option A). See `mega_graph_v4b_fp4_spec.md`.
4. **Swizzle on-the-fly:** checkpoint scales are row-major; the cooperative kernel
   can either (a) read row-major and swizzle into a smem staging buffer, or
   (b) rely on the fact that at tile-load time, we access consecutive rows of a
   single tile — the swizzle only matters for the CUTLASS MoE path, not for
   per-tile cooperative access in the mega-graph.
5. **No pre-swizzling needed** for the mega-graph cooperative kernel because each
   SM owns a contiguous row-stripe. The scale access pattern is sequential
   `[row, block_idx]` for each SM's assigned rows, which is already row-major.
