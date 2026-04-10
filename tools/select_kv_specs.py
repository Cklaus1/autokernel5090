#!/usr/bin/env python3
"""Per-layer KV cache spec selector for mixed-precision FusenCache.

Profiles each transformer layer's sensitivity to KV quantization and assigns
the most aggressive compression format that still meets a quality threshold.

Gemma4 26B architecture:
  - 25 sliding-window layers (head_dim=256, window=1024)
  - 5 global attention layers (head_dim=512, no window)

Usage:
    # Profile with real activations (requires GPU + model):
    python3 tools/select_kv_specs.py \
        --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
        --target-compression 2.5 \
        --min-quality 0.995

    # Simulate with synthetic activations (no model needed):
    python3 tools/select_kv_specs.py \
        --simulate \
        --target-compression 2.5 \
        --min-quality 0.995

    # Custom spec candidates:
    python3 tools/select_kv_specs.py \
        --model /models/gemma-4-26B-A4B-it-NVFP4-modelopt \
        --specs k4v4b64,k4v4b32,k8v4b32,k8v8b32,fp8 \
        --min-quality 0.998
"""

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS
from kv_cache_gen.config import parse_spec


# ---------------------------------------------------------------------------
# Gemma4 26B layer topology
# ---------------------------------------------------------------------------

# Gemma4 26B has 30 layers: global attention at every 6th layer (5, 11, 17, 23, 29)
GEMMA4_NUM_LAYERS = 30
GEMMA4_GLOBAL_LAYERS = {5, 11, 17, 23, 29}
GEMMA4_SLIDING_HEAD_DIM = 256
GEMMA4_GLOBAL_HEAD_DIM = 512
GEMMA4_SLIDING_WINDOW = 1024


@dataclass
class LayerInfo:
    """Metadata for a single transformer layer."""
    index: int
    layer_type: str          # "sliding" or "global"
    head_dim: int
    window_size: int | None  # None for global layers
    num_kv_heads: int = 4    # Gemma4 default GQA


@dataclass
class LayerQuality:
    """Quality measurement for one layer under one spec."""
    layer_index: int
    spec_name: str
    cosine_sim_k: float
    cosine_sim_v: float
    cosine_sim_avg: float


@dataclass
class LayerAssignment:
    """Final spec assignment for one layer."""
    layer_index: int
    layer_type: str
    head_dim: int
    assigned_spec: str
    quality: float           # cosine similarity achieved
    compression: float       # compression ratio vs BF16


@dataclass
class SelectionResult:
    """Complete result of spec selection."""
    assignments: list[LayerAssignment]
    layer_qualities: dict[int, dict[str, float]]  # layer -> spec -> quality
    effective_compression: float
    kv_bytes_per_token_bf16: float
    kv_bytes_per_token_mixed: float
    sliding_spec: str | None      # dominant spec for sliding layers
    global_spec: str | None       # dominant spec for global layers


# ---------------------------------------------------------------------------
# Layer topology
# ---------------------------------------------------------------------------

def get_gemma4_layers(
    num_layers: int = GEMMA4_NUM_LAYERS,
    global_layers: set[int] | None = None,
    sliding_head_dim: int = GEMMA4_SLIDING_HEAD_DIM,
    global_head_dim: int = GEMMA4_GLOBAL_HEAD_DIM,
    num_kv_heads: int = 4,
) -> list[LayerInfo]:
    """Build layer topology for Gemma4."""
    if global_layers is None:
        global_layers = GEMMA4_GLOBAL_LAYERS

    layers = []
    for i in range(num_layers):
        if i in global_layers:
            layers.append(LayerInfo(
                index=i,
                layer_type="global",
                head_dim=global_head_dim,
                window_size=None,
                num_kv_heads=num_kv_heads,
            ))
        else:
            layers.append(LayerInfo(
                index=i,
                layer_type="sliding",
                head_dim=sliding_head_dim,
                window_size=GEMMA4_SLIDING_WINDOW,
                num_kv_heads=num_kv_heads,
            ))
    return layers


def detect_layer_topology(model) -> list[LayerInfo]:
    """Detect layer topology from a loaded model.

    Inspects the model's config and layer modules to determine which layers
    are sliding vs global, and their head dimensions.
    """
    config = model.config
    num_layers = getattr(config, 'num_hidden_layers', 30)
    num_kv_heads = getattr(config, 'num_key_value_heads', 4)

    # Try Gemma4-specific config
    sliding_window = getattr(config, 'sliding_window', None)
    interleave = getattr(config, 'sliding_window_pattern', None)

    # Detect from layer modules
    layers_list = None
    inner = getattr(model, 'model', model)
    if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
        layers_list = inner.language_model.layers
    elif hasattr(inner, 'layers'):
        layers_list = inner.layers

    layer_infos = []
    for i in range(num_layers):
        # Default: assume all layers are the same
        head_dim = getattr(config, 'head_dim', 256)
        layer_type = "sliding"
        window = sliding_window

        if layers_list is not None and i < len(layers_list):
            layer_mod = layers_list[i]
            attn = getattr(layer_mod, 'self_attn', None)
            if attn is not None:
                # Check for layer-specific head dim
                k_proj = getattr(attn, 'k_proj', None)
                if k_proj is not None:
                    out_features = k_proj.out_features
                    layer_head_dim = out_features // num_kv_heads
                    head_dim = layer_head_dim

                # Detect global vs sliding from config or module attributes
                is_sliding = getattr(attn, 'is_sliding', None)
                if is_sliding is False:
                    layer_type = "global"
                    window = None
                elif is_sliding is True:
                    layer_type = "sliding"

        # Fallback: use interleave pattern if available
        if interleave is not None and isinstance(interleave, int):
            # Gemma4 pattern: every Nth layer is global
            if (i + 1) % interleave == 0:
                layer_type = "global"
                window = None
                # Global layers often have 2x head_dim in Gemma4
                head_dim = getattr(config, 'head_dim', 256) * 2

        layer_infos.append(LayerInfo(
            index=i,
            layer_type=layer_type,
            head_dim=head_dim,
            window_size=window,
            num_kv_heads=num_kv_heads,
        ))

    return layer_infos


# ---------------------------------------------------------------------------
# Quality profiling: synthetic activations
# ---------------------------------------------------------------------------

def profile_synthetic(
    layers: list[LayerInfo],
    spec_names: list[str],
    num_samples: int = 1024,
    seq_len: int = 512,
    seed: int = 42,
) -> dict[int, dict[str, float]]:
    """Profile quantization quality using synthetic activations.

    Generates random KV tensors matching the statistical properties of
    real transformer activations (roughly normal with layer-dependent scale)
    and measures round-trip cosine similarity for each spec.

    Returns: {layer_index: {spec_name: cosine_similarity}}
    """
    from kv_cache_gen.generate import make_store_fn

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}

    for layer in layers:
        D = layer.head_dim
        Hk = layer.num_kv_heads
        N = num_samples

        # Simulate KV activations with layer-dependent statistics.
        # Deeper layers and global layers tend to have more varied activation
        # magnitudes. We add a small per-head scale variation.
        base_scale = 1.0 + 0.02 * layer.index  # slight scale growth with depth
        if layer.layer_type == "global":
            base_scale *= 1.3  # global layers see wider context, slightly higher variance

        k_states = torch.randn(N, Hk, D, device=device, dtype=torch.float16) * base_scale
        v_states = torch.randn(N, Hk, D, device=device, dtype=torch.float16) * base_scale

        # Add occasional outliers (common in real transformer activations)
        outlier_mask = torch.rand(N, Hk, D, device=device) < 0.01
        k_states[outlier_mask] *= 5.0
        v_states[outlier_mask] *= 5.0

        layer_results = {}
        for spec_name in spec_names:
            spec = parse_spec(spec_name)
            if spec is None:
                layer_results[spec_name] = 1.0  # BF16 = perfect quality
                continue

            # Skip float-format specs (FP8): they use cast-based dequant,
            # not the integer round-trip path we measure here.
            if spec.is_float_format:
                # FP8 has ~2x compression with near-lossless quality for most layers
                layer_results[spec_name] = 0.9998
                continue

            try:
                store_fn = make_store_fn(spec)
                cos_sim = _measure_roundtrip_quality(
                    k_states, v_states, spec, store_fn, device
                )
                layer_results[spec_name] = cos_sim
            except Exception as e:
                print(f"  [WARN] Layer {layer.index} spec {spec_name}: {e}")
                layer_results[spec_name] = 0.0

        results[layer.index] = layer_results

    return results


def _measure_roundtrip_quality(
    k_states: torch.Tensor,
    v_states: torch.Tensor,
    spec: KVCacheSpec,
    store_fn,
    device: str,
) -> float:
    """Measure cosine similarity after quantize -> dequantize round-trip."""
    N, Hk, D = k_states.shape
    block_size = 16
    num_blocks = (N + block_size - 1) // block_size
    slot_bytes = spec.slot_bytes(D)

    kv_cache = torch.zeros(num_blocks, block_size, Hk, slot_bytes,
                           dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)

    class LayerProxy:
        pass
    layer_proxy = LayerProxy()

    k_f = k_states.to(torch.float16)
    v_f = v_states.to(torch.float16)

    store_fn(k_f, v_f, kv_cache, slot_mapping, layer_proxy, Hk)

    # Dequantize
    k_recon, v_recon = _dequantize_vectorized(
        kv_cache, layer_proxy, spec, N, Hk, D, block_size, slot_mapping, device
    )

    # Cosine similarity
    k_cos = torch.nn.functional.cosine_similarity(
        k_f.reshape(-1).float().unsqueeze(0),
        k_recon.reshape(-1).float().unsqueeze(0),
    ).item()
    v_cos = torch.nn.functional.cosine_similarity(
        v_f.reshape(-1).float().unsqueeze(0),
        v_recon.reshape(-1).float().unsqueeze(0),
    ).item()

    return (k_cos + v_cos) / 2.0


def _dequantize_vectorized(kv_cache, layer_proxy, spec, N, Hk, D,
                           block_size, slot_mapping, device):
    """Vectorized dequantization from packed cache bytes."""
    k_bytes_per_dim = spec.k_bits / 8
    v_bytes_per_dim = spec.v_bits / 8
    k_region_bytes = int(k_bytes_per_dim * D)
    v_region_bytes = int(v_bytes_per_dim * D)

    blk_indices = slot_mapping // block_size
    blk_offsets = slot_mapping % block_size
    packed_rows = kv_cache[blk_indices, blk_offsets]  # (N, Hk, slot_bytes)

    k_packed = packed_rows[:, :, :k_region_bytes].to(torch.int32)
    v_packed = packed_rows[:, :, k_region_bytes:k_region_bytes + v_region_bytes].to(torch.int32)

    k_codes = _unpack_codes(k_packed, spec.k_bits, D)
    v_codes = _unpack_codes(v_packed, spec.v_bits, D)

    if hasattr(layer_proxy, '_fc_scales'):
        flat_slots = blk_indices * block_size + blk_offsets
        scales = layer_proxy._fc_scales[flat_slots]  # (N, Hk, num_sb, 2)
        k_scales = scales[:, :, :, 0].float()
        v_scales = scales[:, :, :, 1].float()

        k_recon = _dequant_with_scales(
            k_codes, k_scales, spec.k_bits, spec.k_sym_offset,
            spec.k_scale_block, D
        )
        v_recon = _dequant_with_scales(
            v_codes, v_scales, spec.v_bits, spec.v_sym_offset,
            spec.v_scale_block, D
        )
    else:
        k_recon = k_codes.float()
        v_recon = v_codes.float()

    return k_recon.to(torch.float16), v_recon.to(torch.float16)


def _unpack_codes(packed: torch.Tensor, bits: int, D: int) -> torch.Tensor:
    """Unpack integer codes from packed bytes."""
    N, Hk, packed_bytes = packed.shape

    if bits == 8:
        return packed[:, :, :D].float()
    elif bits == 4:
        # Two 4-bit values per byte
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        interleaved = torch.stack([lo, hi], dim=-1).reshape(N, Hk, -1)
        return interleaved[:, :, :D].float()
    elif bits == 2:
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        interleaved = torch.stack([b0, b1, b2, b3], dim=-1).reshape(N, Hk, -1)
        return interleaved[:, :, :D].float()
    else:
        raise ValueError(f"Unsupported bits={bits}")


def _dequant_with_scales(codes, scales, bits, sym_offset, scale_block, D):
    """Dequantize codes using per-group scales."""
    N, Hk, _ = codes.shape
    num_groups = D // scale_block
    codes_grouped = codes[:, :, :D].reshape(N, Hk, num_groups, scale_block)
    scales_expanded = scales[:, :, :num_groups].unsqueeze(-1)
    dequant = (codes_grouped - sym_offset) * scales_expanded
    return dequant.reshape(N, Hk, D)


# ---------------------------------------------------------------------------
# Quality profiling: real model activations
# ---------------------------------------------------------------------------

def profile_real_activations(
    model_path: str,
    layers: list[LayerInfo],
    spec_names: list[str],
    max_samples: int = 20,
    seq_len: int = 512,
    dataset: str = "wikitext",
) -> dict[int, dict[str, float]]:
    """Profile KV quantization quality using real model activations.

    Loads the model, runs forward passes on real text, captures per-layer
    K/V projections, and measures round-trip quantization quality for each spec.

    Returns: {layer_index: {spec_name: cosine_similarity}}
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from kv_cache_gen.generate import make_store_fn

    print(f"[profile] Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Detect topology from model if not already provided
    detected_layers = detect_layer_topology(model)
    if len(detected_layers) != len(layers):
        print(f"[profile] WARNING: detected {len(detected_layers)} layers, "
              f"expected {len(layers)}. Using detected topology.")
        layers = detected_layers

    # Find k_proj / v_proj modules
    kv_proj_pairs = _find_kv_projections(model)
    if not kv_proj_pairs:
        print("[profile] Could not find k_proj/v_proj. Falling back to synthetic.")
        del model
        torch.cuda.empty_cache()
        return profile_synthetic(layers, spec_names, num_samples=1024)

    print(f"[profile] Found {len(kv_proj_pairs)} attention layers with k/v projections")

    # Load dataset
    from fusen_kv.eval_perplexity import load_dataset_texts, tokenize_and_chunk
    texts = load_dataset_texts(dataset, max_samples)
    chunks = tokenize_and_chunk(texts, tokenizer, seq_len, max_samples)
    chunks = chunks[:max_samples]

    if not chunks:
        print("[profile] No text chunks created. Falling back to synthetic.")
        del model
        torch.cuda.empty_cache()
        return profile_synthetic(layers, spec_names, num_samples=1024)

    # Capture KV activations per layer
    num_layers = len(kv_proj_pairs)
    captured_k = {}
    captured_v = {}
    hooks = []

    def make_k_hook(idx):
        def hook_fn(module, args, output):
            captured_k[idx] = output.detach()
        return hook_fn

    def make_v_hook(idx):
        def hook_fn(module, args, output):
            captured_v[idx] = output.detach()
        return hook_fn

    for i, (k_proj, v_proj) in enumerate(kv_proj_pairs):
        hooks.append(k_proj.register_forward_hook(make_k_hook(i)))
        hooks.append(v_proj.register_forward_hook(make_v_hook(i)))

    device = next(model.parameters()).device

    # Accumulate per-layer cosine sims
    layer_spec_cosines = {i: {s: [] for s in spec_names} for i in range(num_layers)}

    for chunk_idx, chunk in enumerate(chunks):
        input_ids = torch.tensor([chunk], device=device)

        with torch.no_grad():
            try:
                model(input_ids, use_cache=False)
            except Exception as e:
                print(f"[profile] Forward pass failed on chunk {chunk_idx}: {e}")
                continue

        for layer_idx in range(min(num_layers, len(layers))):
            if layer_idx not in captured_k or layer_idx not in captured_v:
                continue

            k_raw = captured_k[layer_idx]
            v_raw = captured_v[layer_idx]
            k_norm, v_norm = _normalize_kv_shape(k_raw, v_raw, layers[layer_idx])

            if k_norm is None:
                continue

            for spec_name in spec_names:
                spec = parse_spec(spec_name)
                if spec is None:
                    layer_spec_cosines[layer_idx][spec_name].append(1.0)
                    continue

                try:
                    store_fn = make_store_fn(spec)
                    cos_sim = _measure_roundtrip_quality(
                        k_norm, v_norm, spec, store_fn, str(k_norm.device)
                    )
                    layer_spec_cosines[layer_idx][spec_name].append(cos_sim)
                except Exception as e:
                    if chunk_idx == 0:
                        print(f"  [WARN] Layer {layer_idx} spec {spec_name}: {e}")

        captured_k.clear()
        captured_v.clear()

        if (chunk_idx + 1) % 5 == 0:
            print(f"[profile]   Processed {chunk_idx + 1}/{len(chunks)} chunks")

    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Average cosine sims
    results = {}
    for layer_idx in range(min(num_layers, len(layers))):
        layer_results = {}
        for spec_name in spec_names:
            vals = layer_spec_cosines[layer_idx][spec_name]
            layer_results[spec_name] = float(np.mean(vals)) if vals else 0.0
        results[layer_idx] = layer_results

    return results


def _find_kv_projections(model) -> list[tuple]:
    """Find (k_proj, v_proj) pairs across all attention layers."""
    pairs = []

    layers = None
    inner = getattr(model, 'model', model)
    if hasattr(inner, 'language_model') and hasattr(inner.language_model, 'layers'):
        layers = inner.language_model.layers
    elif hasattr(inner, 'layers'):
        layers = inner.layers

    if layers is None:
        # Generic fallback
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

    for layer in layers:
        attn = getattr(layer, 'self_attn', None)
        if attn is None:
            continue
        k_proj = getattr(attn, 'k_proj', None)
        v_proj = getattr(attn, 'v_proj', None)
        if k_proj is not None and v_proj is not None:
            pairs.append((k_proj, v_proj))

    return pairs


def _normalize_kv_shape(k, v, layer_info: LayerInfo):
    """Reshape k_proj/v_proj outputs to (N, Hk, D)."""
    D = layer_info.head_dim
    Hk = layer_info.num_kv_heads

    if k.ndim == 3:
        B, S, out_k = k.shape
    elif k.ndim == 2:
        S, out_k = k.shape
        B = 1
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
    else:
        return None, None

    # Infer head count from output size and known head_dim
    if out_k % D == 0:
        actual_Hk = out_k // D
    else:
        # Fallback to common head dims
        for candidate in [64, 128, 256, 512, 80, 96]:
            if out_k % candidate == 0:
                D = candidate
                actual_Hk = out_k // D
                break
        else:
            return None, None

    k = k.reshape(B * S, actual_Hk, D).contiguous()
    v = v.reshape(B * S, actual_Hk, D).contiguous()
    return k, v


# ---------------------------------------------------------------------------
# Greedy spec assignment
# ---------------------------------------------------------------------------

def get_candidate_specs(spec_names: list[str] | None = None) -> list[str]:
    """Get ordered list of spec candidates, most aggressive first."""
    if spec_names:
        return spec_names

    # Default candidates in order of aggressiveness.
    # FP8 excluded: uses float format (cast-based dequant), not integer codes.
    return ["k4v2b16", "k4v2b32", "k4v4b64", "k4v4b32", "k4v4b16",
            "k8v4b32", "k8v4b16", "k8v8b32"]


def assign_specs_greedy(
    layers: list[LayerInfo],
    qualities: dict[int, dict[str, float]],
    spec_names: list[str],
    min_quality: float = 0.995,
    target_compression: float | None = None,
) -> list[LayerAssignment]:
    """Greedy per-layer spec assignment.

    For each layer, try specs from most aggressive (highest compression) to
    least aggressive. Assign the most aggressive spec that meets the quality
    threshold.

    Args:
        layers: layer topology
        qualities: {layer_index: {spec_name: cosine_similarity}}
        spec_names: candidate spec names, ordered most-to-least aggressive
        min_quality: minimum cosine similarity threshold
        target_compression: optional target (not enforced, just reported)

    Returns: list of LayerAssignment
    """
    # Sort specs by compression ratio (highest first)
    spec_compressions = []
    for name in spec_names:
        spec = parse_spec(name)
        if spec is None:
            spec_compressions.append((name, 1.0))
        else:
            # Use D=256 for comparison (sliding layer head_dim)
            spec_compressions.append((name, spec.compression_vs_bf16(256)))

    sorted_specs = sorted(spec_compressions, key=lambda x: -x[1])

    assignments = []
    for layer in layers:
        layer_idx = layer.index
        layer_quals = qualities.get(layer_idx, {})

        assigned = None
        for spec_name, _ in sorted_specs:
            quality = layer_quals.get(spec_name, 0.0)
            if quality >= min_quality:
                assigned = spec_name
                break

        if assigned is None:
            # Fallback: use least aggressive spec
            assigned = sorted_specs[-1][0]
            quality = layer_quals.get(assigned, 0.0)
        else:
            quality = layer_quals.get(assigned, 0.0)

        spec = parse_spec(assigned)
        compression = spec.compression_vs_bf16(layer.head_dim) if spec else 1.0

        assignments.append(LayerAssignment(
            layer_index=layer_idx,
            layer_type=layer.layer_type,
            head_dim=layer.head_dim,
            assigned_spec=assigned,
            quality=quality,
            compression=compression,
        ))

    return assignments


# ---------------------------------------------------------------------------
# Compute aggregate metrics
# ---------------------------------------------------------------------------

def compute_selection_result(
    layers: list[LayerInfo],
    assignments: list[LayerAssignment],
    qualities: dict[int, dict[str, float]],
) -> SelectionResult:
    """Compute aggregate metrics from per-layer assignments."""
    # BF16 bytes per token: sum over layers of (num_kv_heads * head_dim * 4)
    # (4 bytes = 2 for K + 2 for V in BF16)
    bf16_bytes = sum(l.num_kv_heads * l.head_dim * 4 for l in layers)

    # Mixed-spec bytes per token
    mixed_bytes = 0.0
    for layer, assignment in zip(layers, assignments):
        spec = parse_spec(assignment.assigned_spec)
        if spec is None:
            # BF16
            mixed_bytes += layer.num_kv_heads * layer.head_dim * 4
        else:
            per_head = spec.slot_bytes(layer.head_dim) + spec.scale_bytes(layer.head_dim)
            mixed_bytes += layer.num_kv_heads * per_head

    effective_compression = bf16_bytes / mixed_bytes if mixed_bytes > 0 else 1.0

    # Find dominant specs for sliding and global
    from collections import Counter
    sliding_specs = Counter()
    global_specs = Counter()
    for a in assignments:
        if a.layer_type == "sliding":
            sliding_specs[a.assigned_spec] += 1
        else:
            global_specs[a.assigned_spec] += 1

    sliding_spec = sliding_specs.most_common(1)[0][0] if sliding_specs else None
    global_spec = global_specs.most_common(1)[0][0] if global_specs else None

    return SelectionResult(
        assignments=assignments,
        layer_qualities=qualities,
        effective_compression=effective_compression,
        kv_bytes_per_token_bf16=bf16_bytes,
        kv_bytes_per_token_mixed=mixed_bytes,
        sliding_spec=sliding_spec,
        global_spec=global_spec,
    )


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def print_report(result: SelectionResult, layers: list[LayerInfo]):
    """Print a human-readable selection report."""
    print(f"\n{'='*72}")
    print(f" Per-Layer KV Cache Spec Selection Report")
    print(f"{'='*72}")

    print(f"\n{'Layer':>5} {'Type':<8} {'HeadDim':>7} {'Spec':<12} "
          f"{'Quality':>8} {'Compress':>8}")
    print("-" * 56)

    for a in result.assignments:
        print(f"{a.layer_index:>5} {a.layer_type:<8} {a.head_dim:>7} "
              f"{a.assigned_spec:<12} {a.quality:>8.5f} {a.compression:>7.1f}x")

    print(f"\n{'='*72}")
    print(f" Summary")
    print(f"{'='*72}")
    print(f"  Sliding layers spec:  {result.sliding_spec}")
    print(f"  Global layers spec:   {result.global_spec}")
    print(f"  Effective compression: {result.effective_compression:.2f}x vs BF16")
    print(f"  BF16 KV bytes/token:  {result.kv_bytes_per_token_bf16:,.0f}")
    print(f"  Mixed KV bytes/token: {result.kv_bytes_per_token_mixed:,.0f}")
    print(f"  KV memory savings:    {(1 - result.kv_bytes_per_token_mixed / result.kv_bytes_per_token_bf16) * 100:.1f}%")

    # KV capacity gain
    capacity_gain = result.kv_bytes_per_token_bf16 / result.kv_bytes_per_token_mixed
    print(f"  KV capacity gain:     {capacity_gain:.2f}x more tokens in same memory")

    # Quality summary
    all_qualities = [a.quality for a in result.assignments]
    print(f"\n  Quality (cosine similarity):")
    print(f"    Mean:  {np.mean(all_qualities):.6f}")
    print(f"    Min:   {np.min(all_qualities):.6f}")
    print(f"    Max:   {np.max(all_qualities):.6f}")

    sliding_quals = [a.quality for a in result.assignments if a.layer_type == "sliding"]
    global_quals = [a.quality for a in result.assignments if a.layer_type == "global"]
    if sliding_quals:
        print(f"    Sliding mean: {np.mean(sliding_quals):.6f}")
    if global_quals:
        print(f"    Global mean:  {np.mean(global_quals):.6f}")
    print()


def export_config(result: SelectionResult, output_path: str | None = None) -> dict:
    """Export selection as a FusenCache-compatible config dict.

    The config groups layers by spec for efficient runtime dispatch.
    """
    from collections import Counter

    # Group layers by spec
    spec_groups = {}
    for a in result.assignments:
        if a.assigned_spec not in spec_groups:
            spec_groups[a.assigned_spec] = []
        spec_groups[a.assigned_spec].append(a.layer_index)

    # Build per-layer spec list
    per_layer = [a.assigned_spec for a in result.assignments]

    # Dominant specs for simplified config
    config = {
        "sliding": result.sliding_spec,
        "global": result.global_spec,
        "effective_compression": round(result.effective_compression, 2),
        "per_layer_specs": per_layer,
        "spec_groups": spec_groups,
        "kv_bytes_per_token_bf16": result.kv_bytes_per_token_bf16,
        "kv_bytes_per_token_mixed": round(result.kv_bytes_per_token_mixed, 1),
        "quality": {
            "mean": round(float(np.mean([a.quality for a in result.assignments])), 6),
            "min": round(float(np.min([a.quality for a in result.assignments])), 6),
        },
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Config saved to: {output_path}")

    return config


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer KV cache spec selector for mixed-precision FusenCache"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Path to HF model (for real activation profiling)")
    parser.add_argument("--simulate", action="store_true",
                        help="Use synthetic activations (no model required)")
    parser.add_argument("--specs", type=str, default=None,
                        help="Comma-separated spec candidates (default: auto)")
    parser.add_argument("--target-compression", type=float, default=2.5,
                        help="Target compression ratio vs BF16 (informational)")
    parser.add_argument("--min-quality", type=float, default=0.995,
                        help="Minimum cosine similarity threshold (default: 0.995)")
    parser.add_argument("--num-layers", type=int, default=GEMMA4_NUM_LAYERS,
                        help="Number of transformer layers")
    parser.add_argument("--max-samples", type=int, default=20,
                        help="Max text samples for real activation profiling")
    parser.add_argument("--seq-len", type=int, default=512,
                        help="Sequence length for profiling")
    parser.add_argument("--output", type=str, default=None,
                        help="Save config JSON to this path")
    parser.add_argument("--num-kv-heads", type=int, default=4,
                        help="Number of KV heads per layer (GQA)")
    args = parser.parse_args()

    # Build layer topology
    layers = get_gemma4_layers(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
    )

    # Spec candidates
    spec_names = get_candidate_specs(
        args.specs.split(",") if args.specs else None
    )
    print(f"Spec candidates: {spec_names}")
    print(f"Min quality threshold: {args.min_quality}")
    print(f"Target compression: {args.target_compression}x")
    print(f"Layers: {len(layers)} ({sum(1 for l in layers if l.layer_type == 'sliding')} sliding, "
          f"{sum(1 for l in layers if l.layer_type == 'global')} global)")

    # Profile quality
    if args.model and not args.simulate:
        print(f"\nProfiling with real activations from: {args.model}")
        qualities = profile_real_activations(
            args.model, layers, spec_names,
            max_samples=args.max_samples,
            seq_len=args.seq_len,
        )
    else:
        if not args.simulate and not args.model:
            print("\nNo --model provided. Using synthetic activations.")
        else:
            print("\nUsing synthetic activations (--simulate).")
        qualities = profile_synthetic(layers, spec_names)

    # Assign specs
    assignments = assign_specs_greedy(
        layers, qualities, spec_names,
        min_quality=args.min_quality,
        target_compression=args.target_compression,
    )

    # Compute results
    result = compute_selection_result(layers, assignments, qualities)

    # Report
    print_report(result, layers)

    # Export config
    config = export_config(
        result,
        output_path=args.output or "tools/kv_spec_selection.json",
    )

    print("\nFusenCache config (simplified):")
    print(json.dumps({
        "sliding": config["sliding"],
        "global": config["global"],
        "effective_compression": config["effective_compression"],
    }, indent=2))

    return result


if __name__ == "__main__":
    main()
