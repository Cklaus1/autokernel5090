#!/usr/bin/env python3
"""
Two-Tier "Fast Brain / Slow Brain" Inference Architecture.

Model-level MoE: instead of discarding pruned components forever, keep them
accessible on CPU as a fallback tier.  The fast brain (pruned model on GPU)
handles ~95% of tokens at full speed.  When it detects uncertainty, the slow
brain (pruned experts/layers on CPU) is loaded on-demand.

Architecture:
    Tier 1  Detection    - Router confidence, residual norm, output entropy
    Tier 2  Slow Brain   - CPU-resident pruned experts + layers, GPU LRU cache
    Tier 3  Integration  - TwoTierModel orchestrates fast/slow paths

Usage:
    from tools.two_tier_brain import TwoTierModel, SlowBrainExperts, SlowBrainLayers

    model = TwoTierModel.from_checkpoints(
        fast_model_dir="/root/models/gemma4-pruned-30pct/",
        slow_expert_dir="/root/models/gemma4-slow-experts-30pct/",
        slow_layer_dir="/root/models/gemma4-slow-layers/",
    )
    # During inference:
    output = model.decode_step(hidden_states, layer_idx=0)
"""

from __future__ import annotations

import json
import os
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

try:
    from safetensors import safe_open
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TwoTierConfig:
    """All tunable knobs for the two-tier system."""

    # Detection thresholds
    router_confidence_threshold: float = 0.15   # min router weight to trigger slow expert
    residual_norm_threshold: float = 50.0       # hidden norm above this triggers slow layer
    entropy_threshold: float = 3.5              # output entropy above this triggers slow brain
    entropy_enabled: bool = True                # enable output-entropy detection

    # GPU LRU cache for slow experts
    gpu_cache_max_experts: int = 64             # max pruned experts cached on GPU
    gpu_cache_max_bytes: int = 256 * 1024**2    # 256 MB budget for slow expert GPU cache

    # Async prefetch
    prefetch_enabled: bool = True               # start CPU->GPU transfer early
    prefetch_queue_depth: int = 4               # how many experts to prefetch ahead

    # Graceful degradation
    max_slow_latency_ms: float = 2.0            # if slow path > this, skip it
    skip_slow_on_timeout: bool = True           # skip rather than block

    # Batch splitting
    batch_split_enabled: bool = True            # split batch into fast/slow subsets

    # Model constants (Gemma 4 26B defaults)
    num_original_layers: int = 34               # original model layer count
    num_original_experts: int = 128             # original expert count per MoE layer
    top_k: int = 8                              # router top-k

    # Profiling
    profile: bool = False                       # collect hit-rate statistics


# ─────────────────────────────────────────────────────────────────────────────
# Statistics collector
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TwoTierStats:
    """Runtime statistics for monitoring slow brain usage."""

    total_tokens: int = 0
    slow_expert_hits: int = 0       # tokens that needed >= 1 slow expert
    slow_layer_hits: int = 0        # tokens that needed a slow layer
    slow_expert_cache_hits: int = 0 # slow expert was already on GPU
    slow_expert_cache_misses: int = 0
    slow_expert_skipped: int = 0    # skipped due to timeout
    total_slow_latency_ms: float = 0.0
    total_fast_latency_ms: float = 0.0

    # Per-expert hit counts: {(layer, expert_id): count}
    expert_hit_counts: dict = field(default_factory=dict)

    @property
    def slow_hit_rate(self) -> float:
        if self.total_tokens == 0:
            return 0.0
        return (self.slow_expert_hits + self.slow_layer_hits) / self.total_tokens

    @property
    def cache_hit_rate(self) -> float:
        total = self.slow_expert_cache_hits + self.slow_expert_cache_misses
        if total == 0:
            return 0.0
        return self.slow_expert_cache_hits / total

    @property
    def avg_slow_latency_ms(self) -> float:
        total = self.slow_expert_hits + self.slow_layer_hits
        if total == 0:
            return 0.0
        return self.total_slow_latency_ms / total

    def summary(self) -> str:
        lines = [
            "=== Two-Tier Brain Statistics ===",
            f"Total tokens processed:    {self.total_tokens}",
            f"Slow expert hit rate:      {self.slow_hit_rate:.2%}",
            f"Slow layer hit rate:       {self.slow_layer_hits}/{self.total_tokens}",
            f"GPU cache hit rate:        {self.cache_hit_rate:.2%}",
            f"Slow experts skipped:      {self.slow_expert_skipped}",
            f"Avg slow path latency:     {self.avg_slow_latency_ms:.2f} ms",
            f"Total fast latency:        {self.total_fast_latency_ms:.1f} ms",
            f"Total slow latency:        {self.total_slow_latency_ms:.1f} ms",
        ]
        if self.expert_hit_counts:
            top_experts = sorted(self.expert_hit_counts.items(), key=lambda x: -x[1])[:10]
            lines.append("\nTop 10 most-requested slow experts:")
            for (layer, eid), count in top_experts:
                lines.append(f"  Layer {layer:>2} Expert {eid:>3}: {count} hits")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# GPU LRU Cache for Slow Experts
# ─────────────────────────────────────────────────────────────────────────────

class GPUExpertCache:
    """LRU cache that keeps frequently-used slow experts on GPU.

    Keys are (layer_idx, expert_id) tuples.
    Values are dicts of {proj_name: tensor} on GPU.
    """

    def __init__(self, max_experts: int = 64, max_bytes: int = 256 * 1024**2):
        self.max_experts = max_experts
        self.max_bytes = max_bytes
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self._sizes: dict[tuple[int, int], int] = {}
        self._total_bytes = 0
        self._lock = threading.Lock()

    def get(self, layer_idx: int, expert_id: int) -> Optional[dict[str, torch.Tensor]]:
        key = (layer_idx, expert_id)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, layer_idx: int, expert_id: int, weights: dict[str, torch.Tensor]):
        key = (layer_idx, expert_id)
        size = sum(t.nbytes for t in weights.values())

        with self._lock:
            # Evict until we have room
            while (len(self._cache) >= self.max_experts or
                   self._total_bytes + size > self.max_bytes) and self._cache:
                evict_key, evict_val = self._cache.popitem(last=False)
                evict_size = self._sizes.pop(evict_key, 0)
                self._total_bytes -= evict_size
                # Free GPU memory
                for t in evict_val.values():
                    del t

            self._cache[key] = weights
            self._sizes[key] = size
            self._total_bytes += size

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return (layer_idx, expert_id) in self._cache

    @property
    def num_cached(self) -> int:
        return len(self._cache)

    @property
    def memory_used_mb(self) -> float:
        return self._total_bytes / 1024**2

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._total_bytes = 0


# ─────────────────────────────────────────────────────────────────────────────
# Slow Brain: Pruned Experts (CPU-resident)
# ─────────────────────────────────────────────────────────────────────────────

class SlowBrainExperts:
    """Manages pruned expert weights on CPU, with GPU LRU caching.

    Expects a directory structure:
        slow_expert_dir/
            manifest.json          - maps (layer, original_expert_id) -> file locations
            layer_{L}/
                expert_{E}/
                    gate_proj.weight.pt
                    gate_proj.weight_scale.pt
                    up_proj.weight.pt
                    up_proj.weight_scale.pt
                    down_proj.weight.pt
                    down_proj.weight_scale.pt
                    input_scale.pt   (per-projection)

    Or a single safetensors file with a manifest.
    """

    PROJ_NAMES = ["gate_proj", "up_proj", "down_proj"]
    SUFFIXES = ["weight", "weight_scale"]

    def __init__(
        self,
        slow_dir: str,
        config: TwoTierConfig,
    ):
        self.slow_dir = Path(slow_dir)
        self.config = config
        self.gpu_cache = GPUExpertCache(
            max_experts=config.gpu_cache_max_experts,
            max_bytes=config.gpu_cache_max_bytes,
        )

        # Load manifest
        manifest_path = self.slow_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                raw = json.load(f)
            # Convert string keys to (int, int) from the "experts" sub-dict
            self.manifest: dict[tuple[int, int], dict] = {}
            experts_raw = raw.get("experts", {})
            for k, v in experts_raw.items():
                layer, expert = map(int, k.split(","))
                self.manifest[(layer, expert)] = v
        else:
            self.manifest = {}

        # Pre-load CPU tensor index from safetensors if available
        self._cpu_tensors: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        self._loaded = False

    def load_all_to_cpu(self):
        """Load all pruned expert weights to CPU RAM.

        For Gemma 4 26B with 30% pruning: ~38 experts * 30 layers * ~3MB = ~3.4 GB CPU RAM.
        """
        if self._loaded:
            return

        st_path = self.slow_dir / "slow_experts.safetensors"
        if st_path.exists() and HAS_SAFETENSORS:
            self._load_from_safetensors(st_path)
        else:
            self._load_from_directory()

        self._loaded = True
        total_mb = sum(
            sum(t.nbytes for t in weights.values())
            for weights in self._cpu_tensors.values()
        ) / 1024**2
        print(f"[SlowBrainExperts] Loaded {len(self._cpu_tensors)} experts to CPU ({total_mb:.1f} MB)")

    def _load_from_safetensors(self, path: Path):
        """Load from a single safetensors file."""
        f = safe_open(str(path), framework="pt", device="cpu")
        for key in f.keys():
            # Format: layer_{L}.expert_{E}.{proj}.{suffix}
            parts = key.split(".")
            layer = int(parts[0].split("_")[1])
            expert = int(parts[1].split("_")[1])
            proj = parts[2]
            suffix = parts[3]

            k = (layer, expert)
            if k not in self._cpu_tensors:
                self._cpu_tensors[k] = {}
            self._cpu_tensors[k][f"{proj}.{suffix}"] = f.get_tensor(key)

    def _load_from_directory(self):
        """Load from directory of .pt files."""
        for layer_dir in sorted(self.slow_dir.glob("layer_*")):
            layer = int(layer_dir.name.split("_")[1])
            for expert_dir in sorted(layer_dir.glob("expert_*")):
                expert = int(expert_dir.name.split("_")[1])
                k = (layer, expert)
                self._cpu_tensors[k] = {}
                for pt_file in expert_dir.glob("*.pt"):
                    name = pt_file.stem  # e.g. "gate_proj.weight"
                    self._cpu_tensors[k][name] = torch.load(pt_file, map_location="cpu",
                                                             weights_only=True)

    def has_expert(self, layer_idx: int, expert_id: int) -> bool:
        """Check if this expert exists in the slow brain."""
        return (layer_idx, expert_id) in self._cpu_tensors or \
               (layer_idx, expert_id) in self.manifest

    def get_expert_gpu(
        self,
        layer_idx: int,
        expert_id: int,
        device: torch.device = torch.device("cuda"),
    ) -> dict[str, torch.Tensor]:
        """Get expert weights on GPU, using cache or transferring from CPU.

        Returns dict like:
            {"gate_proj.weight": Tensor, "gate_proj.weight_scale": Tensor, ...}
        """
        # Check GPU cache first
        cached = self.gpu_cache.get(layer_idx, expert_id)
        if cached is not None:
            return cached

        # Transfer from CPU to GPU
        cpu_weights = self._cpu_tensors.get((layer_idx, expert_id))
        if cpu_weights is None:
            raise KeyError(f"Expert ({layer_idx}, {expert_id}) not in slow brain")

        gpu_weights = {k: v.to(device, non_blocking=True) for k, v in cpu_weights.items()}
        # Sync to ensure transfer is complete
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()

        self.gpu_cache.put(layer_idx, expert_id, gpu_weights)
        return gpu_weights

    def prefetch_expert(
        self,
        layer_idx: int,
        expert_id: int,
        device: torch.device = torch.device("cuda"),
        stream: Optional[torch.cuda.Stream] = None,
    ):
        """Async prefetch: start CPU->GPU transfer on a separate CUDA stream."""
        if self.gpu_cache.contains(layer_idx, expert_id):
            return  # Already cached

        cpu_weights = self._cpu_tensors.get((layer_idx, expert_id))
        if cpu_weights is None:
            return

        target_stream = stream or torch.cuda.Stream(device)
        with torch.cuda.stream(target_stream):
            gpu_weights = {k: v.to(device, non_blocking=True) for k, v in cpu_weights.items()}

        # Cache will be populated after stream sync (caller's responsibility)
        # For now, store pending transfer
        self._pending_prefetch = (layer_idx, expert_id, gpu_weights, target_stream)

    def complete_prefetch(self):
        """Complete any pending async prefetch."""
        if hasattr(self, '_pending_prefetch') and self._pending_prefetch is not None:
            layer_idx, expert_id, gpu_weights, stream = self._pending_prefetch
            stream.synchronize()
            self.gpu_cache.put(layer_idx, expert_id, gpu_weights)
            self._pending_prefetch = None


# ─────────────────────────────────────────────────────────────────────────────
# Slow Brain: Pruned Layers (CPU-resident)
# ─────────────────────────────────────────────────────────────────────────────

class SlowBrainLayers:
    """Manages pruned transformer layers on CPU.

    Each pruned layer is stored as a state_dict on CPU and loaded to GPU
    on demand. These are full transformer blocks (attention + MoE/MLP + norms).

    Directory structure:
        slow_layer_dir/
            manifest.json
            layer_{original_idx}.safetensors   (or .pt)
    """

    def __init__(self, slow_dir: str, config: TwoTierConfig):
        self.slow_dir = Path(slow_dir)
        self.config = config
        self._cpu_state_dicts: dict[int, dict[str, torch.Tensor]] = {}
        self._loaded = False

        # Load manifest
        manifest_path = self.slow_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
                self.pruned_layer_ids = set(int(k) for k in self.manifest.get("pruned_layers", []))
        else:
            self.manifest = {}
            self.pruned_layer_ids = set()

    def load_all_to_cpu(self):
        """Load all pruned layer state dicts to CPU RAM."""
        if self._loaded:
            return

        for layer_id in self.pruned_layer_ids:
            st_path = self.slow_dir / f"layer_{layer_id}.safetensors"
            pt_path = self.slow_dir / f"layer_{layer_id}.pt"

            if st_path.exists() and HAS_SAFETENSORS:
                self._cpu_state_dicts[layer_id] = load_file(str(st_path), device="cpu")
            elif pt_path.exists():
                self._cpu_state_dicts[layer_id] = torch.load(pt_path, map_location="cpu",
                                                               weights_only=True)

        self._loaded = True
        total_mb = sum(
            sum(t.nbytes for t in sd.values())
            for sd in self._cpu_state_dicts.values()
        ) / 1024**2
        print(f"[SlowBrainLayers] Loaded {len(self._cpu_state_dicts)} layers to CPU ({total_mb:.1f} MB)")

    def has_layer(self, original_layer_idx: int) -> bool:
        return original_layer_idx in self.pruned_layer_ids

    def get_layer_gpu(
        self,
        original_layer_idx: int,
        device: torch.device = torch.device("cuda"),
    ) -> dict[str, torch.Tensor]:
        """Transfer a pruned layer's state dict to GPU."""
        sd = self._cpu_state_dicts.get(original_layer_idx)
        if sd is None:
            raise KeyError(f"Layer {original_layer_idx} not in slow brain")

        gpu_sd = {k: v.to(device, non_blocking=True) for k, v in sd.items()}
        if device.type == "cuda":
            torch.cuda.current_stream(device).synchronize()
        return gpu_sd

    def free_layer_gpu(self, gpu_state_dict: dict[str, torch.Tensor]):
        """Free GPU memory for a layer after use."""
        for t in gpu_state_dict.values():
            del t
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Detection: When does the fast brain need help?
# ─────────────────────────────────────────────────────────────────────────────

class SlowBrainDetector:
    """Detects when a token needs the slow brain.

    Three signals:
    1. Router confidence - router assigns weight to a pruned expert slot
    2. Residual norm     - hidden state at pruned layer position is unusual
    3. Output entropy    - final logits are uncertain
    """

    def __init__(
        self,
        pruned_expert_ids: dict[int, set[int]],  # {layer: {expert_ids}}
        pruned_layer_ids: set[int],
        config: TwoTierConfig,
    ):
        self.pruned_expert_ids = pruned_expert_ids
        self.pruned_layer_ids = pruned_layer_ids
        self.config = config

        # Per-layer calibrated thresholds (updated during warmup)
        self.residual_norm_thresholds: dict[int, float] = {
            lid: config.residual_norm_threshold for lid in pruned_layer_ids
        }

    def detect_slow_experts(
        self,
        router_logits: torch.Tensor,   # [batch, num_original_experts]
        layer_idx: int,                 # original layer index
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which tokens in the batch need slow experts.

        Returns:
            needs_slow: [batch] bool tensor - True if token needs slow brain
            slow_expert_ids: list of sets - which pruned experts each token needs
        """
        pruned = self.pruned_expert_ids.get(layer_idx, set())
        if not pruned:
            batch = router_logits.shape[0]
            return torch.zeros(batch, dtype=torch.bool, device=router_logits.device), [set()] * batch

        # Apply softmax to get routing probabilities
        router_probs = F.softmax(router_logits, dim=-1)  # [batch, num_experts]
        topk_vals, topk_ids = torch.topk(router_probs, k=self.config.top_k, dim=-1)

        batch = router_logits.shape[0]
        needs_slow = torch.zeros(batch, dtype=torch.bool, device=router_logits.device)
        slow_ids_per_token: list[set[int]] = [set() for _ in range(batch)]

        # Check if any top-k selection includes a pruned expert
        pruned_tensor = torch.tensor(sorted(pruned), device=router_logits.device)
        for b in range(batch):
            for k in range(self.config.top_k):
                eid = topk_ids[b, k].item()
                weight = topk_vals[b, k].item()
                if eid in pruned and weight >= self.config.router_confidence_threshold:
                    needs_slow[b] = True
                    slow_ids_per_token[b].add(eid)

        return needs_slow, slow_ids_per_token

    def detect_slow_layer(
        self,
        hidden_states: torch.Tensor,   # [batch, seq_len, hidden_dim]
        pruned_layer_idx: int,
    ) -> torch.Tensor:
        """Check if hidden states at a pruned layer position are unusual.

        Returns:
            needs_slow: [batch] bool tensor
        """
        threshold = self.residual_norm_thresholds.get(
            pruned_layer_idx, self.config.residual_norm_threshold
        )

        # Compute per-token norm (last dim)
        norms = hidden_states.float().norm(dim=-1).mean(dim=-1)  # [batch]
        return norms > threshold

    def detect_output_entropy(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
    ) -> torch.Tensor:
        """Check if output distribution is too uncertain.

        Returns:
            needs_slow: [batch] bool tensor
        """
        if not self.config.entropy_enabled:
            return torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)

        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)  # [batch]
        return entropy > self.config.entropy_threshold

    def calibrate_thresholds(
        self,
        hidden_states_samples: list[torch.Tensor],
        pruned_layer_idx: int,
        percentile: float = 95.0,
    ):
        """Calibrate residual norm threshold from sample data.

        Run a batch of calibration inputs through the fast brain and collect
        hidden state norms at each pruned layer position.  Set threshold to
        the given percentile so only outliers trigger the slow brain.
        """
        all_norms = []
        for h in hidden_states_samples:
            norms = h.float().norm(dim=-1).mean(dim=-1)
            all_norms.append(norms)

        all_norms = torch.cat(all_norms)
        threshold = torch.quantile(all_norms, percentile / 100.0).item()
        self.residual_norm_thresholds[pruned_layer_idx] = threshold
        return threshold


# ─────────────────────────────────────────────────────────────────────────────
# Expert Forward Pass (MoE GEMM)
# ─────────────────────────────────────────────────────────────────────────────

def expert_forward_fp16(
    hidden: torch.Tensor,                    # [tokens, hidden_dim]
    weights: dict[str, torch.Tensor],        # gate/up/down proj weights on GPU
) -> torch.Tensor:
    """Run a single expert in FP16/BF16 (dequantized path).

    Standard SwiGLU expert: gate * up -> SiLU -> down
    """
    gate = F.linear(hidden, weights["gate_proj.weight"])
    up = F.linear(hidden, weights["up_proj.weight"])
    activated = F.silu(gate) * up
    output = F.linear(activated, weights["down_proj.weight"])
    return output


def expert_forward_nvfp4(
    hidden: torch.Tensor,                    # [tokens, hidden_dim]
    weights: dict[str, torch.Tensor],        # includes weight + weight_scale
) -> torch.Tensor:
    """Run a single expert with NVFP4 quantized weights.

    Uses torch._scaled_mm for FP4 matmul if available, otherwise
    dequantizes to BF16 first.
    """
    # Check if we have the scaled_mm_v2 path
    if hasattr(torch, '_scaled_mm_v2'):
        return _expert_forward_scaled_mm(hidden, weights)

    # Fallback: dequantize and run in BF16
    # This is slower but correct
    return expert_forward_fp16(hidden, _dequantize_expert(weights))


def _dequantize_expert(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Dequantize NVFP4 expert weights to BF16 for fallback path."""
    result = {}
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        w_key = f"{proj}.weight"
        s_key = f"{proj}.weight_scale"
        if w_key in weights and s_key in weights:
            # Simple dequant: w_fp4 * scale -> bf16
            w = weights[w_key].to(torch.bfloat16)
            s = weights[s_key].to(torch.bfloat16)
            # Scale shape depends on group size, broadcast
            if s.ndim == 2 and w.ndim == 2:
                group_size = w.shape[1] // s.shape[1] if s.shape[1] > 0 else 1
                s = s.repeat_interleave(group_size, dim=1)[:, :w.shape[1]]
            result[w_key] = w * s
        elif w_key in weights:
            result[w_key] = weights[w_key].to(torch.bfloat16)
    return result


def _expert_forward_scaled_mm(
    hidden: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Expert forward using torch._scaled_mm_v2 for NVFP4."""
    # Placeholder for the real scaled_mm_v2 path
    # In production, this would use the fused FP4 GEMM
    return expert_forward_fp16(hidden, _dequantize_expert(weights))


# ─────────────────────────────────────────────────────────────────────────────
# Two-Tier Model: Orchestrates fast brain + slow brain
# ─────────────────────────────────────────────────────────────────────────────

class TwoTierModel:
    """Orchestrates fast brain (pruned GPU model) + slow brain (CPU fallback).

    The fast brain is a standard pruned model running on GPU.
    The slow brain provides pruned experts and layers on demand.

    This class does NOT wrap the full model forward pass. Instead, it provides
    hooks that integrate into the existing model's forward pass:

        1. before_moe_layer()  - check router, load slow experts if needed
        2. run_slow_experts()  - execute pruned experts that were requested
        3. before_pruned_layer_position() - check if slow layer needed
        4. run_slow_layer()    - execute a pruned layer
        5. after_output()      - check output entropy, optionally re-run with slow brain
    """

    def __init__(
        self,
        slow_experts: SlowBrainExperts,
        slow_layers: SlowBrainLayers,
        pruned_expert_ids: dict[int, set[int]],  # {orig_layer: {expert_ids}}
        pruned_layer_ids: set[int],
        layer_index_map: dict[int, int],  # original_idx -> fast_brain_idx
        config: Optional[TwoTierConfig] = None,
    ):
        self.slow_experts = slow_experts
        self.slow_layers = slow_layers
        self.pruned_expert_ids = pruned_expert_ids
        self.pruned_layer_ids = pruned_layer_ids
        self.layer_index_map = layer_index_map
        self.config = config or TwoTierConfig()

        self.detector = SlowBrainDetector(
            pruned_expert_ids=pruned_expert_ids,
            pruned_layer_ids=pruned_layer_ids,
            config=self.config,
        )
        self.stats = TwoTierStats()

        # Pre-load slow brain weights to CPU
        self.slow_experts.load_all_to_cpu()
        self.slow_layers.load_all_to_cpu()

        # Prefetch stream
        self._prefetch_stream = None
        if self.config.prefetch_enabled and torch.cuda.is_available():
            self._prefetch_stream = torch.cuda.Stream()

    @classmethod
    def from_checkpoints(
        cls,
        fast_model_dir: str,
        slow_expert_dir: str,
        slow_layer_dir: str,
        config: Optional[TwoTierConfig] = None,
    ) -> "TwoTierModel":
        """Build TwoTierModel from checkpoint directories.

        Reads manifest files to determine which experts/layers are pruned.
        """
        cfg = config or TwoTierConfig()

        # Load pruned expert manifest
        expert_manifest_path = Path(slow_expert_dir) / "manifest.json"
        pruned_expert_ids: dict[int, set[int]] = {}
        if expert_manifest_path.exists():
            with open(expert_manifest_path) as f:
                raw = json.load(f)
            for k in raw.get("experts", {}):
                layer, expert = map(int, k.split(","))
                if layer not in pruned_expert_ids:
                    pruned_expert_ids[layer] = set()
                pruned_expert_ids[layer].add(expert)

        # Load pruned layer manifest
        layer_manifest_path = Path(slow_layer_dir) / "manifest.json"
        pruned_layer_ids: set[int] = set()
        layer_index_map: dict[int, int] = {}
        if layer_manifest_path.exists():
            with open(layer_manifest_path) as f:
                raw = json.load(f)
            pruned_layer_ids = set(int(x) for x in raw.get("pruned_layers", []))
            layer_index_map = {int(k): int(v) for k, v in raw.get("layer_map", {}).items()}

        slow_experts = SlowBrainExperts(slow_expert_dir, cfg)
        slow_layers = SlowBrainLayers(slow_layer_dir, cfg)

        return cls(
            slow_experts=slow_experts,
            slow_layers=slow_layers,
            pruned_expert_ids=pruned_expert_ids,
            pruned_layer_ids=pruned_layer_ids,
            layer_index_map=layer_index_map,
            config=cfg,
        )

    # ─── Hook: Before MoE Layer ───────────────────────────────────────────

    def before_moe_layer(
        self,
        router_logits: torch.Tensor,   # [batch, num_original_experts]
        original_layer_idx: int,
    ) -> tuple[torch.Tensor, list[set[int]]]:
        """Check if any tokens need slow experts for this layer.

        Call this with the ORIGINAL router (before pruning remaps expert IDs).
        Returns (needs_slow, slow_expert_ids_per_token).
        """
        return self.detector.detect_slow_experts(router_logits, original_layer_idx)

    # ─── Hook: Run Slow Experts ───────────────────────────────────────────

    def run_slow_experts(
        self,
        hidden_states: torch.Tensor,      # [tokens_needing_slow, hidden_dim]
        original_layer_idx: int,
        expert_ids: set[int],              # which pruned experts to run
        router_weights: dict[int, float],  # {expert_id: routing_weight}
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """Execute pruned experts and return weighted sum of outputs.

        Only called for tokens that need the slow brain.
        """
        t0 = time.perf_counter()
        output = torch.zeros_like(hidden_states)

        for eid in expert_ids:
            if not self.slow_experts.has_expert(original_layer_idx, eid):
                continue

            # Check timeout
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if self.config.skip_slow_on_timeout and elapsed_ms > self.config.max_slow_latency_ms:
                self.stats.slow_expert_skipped += 1
                break

            # Get expert weights (from GPU cache or CPU transfer)
            cache_hit = self.slow_experts.gpu_cache.contains(original_layer_idx, eid)
            if cache_hit:
                self.stats.slow_expert_cache_hits += 1
            else:
                self.stats.slow_expert_cache_misses += 1

            weights = self.slow_experts.get_expert_gpu(original_layer_idx, eid, device)

            # Run expert
            expert_output = expert_forward_nvfp4(hidden_states, weights)

            # Weight by router probability
            w = router_weights.get(eid, 1.0 / len(expert_ids))
            output += w * expert_output

            # Track hit counts
            hit_key = (original_layer_idx, eid)
            self.stats.expert_hit_counts[hit_key] = \
                self.stats.expert_hit_counts.get(hit_key, 0) + 1

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats.total_slow_latency_ms += elapsed_ms
        self.stats.slow_expert_hits += 1

        return output

    # ─── Hook: Before Pruned Layer Position ────────────────────────────────

    def check_pruned_layer(
        self,
        hidden_states: torch.Tensor,   # [batch, seq_len, hidden_dim]
        original_layer_idx: int,
    ) -> torch.Tensor:
        """Check if a pruned layer should be executed.

        Returns needs_slow: [batch] bool tensor.
        """
        if original_layer_idx not in self.pruned_layer_ids:
            return torch.zeros(hidden_states.shape[0], dtype=torch.bool,
                              device=hidden_states.device)

        return self.detector.detect_slow_layer(hidden_states, original_layer_idx)

    # ─── Hook: Run Slow Layer ──────────────────────────────────────────────

    def run_slow_layer(
        self,
        hidden_states: torch.Tensor,   # [batch, seq_len, hidden_dim]
        original_layer_idx: int,
        layer_forward_fn,              # callable(state_dict, hidden) -> hidden
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """Execute a pruned layer for tokens that need it.

        `layer_forward_fn` is a callable that takes (state_dict, hidden_states)
        and returns the transformed hidden states.  This is model-specific.
        """
        t0 = time.perf_counter()

        gpu_sd = self.slow_layers.get_layer_gpu(original_layer_idx, device)
        output = layer_forward_fn(gpu_sd, hidden_states)

        # Free GPU memory immediately
        self.slow_layers.free_layer_gpu(gpu_sd)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.stats.total_slow_latency_ms += elapsed_ms
        self.stats.slow_layer_hits += 1

        return output

    # ─── Hook: After Output ────────────────────────────────────────────────

    def check_output_entropy(
        self,
        logits: torch.Tensor,  # [batch, vocab_size]
    ) -> torch.Tensor:
        """Check if output is too uncertain and needs slow brain re-evaluation."""
        return self.detector.detect_output_entropy(logits)

    # ─── Prefetch ──────────────────────────────────────────────────────────

    def prefetch_likely_experts(
        self,
        router_logits: torch.Tensor,   # [batch, num_original_experts]
        original_layer_idx: int,
        next_layer_idx: int,            # prefetch for NEXT layer
    ):
        """Speculatively prefetch slow experts for the next layer.

        Called during the current layer's MoE computation to overlap
        CPU->GPU transfer with GPU compute.
        """
        if not self.config.prefetch_enabled or self._prefetch_stream is None:
            return

        pruned = self.pruned_expert_ids.get(next_layer_idx, set())
        if not pruned:
            return

        # Look at current router logits to predict next layer's needs
        # (heuristic: same experts tend to be needed in adjacent layers)
        router_probs = F.softmax(router_logits, dim=-1)
        mean_probs = router_probs.mean(dim=0)  # [num_experts]

        # Prefetch top-N pruned experts by average probability
        pruned_probs = [(eid, mean_probs[eid].item()) for eid in pruned
                        if eid < mean_probs.shape[0]]
        pruned_probs.sort(key=lambda x: -x[1])

        for eid, prob in pruned_probs[:self.config.prefetch_queue_depth]:
            if prob > self.config.router_confidence_threshold * 0.5:
                self.slow_experts.prefetch_expert(
                    next_layer_idx, eid,
                    stream=self._prefetch_stream,
                )

    def complete_prefetch(self):
        """Complete any pending async prefetches."""
        self.slow_experts.complete_prefetch()

    # ─── Batch Splitting ───────────────────────────────────────────────────

    def split_batch(
        self,
        hidden_states: torch.Tensor,   # [batch, seq_len, hidden_dim]
        needs_slow: torch.Tensor,      # [batch] bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split batch into fast-only and slow-needing subsets.

        Returns:
            fast_hidden:   [N_fast, seq_len, hidden_dim]
            slow_hidden:   [N_slow, seq_len, hidden_dim]
            fast_indices:  [N_fast] original batch indices
            slow_indices:  [N_slow] original batch indices
        """
        fast_mask = ~needs_slow
        fast_indices = fast_mask.nonzero(as_tuple=True)[0]
        slow_indices = needs_slow.nonzero(as_tuple=True)[0]

        fast_hidden = hidden_states[fast_indices]
        slow_hidden = hidden_states[slow_indices]

        return fast_hidden, slow_hidden, fast_indices, slow_indices

    def merge_batch(
        self,
        fast_output: torch.Tensor,     # [N_fast, ...]
        slow_output: torch.Tensor,     # [N_slow, ...]
        fast_indices: torch.Tensor,
        slow_indices: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Merge fast and slow outputs back into original batch order."""
        shape = list(fast_output.shape)
        shape[0] = batch_size
        merged = torch.zeros(shape, dtype=fast_output.dtype, device=fast_output.device)
        merged[fast_indices] = fast_output
        merged[slow_indices] = slow_output
        return merged

    # ─── Full Decode Step (Reference Implementation) ───────────────────────

    def decode_step_reference(
        self,
        hidden_states: torch.Tensor,       # [batch, 1, hidden_dim]
        fast_model_layers: list,            # fast brain's layer modules
        original_to_fast: dict[int, int],   # orig layer idx -> fast layer idx
        num_original_layers: int,
    ) -> torch.Tensor:
        """Reference implementation of a full decode step with two-tier logic.

        This shows the complete flow. In production, you'd integrate these
        hooks into your existing model's forward() method instead.
        """
        self.stats.total_tokens += hidden_states.shape[0]
        device = hidden_states.device

        for orig_idx in range(num_original_layers):
            # --- Check for pruned layers that come at this position ---
            if orig_idx in self.pruned_layer_ids:
                needs_layer = self.check_pruned_layer(hidden_states, orig_idx)
                if needs_layer.any():
                    # For simplicity, run slow layer for entire batch
                    # (batch splitting for layers is complex due to residual connections)
                    hidden_states = self.run_slow_layer(
                        hidden_states, orig_idx,
                        layer_forward_fn=lambda sd, h: h,  # placeholder
                        device=device,
                    )
                continue  # pruned layers aren't in the fast brain

            # --- Run fast brain layer ---
            fast_idx = original_to_fast.get(orig_idx)
            if fast_idx is None:
                continue

            fast_layer = fast_model_layers[fast_idx]

            # If this is an MoE layer, check for slow expert needs
            if hasattr(fast_layer, 'router') and hasattr(fast_layer, 'experts'):
                # Get router logits using original expert count
                router_logits = fast_layer.router(hidden_states.squeeze(1))

                needs_slow, slow_ids = self.before_moe_layer(router_logits, orig_idx)

                if needs_slow.any() and self.config.batch_split_enabled:
                    # Split batch
                    fast_h, slow_h, fast_idx_b, slow_idx_b = self.split_batch(
                        hidden_states, needs_slow
                    )

                    # Fast path: normal forward
                    fast_out = fast_layer(fast_h)

                    # Slow path: fast layer + slow experts
                    slow_out = fast_layer(slow_h)  # Run fast experts
                    for b_local in range(slow_h.shape[0]):
                        b_orig = slow_idx_b[b_local].item()
                        if slow_ids[b_orig]:
                            # Get router weights for slow experts
                            probs = F.softmax(router_logits[b_orig], dim=-1)
                            rw = {eid: probs[eid].item() for eid in slow_ids[b_orig]}

                            slow_expert_out = self.run_slow_experts(
                                slow_h[b_local:b_local+1].squeeze(1),
                                orig_idx,
                                slow_ids[b_orig],
                                rw,
                                device,
                            )
                            slow_out[b_local] += slow_expert_out.unsqueeze(1)

                    hidden_states = self.merge_batch(
                        fast_out, slow_out, fast_idx_b, slow_idx_b, hidden_states.shape[0]
                    )
                else:
                    hidden_states = fast_layer(hidden_states)
            else:
                hidden_states = fast_layer(hidden_states)

            # Prefetch for next MoE layer
            if self.config.prefetch_enabled:
                next_orig = orig_idx + 1
                while next_orig < num_original_layers and next_orig in self.pruned_layer_ids:
                    next_orig += 1
                if next_orig < num_original_layers and hasattr(fast_layer, 'router'):
                    self.prefetch_likely_experts(
                        router_logits, orig_idx, next_orig
                    )

        self.complete_prefetch()
        return hidden_states

    def print_stats(self):
        """Print runtime statistics."""
        print(self.stats.summary())

    def reset_stats(self):
        """Reset all statistics."""
        self.stats = TwoTierStats()


# ─────────────────────────────────────────────────────────────────────────────
# Generalization: Small+Large, Quantized+Full, Local+Cloud
# ─────────────────────────────────────────────────────────────────────────────

class TwoTierGeneric:
    """Generic two-tier architecture template.

    Extends beyond pruning to other fast/slow combinations:
    - Small model (9B) + Large model (26B)
    - Quantized (FP4) + Full precision (BF16)
    - Local (on-device) + Cloud (API call)

    The detection logic is the same: confidence, entropy, norm.
    Only the "slow brain execution" differs.
    """

    def __init__(
        self,
        fast_fn,        # callable(hidden) -> output (fast path)
        slow_fn,        # callable(hidden) -> output (slow path)
        detect_fn,      # callable(hidden_or_logits) -> needs_slow [batch] bool
        config: Optional[TwoTierConfig] = None,
    ):
        self.fast_fn = fast_fn
        self.slow_fn = slow_fn
        self.detect_fn = detect_fn
        self.config = config or TwoTierConfig()
        self.stats = TwoTierStats()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run two-tier forward: fast path for most, slow path for uncertain."""
        self.stats.total_tokens += hidden.shape[0]

        # Detect which tokens need slow brain
        needs_slow = self.detect_fn(hidden)

        if not needs_slow.any():
            return self.fast_fn(hidden)

        if self.config.batch_split_enabled:
            fast_mask = ~needs_slow
            slow_mask = needs_slow

            fast_indices = fast_mask.nonzero(as_tuple=True)[0]
            slow_indices = slow_mask.nonzero(as_tuple=True)[0]

            outputs = torch.zeros_like(hidden)

            if fast_indices.numel() > 0:
                t0 = time.perf_counter()
                outputs[fast_indices] = self.fast_fn(hidden[fast_indices])
                self.stats.total_fast_latency_ms += (time.perf_counter() - t0) * 1000

            if slow_indices.numel() > 0:
                t0 = time.perf_counter()
                outputs[slow_indices] = self.slow_fn(hidden[slow_indices])
                elapsed_ms = (time.perf_counter() - t0) * 1000
                self.stats.total_slow_latency_ms += elapsed_ms
                self.stats.slow_expert_hits += slow_indices.numel()

            return outputs
        else:
            # No batch splitting: run slow path for entire batch
            return self.slow_fn(hidden)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Tier Brain inference system")
    parser.add_argument("--fast-model-dir", required=True, help="Path to pruned fast brain checkpoint")
    parser.add_argument("--slow-expert-dir", required=True, help="Path to slow brain experts")
    parser.add_argument("--slow-layer-dir", default="", help="Path to slow brain layers")
    parser.add_argument("--info", action="store_true", help="Print configuration info")
    parser.add_argument("--test-detection", action="store_true", help="Test detection with random data")
    args = parser.parse_args()

    config = TwoTierConfig()

    if args.info:
        print("Two-Tier Brain Configuration:")
        for k, v in vars(config).items():
            print(f"  {k}: {v}")

    if args.test_detection:
        print("\n=== Testing Detection Logic ===")
        # Simulate router logits
        batch = 16
        num_experts = 128
        pruned_per_layer = {i: set(range(90, 128)) for i in range(30)}
        pruned_layers = {2, 4, 8}

        detector = SlowBrainDetector(pruned_per_layer, pruned_layers, config)

        # Test router confidence detection
        router_logits = torch.randn(batch, num_experts)
        # Make some tokens strongly prefer pruned experts
        router_logits[0, 95] = 10.0
        router_logits[3, 100] = 8.0
        router_logits[7, 110] = 12.0

        needs_slow, slow_ids = detector.detect_slow_experts(router_logits, layer_idx=5)
        print(f"Router detection: {needs_slow.sum().item()}/{batch} tokens need slow brain")
        for i in range(batch):
            if needs_slow[i]:
                print(f"  Token {i}: needs experts {slow_ids[i]}")

        # Test residual norm detection
        hidden = torch.randn(batch, 1, 2816)
        hidden[2] *= 5.0  # Make one token an outlier
        needs_layer = detector.detect_slow_layer(hidden, pruned_layer_idx=2)
        print(f"\nLayer detection: {needs_layer.sum().item()}/{batch} tokens need slow layer")

        # Test entropy detection
        logits = torch.randn(batch, 32000)
        logits[5] *= 0.01  # Make one token very uncertain (flat distribution)
        needs_entropy = detector.detect_output_entropy(logits)
        print(f"\nEntropy detection: {needs_entropy.sum().item()}/{batch} tokens need re-eval")

        print("\n=== Detection Test Complete ===")
