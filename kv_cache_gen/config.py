"""Serialize KV cache specs to/from strings, dicts, and YAML.

Provides the user-facing config format for specifying KV cache formats
via CLI args, config files, or environment variables.

Usage:
    # From string (CLI args)
    spec = parse_spec("k4v4kb64vb64")
    spec = parse_spec("auto")  # → adaptive selection
    spec = parse_spec("fp8")   # → fp8_e4m3

    # From dict (YAML/JSON config)
    spec = spec_from_dict({"k_bits": 4, "v_bits": 4, "scale_block": 64})

    # To string
    name = spec_to_string(spec)  # → "k4v4kb64vb64"

    # To dict (for serialization)
    d = spec_to_dict(spec)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kv_cache_gen.spec import KVCacheSpec, PREDEFINED_SPECS


# Aliases for common shorthand names
_ALIASES = {
    "auto": None,  # sentinel for adaptive selection
    "fp8": "fp8_e4m3",
    "fp8e5m2": "fp8_e5m2",
    "fp8e4m3": "fp8_e4m3",
    "bf16": None,  # no quantization
    "int8": "k8v8b32",
    "int4": "k4v4b64",
    # Short names without scale block → use best default
    "k8v8": "k8v8b32",
    "k8v4": "k8v4b32",
    "k4v4": "k4v4b64",
    "k4v2": "k4v2b32",
    "k8v2": "k8v2b16",
}


def parse_spec(name: str) -> KVCacheSpec | None:
    """Parse a spec name string into a KVCacheSpec.

    Accepts:
        - Predefined names: "k4v4b64", "k8v8b32", "fp8_e5m2", etc.
        - Full names: "k4v4kb64vb64", "k8v4kb16vb32"
        - Aliases: "auto", "fp8", "int4", "k4v4"
        - Parameterized: "k4v4kb32vb64" (parsed into spec)

    Returns None for "auto" and "bf16" (no quantization).
    """
    name = name.strip().lower()

    # Check aliases
    if name in _ALIASES:
        resolved = _ALIASES[name]
        if resolved is None:
            return None
        name = resolved

    # Check predefined specs
    if name in PREDEFINED_SPECS:
        return PREDEFINED_SPECS[name]

    # Try parsing parameterized name: k{K}v{V}kb{KB}vb{VB}
    import re
    m = re.match(r"k(\d+)v(\d+)kb(\d+)vb(\d+)", name)
    if m:
        k_bits, v_bits, k_block, v_block = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        offset_map = {2: 1.5, 4: 7.5, 8: 127.5}
        if k_bits not in offset_map or v_bits not in offset_map:
            raise ValueError(f"Unsupported bit width in '{name}': k_bits={k_bits}, v_bits={v_bits}")
        return KVCacheSpec(
            name=name,
            k_bits=k_bits, k_sym_offset=offset_map[k_bits], k_scale_block=k_block,
            v_bits=v_bits, v_sym_offset=offset_map[v_bits], v_scale_block=v_block,
        )

    # Try short name: k{K}v{V}b{B} (same scale block for K and V)
    m = re.match(r"k(\d+)v(\d+)b(\d+)", name)
    if m:
        k_bits, v_bits, block = int(m.group(1)), int(m.group(2)), int(m.group(3))
        offset_map = {2: 1.5, 4: 7.5, 8: 127.5}
        if k_bits not in offset_map or v_bits not in offset_map:
            raise ValueError(f"Unsupported bit width in '{name}': k_bits={k_bits}, v_bits={v_bits}")
        return KVCacheSpec(
            name=name,
            k_bits=k_bits, k_sym_offset=offset_map[k_bits], k_scale_block=block,
            v_bits=v_bits, v_sym_offset=offset_map[v_bits], v_scale_block=block,
        )

    raise ValueError(
        f"Unknown KV cache spec: '{name}'. "
        f"Valid options: {', '.join(sorted(list(PREDEFINED_SPECS.keys()) + list(_ALIASES.keys())))}"
    )


def spec_to_string(spec: KVCacheSpec) -> str:
    """Convert a spec to its canonical string name."""
    return spec.name


def spec_to_dict(spec: KVCacheSpec) -> dict:
    """Convert a spec to a serializable dict."""
    return {
        "name": spec.name,
        "k_bits": spec.k_bits,
        "v_bits": spec.v_bits,
        "k_sym_offset": spec.k_sym_offset,
        "v_sym_offset": spec.v_sym_offset,
        "k_scale_block": spec.k_scale_block,
        "v_scale_block": spec.v_scale_block,
        "compression_vs_bf16": spec.compression_vs_bf16(256),
    }


def spec_from_dict(d: dict) -> KVCacheSpec:
    """Create a spec from a dict (YAML/JSON config)."""
    # If just a name, resolve it
    if "name" in d and len(d) == 1:
        return parse_spec(d["name"])

    # Shorthand: {"k_bits": 4, "v_bits": 4, "scale_block": 64}
    if "scale_block" in d and "k_scale_block" not in d:
        d["k_scale_block"] = d.pop("scale_block")
        d["v_scale_block"] = d.get("v_scale_block", d["k_scale_block"])

    offset_map = {2: 1.5, 4: 7.5, 8: 127.5}
    k_bits = d["k_bits"]
    v_bits = d["v_bits"]

    return KVCacheSpec(
        name=d.get("name", f"k{k_bits}v{v_bits}kb{d['k_scale_block']}vb{d.get('v_scale_block', d['k_scale_block'])}"),
        k_bits=k_bits,
        k_sym_offset=d.get("k_sym_offset", offset_map.get(k_bits, 0)),
        k_scale_block=d["k_scale_block"],
        v_bits=v_bits,
        v_sym_offset=d.get("v_sym_offset", offset_map.get(v_bits, 0)),
        v_scale_block=d.get("v_scale_block", d["k_scale_block"]),
    )


def list_specs() -> list[dict]:
    """List all available specs with their properties."""
    results = []
    for name, spec in sorted(PREDEFINED_SPECS.items()):
        results.append({
            "name": name,
            "compression": f"{spec.compression_vs_bf16(256):.1f}x",
            "k_bits": spec.k_bits,
            "v_bits": spec.v_bits,
            "k_scale_block": spec.k_scale_block,
            "v_scale_block": spec.v_scale_block,
            "float_format": spec.is_float_format,
        })
    return results


if __name__ == "__main__":
    # Demo: parse various spec strings
    test_names = [
        "k4v4kb64vb64", "k8v8", "int4", "fp8", "auto",
        "k4v4b32", "k8v4kb16vb32",
    ]
    print("Spec parsing demo:")
    print(f"{'Input':<20} {'Resolved':<20} {'Comp':>6}")
    print("-" * 48)
    for name in test_names:
        spec = parse_spec(name)
        if spec:
            print(f"{name:<20} {spec.name:<20} {spec.compression_vs_bf16(256):>5.1f}x")
        else:
            print(f"{name:<20} {'(none/auto)':<20}")

    print("\nAll available specs:")
    for s in list_specs():
        print(f"  {s['name']:<16} {s['compression']:>6} k{s['k_bits']}v{s['v_bits']} "
              f"blocks={s['k_scale_block']}/{s['v_scale_block']}"
              f"{' (float)' if s['float_format'] else ''}")
