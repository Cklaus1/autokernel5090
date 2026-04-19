#!/usr/bin/env python3
"""
Phase 0 decision gate: real Qwen3 MoE layer with FP4 expert weights.

Measures effective HBM bandwidth of the FP4-dequant + WMMA path.

Decision:
  >= 50% of 1,792 GB/s peak -> PASS, proceed to v7 Phase 1
  30-50%                    -> MARGINAL, stop and flag v7.1 redesign
  < 30%                     -> v6 pathology, STOP

Correctness gate: cos >= 0.999 vs PyTorch BF16 reference for
single-expert forward (one random expert).

Pins GPU 1 (as dispatched); Phase 0 needs no shared GPU with other agents.
"""
import ctypes
import math
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch
from safetensors.torch import safe_open

DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)
print(f"[INFO] device={DEVICE}  name={torch.cuda.get_device_name(DEVICE)}")

SO_PATH = "/tmp/build_mega_graph_v7_phase0/libmega_graph_v7_phase0.so"
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found. Run build_mega_graph_v7_phase0.py.",
          file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(SO_PATH)
for fn in ("v7p0_num_sms", "v7p0_hidden", "v7p0_inter", "v7p0_top_k",
           "v7p0_fp4_block", "v7p0_smem_bytes"):
    getattr(lib, fn).restype = ctypes.c_int
for fn in ("v7p0_gate_up_fp4_bytes_per_exp", "v7p0_gate_up_sf_bytes_per_exp",
           "v7p0_down_fp4_bytes_per_exp", "v7p0_down_sf_bytes_per_exp"):
    getattr(lib, fn).restype = ctypes.c_size_t

lib.v7p0_launch.restype = ctypes.c_int
lib.v7p0_launch.argtypes = [
    ctypes.c_void_p,  # hidden
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # gate fp4/sf/gs
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # up
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # down
    ctypes.c_void_p,  # mlp_scratch
    ctypes.c_void_p,  # expert_out
    ctypes.c_int,     # num_iters
    ctypes.c_void_p,  # stream
]

NUM_SMS    = lib.v7p0_num_sms()
HIDDEN     = lib.v7p0_hidden()       # 2048
INTER      = lib.v7p0_inter()         # 768
TOP_K      = lib.v7p0_top_k()         # 8
FP4_BLOCK  = lib.v7p0_fp4_block()     # 16
SMEM_BYTES = lib.v7p0_smem_bytes()
GU_FP4_B   = lib.v7p0_gate_up_fp4_bytes_per_exp()
GU_SF_B    = lib.v7p0_gate_up_sf_bytes_per_exp()
D_FP4_B    = lib.v7p0_down_fp4_bytes_per_exp()
D_SF_B     = lib.v7p0_down_sf_bytes_per_exp()
print(f"[INFO] NUM_SMS={NUM_SMS}  HIDDEN={HIDDEN}  INTER={INTER}  "
      f"TOP_K={TOP_K}  FP4_BLOCK={FP4_BLOCK}  SMEM={SMEM_BYTES}B")
print(f"[INFO] per-exp weight bytes: gate+up FP4={GU_FP4_B}, SF={GU_SF_B}; "
      f"down FP4={D_FP4_B}, SF={D_SF_B}")

# ---------------------------------------------------------------------------
# Load 8 real experts from Qwen3-30B-A3B-NVFP4, layer 0.
# ---------------------------------------------------------------------------
MODEL_DIR = "/root/models/Qwen3-30B-A3B-NVFP4"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")

import json
with open(INDEX_PATH) as f:
    index = json.load(f)
WMAP = index["weight_map"]

# Use experts 0..7 for the microbench (fixed top-k selection).
ACTIVE_EXPERTS = list(range(TOP_K))
LAYER = 0

def fetch(key):
    shard = WMAP[key]
    with safe_open(os.path.join(MODEL_DIR, shard), framework="pt", device="cpu") as sf:
        return sf.get_tensor(key)

print(f"[LOAD] reading {TOP_K} real expert weights from layer {LAYER} ...")
t0 = time.perf_counter()
gate_fp4_list, gate_sf_list, gate_gs_list = [], [], []
up_fp4_list,   up_sf_list,   up_gs_list   = [], [], []
down_fp4_list, down_sf_list, down_gs_list = [], [], []
for e in ACTIVE_EXPERTS:
    base = f"model.layers.{LAYER}.mlp.experts.{e}"
    gate_fp4 = fetch(f"{base}.gate_proj.weight")         # uint8 [768, 1024]
    gate_sf  = fetch(f"{base}.gate_proj.weight_scale")    # fp8e4m3 [768, 128]
    gate_gs  = fetch(f"{base}.gate_proj.weight_scale_2")  # fp32 scalar
    up_fp4   = fetch(f"{base}.up_proj.weight")
    up_sf    = fetch(f"{base}.up_proj.weight_scale")
    up_gs    = fetch(f"{base}.up_proj.weight_scale_2")
    down_fp4 = fetch(f"{base}.down_proj.weight")          # uint8 [2048, 384]
    down_sf  = fetch(f"{base}.down_proj.weight_scale")    # fp8e4m3 [2048, 48]
    down_gs  = fetch(f"{base}.down_proj.weight_scale_2")

    assert gate_fp4.shape == (INTER, HIDDEN // 2), gate_fp4.shape
    assert gate_sf.shape  == (INTER, HIDDEN // FP4_BLOCK), gate_sf.shape
    assert down_fp4.shape == (HIDDEN, INTER // 2), down_fp4.shape
    assert down_sf.shape  == (HIDDEN, INTER // FP4_BLOCK), down_sf.shape
    assert gate_sf.dtype == torch.float8_e4m3fn
    assert gate_fp4.dtype == torch.uint8
    assert gate_gs.dtype == torch.float32

    gate_fp4_list.append(gate_fp4)
    gate_sf_list.append(gate_sf)
    gate_gs_list.append(gate_gs.reshape(()))
    up_fp4_list.append(up_fp4)
    up_sf_list.append(up_sf)
    up_gs_list.append(up_gs.reshape(()))
    down_fp4_list.append(down_fp4)
    down_sf_list.append(down_sf)
    down_gs_list.append(down_gs.reshape(()))
print(f"[LOAD] done in {time.perf_counter()-t0:.2f}s")

# Stack to contiguous per-expert tensors, move to device as uint8.
def stack_fp4(lst):
    t = torch.stack(lst, dim=0).contiguous()
    return t.view(torch.uint8) if t.dtype == torch.uint8 else t

gate_fp4_all = torch.stack(gate_fp4_list, dim=0).contiguous().to(DEVICE)  # [8, 768, 1024]
gate_sf_all  = torch.stack(gate_sf_list,  dim=0).contiguous()              # [8, 768, 128] fp8e4m3
gate_sf_all  = gate_sf_all.view(torch.uint8).to(DEVICE)  # pass as uint8 bytes to kernel
gate_gs_all  = torch.stack(gate_gs_list,  dim=0).contiguous().to(DEVICE)   # [8] fp32

up_fp4_all   = torch.stack(up_fp4_list, dim=0).contiguous().to(DEVICE)
up_sf_all    = torch.stack(up_sf_list,  dim=0).contiguous().view(torch.uint8).to(DEVICE)
up_gs_all    = torch.stack(up_gs_list,  dim=0).contiguous().to(DEVICE)

down_fp4_all = torch.stack(down_fp4_list, dim=0).contiguous().to(DEVICE)   # [8, 2048, 384]
down_sf_all  = torch.stack(down_sf_list,  dim=0).contiguous().view(torch.uint8).to(DEVICE)
down_gs_all  = torch.stack(down_gs_list,  dim=0).contiguous().to(DEVICE)

print(f"[DEV] gate_fp4_all {tuple(gate_fp4_all.shape)} {gate_fp4_all.dtype} "
      f"{gate_fp4_all.numel()} bytes = {gate_fp4_all.numel()/1e6:.2f} MB")

# ---------------------------------------------------------------------------
# Reference dequant (used for correctness check).
# ---------------------------------------------------------------------------
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32, device=DEVICE,
)

def dequant_fp4(weight_u8, weight_sf_u8, weight_gs):
    """Dequant [N, K/2] FP4 + [N, K/16] FP8 + scalar -> [N, K] bf16.

    Note weight_sf_u8 is uint8 view of fp8_e4m3fn tensor; we need the fp8 values.
    """
    # Reinterpret to fp8
    N, halfK = weight_u8.shape
    K = halfK * 2
    sf_fp8 = weight_sf_u8.view(torch.float8_e4m3fn)
    scale_f32 = sf_fp8.to(torch.float32)  # [N, K/16]
    # Unpack nibbles
    pi = weight_u8.to(torch.int32)
    lo = pi & 0xF
    hi = (pi >> 4) & 0xF
    lo_v = FP4_TABLE[lo]
    hi_v = FP4_TABLE[hi]
    vals = torch.stack([lo_v, hi_v], dim=-1).reshape(N, K)  # [N, K]
    scale_exp = scale_f32.unsqueeze(-1).expand(N, K // FP4_BLOCK, FP4_BLOCK).reshape(N, K)
    deq = vals * scale_exp * weight_gs.item()
    return deq.to(torch.bfloat16)

# ---------------------------------------------------------------------------
# PyTorch reference forward: silu(x @ Wg) * (x @ Wu) @ Wd for each active expert.
# Qwen3 weight layout: [N_out, K_in], so for linear(x, W): y = x @ W.T.
# ---------------------------------------------------------------------------
torch.manual_seed(0xCAFE)
hidden_bf16 = (torch.randn(HIDDEN, dtype=torch.float32, device=DEVICE) * 0.01).to(torch.bfloat16)

ref_expert_out = torch.zeros(TOP_K, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
ref_mlp_scratch = torch.zeros(TOP_K, INTER, dtype=torch.bfloat16, device=DEVICE)

for e_slot in range(TOP_K):
    Wg = dequant_fp4(gate_fp4_all[e_slot], gate_sf_all[e_slot], gate_gs_all[e_slot])   # [768, 2048]
    Wu = dequant_fp4(up_fp4_all[e_slot],   up_sf_all[e_slot],   up_gs_all[e_slot])     # [768, 2048]
    Wd = dequant_fp4(down_fp4_all[e_slot], down_sf_all[e_slot], down_gs_all[e_slot])   # [2048, 768]

    x = hidden_bf16.to(torch.float32)
    g = x @ Wg.to(torch.float32).T   # [INTER]
    u = x @ Wu.to(torch.float32).T
    silu_g = g * torch.sigmoid(g)
    mid = (silu_g * u).to(torch.bfloat16)
    ref_mlp_scratch[e_slot] = mid
    y = mid.to(torch.float32) @ Wd.to(torch.float32).T  # [HIDDEN]
    ref_expert_out[e_slot] = y.to(torch.bfloat16)

# ---------------------------------------------------------------------------
# Allocate outputs and launch kernel (correctness run: num_iters=1).
# ---------------------------------------------------------------------------
mlp_scratch = torch.zeros(TOP_K, INTER, dtype=torch.bfloat16, device=DEVICE)
expert_out  = torch.zeros(TOP_K, HIDDEN, dtype=torch.bfloat16, device=DEVICE)

stream = torch.cuda.current_stream().cuda_stream

def launch(niters):
    return lib.v7p0_launch(
        hidden_bf16.data_ptr(),
        gate_fp4_all.data_ptr(), gate_sf_all.data_ptr(), gate_gs_all.data_ptr(),
        up_fp4_all.data_ptr(),   up_sf_all.data_ptr(),   up_gs_all.data_ptr(),
        down_fp4_all.data_ptr(), down_sf_all.data_ptr(), down_gs_all.data_ptr(),
        mlp_scratch.data_ptr(),
        expert_out.data_ptr(),
        niters,
        stream,
    )

rc = launch(1)
torch.cuda.synchronize()
if rc != 0:
    print(f"[ERROR] launch returned cudaError {rc}:",
          ctypes.string_at(lib.v7p0_launch.__name__) if False else "")
    print("[ERROR] launch failed")
    sys.exit(3)

# ---------------------------------------------------------------------------
# Correctness check — cos vs reference across all TOP_K experts.
# ---------------------------------------------------------------------------
def cos(a, b):
    a = a.to(torch.float32).reshape(-1)
    b = b.to(torch.float32).reshape(-1)
    return (a @ b) / (a.norm() * b.norm() + 1e-12)

cos_mlp_all = [cos(mlp_scratch[e], ref_mlp_scratch[e]).item() for e in range(TOP_K)]
cos_out_all = [cos(expert_out[e], ref_expert_out[e]).item()  for e in range(TOP_K)]
mlp_max_abs = (mlp_scratch.to(torch.float32) - ref_mlp_scratch.to(torch.float32)).abs().max().item()
out_max_abs = (expert_out.to(torch.float32) - ref_expert_out.to(torch.float32)).abs().max().item()
print(f"[COR] mlp_scratch cos per expert: {['%.4f'%c for c in cos_mlp_all]}")
print(f"[COR] expert_out  cos per expert: {['%.4f'%c for c in cos_out_all]}")
print(f"[COR] mlp max_abs={mlp_max_abs:.4e}   out max_abs={out_max_abs:.4e}")

cos_ok = all(c >= 0.999 for c in cos_out_all) and all(c >= 0.999 for c in cos_mlp_all)
print(f"[COR] Correctness {'PASS' if cos_ok else 'FAIL'} (cos>=0.999 required)")
if not cos_ok:
    # Still report BW so we can see if numerical bug just skews precision.
    print("[COR] proceeding to BW measurement anyway for diagnostic info.")

# ---------------------------------------------------------------------------
# Bandwidth measurement: steady-state, via in-kernel iteration loop.
# ---------------------------------------------------------------------------
# Weight bytes read per MoE block iteration (phase B + C):
#   Phase B (gate+up): 8 experts * (GU_FP4 + GU_SF) bytes
#   Phase C (down)   : 8 experts * (D_FP4 + D_SF) bytes
#   hidden/mlp activations are ~small (<< MB): neglected for HBM denominator
weights_bytes_per_iter = TOP_K * (GU_FP4_B + GU_SF_B) * 2 + TOP_K * (D_FP4_B + D_SF_B)
print(f"[BW] weight bytes per iter: {weights_bytes_per_iter/1e6:.2f} MB "
      f"(gate+up={TOP_K*(GU_FP4_B+GU_SF_B)*2/1e6:.2f}, down={TOP_K*(D_FP4_B+D_SF_B)/1e6:.2f})")

# Warmup
for _ in range(3):
    launch(4)
torch.cuda.synchronize()

# Bench
N_BENCH = 50
N_ITERS = 8   # per launch
t_start = torch.cuda.Event(enable_timing=True)
t_stop  = torch.cuda.Event(enable_timing=True)

t_start.record()
for _ in range(N_BENCH):
    launch(N_ITERS)
t_stop.record()
torch.cuda.synchronize()
elapsed_ms = t_start.elapsed_time(t_stop)
total_iters = N_BENCH * N_ITERS
per_iter_us = (elapsed_ms * 1000.0) / total_iters
bw_gbps = (weights_bytes_per_iter / (per_iter_us * 1e-6)) / 1e9

PEAK_HBM_GBPS = 1792.0
pct_peak = 100.0 * bw_gbps / PEAK_HBM_GBPS

print(f"[BW] {N_BENCH}*{N_ITERS}={total_iters} MoE iters in {elapsed_ms:.2f} ms")
print(f"[BW] per-iter: {per_iter_us:.2f} us")
print(f"[BW] effective HBM throughput: {bw_gbps:.1f} GB/s ({pct_peak:.1f}% of {PEAK_HBM_GBPS} peak)")

# ---------------------------------------------------------------------------
# Decision gate.
# ---------------------------------------------------------------------------
print()
print("=== Phase 0 Decision Gate ===")
if pct_peak >= 50.0:
    verdict = "PASS"
    print(f"[DECISION] PASS ({pct_peak:.1f}% >= 50%) — proceed to v7 Phase 1.")
elif pct_peak >= 30.0:
    verdict = "MARGINAL"
    print(f"[DECISION] MARGINAL ({pct_peak:.1f}% in 30-50%) — stop, flag v7.1 redesign.")
else:
    verdict = "KILL"
    print(f"[DECISION] KILL ({pct_peak:.1f}% < 30%) — v6 pathology repeating, STOP.")

# ---------------------------------------------------------------------------
# Emit results.tsv row
# ---------------------------------------------------------------------------
tsv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SO_PATH))),
                        "autokernel", "results.tsv")
# More robust: locate repo root.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
tsv_path = os.path.join(repo_root, "results.tsv")
row = [
    "phase0_v7",
    "W6_v7_phase0_microbench",
    "mega_graph_v7_phase0",
    f"{bw_gbps:.1f}",
    f"{per_iter_us:.2f}",
    f"{pct_peak:.1f}",
    "n/a",
    "PASS" if cos_ok else "FAIL",
    "n/a",
    f"Qwen3 real-MoE layer FP4+WMMA. {verdict}. cos={min(cos_out_all):.4f} max_abs={out_max_abs:.3e}.",
]
try:
    # Append row.
    with open(tsv_path, "a") as f:
        f.write("\t".join(row) + "\n")
    print(f"[TSV] appended row to {tsv_path}")
except Exception as ex:
    print(f"[TSV] append failed: {ex}")

# Return code: correctness failure => exit 3; verdict KILL => exit 2.
if not cos_ok:
    sys.exit(3)
if verdict == "KILL":
    sys.exit(2)
sys.exit(0)
