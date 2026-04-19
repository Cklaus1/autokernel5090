#!/usr/bin/env python3
"""
Correctness + microbench for mega_graph_gemma4_30layer_v6.cu.

v6 changes vs v5a: MLP gate/up/down weights are FP4 packed + FP8 block
scales + fp32 global scales. Attention/QKV/O and RMSNorm stay BF16.

Synthetic test harness:
  * Start with random BF16 weights for all layers (same as v5a).
  * Quantize MLP weights (gate/up/down) to FP4 per the modelopt convention.
  * Dequantize back to BF16 to use as the ground-truth reference. This
    makes the reference INCLUDE the FP4 rounding error, so the kernel
    should match within WMMA fp32-accum tolerances — the same tolerance
    band that passes v5a.
  * Reference kernel: same eager pipeline as v5a but with the dequantized
    (BF16-from-FP4) MLP weights.
  * Bench both v6 (FP4 weights on GPU) and the BF16 eager reference.

Shape:
  HIDDEN=2048, NUM_HEADS=16, HEAD_DIM=128, INTER_DIM=8192, MAX_SEQ=256,
  seq_len=256, NUM_LAYERS=30, M=1 decode.
"""
import ctypes
import os
import sys

# Pin to GPU 1 before importing torch, without relying on the parent shell.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

import torch

# ---------------------------------------------------------------------------
# Device — pinned via CUDA_VISIBLE_DEVICES=1 (forced above if not already set).
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda:0")
torch.cuda.set_device(DEVICE)
print(f"[INFO] device={DEVICE}  name={torch.cuda.get_device_name(DEVICE)}")

# ---------------------------------------------------------------------------
# Load shared lib.
# ---------------------------------------------------------------------------
SO_PATH = "/tmp/build_mega_graph_gemma4_30_v6/libmega_graph_gemma4_30_v6.so"
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found. Run build_mega_graph_gemma4_30layer_v6.py.",
          file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(SO_PATH)
for fn in ("mgg4_30_v6_num_sms", "mgg4_30_v6_hidden", "mgg4_30_v6_inter_dim",
           "mgg4_30_v6_num_heads", "mgg4_30_v6_head_dim", "mgg4_30_v6_max_seq",
           "mgg4_30_v6_num_layers", "mgg4_30_v6_fp4_block", "mgg4_30_v6_smem_bytes"):
    getattr(lib, fn).restype = ctypes.c_int
lib.mgg4_30_v6_layer_weights_size.restype = ctypes.c_size_t

lib.mgg4_30_v6_launch.restype = ctypes.c_int
lib.mgg4_30_v6_launch.argtypes = [
    ctypes.c_void_p,  # hidden
    ctypes.c_void_p,  # layers_device
    ctypes.c_void_p,  # q_scratch
    ctypes.c_void_p,  # attn_out
    ctypes.c_void_p,  # mlp_scratch
    ctypes.c_int,     # seq_len
    ctypes.c_int,     # num_layers_run
    ctypes.c_void_p,  # stream
]

NUM_SMS     = lib.mgg4_30_v6_num_sms()
HIDDEN      = lib.mgg4_30_v6_hidden()
INTER_DIM   = lib.mgg4_30_v6_inter_dim()
NUM_HEADS   = lib.mgg4_30_v6_num_heads()
HEAD_DIM    = lib.mgg4_30_v6_head_dim()
MAX_SEQ     = lib.mgg4_30_v6_max_seq()
NUM_LAYERS  = lib.mgg4_30_v6_num_layers()
FP4_BLOCK   = lib.mgg4_30_v6_fp4_block()
LW_SIZE     = lib.mgg4_30_v6_layer_weights_size()
SMEM_BYTES  = lib.mgg4_30_v6_smem_bytes()
print(f"[INFO] NUM_SMS={NUM_SMS}  HIDDEN={HIDDEN}  INTER_DIM={INTER_DIM}  "
      f"NUM_HEADS={NUM_HEADS}  HEAD_DIM={HEAD_DIM}  MAX_SEQ={MAX_SEQ}  "
      f"NUM_LAYERS={NUM_LAYERS}  FP4_BLOCK={FP4_BLOCK}  "
      f"SMEM={SMEM_BYTES}B  LW={LW_SIZE}B")

# ---------------------------------------------------------------------------
# FP4 constants.
# ---------------------------------------------------------------------------
FP4_E2M1_MAX = 6.0
FP4_BOUNDARIES = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
    dtype=torch.float32, device=DEVICE,
)
# sign(1 bit) | magnitude code(3 bits). Magnitudes: {0,0.5,1,1.5,2,3,4,6}
FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32, device=DEVICE,
)


def encode_fp4_nibbles(x: torch.Tensor) -> torch.Tensor:
    """Encode float32 tensor to 4-bit FP4-E2M1 nibbles (0..15).
    Matches PTX cvt.rn.satfinite.e2m1x2.f32 (round-nearest, saturate).
    Returns int32 tensor of same shape."""
    sign = (x < 0).to(torch.int32)
    ax = x.abs().clamp(max=FP4_E2M1_MAX)
    code = torch.zeros_like(x, dtype=torch.int32)
    for b in FP4_BOUNDARIES.tolist():
        code += (ax > b).to(torch.int32)
    return (sign << 3) | code


def quantize_bf16_to_fp4(
    W_bf16: torch.Tensor,  # [K, N] BF16
):
    """Quantize a BF16 weight tensor to the modelopt FP4 triple.

    Packing: the FP4 block dimension is K (the inner/reduction dim).
    - weight_fp4: uint8, shape [K, N/2]. Byte at (k, n/2) holds:
        low nibble = W[k, 2*(n/2)]       (even N)
        high nibble = W[k, 2*(n/2)+1]    (odd N)
    - weight_scale: fp8_e4m3fn, shape [K/16, N]. One scale per
        16 consecutive K-elements per N-column.
    - weight_scale_2: fp32 scalar global scale = 1 / weight_global_scale.

    Returns (weight_fp4, weight_scale, weight_scale_2_float, dequant_bf16).
    """
    K, N = W_bf16.shape
    assert K % FP4_BLOCK == 0 and N % 2 == 0

    Wf = W_bf16.to(torch.float32)

    # Per-block (16 K-elems) × per-column abs max.
    # Reshape K -> (K/16, 16), keep N as-is. abs-max over the 16-dim.
    Wblk = Wf.reshape(K // FP4_BLOCK, FP4_BLOCK, N)
    block_max = Wblk.abs().amax(dim=1)  # [K/16, N]

    # Global scale: pick weight_scale_2 = 1.0 for simplicity (synthetic).
    # Block scales absorb all magnitude normalization.
    weight_scale_2 = 1.0

    # Per-block fp32 scale = block_max / FP4_MAX.
    fp32_scale = (block_max / FP4_E2M1_MAX).clamp(min=1e-10)  # [K/16, N]
    # Encode as FP8-E4M3 then read back for exact bit consistency with GPU.
    fp8_scale = fp32_scale.to(torch.float8_e4m3fn)
    fp32_readback = fp8_scale.to(torch.float32)  # [K/16, N]

    # Compute each element's "quant-space" value = W / (scale_fp32_readback)
    scale_expanded = fp32_readback.unsqueeze(1).expand(K // FP4_BLOCK, FP4_BLOCK, N)
    scale_expanded = scale_expanded.reshape(K, N)  # [K, N]
    # Avoid /0 for empty blocks.
    safe_scale = scale_expanded.clamp(min=1e-10)
    W_scaled = (Wf / safe_scale).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)

    nibbles = encode_fp4_nibbles(W_scaled)  # [K, N] int32 in 0..15

    # Pack along N: low nibble = even N, high nibble = odd N.
    lo = (nibbles[:, 0::2] & 0xF).to(torch.uint8)
    hi = (nibbles[:, 1::2] & 0xF).to(torch.uint8)
    weight_fp4 = ((hi << 4) | lo)  # [K, N/2] uint8

    # Dequant reference: apply inverse mapping.
    dequant_f32 = FP4_E2M1_TABLE[nibbles] * scale_expanded * weight_scale_2
    dequant_bf16 = dequant_f32.to(torch.bfloat16)

    return weight_fp4, fp8_scale, weight_scale_2, dequant_bf16


# ---------------------------------------------------------------------------
# Build weights (same init scheme as v5a, then quantize MLP to FP4).
# ---------------------------------------------------------------------------
torch.manual_seed(0xC0FFEE)
SEQ_LEN = 256


def rand_bf16(*shape):
    return (torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.01).to(torch.bfloat16)


def rand_norm(*shape):
    return (1.0 + torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.01).to(torch.bfloat16)


print(f"[INFO] allocating weights + KV ...")
weights = []
K_caches = []
V_caches = []
for L in range(NUM_LAYERS):
    # BF16 attention weights — unchanged from v5a.
    Wq = rand_bf16(HIDDEN, HIDDEN)
    Wk = rand_bf16(HIDDEN, HIDDEN)
    Wv = rand_bf16(HIDDEN, HIDDEN)
    Wo = rand_bf16(HIDDEN, HIDDEN)
    norm_in = rand_norm(HIDDEN)
    norm_post = rand_norm(HIDDEN)
    # MLP weights — start as BF16, then quantize.
    W_gate_bf16 = rand_bf16(HIDDEN, INTER_DIM)     # [K=2048, N=8192]
    W_up_bf16   = rand_bf16(HIDDEN, INTER_DIM)     # [K=2048, N=8192]
    W_down_bf16 = rand_bf16(INTER_DIM, HIDDEN)     # [K=8192, N=2048]

    gate_fp4, gate_sf, gate_gs, gate_dq = quantize_bf16_to_fp4(W_gate_bf16)
    up_fp4,   up_sf,   up_gs,   up_dq   = quantize_bf16_to_fp4(W_up_bf16)
    down_fp4, down_sf, down_gs, down_dq = quantize_bf16_to_fp4(W_down_bf16)

    w = {
        "norm_in":      norm_in,
        "Wq":           Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
        "norm_post":    norm_post,
        # FP4 GPU buffers:
        "W_gate_fp4":   gate_fp4, "W_gate_sf": gate_sf, "W_gate_gs": gate_gs,
        "W_up_fp4":     up_fp4,   "W_up_sf":   up_sf,   "W_up_gs":   up_gs,
        "W_down_fp4":   down_fp4, "W_down_sf": down_sf, "W_down_gs": down_gs,
        # Dequantized BF16 (the reference sees these):
        "W_gate_ref":   gate_dq,
        "W_up_ref":     up_dq,
        "W_down_ref":   down_dq,
    }
    K = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    V = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    K[:SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
    V[:SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
    weights.append(w)
    K_caches.append(K)
    V_caches.append(V)
hidden_init = rand_bf16(HIDDEN)
print(f"[INFO] weights allocated (MLP quantized to FP4)")

# ---------------------------------------------------------------------------
# Reference (uses BF16 dequantized MLP weights so the reference matches the
# FP4 storage used by the kernel).
# ---------------------------------------------------------------------------
def rmsnorm_ref(x, w, eps=1e-6):
    xf = x.to(torch.float32)
    var = (xf * xf).mean()
    inv = torch.rsqrt(var + eps)
    return (xf * inv * w.to(torch.float32)).to(torch.bfloat16)


def attention_ref(q, K, V, seq_len):
    q_h = q.view(NUM_HEADS, HEAD_DIM).to(torch.float32)
    Kh = K[:seq_len].view(seq_len, NUM_HEADS, HEAD_DIM).to(torch.float32)
    Vh = V[:seq_len].view(seq_len, NUM_HEADS, HEAD_DIM).to(torch.float32)
    inv_sqrt_d = 1.0 / (HEAD_DIM ** 0.5)
    scores = torch.einsum("hd,thd->ht", q_h, Kh) * inv_sqrt_d
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("ht,thd->hd", p, Vh)
    return out.reshape(HIDDEN).to(torch.bfloat16)


def layer_ref(hidden, W, K_cache, V_cache, seq_len):
    x = rmsnorm_ref(hidden, W["norm_in"])
    xf = x.to(torch.float32)
    q = (xf @ W["Wq"].to(torch.float32)).to(torch.bfloat16)
    k_new = (xf @ W["Wk"].to(torch.float32)).to(torch.bfloat16)
    v_new = (xf @ W["Wv"].to(torch.float32)).to(torch.bfloat16)
    K_cache = K_cache.clone(); V_cache = V_cache.clone()
    K_cache[seq_len - 1] = k_new; V_cache[seq_len - 1] = v_new
    attn = attention_ref(q, K_cache, V_cache, seq_len)
    attn_out = (attn.to(torch.float32) @ W["Wo"].to(torch.float32)).to(torch.bfloat16)
    hidden = (hidden.to(torch.float32) + attn_out.to(torch.float32)).to(torch.bfloat16)
    x2 = rmsnorm_ref(hidden, W["norm_post"])
    xf2 = x2.to(torch.float32)
    # Use BF16-dequantized versions of the FP4 weights.
    gate = xf2 @ W["W_gate_ref"].to(torch.float32)
    up   = xf2 @ W["W_up_ref"].to(torch.float32)
    silu = gate * torch.sigmoid(gate)
    h = (silu * up)
    out = (h @ W["W_down_ref"].to(torch.float32)).to(torch.bfloat16)
    hidden = (hidden.to(torch.float32) + out.to(torch.float32)).to(torch.bfloat16)
    return hidden, K_cache, V_cache


def run_eager_ref():
    h = hidden_init.clone()
    kvs = []
    for L in range(NUM_LAYERS):
        h, K_after, V_after = layer_ref(h, weights[L], K_caches[L], V_caches[L], SEQ_LEN)
        kvs.append((K_after, V_after))
    return h, kvs


# ---------------------------------------------------------------------------
# Build LayerWeights device array (v6 layout).
# ---------------------------------------------------------------------------
class LayerWeightsC(ctypes.Structure):
    _fields_ = [
        ("input_norm",      ctypes.c_void_p),
        ("Wq",              ctypes.c_void_p),
        ("Wk",              ctypes.c_void_p),
        ("Wv",              ctypes.c_void_p),
        ("Wo",              ctypes.c_void_p),
        ("post_attn_norm",  ctypes.c_void_p),
        ("W_gate_fp4",      ctypes.c_void_p),
        ("W_gate_sf",       ctypes.c_void_p),
        ("W_gate_gs",       ctypes.c_float),
        ("W_up_fp4",        ctypes.c_void_p),
        ("W_up_sf",         ctypes.c_void_p),
        ("W_up_gs",         ctypes.c_float),
        ("W_down_fp4",      ctypes.c_void_p),
        ("W_down_sf",       ctypes.c_void_p),
        ("W_down_gs",       ctypes.c_float),
        ("K_cache",         ctypes.c_void_p),
        ("V_cache",         ctypes.c_void_p),
    ]


assert ctypes.sizeof(LayerWeightsC) == LW_SIZE, (
    f"LayerWeightsC python sizeof {ctypes.sizeof(LayerWeightsC)} != C sizeof {LW_SIZE}"
)


def build_layers_array(K_bufs, V_bufs):
    arr = (LayerWeightsC * NUM_LAYERS)()
    for L in range(NUM_LAYERS):
        w = weights[L]
        arr[L] = LayerWeightsC(
            input_norm=w["norm_in"].data_ptr(),
            Wq=w["Wq"].data_ptr(),
            Wk=w["Wk"].data_ptr(),
            Wv=w["Wv"].data_ptr(),
            Wo=w["Wo"].data_ptr(),
            post_attn_norm=w["norm_post"].data_ptr(),
            W_gate_fp4=w["W_gate_fp4"].data_ptr(),
            W_gate_sf=w["W_gate_sf"].data_ptr(),
            W_gate_gs=float(w["W_gate_gs"]),
            W_up_fp4=w["W_up_fp4"].data_ptr(),
            W_up_sf=w["W_up_sf"].data_ptr(),
            W_up_gs=float(w["W_up_gs"]),
            W_down_fp4=w["W_down_fp4"].data_ptr(),
            W_down_sf=w["W_down_sf"].data_ptr(),
            W_down_gs=float(w["W_down_gs"]),
            K_cache=K_bufs[L].data_ptr(),
            V_cache=V_bufs[L].data_ptr(),
        )
    return arr


def run_mega_graph(num_layers_run=None):
    if num_layers_run is None:
        num_layers_run = NUM_LAYERS
    K_bufs = [K.clone() for K in K_caches]
    V_bufs = [V.clone() for V in V_caches]
    h = hidden_init.clone()
    q_scratch   = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    attn_out    = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    mlp_scratch = torch.zeros(INTER_DIM, device=DEVICE, dtype=torch.bfloat16)

    layers_host = build_layers_array(K_bufs, V_bufs)
    layers_dev = torch.empty(LW_SIZE * NUM_LAYERS, device=DEVICE, dtype=torch.uint8)
    import ctypes as C
    src = C.string_at(C.addressof(layers_host), LW_SIZE * NUM_LAYERS)
    src_t = torch.frombuffer(bytearray(src), dtype=torch.uint8)
    layers_dev.copy_(src_t.to(DEVICE))

    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    rc = lib.mgg4_30_v6_launch(
        h.data_ptr(),
        layers_dev.data_ptr(),
        q_scratch.data_ptr(),
        attn_out.data_ptr(),
        mlp_scratch.data_ptr(),
        SEQ_LEN, num_layers_run,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"mgg4_30_v6_launch returned CUDA error {rc}")
    torch.cuda.synchronize(DEVICE)
    return h, list(zip(K_bufs, V_bufs)), layers_dev


# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
print("[RUN] v6 FP4-MLP mega-graph kernel ...")
h_mg, kv_mg, layers_dev = run_mega_graph()
print("[RUN] eager reference (BF16-dequantized MLP weights) ...")
h_ref, kv_ref = run_eager_ref()

diff = (h_mg.to(torch.float32) - h_ref.to(torch.float32)).abs()
max_abs = diff.max().item()
mean_abs = diff.mean().item()
ref_max = h_ref.to(torch.float32).abs().max().item()
print(f"[CORR] hidden  max_abs_diff={max_abs:.4e}  mean={mean_abs:.4e}  ref_max={ref_max:.4e}")

for L in (0, 15, 29):
    k_diff = (kv_mg[L][0][SEQ_LEN - 1].to(torch.float32) -
              kv_ref[L][0][SEQ_LEN - 1].to(torch.float32)).abs().max().item()
    v_diff = (kv_mg[L][1][SEQ_LEN - 1].to(torch.float32) -
              kv_ref[L][1][SEQ_LEN - 1].to(torch.float32)).abs().max().item()
    print(f"[CORR] layer {L:2d}  K_new={k_diff:.3e}  V_new={v_diff:.3e}")

# v6 tolerates the same error band as v5a.
CORR_THRESH_TIGHT = 1.0e-1
CORR_THRESH_REL   = 0.05
rel_err = max_abs / max(ref_max, 1e-9)
corr_pass_tight = max_abs < CORR_THRESH_TIGHT
corr_pass_rel   = rel_err < CORR_THRESH_REL
corr_pass = corr_pass_tight or corr_pass_rel
print(f"[CORR] tight(<{CORR_THRESH_TIGHT})={'PASS' if corr_pass_tight else 'FAIL'} "
      f"rel(<5%)={'PASS' if corr_pass_rel else 'FAIL'}  "
      f"rel_err={rel_err*100:.2f}%")

if not corr_pass:
    print("[CORR] FAIL (both tight and relaxed) — aborting bench.")
    sys.exit(4)

# ---------------------------------------------------------------------------
# Microbench.
# ---------------------------------------------------------------------------
def bench_mega_graph(n_warmup=5, n_iters=200):
    K_bufs = [K.clone() for K in K_caches]
    V_bufs = [V.clone() for V in V_caches]
    h_buf = hidden_init.clone()
    q_scratch   = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    attn_out    = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    mlp_scratch = torch.zeros(INTER_DIM, device=DEVICE, dtype=torch.bfloat16)

    layers_host = build_layers_array(K_bufs, V_bufs)
    layers_dev_buf = torch.empty(LW_SIZE * NUM_LAYERS, device=DEVICE, dtype=torch.uint8)
    import ctypes as C
    src = C.string_at(C.addressof(layers_host), LW_SIZE * NUM_LAYERS)
    src_t = torch.frombuffer(bytearray(src), dtype=torch.uint8)
    layers_dev_buf.copy_(src_t.to(DEVICE))

    stream = torch.cuda.current_stream(DEVICE).cuda_stream

    def call():
        lib.mgg4_30_v6_launch(
            h_buf.data_ptr(), layers_dev_buf.data_ptr(),
            q_scratch.data_ptr(),
            attn_out.data_ptr(), mlp_scratch.data_ptr(),
            SEQ_LEN, NUM_LAYERS, stream,
        )

    for _ in range(n_warmup):
        call()
    torch.cuda.synchronize(DEVICE)

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(n_iters):
        call()
    ev1.record()
    torch.cuda.synchronize(DEVICE)
    return ev0.elapsed_time(ev1) * 1000.0 / n_iters


def bench_eager(n_warmup=3, n_iters=30):
    import torch.nn.functional as F
    Wlinear = []
    for L in range(NUM_LAYERS):
        Wlinear.append({
            "Wq": weights[L]["Wq"].t().contiguous(),
            "Wk": weights[L]["Wk"].t().contiguous(),
            "Wv": weights[L]["Wv"].t().contiguous(),
            "Wo": weights[L]["Wo"].t().contiguous(),
            "Wg": weights[L]["W_gate_ref"].t().contiguous(),
            "Wu": weights[L]["W_up_ref"].t().contiguous(),
            "Wd": weights[L]["W_down_ref"].t().contiguous(),
            "n_in":   weights[L]["norm_in"],
            "n_post": weights[L]["norm_post"],
        })
    inv_sqrt_d = 1.0 / (HEAD_DIM ** 0.5)

    def rmsnorm_fast(x, w, eps=1e-6):
        xf = x.to(torch.float32)
        var = (xf * xf).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + eps)
        return (xf * inv * w.to(torch.float32)).to(torch.bfloat16)

    def layer_fast(h, WL, K, V):
        x = rmsnorm_fast(h, WL["n_in"])
        q = F.linear(x, WL["Wq"])
        k_new = F.linear(x, WL["Wk"])
        v_new = F.linear(x, WL["Wv"])
        K[SEQ_LEN - 1] = k_new; V[SEQ_LEN - 1] = v_new
        q_h = q.view(NUM_HEADS, HEAD_DIM)
        Kh = K[:SEQ_LEN].view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        Vh = V[:SEQ_LEN].view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        scores = torch.einsum("hd,thd->ht", q_h.float(), Kh.float()) * inv_sqrt_d
        p = torch.softmax(scores, dim=-1)
        attn = torch.einsum("ht,thd->hd", p, Vh.float()).to(torch.bfloat16).reshape(HIDDEN)
        attn_out = F.linear(attn, WL["Wo"])
        h = (h.float() + attn_out.float()).to(torch.bfloat16)
        x2 = rmsnorm_fast(h, WL["n_post"])
        g = F.linear(x2, WL["Wg"]); u = F.linear(x2, WL["Wu"])
        inter = (torch.sigmoid(g.float()) * g.float() * u.float()).to(torch.bfloat16)
        out = F.linear(inter, WL["Wd"])
        h = (h.float() + out.float()).to(torch.bfloat16)
        return h

    def one_step():
        h = hidden_init.clone()
        K_bufs = [K.clone() for K in K_caches]
        V_bufs = [V.clone() for V in V_caches]
        for L in range(NUM_LAYERS):
            h = layer_fast(h, Wlinear[L], K_bufs[L], V_bufs[L])
        return h

    for _ in range(n_warmup):
        one_step()
    torch.cuda.synchronize(DEVICE)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(n_iters):
        one_step()
    ev1.record()
    torch.cuda.synchronize(DEVICE)
    return ev0.elapsed_time(ev1) * 1000.0 / n_iters


us_mg    = bench_mega_graph()
us_eager = bench_eager()
print()
print(f"[BENCH] mega_graph_v6 30-layer:  {us_mg:10.1f} us/iter")
print(f"[BENCH] eager         30-layer:  {us_eager:10.1f} us/iter")
ratio = us_eager / us_mg if us_mg > 0 else 0.0
print(f"[BENCH] speedup v6 vs eager:     {ratio:.2f}x")

# Baselines.
us_v5a = 8809.0
us_v3  = 9887.0
v5a_ratio = us_v5a / us_mg if us_mg > 0 else 0.0
v3_ratio  = us_v3  / us_mg if us_mg > 0 else 0.0
print(f"[BENCH] speedup v6 vs v5a:       {v5a_ratio:.2f}x (v5a={us_v5a:.0f}us)")
print(f"[BENCH] speedup v6 vs v3:        {v3_ratio:.2f}x  (v3={us_v3:.0f}us)")

# ---------------------------------------------------------------------------
# Bytes moved.
#   Attention: 4 * H*H * 2 (BF16) per layer = 32 MB/layer
#   MLP FP4:   2 * H*IN * 0.5 (gate+up packed) + 1 * IN*H * 0.5 (down packed)
#              + 3 * H*IN/16 (fp8 scales, same in each direction)
#              = 3 * H*IN * 0.5 + 3 * H*IN / 16
#              = (3*2048*8192)*(0.5 + 1/16) = ~28 MB/layer (plain)
#   Norms:     2 * H * 2 = ~8 KB / layer (negligible)
#   KV cache:  2 * SEQ * H * 2 = 2 MB / layer
#
# Total per-layer ~= 32 + 28 + 2 = 62 MB; 30 layers ≈ 1.86 GB.
# ---------------------------------------------------------------------------
bytes_attn = 4 * HIDDEN * HIDDEN * 2
bytes_mlp_fp4 = 3 * HIDDEN * INTER_DIM // 2 + 3 * (HIDDEN * INTER_DIM // FP4_BLOCK)
bytes_kv = 2 * SEQ_LEN * HIDDEN * 2
bytes_per_layer = bytes_attn + bytes_mlp_fp4 + bytes_kv
bytes_total = NUM_LAYERS * bytes_per_layer
hbm_peak_gbs = 1792.0
pct_bw = bytes_total / (us_mg * 1e-6) / (hbm_peak_gbs * 1e9) * 100.0
print(f"[BENCH] bytes moved ~= {bytes_total/1e9:.2f} GB, pct_peak_bw={pct_bw:.1f}%")

# v6 task gates:
#   BIG_WIN  = <= 4.0 ms   (>= 2.2x vs v5a)
#   PASS     = <= 5.5 ms   (>= 1.6x vs v5a)
#   PARTIAL  = 5.5-7.0 ms  (1.26-1.6x)
#   KILL     = >= 7.0 ms   (<1.26x)
us_ms = us_mg / 1000.0
if us_ms <= 4.0:
    verdict = "PASS"; tag = "BIG_WIN"
elif us_ms <= 5.5:
    verdict = "PASS"; tag = "PASS"
elif us_ms <= 7.0:
    verdict = "PARTIAL"; tag = "PARTIAL"
else:
    verdict = "KILL"; tag = "KILL"
print(f"[VERDICT] {verdict} ({tag}) "
      f"(KILL >= 7.0ms ; PASS <= 5.5ms ; BIG_WIN <= 4.0ms)")

summary = (
    f"30L mega-graph v6 FP4-MLP M=1 H={HIDDEN} seq={SEQ_LEN}: "
    f"v6={us_mg:.1f}us eager={us_eager:.1f}us speedup={ratio:.2f}x "
    f"(vs v5a {us_v5a:.0f}us = {v5a_ratio:.2f}x ; vs v3 {us_v3:.0f}us = {v3_ratio:.2f}x) "
    f"max_abs={max_abs:.2e} corr={'PASS' if corr_pass else 'FAIL'} "
    f"bw={pct_bw:.1f}%"
)
print(f"[SUMMARY] {summary}")

print("=== RESULTS_TSV_ROW ===")
tsv_row = "\t".join([
    "mega_graph_30layer_v6",
    "W3_P1_v6_fp4_mlp_dense",
    "kernel_bench",
    "0",
    f"{us_mg:.2f}",
    f"{pct_bw:.1f}",
    f"{ratio:.2f}",
    "PASS" if (corr_pass and verdict == "PASS") else ("PARTIAL" if verdict == "PARTIAL" else "KILL"),
    "0",
    summary,
])
print(tsv_row)
print("=== END ===")
