#!/usr/bin/env python3
"""
Correctness + microbench for mega_graph_gemma4.cu.

Runs a 2-layer decoder step:
  - `mega_graph` path: one cooperative kernel launch, 2 barriers/layer.
  - `eager` path: equivalent PyTorch ops, each kernel launched separately.

Correctness: max-abs-diff between the two outputs must be < 5e-3.
Microbench: CUDA-event-timed latency of each path, plus projection to 30 layers.

Runs on GPU 1 only (set CUDA_VISIBLE_DEVICES=1 externally or the script will
select device index 1 if two are visible).
"""
import ctypes
import os
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Device selection — stick to GPU 1.
# ---------------------------------------------------------------------------
# If the user pinned via CUDA_VISIBLE_DEVICES, honor that. Otherwise pick cuda:1.
if "CUDA_VISIBLE_DEVICES" in os.environ:
    DEVICE = torch.device("cuda:0")
else:
    assert torch.cuda.device_count() >= 2, "Need at least 2 CUDA devices visible"
    DEVICE = torch.device("cuda:1")
torch.cuda.set_device(DEVICE)
print(f"[INFO] device={DEVICE}  name={torch.cuda.get_device_name(DEVICE)}")

# ---------------------------------------------------------------------------
# Load kernel shared lib
# ---------------------------------------------------------------------------
SO_PATH = "/tmp/build_mega_graph_gemma4/libmega_graph_gemma4.so"
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found. Run build_mega_graph_gemma4.py first.",
          file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(SO_PATH)
lib.mgg4_num_sms.restype = ctypes.c_int
lib.mgg4_hidden.restype = ctypes.c_int
lib.mgg4_inter_dim.restype = ctypes.c_int
lib.mgg4_num_heads.restype = ctypes.c_int
lib.mgg4_head_dim.restype = ctypes.c_int
lib.mgg4_max_seq.restype = ctypes.c_int

lib.mgg4_launch_2layer.restype = ctypes.c_int
lib.mgg4_launch_2layer.argtypes = [ctypes.c_void_p] * 28 + [ctypes.c_int, ctypes.c_void_p]

NUM_SMS   = lib.mgg4_num_sms()
HIDDEN    = lib.mgg4_hidden()
INTER_DIM = lib.mgg4_inter_dim()
NUM_HEADS = lib.mgg4_num_heads()
HEAD_DIM  = lib.mgg4_head_dim()
MAX_SEQ   = lib.mgg4_max_seq()
print(f"[INFO] NUM_SMS={NUM_SMS}  HIDDEN={HIDDEN}  INTER_DIM={INTER_DIM}  "
      f"NUM_HEADS={NUM_HEADS}  HEAD_DIM={HEAD_DIM}  MAX_SEQ={MAX_SEQ}")


# ---------------------------------------------------------------------------
# Build reference weights + KV caches
# ---------------------------------------------------------------------------
torch.manual_seed(0xC0FFEE)
SEQ_LEN = 16  # includes the new token

def rand_bf16(*shape):
    # Smaller init so that accumulated BF16 matmuls don't blow up.
    return (torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.02).to(torch.bfloat16)

def randn_norm(*shape):
    # RMSNorm weight is usually near 1.0 — use 1 + small noise.
    return (1.0 + torch.randn(*shape, device=DEVICE, dtype=torch.float32) * 0.02).to(torch.bfloat16)

weights = {}
for L in (0, 1):
    weights[(L, "norm_in")]    = randn_norm(HIDDEN)
    weights[(L, "Wq")]          = rand_bf16(HIDDEN, HIDDEN)
    weights[(L, "Wk")]          = rand_bf16(HIDDEN, HIDDEN)
    weights[(L, "Wv")]          = rand_bf16(HIDDEN, HIDDEN)
    weights[(L, "Wo")]          = rand_bf16(HIDDEN, HIDDEN)
    weights[(L, "norm_post")]  = randn_norm(HIDDEN)
    weights[(L, "W_gate")]      = rand_bf16(HIDDEN, INTER_DIM)
    weights[(L, "W_up")]        = rand_bf16(HIDDEN, INTER_DIM)
    weights[(L, "W_down")]      = rand_bf16(INTER_DIM, HIDDEN)

# KV cache (per-layer) seeded with random history.
K0 = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
V0 = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
K1 = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
V1 = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
# Fill the first SEQ_LEN-1 positions with "prior tokens" K,V so attention has
# something to attend to. Leave slot SEQ_LEN-1 for the kernel to fill.
K0[: SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
V0[: SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
K1[: SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
V1[: SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)

# Initial hidden vector (the current token's embedding).
hidden_init = rand_bf16(HIDDEN)


# ---------------------------------------------------------------------------
# Reference (PyTorch eager, matches the kernel's math exactly)
# ---------------------------------------------------------------------------
def rmsnorm_ref(x, w, eps=1e-6):
    # x: [HIDDEN] bf16
    xf = x.to(torch.float32)
    var = (xf * xf).mean()
    inv = torch.rsqrt(var + eps)
    return (xf * inv * w.to(torch.float32)).to(torch.bfloat16)

def attention_ref(q, K, V, seq_len):
    # q [HIDDEN], K/V [MAX_SEQ, HIDDEN]; returns [HIDDEN]
    q_h = q.view(NUM_HEADS, HEAD_DIM).to(torch.float32)
    Kh = K[:seq_len].view(seq_len, NUM_HEADS, HEAD_DIM).to(torch.float32)
    Vh = V[:seq_len].view(seq_len, NUM_HEADS, HEAD_DIM).to(torch.float32)
    inv_sqrt_d = 1.0 / (HEAD_DIM ** 0.5)
    # scores [NUM_HEADS, seq_len]
    scores = torch.einsum("hd,thd->ht", q_h, Kh) * inv_sqrt_d
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("ht,thd->hd", p, Vh)
    return out.reshape(HIDDEN).to(torch.bfloat16)

def layer_ref(hidden, W, K_cache, V_cache, seq_len):
    # 1) pre-attn norm
    x = rmsnorm_ref(hidden, W["norm_in"])
    # 2) q,k,v projections
    xf = x.to(torch.float32)
    q = (xf @ W["Wq"].to(torch.float32)).to(torch.bfloat16)
    k_new = (xf @ W["Wk"].to(torch.float32)).to(torch.bfloat16)
    v_new = (xf @ W["Wv"].to(torch.float32)).to(torch.bfloat16)
    # Write into cache at seq_len-1
    K_cache = K_cache.clone(); V_cache = V_cache.clone()
    K_cache[seq_len - 1] = k_new
    V_cache[seq_len - 1] = v_new
    # 3) attention
    attn = attention_ref(q, K_cache, V_cache, seq_len)
    # 4) o_proj + residual
    attn_out = (attn.to(torch.float32) @ W["Wo"].to(torch.float32)).to(torch.bfloat16)
    hidden = (hidden.to(torch.float32) + attn_out.to(torch.float32)).to(torch.bfloat16)
    # 5) post-attn norm
    x2 = rmsnorm_ref(hidden, W["norm_post"])
    xf2 = x2.to(torch.float32)
    # 6) moe (swiGLU): silu(gate) * up, then @ W_down
    gate = xf2 @ W["W_gate"].to(torch.float32)
    up   = xf2 @ W["W_up"].to(torch.float32)
    silu = gate * torch.sigmoid(gate)
    h = (silu * up)
    out = (h @ W["W_down"].to(torch.float32)).to(torch.bfloat16)
    hidden = (hidden.to(torch.float32) + out.to(torch.float32)).to(torch.bfloat16)
    return hidden, K_cache, V_cache


def run_eager():
    h = hidden_init.clone()
    W0 = {
        "norm_in": weights[(0, "norm_in")],
        "Wq": weights[(0, "Wq")], "Wk": weights[(0, "Wk")],
        "Wv": weights[(0, "Wv")], "Wo": weights[(0, "Wo")],
        "norm_post": weights[(0, "norm_post")],
        "W_gate": weights[(0, "W_gate")], "W_up": weights[(0, "W_up")],
        "W_down": weights[(0, "W_down")],
    }
    W1 = {k[1]: v for k, v in weights.items() if k[0] == 1}
    h, K0f, V0f = layer_ref(h, W0, K0, V0, SEQ_LEN)
    h, K1f, V1f = layer_ref(h, W1, K1, V1, SEQ_LEN)
    return h, (K0f, V0f, K1f, V1f)


# ---------------------------------------------------------------------------
# Mega-graph path: allocate workspaces, launch, read back.
# ---------------------------------------------------------------------------
def run_mega_graph():
    # Clone KV caches so eager & kernel don't share state.
    K0w = K0.clone(); V0w = V0.clone()
    K1w = K1.clone(); V1w = V1.clone()
    h = hidden_init.clone()

    normed      = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    q_scratch   = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    attn_out    = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    mlp_scratch = torch.zeros(INTER_DIM, device=DEVICE, dtype=torch.bfloat16)
    reduce_scratch = torch.zeros(NUM_SMS, device=DEVICE, dtype=torch.float32)

    def p(t):
        return ctypes.c_void_p(t.data_ptr())

    args = [
        p(h),
        p(weights[(0, "norm_in")]),
        p(weights[(0, "Wq")]), p(weights[(0, "Wk")]),
        p(weights[(0, "Wv")]), p(weights[(0, "Wo")]),
        p(weights[(0, "norm_post")]),
        p(weights[(0, "W_gate")]), p(weights[(0, "W_up")]),
        p(weights[(0, "W_down")]),
        p(weights[(1, "norm_in")]),
        p(weights[(1, "Wq")]), p(weights[(1, "Wk")]),
        p(weights[(1, "Wv")]), p(weights[(1, "Wo")]),
        p(weights[(1, "norm_post")]),
        p(weights[(1, "W_gate")]), p(weights[(1, "W_up")]),
        p(weights[(1, "W_down")]),
        p(K0w), p(V0w), p(K1w), p(V1w),
        p(normed), p(q_scratch), p(attn_out), p(mlp_scratch),
        p(reduce_scratch),
    ]
    assert len(args) == 28
    seq_len_c = ctypes.c_int(SEQ_LEN)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    rc = lib.mgg4_launch_2layer(*args, seq_len_c, ctypes.c_void_p(stream))
    if rc != 0:
        raise RuntimeError(f"mgg4_launch_2layer returned CUDA error {rc}")
    torch.cuda.synchronize(DEVICE)
    return h, (K0w, V0w, K1w, V1w)


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
print("[RUN] Running mega-graph kernel ...")
h_mg, kv_mg = run_mega_graph()
print("[RUN] Running eager reference ...")
h_ref, kv_ref = run_eager()

diff = (h_mg.to(torch.float32) - h_ref.to(torch.float32)).abs()
max_abs = diff.max().item()
mean_abs = diff.mean().item()
ref_max = h_ref.to(torch.float32).abs().max().item()
print(f"[CORR] hidden  max_abs_diff={max_abs:.4e}  mean={mean_abs:.4e}  ref_max={ref_max:.4e}")

# Also verify KV cache slots (the new-token-K/V the kernel wrote).
k0_diff = (kv_mg[0][SEQ_LEN - 1].to(torch.float32) -
           kv_ref[0][SEQ_LEN - 1].to(torch.float32)).abs().max().item()
v0_diff = (kv_mg[1][SEQ_LEN - 1].to(torch.float32) -
           kv_ref[1][SEQ_LEN - 1].to(torch.float32)).abs().max().item()
print(f"[CORR] K0[new]  max_abs_diff={k0_diff:.4e}")
print(f"[CORR] V0[new]  max_abs_diff={v0_diff:.4e}")

CORR_THRESH = 5e-3
corr_pass = (max_abs < CORR_THRESH) and (k0_diff < CORR_THRESH) and (v0_diff < CORR_THRESH)
print(f"[CORR] verdict: {'PASS' if corr_pass else 'FAIL'} "
      f"(threshold {CORR_THRESH:.0e})")

if not corr_pass:
    print("[CORR] FAIL — aborting bench.")
    sys.exit(4)


# ---------------------------------------------------------------------------
# Microbench
# ---------------------------------------------------------------------------
def bench_mega_graph(n_warmup=10, n_iters=100):
    # Pre-allocate scratch once.
    normed      = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    q_scratch   = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    attn_out    = torch.zeros(HIDDEN,    device=DEVICE, dtype=torch.bfloat16)
    mlp_scratch = torch.zeros(INTER_DIM, device=DEVICE, dtype=torch.bfloat16)
    reduce_scratch = torch.zeros(NUM_SMS, device=DEVICE, dtype=torch.float32)
    K0w = K0.clone(); V0w = V0.clone(); K1w = K1.clone(); V1w = V1.clone()
    h_buf = hidden_init.clone()

    def p(t): return ctypes.c_void_p(t.data_ptr())
    args = [
        p(h_buf),
        p(weights[(0, "norm_in")]),
        p(weights[(0, "Wq")]), p(weights[(0, "Wk")]),
        p(weights[(0, "Wv")]), p(weights[(0, "Wo")]),
        p(weights[(0, "norm_post")]),
        p(weights[(0, "W_gate")]), p(weights[(0, "W_up")]),
        p(weights[(0, "W_down")]),
        p(weights[(1, "norm_in")]),
        p(weights[(1, "Wq")]), p(weights[(1, "Wk")]),
        p(weights[(1, "Wv")]), p(weights[(1, "Wo")]),
        p(weights[(1, "norm_post")]),
        p(weights[(1, "W_gate")]), p(weights[(1, "W_up")]),
        p(weights[(1, "W_down")]),
        p(K0w), p(V0w), p(K1w), p(V1w),
        p(normed), p(q_scratch), p(attn_out), p(mlp_scratch),
        p(reduce_scratch),
    ]
    seq_len_c = ctypes.c_int(SEQ_LEN)
    stream = torch.cuda.current_stream(DEVICE).cuda_stream

    for _ in range(n_warmup):
        lib.mgg4_launch_2layer(*args, seq_len_c, ctypes.c_void_p(stream))
    torch.cuda.synchronize(DEVICE)

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(n_iters):
        lib.mgg4_launch_2layer(*args, seq_len_c, ctypes.c_void_p(stream))
    ev1.record()
    torch.cuda.synchronize(DEVICE)
    return ev0.elapsed_time(ev1) * 1000.0 / n_iters  # us/iter


def bench_eager(n_warmup=10, n_iters=100):
    # "Realistic" eager baseline: cuBLAS BF16 matmuls via F.linear, BF16
    # rmsnorm via torch ops. Gives each op its own kernel launch, which is
    # what the piecewise-graph path is competing against.
    import torch.nn.functional as F

    # Pre-transpose weights to [out, in] for F.linear.
    def T(w):  # weights stored as [in, out]; F.linear wants [out, in]
        return w.t().contiguous()

    W0q = T(weights[(0, "Wq")]);  W0k = T(weights[(0, "Wk")])
    W0v = T(weights[(0, "Wv")]);  W0o = T(weights[(0, "Wo")])
    W0g = T(weights[(0, "W_gate")]); W0u = T(weights[(0, "W_up")])
    W0d = T(weights[(0, "W_down")])
    W1q = T(weights[(1, "Wq")]);  W1k = T(weights[(1, "Wk")])
    W1v = T(weights[(1, "Wv")]);  W1o = T(weights[(1, "Wo")])
    W1g = T(weights[(1, "W_gate")]); W1u = T(weights[(1, "W_up")])
    W1d = T(weights[(1, "W_down")])
    n0i = weights[(0, "norm_in")]; n0p = weights[(0, "norm_post")]
    n1i = weights[(1, "norm_in")]; n1p = weights[(1, "norm_post")]

    inv_sqrt_d = 1.0 / (HEAD_DIM ** 0.5)

    def rmsnorm_fast(x, w, eps=1e-6):
        xf = x.to(torch.float32)
        var = (xf * xf).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(var + eps)
        return (xf * inv * w.to(torch.float32)).to(torch.bfloat16)

    def layer_fast(h, Wq, Wk, Wv, Wo, Wg, Wu, Wd, nin, npost, K, V):
        x = rmsnorm_fast(h, nin)
        q = F.linear(x, Wq)           # [HIDDEN]
        k_new = F.linear(x, Wk); v_new = F.linear(x, Wv)
        K[SEQ_LEN - 1] = k_new; V[SEQ_LEN - 1] = v_new
        # Attention: reshape to heads.
        q_h = q.view(NUM_HEADS, HEAD_DIM)
        Kh = K[:SEQ_LEN].view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        Vh = V[:SEQ_LEN].view(SEQ_LEN, NUM_HEADS, HEAD_DIM)
        scores = torch.einsum("hd,thd->ht", q_h.float(), Kh.float()) * inv_sqrt_d
        p = torch.softmax(scores, dim=-1)
        attn = torch.einsum("ht,thd->hd", p, Vh.float()).to(torch.bfloat16).reshape(HIDDEN)
        attn_out = F.linear(attn, Wo)
        h = (h.float() + attn_out.float()).to(torch.bfloat16)
        x2 = rmsnorm_fast(h, npost)
        g = F.linear(x2, Wg); u = F.linear(x2, Wu)
        inter = (torch.sigmoid(g.float()) * g.float() * u.float()).to(torch.bfloat16)
        out = F.linear(inter, Wd)
        h = (h.float() + out.float()).to(torch.bfloat16)
        return h

    def one_step():
        h = hidden_init.clone()
        K0w = K0.clone(); V0w = V0.clone()
        K1w = K1.clone(); V1w = V1.clone()
        h = layer_fast(h, W0q, W0k, W0v, W0o, W0g, W0u, W0d, n0i, n0p, K0w, V0w)
        h = layer_fast(h, W1q, W1k, W1v, W1o, W1g, W1u, W1d, n1i, n1p, K1w, V1w)
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
print(f"[BENCH] mega_graph 2-layer: {us_mg:8.1f} us/iter")
print(f"[BENCH] eager      2-layer: {us_eager:8.1f} us/iter")
ratio = us_eager / us_mg if us_mg > 0 else 0.0
print(f"[BENCH] speedup mega vs eager: {ratio:.2f}x")

# % bandwidth — rough, using weight bytes per token * HBM peak.
# Weight bytes: per layer 4 HIDDEN^2 (Wq,Wk,Wv,Wo) + 3 * HIDDEN*INTER_DIM
# Plus 2 * norm weights (HIDDEN each). All bf16.
bytes_per_layer = (4 * HIDDEN * HIDDEN + 3 * HIDDEN * INTER_DIM + 2 * HIDDEN) * 2
bytes_total = 2 * bytes_per_layer  # 2 layers
# Plus KV reads for attention: 2 * seq_len * HIDDEN * 2 bytes per layer
bytes_total += 2 * (2 * SEQ_LEN * HIDDEN * 2)
hbm_peak_gbs = 1792.0
pct_bw = bytes_total / (us_mg * 1e-6) / (hbm_peak_gbs * 1e9) * 100.0
print(f"[BENCH] bytes moved (weights+KV) ≈ {bytes_total/1e6:.2f} MB, "
      f"pct_peak_bw={pct_bw:.1f}%")

# Projection to 30 layers: fixed overhead scales linearly with layer count
# for barrier and compute. Naive linear projection:
us_30layer_proj = us_mg / 2.0 * 30.0
eager_30layer_proj = us_eager / 2.0 * 30.0
print(f"[PROJ] 30-layer mega-graph (linear)  ≈ {us_30layer_proj:8.1f} us")
print(f"[PROJ] 30-layer eager      (linear)  ≈ {eager_30layer_proj:8.1f} us")

# Pass/fail verdict (from task spec):
#   > 1.5x slower than eager → KILL.
KILL_RATIO = 1.5
verdict = "PASS"
if us_mg > KILL_RATIO * us_eager:
    verdict = "KILL"
print(f"[VERDICT] {verdict}  (kill if mega > {KILL_RATIO}x eager)")

# Emit a one-line summary for results.tsv (caller will prepend tab fields).
summary = (
    f"2-layer mega-graph M=1 HIDDEN={HIDDEN} seq_len={SEQ_LEN}: "
    f"mega={us_mg:.1f}us eager={us_eager:.1f}us speedup={ratio:.2f}x "
    f"max_abs={max_abs:.2e} corr={'PASS' if corr_pass else 'FAIL'} "
    f"30L-proj={us_30layer_proj/1000:.2f}ms"
)
print(f"[SUMMARY] {summary}")

# Print machine-readable block so the outer script can pick fields.
print("=== RESULTS_TSV_ROW ===")
tsv_row = "\t".join([
    "mega_graph_2layer_prototype",
    "megagraph_v1",
    "kernel_bench",
    "0",
    f"{us_mg:.2f}",
    f"{pct_bw:.1f}",
    f"{ratio:.2f}",
    "PASS" if corr_pass and verdict == "PASS" else "FAIL",
    "0",
    summary,
])
print(tsv_row)
print("=== END ===")
