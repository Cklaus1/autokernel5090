#!/usr/bin/env python3
"""
Correctness + microbench for mega_graph_gemma4_30layer_v5b.cu.

v5b: barrier fusion 5 -> 4 per layer. Attention stage extended with
partial-O-proj (per-head slab), and combined "reduce-partials + rmsnorm_post
+ mlp_gate_up" stage replaces v5a's two separate stages.

Shape identical to v5a:
  HIDDEN=2048, NUM_HEADS=16, HEAD_DIM=128, INTER_DIM=8192, MAX_SEQ=256,
  seq_len=256, NUM_LAYERS=30, M=1 decode.
"""
import ctypes
import os
import sys

import torch

# ---------------------------------------------------------------------------
# Device — pinned via CUDA_VISIBLE_DEVICES=1.
# ---------------------------------------------------------------------------
if "CUDA_VISIBLE_DEVICES" in os.environ:
    DEVICE = torch.device("cuda:0")
else:
    assert torch.cuda.device_count() >= 2, "Need >=2 CUDA devices"
    DEVICE = torch.device("cuda:1")
torch.cuda.set_device(DEVICE)
print(f"[INFO] device={DEVICE}  name={torch.cuda.get_device_name(DEVICE)}")

# ---------------------------------------------------------------------------
# Load shared lib.
# ---------------------------------------------------------------------------
SO_PATH = "/tmp/build_mega_graph_gemma4_30_v5b/libmega_graph_gemma4_30_v5b.so"
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found. Run build_mega_graph_gemma4_30layer_v5b.py.",
          file=sys.stderr)
    sys.exit(2)

lib = ctypes.CDLL(SO_PATH)
for fn in ("mgg4_30_v5b_num_sms", "mgg4_30_v5b_hidden", "mgg4_30_v5b_inter_dim",
           "mgg4_30_v5b_num_heads", "mgg4_30_v5b_head_dim", "mgg4_30_v5b_max_seq",
           "mgg4_30_v5b_num_layers", "mgg4_30_v5b_smem_bytes"):
    getattr(lib, fn).restype = ctypes.c_int
lib.mgg4_30_v5b_layer_weights_size.restype = ctypes.c_size_t

lib.mgg4_30_v5b_launch.restype = ctypes.c_int
lib.mgg4_30_v5b_launch.argtypes = [
    ctypes.c_void_p,  # hidden
    ctypes.c_void_p,  # layers_device
    ctypes.c_void_p,  # q_scratch
    ctypes.c_void_p,  # attn_out
    ctypes.c_void_p,  # mlp_scratch
    ctypes.c_void_p,  # o_partials (NEW)
    ctypes.c_int,     # seq_len
    ctypes.c_int,     # num_layers_run
    ctypes.c_void_p,  # stream
]

NUM_SMS     = lib.mgg4_30_v5b_num_sms()
HIDDEN      = lib.mgg4_30_v5b_hidden()
INTER_DIM   = lib.mgg4_30_v5b_inter_dim()
NUM_HEADS   = lib.mgg4_30_v5b_num_heads()
HEAD_DIM    = lib.mgg4_30_v5b_head_dim()
MAX_SEQ     = lib.mgg4_30_v5b_max_seq()
NUM_LAYERS  = lib.mgg4_30_v5b_num_layers()
LW_SIZE     = lib.mgg4_30_v5b_layer_weights_size()
SMEM_BYTES  = lib.mgg4_30_v5b_smem_bytes()
print(f"[INFO] NUM_SMS={NUM_SMS}  HIDDEN={HIDDEN}  INTER_DIM={INTER_DIM}  "
      f"NUM_HEADS={NUM_HEADS}  HEAD_DIM={HEAD_DIM}  MAX_SEQ={MAX_SEQ}  "
      f"NUM_LAYERS={NUM_LAYERS}  SMEM={SMEM_BYTES}B  LW={LW_SIZE}B")

# ---------------------------------------------------------------------------
# Build weights (same init scheme as v5a).
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
    w = {
        "norm_in":   rand_norm(HIDDEN),
        "Wq":        rand_bf16(HIDDEN, HIDDEN),
        "Wk":        rand_bf16(HIDDEN, HIDDEN),
        "Wv":        rand_bf16(HIDDEN, HIDDEN),
        "Wo":        rand_bf16(HIDDEN, HIDDEN),
        "norm_post": rand_norm(HIDDEN),
        "W_gate":    rand_bf16(HIDDEN, INTER_DIM),
        "W_up":      rand_bf16(HIDDEN, INTER_DIM),
        "W_down":    rand_bf16(INTER_DIM, HIDDEN),
    }
    K = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    V = torch.zeros(MAX_SEQ, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    K[:SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
    V[:SEQ_LEN - 1] = rand_bf16(SEQ_LEN - 1, HIDDEN)
    weights.append(w)
    K_caches.append(K)
    V_caches.append(V)
hidden_init = rand_bf16(HIDDEN)
print(f"[INFO] weights allocated")

# ---------------------------------------------------------------------------
# Reference (matches v5a).
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
    gate = xf2 @ W["W_gate"].to(torch.float32)
    up   = xf2 @ W["W_up"].to(torch.float32)
    silu = gate * torch.sigmoid(gate)
    h = (silu * up)
    out = (h @ W["W_down"].to(torch.float32)).to(torch.bfloat16)
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
# Build LayerWeights device array.
# ---------------------------------------------------------------------------
class LayerWeightsC(ctypes.Structure):
    _fields_ = [
        ("input_norm",     ctypes.c_void_p),
        ("Wq",              ctypes.c_void_p),
        ("Wk",              ctypes.c_void_p),
        ("Wv",              ctypes.c_void_p),
        ("Wo",              ctypes.c_void_p),
        ("post_attn_norm", ctypes.c_void_p),
        ("W_gate",          ctypes.c_void_p),
        ("W_up",            ctypes.c_void_p),
        ("W_down",          ctypes.c_void_p),
        ("K_cache",         ctypes.c_void_p),
        ("V_cache",         ctypes.c_void_p),
    ]

assert ctypes.sizeof(LayerWeightsC) == LW_SIZE

def build_layers_array(K_bufs, V_bufs):
    arr = (LayerWeightsC * NUM_LAYERS)()
    for L in range(NUM_LAYERS):
        arr[L] = LayerWeightsC(
            input_norm=weights[L]["norm_in"].data_ptr(),
            Wq=weights[L]["Wq"].data_ptr(),
            Wk=weights[L]["Wk"].data_ptr(),
            Wv=weights[L]["Wv"].data_ptr(),
            Wo=weights[L]["Wo"].data_ptr(),
            post_attn_norm=weights[L]["norm_post"].data_ptr(),
            W_gate=weights[L]["W_gate"].data_ptr(),
            W_up=weights[L]["W_up"].data_ptr(),
            W_down=weights[L]["W_down"].data_ptr(),
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
    # NEW: o_partials [NUM_HEADS, HIDDEN]
    o_partials  = torch.zeros(NUM_HEADS, HIDDEN, device=DEVICE, dtype=torch.bfloat16)

    layers_host = build_layers_array(K_bufs, V_bufs)
    layers_dev = torch.empty(LW_SIZE * NUM_LAYERS, device=DEVICE, dtype=torch.uint8)
    import ctypes as C
    src = C.string_at(C.addressof(layers_host), LW_SIZE * NUM_LAYERS)
    src_t = torch.frombuffer(bytearray(src), dtype=torch.uint8)
    layers_dev.copy_(src_t.to(DEVICE))

    stream = torch.cuda.current_stream(DEVICE).cuda_stream
    rc = lib.mgg4_30_v5b_launch(
        h.data_ptr(),
        layers_dev.data_ptr(),
        q_scratch.data_ptr(),
        attn_out.data_ptr(),
        mlp_scratch.data_ptr(),
        o_partials.data_ptr(),
        SEQ_LEN, num_layers_run,
        stream,
    )
    if rc != 0:
        raise RuntimeError(f"mgg4_30_v5b_launch returned CUDA error {rc}")
    torch.cuda.synchronize(DEVICE)
    return h, list(zip(K_bufs, V_bufs)), layers_dev

# ---------------------------------------------------------------------------
# Correctness.
# ---------------------------------------------------------------------------
print("[RUN] v5b barrier-fusion mega-graph kernel ...")
h_mg, kv_mg, layers_dev = run_mega_graph()
print("[RUN] eager reference ...")
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

# Same tolerance as v5a.
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
    o_partials  = torch.zeros(NUM_HEADS, HIDDEN, device=DEVICE, dtype=torch.bfloat16)

    layers_host = build_layers_array(K_bufs, V_bufs)
    layers_dev_buf = torch.empty(LW_SIZE * NUM_LAYERS, device=DEVICE, dtype=torch.uint8)
    import ctypes as C
    src = C.string_at(C.addressof(layers_host), LW_SIZE * NUM_LAYERS)
    src_t = torch.frombuffer(bytearray(src), dtype=torch.uint8)
    layers_dev_buf.copy_(src_t.to(DEVICE))

    stream = torch.cuda.current_stream(DEVICE).cuda_stream

    def call():
        lib.mgg4_30_v5b_launch(
            h_buf.data_ptr(), layers_dev_buf.data_ptr(),
            q_scratch.data_ptr(),
            attn_out.data_ptr(), mlp_scratch.data_ptr(),
            o_partials.data_ptr(),
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
    return ev0.elapsed_time(ev1) * 1000.0 / n_iters  # us/iter

us_mg = bench_mega_graph()
print()
print(f"[BENCH] mega_graph_v5b 30-layer: {us_mg:10.1f} us/iter")

# Baselines:
# v5a: 8821 us (measured this session)
# v3:  9887 us
us_v5a = 8821.0
us_v3  = 9887.0
v5a_ratio = us_v5a / us_mg if us_mg > 0 else 0.0
v3_ratio  = us_v3 / us_mg if us_mg > 0 else 0.0
print(f"[BENCH] speedup v5b vs v5a:      {v5a_ratio:.3f}x (v5a={us_v5a:.0f}us)")
print(f"[BENCH] speedup v5b vs v3:       {v3_ratio:.2f}x (v3={us_v3:.0f}us)")

# Per-barrier cost measurement.
# v5a had 5 barriers/layer x 30 layers = 150 barriers.
# v5b has 4 barriers/layer x 30 layers = 120 barriers (30 removed).
delta_us = us_v5a - us_mg
per_barrier_us = delta_us / 30.0
print(f"[BARRIER_COST] v5a - v5b = {delta_us:+.1f}us  ({per_barrier_us:+.2f}us per removed barrier)")

# Bytes moved (same as v5a + o_partials traffic).
bytes_per_layer_weights = (4 * HIDDEN * HIDDEN + 3 * HIDDEN * INTER_DIM + 2 * HIDDEN) * 2
bytes_kv_per_layer = 2 * SEQ_LEN * HIDDEN * 2
# Additional: o_partials write+read per layer = 2 * NUM_HEADS * HIDDEN * 2
bytes_oprt_per_layer = 2 * NUM_HEADS * HIDDEN * 2
bytes_total = NUM_LAYERS * (bytes_per_layer_weights + bytes_kv_per_layer + bytes_oprt_per_layer)
hbm_peak_gbs = 1792.0
pct_bw = bytes_total / (us_mg * 1e-6) / (hbm_peak_gbs * 1e9) * 100.0
print(f"[BENCH] bytes moved ~= {bytes_total/1e9:.2f} GB, pct_peak_bw={pct_bw:.1f}%")

# v5b task gates (from roadmap):
#   BIG_WIN  = <= 6.9 ms   (>= 1.28x vs v5a)
#   PASS     = <= 7.9 ms   (>= 1.11x vs v5a)
#   PARTIAL  = 7.9-8.4 ms
#   KILL     = >= 8.4 ms   (< 1.05x vs v5a)
us_ms = us_mg / 1000.0
if us_ms <= 6.9:
    verdict = "PASS"; tag = "BIG_WIN"
elif us_ms <= 7.9:
    verdict = "PASS"; tag = "PASS"
elif us_ms <= 8.4:
    verdict = "PARTIAL"; tag = "PARTIAL"
else:
    verdict = "KILL"; tag = "KILL"
print(f"[VERDICT] {verdict} ({tag}) "
      f"(KILL >= 8.4ms ; PARTIAL 7.9-8.4ms ; PASS <= 7.9ms ; BIG_WIN <= 6.9ms)")

# Barrier-cost hypothesis:
# If v5a.1 revealed ~15 µs/barrier and we removed 30 barriers, we expect ~450 µs saving.
# CONFIRMED if delta_us in [300, 700] us range.
# AMBIGUOUS if delta_us in [100, 300] or [700, 1200] range.
# REJECTED otherwise.
if 300.0 <= delta_us <= 700.0:
    hypothesis = "CONFIRMED"
elif (100.0 <= delta_us < 300.0) or (700.0 < delta_us <= 1200.0):
    hypothesis = "AMBIGUOUS"
else:
    hypothesis = "REJECTED"
print(f"[HYPOTHESIS] barrier-cost-dominant: {hypothesis}  "
      f"(observed delta {delta_us:+.1f}us vs expected ~450us for 30 barriers removed)")

summary = (
    f"30L mega-graph v5b M=1 H={HIDDEN} seq={SEQ_LEN}: "
    f"v5b={us_mg:.1f}us (vs v5a {us_v5a:.0f}us = {v5a_ratio:.3f}x) "
    f"(vs v3 {us_v3:.0f}us = {v3_ratio:.2f}x) "
    f"max_abs={max_abs:.2e} corr={'PASS' if corr_pass else 'FAIL'} "
    f"bw={pct_bw:.1f}% barrier_delta={delta_us:+.1f}us "
    f"per_barrier={per_barrier_us:+.2f}us hypothesis={hypothesis}"
)
print(f"[SUMMARY] {summary}")

print("=== RESULTS_TSV_ROW ===")
tsv_row = "\t".join([
    "mega_graph_30layer_v5b",
    "W1_3a_v5b_barrier_fusion",
    "kernel_bench",
    "0",
    f"{us_mg:.2f}",
    f"{pct_bw:.1f}",
    f"{v5a_ratio:.3f}",
    "PASS" if (corr_pass and verdict == "PASS")
       else ("PARTIAL" if verdict == "PARTIAL"
       else ("BIG_WIN" if (corr_pass and tag == "BIG_WIN") else "KILL")),
    "0",
    summary,
])
print(tsv_row)
print("=== END ===")
