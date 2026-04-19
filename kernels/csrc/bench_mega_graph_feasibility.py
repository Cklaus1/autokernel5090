#!/usr/bin/env python3
"""
Feasibility microbench for the mega-graph cooperative kernel.

Hypothesis being tested:
  Running a full 30-layer decoder forward as ONE cooperative kernel, with
  grid.sync() at layer boundaries, is within 2x of a CUDA-graph-replayed
  equivalent (one no-op kernel per barrier, captured and replayed).

If the ratio is within 2x → GO for full implementation.
If the ratio is > 2x     → BLOCKED, fall back to piecewise.

Design:
  - No torch dependency. Uses ctypes against the skeleton .so.
  - Runs on GPU 1 (env CUDA_VISIBLE_DEVICES=1) so we don't touch the vllm-ad
    workload on GPU 0.
  - ≤ 30 seconds wall clock. Many short launches.

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 kernels/csrc/bench_mega_graph_feasibility.py
"""
import ctypes
import os
import sys
import time

# --- Config ------------------------------------------------------------------
LAYER_COUNT = 30               # Gemma4 26B depth
BARRIERS_PER_LAYER = 3         # sweep-friendly; 2 is the production target
N_WARMUP = 5
N_ITERS = 200                  # ~ seconds total given microsecond latencies

SO_PATH = "/tmp/build_mega_graph/libmega_graph.so"

# --- Minimal CUDA Driver / Runtime bindings via ctypes -----------------------
# We avoid pulling torch to keep the harness lean and the numbers clean.

libcudart = None
for cand in (
    "/usr/local/cuda-12.8/lib64/libcudart.so",
    "/usr/local/cuda/lib64/libcudart.so",
    "libcudart.so",
    "libcudart.so.12",
):
    try:
        libcudart = ctypes.CDLL(cand)
        break
    except OSError:
        continue
if libcudart is None:
    print("ERROR: could not load libcudart.so", file=sys.stderr)
    sys.exit(2)

cudaMalloc = libcudart.cudaMalloc
cudaMalloc.restype = ctypes.c_int
cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudaFree = libcudart.cudaFree
cudaFree.restype = ctypes.c_int
cudaFree.argtypes = [ctypes.c_void_p]
cudaMemset = libcudart.cudaMemset
cudaMemset.restype = ctypes.c_int
cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
cudaDeviceSynchronize = libcudart.cudaDeviceSynchronize
cudaDeviceSynchronize.restype = ctypes.c_int
cudaDeviceSynchronize.argtypes = []
cudaStreamCreate = libcudart.cudaStreamCreate
cudaStreamCreate.restype = ctypes.c_int
cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudaStreamDestroy = libcudart.cudaStreamDestroy
cudaStreamDestroy.restype = ctypes.c_int
cudaStreamDestroy.argtypes = [ctypes.c_void_p]
cudaStreamSynchronize = libcudart.cudaStreamSynchronize
cudaStreamSynchronize.restype = ctypes.c_int
cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
cudaEventCreate = libcudart.cudaEventCreate
cudaEventCreate.restype = ctypes.c_int
cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
cudaEventDestroy = libcudart.cudaEventDestroy
cudaEventDestroy.restype = ctypes.c_int
cudaEventDestroy.argtypes = [ctypes.c_void_p]
cudaEventRecord = libcudart.cudaEventRecord
cudaEventRecord.restype = ctypes.c_int
cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cudaEventElapsedTime = libcudart.cudaEventElapsedTime
cudaEventElapsedTime.restype = ctypes.c_int
cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_void_p, ctypes.c_void_p]
cudaEventSynchronize = libcudart.cudaEventSynchronize
cudaEventSynchronize.restype = ctypes.c_int
cudaEventSynchronize.argtypes = [ctypes.c_void_p]

# Graph capture bindings.
cudaStreamBeginCapture = libcudart.cudaStreamBeginCapture
cudaStreamBeginCapture.restype = ctypes.c_int
cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
cudaStreamEndCapture = libcudart.cudaStreamEndCapture
cudaStreamEndCapture.restype = ctypes.c_int
cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
cudaGraphInstantiate = libcudart.cudaGraphInstantiate
cudaGraphInstantiate.restype = ctypes.c_int
cudaGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                 ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_char_p, ctypes.c_size_t]
cudaGraphLaunch = libcudart.cudaGraphLaunch
cudaGraphLaunch.restype = ctypes.c_int
cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
cudaGraphDestroy = libcudart.cudaGraphDestroy
cudaGraphDestroy.restype = ctypes.c_int
cudaGraphDestroy.argtypes = [ctypes.c_void_p]
cudaGraphExecDestroy = libcudart.cudaGraphExecDestroy
cudaGraphExecDestroy.restype = ctypes.c_int
cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]


def chk(rc, where):
    if rc != 0:
        print(f"ERROR: CUDA rc={rc} at {where}", file=sys.stderr)
        sys.exit(3)


# --- Skeleton .so ------------------------------------------------------------
if not os.path.exists(SO_PATH):
    print(f"ERROR: {SO_PATH} not found. Run build_mega_graph.py first.",
          file=sys.stderr)
    sys.exit(2)

mg = ctypes.CDLL(SO_PATH)

mg.mega_graph_num_sms.restype = ctypes.c_int
mg.mega_graph_supports_cooperative.restype = ctypes.c_int
mg.mega_graph_launch_coop_bench.restype = ctypes.c_int
mg.mega_graph_launch_coop_bench.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
mg.mega_graph_launch_noop_step.restype = ctypes.c_int
mg.mega_graph_launch_noop_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


# --- Setup -------------------------------------------------------------------
num_sms = mg.mega_graph_num_sms()
coop_ok = mg.mega_graph_supports_cooperative()
print(f"[INFO] num_sms={num_sms}  cooperative_launch={bool(coop_ok)}")
if not coop_ok:
    print("BLOCKED: device does not advertise cooperative launch.")
    sys.exit(0)

# One int side-effect buffer on device.
d_counter = ctypes.c_void_p()
chk(cudaMalloc(ctypes.byref(d_counter), 4), "cudaMalloc")
chk(cudaMemset(d_counter, 0, 4), "cudaMemset")

stream = ctypes.c_void_p()
chk(cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")

start_evt = ctypes.c_void_p()
stop_evt = ctypes.c_void_p()
chk(cudaEventCreate(ctypes.byref(start_evt)), "cudaEventCreate start")
chk(cudaEventCreate(ctypes.byref(stop_evt)), "cudaEventCreate stop")


# --- Path 1: Cooperative kernel (one launch, N layers × K barriers) ---------
# Warmup.
for _ in range(N_WARMUP):
    rc = mg.mega_graph_launch_coop_bench(
        LAYER_COUNT, BARRIERS_PER_LAYER, d_counter, stream)
    chk(rc, "coop warmup")
chk(cudaStreamSynchronize(stream), "coop warmup sync")

# Timed.
chk(cudaEventRecord(start_evt, stream), "coop start record")
for _ in range(N_ITERS):
    rc = mg.mega_graph_launch_coop_bench(
        LAYER_COUNT, BARRIERS_PER_LAYER, d_counter, stream)
    chk(rc, "coop timed")
chk(cudaEventRecord(stop_evt, stream), "coop stop record")
chk(cudaEventSynchronize(stop_evt), "coop event sync")

ms_coop_total = ctypes.c_float(0.0)
chk(cudaEventElapsedTime(ctypes.byref(ms_coop_total), start_evt, stop_evt),
    "coop elapsed")
us_per_coop_launch = ms_coop_total.value * 1000.0 / N_ITERS
us_per_coop_barrier = us_per_coop_launch / (LAYER_COUNT * BARRIERS_PER_LAYER)
print(f"[COOP]  launch time: {us_per_coop_launch:7.1f} us  "
      f"({us_per_coop_barrier:6.2f} us/barrier, "
      f"{LAYER_COUNT} layers × {BARRIERS_PER_LAYER} barriers)")


# --- Path 2: CUDA graph (capture one no-op-per-barrier sequence, replay) ----
# Capture: launch the tiny noop_step_kernel once per barrier. Under graph
# replay each "step" is a graph node; this models the piecewise cost vLLM
# pays today.
# cudaStreamCaptureModeThreadLocal = 1
chk(cudaStreamBeginCapture(stream, 1), "begin capture")
for _ in range(LAYER_COUNT * BARRIERS_PER_LAYER):
    rc = mg.mega_graph_launch_noop_step(d_counter, stream)
    chk(rc, "noop step capture")
graph = ctypes.c_void_p()
chk(cudaStreamEndCapture(stream, ctypes.byref(graph)), "end capture")

graph_exec = ctypes.c_void_p()
chk(cudaGraphInstantiate(ctypes.byref(graph_exec), graph, None, None, 0),
    "graph instantiate")

# Warmup replay.
for _ in range(N_WARMUP):
    chk(cudaGraphLaunch(graph_exec, stream), "graph warmup")
chk(cudaStreamSynchronize(stream), "graph warmup sync")

# Timed.
chk(cudaEventRecord(start_evt, stream), "graph start record")
for _ in range(N_ITERS):
    chk(cudaGraphLaunch(graph_exec, stream), "graph timed")
chk(cudaEventRecord(stop_evt, stream), "graph stop record")
chk(cudaEventSynchronize(stop_evt), "graph event sync")

ms_graph_total = ctypes.c_float(0.0)
chk(cudaEventElapsedTime(ctypes.byref(ms_graph_total), start_evt, stop_evt),
    "graph elapsed")
us_per_graph_launch = ms_graph_total.value * 1000.0 / N_ITERS
us_per_graph_barrier = us_per_graph_launch / (LAYER_COUNT * BARRIERS_PER_LAYER)
print(f"[GRAPH] launch time: {us_per_graph_launch:7.1f} us  "
      f"({us_per_graph_barrier:6.2f} us/barrier, "
      f"{LAYER_COUNT * BARRIERS_PER_LAYER} nodes)")


# --- Verdict -----------------------------------------------------------------
ratio = us_per_coop_launch / us_per_graph_launch if us_per_graph_launch > 0 else float("inf")
print()
print(f"[RESULT] cooperative_us={us_per_coop_launch:.1f}  "
      f"graph_us={us_per_graph_launch:.1f}  ratio={ratio:.2f}x")

# Go/no-go:
#   < 2.0x  → PASS (competitive with graph replay, worth pursuing since it
#              *also* fixes the SM120 graph-size bug)
#   >= 2.0x → BLOCKED (cooperative overhead swamps the graph win)
verdict = "PASS" if ratio < 2.0 else "BLOCKED"

# Also surface the absolute per-barrier cost, which is the input to the
# plan's design-decision math (Section 2.2).
print(f"[RESULT] verdict={verdict}  "
      f"per-barrier-us coop={us_per_coop_barrier:.2f} "
      f"graph={us_per_graph_barrier:.2f}")
print(f"[RESULT] 30-layer × 2-barrier projected overhead: "
      f"{60 * us_per_coop_barrier:.0f} us")

# --- Cleanup -----------------------------------------------------------------
cudaGraphExecDestroy(graph_exec)
cudaGraphDestroy(graph)
cudaEventDestroy(start_evt)
cudaEventDestroy(stop_evt)
cudaStreamDestroy(stream)
cudaFree(d_counter)

# Expose numbers as exit-friendly values for the caller (non-zero rc = BLOCKED).
sys.exit(0 if verdict == "PASS" else 1)
