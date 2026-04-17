"""
Generate a comprehensive graph covering all 225 experiments across
kernel optimization, serving, DFlash, and batch throughput campaigns.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------
# Dark theme
# --------------------------------------------------------------------------
BG = "#0d1117"
PANEL_BG = "#161b22"
GRID = "#21262d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
CYAN = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
ORANGE = "#db6d28"
PURPLE = "#bc8cff"
PINK = "#f778ba"
TEAL = "#39d353"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL_BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.alpha": 0.4,
    "font.family": "monospace",
    "font.size": 10,
})

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
results = pd.read_csv(os.path.join(SCRIPT_DIR, "results.tsv"), sep="\t")
sglang = pd.read_csv(os.path.join(SCRIPT_DIR, "sglang_results.tsv"), sep="\t")

# --------------------------------------------------------------------------
# Helper: build frontier from sorted (experiment, value) pairs
# --------------------------------------------------------------------------
def build_frontier(df, exp_col, val_col):
    frontier_x, frontier_y = [], []
    best = 0
    for _, r in df.sort_values(exp_col).iterrows():
        if r[val_col] > best:
            best = r[val_col]
            frontier_x.append(r[exp_col])
            frontier_y.append(r[val_col])
    return frontier_x, frontier_y

# --------------------------------------------------------------------------
# Prepare kernel data
# --------------------------------------------------------------------------
w4a16 = results[results["kernel_type"] == "quantized_matmul_w4a16"].copy()
w4a16 = w4a16[w4a16["throughput_tflops"] > 0]
w4a16_kept = w4a16[
    ~w4a16["description"].str.contains("revert|REVERT", case=False, na=False)
    & (w4a16["correctness"] == "PASS")
]

nvfp4 = results[results["kernel_type"] == "nvfp4_matmul"].copy()
nvfp4 = nvfp4[nvfp4["throughput_tflops"] > 0]
nvfp4_kept = nvfp4[~nvfp4["description"].str.contains("revert|REVERT", case=False, na=False)]

# --------------------------------------------------------------------------
# Prepare decode data
# --------------------------------------------------------------------------
decode_points = []
for _, r in results.iterrows():
    kt = str(r.get("kernel_type", ""))
    desc = str(r.get("description", ""))
    tag = str(r.get("tag", ""))
    exp = r["experiment"]
    if kt == "9b_serving" and r["throughput_tflops"] > 0:
        decode_points.append({"experiment": exp, "decode": r["throughput_tflops"],
                              "source": "vLLM serving", "desc": desc})
    elif kt in ("vllm_overhead", "vllm_perf") and r["throughput_tflops"] > 0:
        decode_points.append({"experiment": exp, "decode": r["throughput_tflops"],
                              "source": "MTP speculative", "desc": desc})
for _, r in sglang.iterrows():
    if r["decode_tok_s"] > 0:
        source = "SGLang (INVALID)" if 102 <= r["experiment"] <= 121 else "DFlash on vLLM"
        decode_points.append({"experiment": r["experiment"], "decode": r["decode_tok_s"],
                              "source": source, "desc": str(r.get("description", ""))})
decode_df = pd.DataFrame(decode_points)

# --------------------------------------------------------------------------
# Prepare batch data
# --------------------------------------------------------------------------
batch_points = []
for _, r in results.iterrows():
    kt = str(r.get("kernel_type", ""))
    desc = str(r.get("description", ""))
    exp = r["experiment"]
    if kt == "9b_serving" and r["latency_us"] > 0:
        batch_points.append({"experiment": exp, "batch": r["latency_us"],
                             "source": "vLLM config tuning", "desc": desc})
    elif kt == "model_benchmark" and r["throughput_tflops"] > 100:
        batch_points.append({"experiment": exp, "batch": r["throughput_tflops"],
                             "source": "vLLM batch sweep", "desc": desc})
for _, r in sglang.iterrows():
    if r["batch32_tok_s"] > 0:
        source = "SGLang (INVALID)" if 102 <= r["experiment"] <= 121 else "DFlash + memory opt"
        batch_points.append({"experiment": r["experiment"], "batch": r["batch32_tok_s"],
                             "source": source, "desc": str(r.get("description", ""))})
batch_df = pd.DataFrame(batch_points)

# ==========================================================================
# Create figure: 2x2 grid (top-left kernel, top-right kernel zoom,
#                           mid decode, bottom batch)
# Actually: 4 panels — kernel split into W4A16 and NVFP4 side by side
# ==========================================================================
fig = plt.figure(figsize=(20, 18))
gs = fig.add_gridspec(3, 2, hspace=0.32, wspace=0.25,
                      height_ratios=[1, 1, 1])

ax_w4 = fig.add_subplot(gs[0, 0])
ax_nv = fig.add_subplot(gs[0, 1])
ax_dec = fig.add_subplot(gs[1, :])
ax_bat = fig.add_subplot(gs[2, :])

# ---- Title ----
fig.suptitle(
    "AutoKernel: Full 225-Experiment Optimization Campaign",
    fontsize=20, fontweight="bold", color=CYAN, y=0.975,
)
fig.text(0.5, 0.955,
         "RTX 5090 Blackwell (SM120)  \u2022  Qwen3.5-9B NVFP4  \u2022  vLLM 0.17.0  \u2022  March 2026",
         ha="center", fontsize=12, color=MUTED)

# ==========================================================================
# Panel 1a: W4A16 Kernel TFLOPS
# ==========================================================================
ax_w4.set_title("W4A16 Quantized Matmul", fontsize=13, fontweight="bold", pad=8, color=YELLOW)

# All points faded
ax_w4.scatter(w4a16["experiment"], w4a16["throughput_tflops"],
              c=YELLOW, alpha=0.15, s=18, zorder=2)
# Frontier
fx, fy = build_frontier(w4a16_kept, "experiment", "throughput_tflops")
if fx:
    ax_w4.plot(fx, fy, "-o", color=YELLOW, linewidth=2.5, markersize=7,
               zorder=3, markeredgecolor="#fff", markeredgewidth=0.5)
    # Fill under frontier
    ax_w4.fill_between(fx, 0, fy, color=YELLOW, alpha=0.06)

# Ceiling
ax_w4.axhline(y=419, color=RED, linestyle="--", alpha=0.5, linewidth=1)
ax_w4.text(95, 425, "FP16 Peak: 419", color=RED, fontsize=8, alpha=0.7, ha="right")

# Milestone annotations
milestones_w4 = [
    (0, 15.1, "15 TFLOPS\nbaseline", "right", 12, 55),
    (21, 136.8, "Autotune +\nL2 swizzle", "right", 8, 45),
    (36, 170.4, "Flat K loop", "right", 5, 35),
    (61, 196.1, "Split dequant\n+ cuBLAS", "left", -12, 35),
    (75, 290.0, "FP16 accum", "left", -10, 25),
    (89, 328.9, "329 TFLOPS\n21.7\u00d7", "left", -8, 35),
]
for exp, val, label, ha, dx, dy in milestones_w4:
    ax_w4.annotate(label, xy=(exp, val), xytext=(exp + dx, val + dy),
                   fontsize=8, color=YELLOW, fontweight="bold" if val > 300 else "normal",
                   arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1, alpha=0.7),
                   ha=ha)

ax_w4.set_xlabel("Experiment #", fontsize=10)
ax_w4.set_ylabel("TFLOPS", fontsize=11)
ax_w4.set_ylim(0, 460)
ax_w4.set_xlim(-3, 100)
ax_w4.grid(True, alpha=0.2)

# Speedup badge
ax_w4.text(0.97, 0.03, "21.7\u00d7", transform=ax_w4.transAxes,
           fontsize=28, fontweight="bold", color=YELLOW, alpha=0.15,
           ha="right", va="bottom")

# ==========================================================================
# Panel 1b: NVFP4 Kernel TFLOPS
# ==========================================================================
ax_nv.set_title("NVFP4 FP4 Tensor Core Matmul", fontsize=13, fontweight="bold", pad=8, color=GREEN)

ax_nv.scatter(nvfp4["experiment"], nvfp4["throughput_tflops"],
              c=GREEN, alpha=0.15, s=18, zorder=2)
fx, fy = build_frontier(nvfp4_kept, "experiment", "throughput_tflops")
if fx:
    ax_nv.plot(fx, fy, "-o", color=GREEN, linewidth=2.5, markersize=7,
               zorder=3, markeredgecolor="#fff", markeredgewidth=0.5)
    ax_nv.fill_between(fx, 0, fy, color=GREEN, alpha=0.06)

# Ceilings
ax_nv.axhline(y=419, color=RED, linestyle="--", alpha=0.4, linewidth=1)
ax_nv.text(13, 440, "FP16 Peak: 419", color=RED, fontsize=8, alpha=0.6)
ax_nv.axhline(y=838, color=PINK, linestyle="--", alpha=0.4, linewidth=1)
ax_nv.text(13, 860, "FP4 Peak: 838", color=PINK, fontsize=8, alpha=0.6)

# Milestones
milestones_nv = [
    (3, 240.5, "240 TFLOPS\nsearchsorted", "right", 1.5, 60),
    (6, 899.9, "CUDA quant\nkernel", "right", 1, 50),
    (9, 987.3, "half2\nvectorized", "left", -2, 50),
    (10, 1260.5, "1,261 TFLOPS\n5.7\u00d7 cuBLAS\n300% FP16 peak", "left", -3, 50),
]
for exp, val, label, ha, dx, dy in milestones_nv:
    ax_nv.annotate(label, xy=(exp, val), xytext=(exp + dx, val + dy),
                   fontsize=8, color=GREEN, fontweight="bold" if val > 1200 else "normal",
                   arrowprops=dict(arrowstyle="->", color=GREEN, lw=1, alpha=0.7),
                   ha=ha)

ax_nv.set_xlabel("Experiment #", fontsize=10)
ax_nv.set_ylabel("TFLOPS", fontsize=11)
ax_nv.set_ylim(0, 1500)
ax_nv.set_xlim(2, 14)
ax_nv.grid(True, alpha=0.2)

ax_nv.text(0.97, 0.03, "5.7\u00d7", transform=ax_nv.transAxes,
           fontsize=28, fontweight="bold", color=GREEN, alpha=0.15,
           ha="right", va="bottom")

# ==========================================================================
# Panel 2: Decode tok/s
# ==========================================================================
ax_dec.set_title("Single-User Decode Speed", fontsize=14, fontweight="bold", pad=10)

# Invalidation zone (draw first so it's behind)
ax_dec.axvspan(102, 121, alpha=0.06, color=RED, zorder=0)
ax_dec.text(111.5, 12, "SGLang NVFP4\nINVALIDATED", color=RED, fontsize=8,
            ha="center", alpha=0.6, style="italic", zorder=1)

# Phase dividers
for x, label in [(38, "Campaign 2:\nServing"), (93, "MTP"), (126, "Campaign 3:\nDFlash")]:
    ax_dec.axvline(x=x, color=GRID, linestyle=":", alpha=0.4, linewidth=1)
    ax_dec.text(x + 1, 205, label, color=MUTED, fontsize=7, alpha=0.6, va="top")

# SGLang invalid
sg_inv = decode_df[decode_df["source"] == "SGLang (INVALID)"]
if len(sg_inv) > 0:
    ax_dec.scatter(sg_inv["experiment"], sg_inv["decode"],
                   c=RED, alpha=0.25, s=25, marker="x", zorder=2)

# vLLM serving
vllm_s = decode_df[decode_df["source"] == "vLLM serving"]
if len(vllm_s) > 0:
    ax_dec.scatter(vllm_s["experiment"], vllm_s["decode"],
                   c=CYAN, alpha=0.4, s=20, zorder=2)
    fx, fy = build_frontier(vllm_s, "experiment", "decode")
    ax_dec.plot(fx, fy, "-o", color=CYAN, linewidth=2, markersize=5,
                zorder=3, markeredgecolor="#fff", markeredgewidth=0.5)
    ax_dec.fill_between(fx, 0, fy, color=CYAN, alpha=0.04)

# MTP
mtp = decode_df[decode_df["source"] == "MTP speculative"]
if len(mtp) > 0:
    ax_dec.scatter(mtp["experiment"], mtp["decode"],
                   c=PURPLE, alpha=0.5, s=35, zorder=3, marker="D")

# DFlash
dfl = decode_df[decode_df["source"] == "DFlash on vLLM"]
if len(dfl) > 0:
    ax_dec.scatter(dfl["experiment"], dfl["decode"],
                   c=GREEN, alpha=0.4, s=20, zorder=2)
    fx, fy = build_frontier(dfl, "experiment", "decode")
    ax_dec.plot(fx, fy, "-o", color=GREEN, linewidth=2, markersize=5,
                zorder=3, markeredgecolor="#fff", markeredgewidth=0.5)

# BW ceiling
ax_dec.axhline(y=191, color=RED, linestyle="--", alpha=0.5, linewidth=1)
ax_dec.text(178, 194, "HW ceiling: 191 tok/s", color=RED, fontsize=8, alpha=0.6, ha="right")

# Annotations
ax_dec.annotate("91 tok/s baseline", xy=(38, 91), xytext=(33, 50),
                fontsize=9, color=CYAN, arrowprops=dict(arrowstyle="->", color=CYAN, lw=1, alpha=0.7))
ax_dec.annotate("122 tok/s\n+34% optimized", xy=(76, 122), xytext=(62, 65),
                fontsize=9, color=CYAN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1, alpha=0.7))

# MTP peak
if len(mtp) > 0:
    mp = mtp.loc[mtp["decode"].idxmax()]
    ax_dec.annotate(f"MTP3: {mp['decode']:.0f} tok/s",
                    xy=(mp["experiment"], mp["decode"]),
                    xytext=(mp["experiment"] - 8, mp["decode"] + 20),
                    fontsize=9, color=PURPLE, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))

# DFlash peak
if len(dfl) > 0:
    dp = dfl.loc[dfl["decode"].idxmax()]
    ax_dec.annotate(f"170 tok/s\nDFlash draft=6\n89% of HW ceiling",
                    xy=(dp["experiment"], dp["decode"]),
                    xytext=(dp["experiment"] + 8, dp["decode"] - 45),
                    fontsize=10, color=GREEN, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=GREEN, alpha=0.8))

ax_dec.set_ylabel("tok/s", fontsize=11)
ax_dec.set_xlabel("Experiment #", fontsize=10)
ax_dec.set_ylim(0, 215)
ax_dec.set_xlim(33, 180)
ax_dec.grid(True, alpha=0.2)

# Legend
legend_items = [
    mpatches.Patch(color=CYAN, alpha=0.7, label="vLLM serving config"),
    mpatches.Patch(color=PURPLE, alpha=0.7, label="MTP speculative decode"),
    mpatches.Patch(color=GREEN, alpha=0.7, label="DFlash on vLLM"),
    mpatches.Patch(color=RED, alpha=0.3, label="SGLang NVFP4 (invalid)"),
]
ax_dec.legend(handles=legend_items, loc="lower right", fontsize=9,
              framealpha=0.3, ncol=2, edgecolor=GRID)

# ==========================================================================
# Panel 3: Batch throughput
# ==========================================================================
ax_bat.set_title("Multi-User Batch Throughput", fontsize=14, fontweight="bold", pad=10)

# Invalidation zone
ax_bat.axvspan(102, 121, alpha=0.06, color=RED, zorder=0)

# Phase dividers
for x, label in [(38, "Config tuning"), (79, "Batch sweeps"), (146, "Memory optimization")]:
    ax_bat.axvline(x=x, color=GRID, linestyle=":", alpha=0.4, linewidth=1)
    ax_bat.text(x + 1, 9200, label, color=MUTED, fontsize=7, alpha=0.6, va="top")

# SGLang invalid
sg_b = batch_df[batch_df["source"] == "SGLang (INVALID)"]
if len(sg_b) > 0:
    ax_bat.scatter(sg_b["experiment"], sg_b["batch"],
                   c=RED, alpha=0.2, s=25, marker="x", zorder=2)

# vLLM config tuning
vb_cfg = batch_df[batch_df["source"] == "vLLM config tuning"]
if len(vb_cfg) > 0:
    ax_bat.scatter(vb_cfg["experiment"], vb_cfg["batch"],
                   c=CYAN, alpha=0.5, s=22, zorder=2)
    fx, fy = build_frontier(vb_cfg, "experiment", "batch")
    ax_bat.plot(fx, fy, "-", color=CYAN, linewidth=1.5, alpha=0.6, zorder=3)

# vLLM batch sweeps
vb_sw = batch_df[batch_df["source"] == "vLLM batch sweep"]
if len(vb_sw) > 0:
    ax_bat.scatter(vb_sw["experiment"], vb_sw["batch"],
                   c=YELLOW, alpha=0.6, s=35, zorder=3, marker="s")

# DFlash + memory opt
df_b = batch_df[batch_df["source"] == "DFlash + memory opt"]
if len(df_b) > 0:
    ax_bat.scatter(df_b["experiment"], df_b["batch"],
                   c=GREEN, alpha=0.5, s=25, zorder=2)
    fx, fy = build_frontier(df_b, "experiment", "batch")
    ax_bat.plot(fx, fy, "-o", color=GREEN, linewidth=2, markersize=5, alpha=0.7, zorder=3)

# Overall frontier (dashed orange)
valid_b = batch_df[~batch_df["source"].str.contains("INVALID")]
fx, fy = build_frontier(valid_b, "experiment", "batch")
if fx:
    ax_bat.plot(fx, fy, "--", color=ORANGE, linewidth=1.5, alpha=0.5, zorder=4)

# Annotations
ax_bat.annotate("420 baseline", xy=(38, 420), xytext=(40, 1800),
                fontsize=9, color=CYAN,
                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1, alpha=0.7))
ax_bat.annotate("5,941\nbatch=96", xy=(79, 5941), xytext=(72, 4200),
                fontsize=9, color=YELLOW,
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1, alpha=0.7))
ax_bat.annotate("7,075\nFP8 KV + batch=232", xy=(83, 7075), xytext=(86, 8600),
                fontsize=9, color=YELLOW, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1, alpha=0.7))
ax_bat.annotate("8,245 tok/s\nMamba FP16 + ctx=1024\n128 concurrent users",
                xy=(158, 8245), xytext=(138, 6200),
                fontsize=10, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", fc=PANEL_BG, ec=GREEN, alpha=0.8))
ax_bat.annotate("7,981 server\n(128 users real)", xy=(172, 7981), xytext=(173, 5000),
                fontsize=9, color=CYAN,
                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1, alpha=0.7))

ax_bat.set_ylabel("tok/s (total)", fontsize=11)
ax_bat.set_xlabel("Experiment #", fontsize=10)
ax_bat.set_ylim(0, 9800)
ax_bat.set_xlim(33, 180)
ax_bat.grid(True, alpha=0.2)

legend_items_b = [
    mpatches.Patch(color=CYAN, alpha=0.7, label="vLLM config tuning"),
    plt.Line2D([0], [0], marker="s", color=YELLOW, alpha=0.7,
               linestyle="None", markersize=6, label="vLLM batch sweep"),
    mpatches.Patch(color=GREEN, alpha=0.7, label="DFlash + memory opt"),
    mpatches.Patch(color=RED, alpha=0.3, label="SGLang (invalid)"),
]
ax_bat.legend(handles=legend_items_b, loc="center left", fontsize=9,
              framealpha=0.3, ncol=1, edgecolor=GRID)

# --------------------------------------------------------------------------
# Summary stats bar at bottom
# --------------------------------------------------------------------------
stats_lines = [
    "225 experiments  \u2022  127 kept (56%)  \u2022  98 reverted (44%)  \u2022  15 bugs discovered & fixed",
    "W4A16: 15 \u2192 329 TFLOPS (21.7\u00d7)    NVFP4: 240 \u2192 1,261 TFLOPS (5.7\u00d7 cuBLAS, 300% FP16 peak)",
    "Decode: 91 \u2192 170 tok/s (89% HW ceiling)    Batch: 420 \u2192 8,245 tok/s    Cost efficiency: 8.5\u00d7 vs A100",
]
for i, line in enumerate(stats_lines):
    fig.text(0.5, 0.018 - i * 0.014, line, ha="center", fontsize=9.5,
             color=MUTED if i > 0 else TEXT, family="monospace",
             fontweight="bold" if i == 0 else "normal")

# --------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------
out = os.path.join(SCRIPT_DIR, "autokernel_full_campaign.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.4)
print(f"Saved: {out}")
plt.close()
