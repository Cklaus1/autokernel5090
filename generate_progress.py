"""
Generate progress.png -- example chart showing what AutoKernel produces
after an overnight optimization run.

Usage: python generate_progress.py
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "progress.png")

# Colors
BG       = "#FFFFFF"
GRID     = "#E8ECEF"
TEXT     = "#1A1A2E"
TEXT_SEC = "#5A6378"
KEPT     = "#10B981"
KEPT_E   = "#059669"
REVERT   = "#CBD5E1"
CRASH    = "#EF4444"
FRONT    = "#10B981"
BASE_C   = "#6366F1"
PH1      = "#3B82F6"
PH2      = "#8B5CF6"
PH3      = "#F59E0B"
PH4      = "#EF4444"
PH5      = "#94A3B8"
ANNO     = "#1E293B"
ARROW    = "#10B981"
SBOX     = "#F8FAFC"
SEDGE    = "#E2E8F0"


def generate_experiments():
    random.seed(42)
    np.random.seed(42)

    pytorch_baseline = 142.5

    scenario = [
        ("baseline (BLOCK 64x64x32)", "keep", 18.3),
        ("increase BLOCK_M to 128", "keep", 32.7),
        ("increase BLOCK_N to 128", "keep", 48.1),
        ("increase BLOCK_K to 64", "keep", 56.3),
        ("try BLOCK 256x128x64", "crash", 0),
        ("try BLOCK 128x128x64", "keep", 61.8),
        ("try BLOCK 128x64x64", "revert", 55.2),
        ("try BLOCK 128x128x32", "revert", 59.1),
        ("try BLOCK 128x128x128", "revert", 52.0),
        ("add num_warps=8", "keep", 67.4),
        ("try num_warps=4", "revert", 63.8),
        ("try num_warps=16", "crash", 0),
        ("BLOCK 128x128x64, warps=8", "keep", 72.1),
        ("add num_stages=3 pipelining", "keep", 85.6),
        ("increase to num_stages=4", "keep", 91.2),
        ("increase to num_stages=5", "revert", 88.7),
        ("add L2 cache swizzling (GROUP_M=8)", "keep", 98.4),
        ("increase GROUP_M to 16", "revert", 96.1),
        ("try GROUP_M=4", "revert", 93.2),
        ("pad shared memory to avoid bank conflicts", "revert", 97.8),
        ("coalesced B matrix loads (transpose)", "keep", 105.3),
        ("vectorized 128-bit loads for A", "keep", 112.8),
        ("try 64-bit loads", "revert", 102.4),
        ("prefetch next K tile", "keep", 118.5),
        ("double-buffer prefetch", "revert", 116.2),
        ("reduce register pressure (smaller acc tile)", "revert", 108.9),
        ("tune launch grid ordering", "revert", 117.1),
        ("persistent kernel (grid = SM count)", "crash", 0),
        ("persistent kernel (fixed sync)", "keep", 126.3),
        ("enable TF32 accumulation", "revert", 124.8),
        ("use tl.dot with allow_tf32=True", "revert", 125.1),
        ("fuse bias addition into kernel", "keep", 131.7),
        ("try split-K with K_SPLITS=2", "revert", 119.4),
        ("split-K with K_SPLITS=4", "revert", 115.8),
        ("split-K with atomic add", "crash", 0),
        ("revert to non-split-K, tune epilogue", "keep", 134.2),
        ("fuse ReLU activation into store", "keep", 137.8),
        ("try GELU fusion", "revert", 135.1),
        ("streamline mask computation", "keep", 141.3),
        ("remove unnecessary sync barriers", "revert", 138.9),
        ("unroll inner K loop by 2", "keep", 145.6),
        ("unroll inner K loop by 4", "revert", 143.2),
        ("output stationary tiling", "revert", 140.8),
        ("A-stationary with larger M tile", "revert", 139.5),
        ("mixed precision accumulator (fp32 inner, bf16 outer)", "keep", 148.9),
        ("tune EVEN_K masking branch", "keep", 152.3),
        ("@triton.autotune with 4 configs", "keep", 158.7),
        ("expand autotune to 8 configs", "revert", 156.4),
        ("autotune: add large-K specialization", "keep", 162.1),
        ("warp specialization (2 load + 2 compute)", "crash", 0),
        ("warp specialization (simpler variant)", "crash", 0),
        ("warp specialization (minimal)", "revert", 155.3),
        ("back to standard, optimize autotune configs", "keep", 164.8),
        ("cooperative matrix multiply", "revert", 161.2),
        ("register tiling 2x2", "keep", 168.3),
        ("register tiling 4x2", "revert", 165.7),
        ("register tiling 2x4", "keep", 171.9),
        ("register tiling 4x4", "crash", 0),
        ("tune register tile 2x4 + pipeline", "keep", 174.5),
        ("reduce shared memory usage 20%", "revert", 172.1),
        ("increase occupancy via smaller block", "revert", 169.8),
        ("micro-benchmark guided BLOCK selection", "keep", 176.2),
        ("L2 residency control hints", "revert", 174.8),
        ("async copy via cp.async", "keep", 179.1),
        ("tune cp.async staging depth", "revert", 177.4),
        ("final pipeline depth tuning", "keep", 181.3),
        ("micro-optimize pointer arithmetic", "keep", 182.8),
        ("strength reduction on stride calc", "revert", 181.9),
        ("branch elimination in K loop tail", "keep", 183.7),
        ("try non-power-of-2 BLOCK_K=48", "crash", 0),
        ("BLOCK_K=96 (non-pow2)", "revert", 180.4),
        ("revert to pow2, tune warmup", "revert", 183.1),
        ("instruction scheduling hints", "revert", 182.5),
        ("pack Q/K into single load", "crash", 0),
        ("simplify masking for aligned sizes", "keep", 184.2),
        ("special case for square matrices", "revert", 183.9),
        ("tune autotune key ranges", "revert", 183.5),
        ("increase num_stages for large K", "revert", 183.0),
        ("thread coarsening factor=2", "revert", 182.1),
        ("tune grid launch heuristic", "keep", 184.8),
        ("minimize register spill", "revert", 184.3),
        ("cache-oblivious tile ordering", "revert", 183.7),
        ("vectorized epilogue stores", "keep", 185.6),
        ("align output pointer", "revert", 185.2),
        ("final: cleanup + simplify code", "keep", 185.9),
        ("remove dead code paths", "keep", 186.1),
        ("try WAR hazard avoidance", "revert", 185.4),
        ("experiment with EVEN_N fast path", "revert", 185.7),
        ("combine best autotune configs", "keep", 186.5),
        ("strip unused constexpr params", "revert", 186.2),
        ("final micro-tune epilogue", "revert", 186.0),
        ("radix-sort block index for L2", "revert", 185.3),
        ("try Hilbert curve block ordering", "revert", 184.9),
        ("cleanup: remove experimental code", "keep", 186.8),
        ("final polish", "keep", 187.1),
    ]

    experiments = []
    for i, (desc, cat, tp) in enumerate(scenario):
        experiments.append({
            "experiment": i,
            "description": desc,
            "category": cat,
            "throughput_tflops": tp if cat != "crash" else 0,
        })

    return experiments, pytorch_baseline


def make_plot():
    experiments, pytorch_baseline = generate_experiments()

    xs_kept, ys_kept = [], []
    xs_revert, ys_revert = [], []
    xs_crash = []

    for exp in experiments:
        x = exp["experiment"]
        y = exp["throughput_tflops"]
        cat = exp["category"]
        if cat == "keep":
            xs_kept.append(x)
            ys_kept.append(y)
        elif cat == "revert":
            xs_revert.append(x)
            ys_revert.append(y)
        elif cat == "crash":
            xs_crash.append(x)

    # Running maximum
    frontier_x, frontier_y = [], []
    running_max = 0
    for exp in experiments:
        y = exp["throughput_tflops"]
        if y > running_max:
            running_max = y
        frontier_x.append(exp["experiment"])
        frontier_y.append(running_max)

    # Figure
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.unicode_minus": False,
    })

    fig, ax = plt.subplots(figsize=(16, 8.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # Grid
    ax.grid(True, axis="y", alpha=0.5, color=GRID, linewidth=0.7, linestyle="-")
    ax.grid(True, axis="x", alpha=0.25, color=GRID, linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)

    # Phase bands
    phases = [
        (0, 12, "Block Size Tuning", PH1),
        (13, 28, "Memory Optimization", PH2),
        (29, 45, "Compute Optimization", PH3),
        (46, 65, "Advanced Techniques", PH4),
        (66, 94, "Diminishing Returns", PH5),
    ]
    for start, end, label, color in phases:
        ax.axvspan(start - 0.5, end + 0.5, alpha=0.05, color=color, zorder=0)
        mid = (start + end) / 2
        ax.text(mid, 198, label, fontsize=8, color=color, alpha=0.85,
                ha="center", va="top", fontweight="semibold",
                path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    # PyTorch baseline
    ax.axhline(y=pytorch_baseline, color=BASE_C, linestyle="--",
               linewidth=1.5, alpha=0.5, zorder=1)
    ax.text(93, pytorch_baseline + 2.5,
            f"torch.matmul (cuBLAS) = {pytorch_baseline:.1f} TFLOPS",
            fontsize=8.5, color=BASE_C, alpha=0.8, ha="right", va="bottom",
            fontweight="medium",
            path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    # Frontier
    ax.fill_between(frontier_x, 0, frontier_y, alpha=0.06, color=FRONT, zorder=1)
    ax.plot(frontier_x, frontier_y, color=FRONT, linewidth=2.2, alpha=0.65,
            zorder=3, solid_capstyle="round")

    # Reverted
    if xs_revert:
        ax.scatter(xs_revert, ys_revert, c=REVERT, s=20, alpha=0.6, zorder=2,
                   edgecolors="white", linewidths=0.3)

    # Crashes
    if xs_crash:
        ax.scatter(xs_crash, [3] * len(xs_crash), marker="X", c=CRASH, s=35,
                   linewidths=0.5, alpha=0.65, zorder=3, edgecolors="#B91C1C")

    # Kept
    if xs_kept:
        ax.scatter(xs_kept, ys_kept, c=KEPT, s=50, alpha=0.9, zorder=4,
                   edgecolors=KEPT_E, linewidths=0.8)

    # Breakthrough annotations
    breakthroughs = [
        (2,  48.1,  "BLOCK 128x128",       -35, 18),
        (13, 85.6,  "Pipelining",           -40, 15),
        (16, 98.4,  "L2 Swizzling",          12, 12),
        (22, 118.5, "K-tile Prefetch",       -45, 12),
        (28, 126.3, "Persistent Kernel",      12, 10),
        (46, 158.7, "Autotune",              -40, 14),
        (56, 171.9, "Register Tiling 2x4",    12, 10),
        (63, 179.1, "Async Copy (cp.async)", -50,  8),
    ]
    for x, y, label, xoff, yoff in breakthroughs:
        ax.annotate(
            label, xy=(x, y),
            xytext=(xoff, yoff), textcoords="offset points",
            fontsize=7.5, color=ANNO, alpha=0.85, fontweight="semibold",
            arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.0,
                            alpha=0.5, mutation_scale=8,
                            connectionstyle="arc3,rad=-0.15"),
            zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=BG,
                      edgecolor="none", alpha=0.8),
        )

    # Final result callout
    best_tp = max(ys_kept)
    speedup = best_tp / pytorch_baseline
    ax.annotate(
        f"{best_tp:.1f} TFLOPS\n({speedup:.2f}x vs cuBLAS)",
        xy=(94, best_tp),
        xytext=(15, -25), textcoords="offset points",
        fontsize=9, color=KEPT_E, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=KEPT_E, lw=1.2,
                        mutation_scale=10),
        zorder=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECFDF5",
                  edgecolor=KEPT_E, alpha=0.95, linewidth=0.8),
    )

    # Title
    n_total = len(experiments)
    n_kept = len(xs_kept)

    fig.suptitle(
        "AutoKernel: Autonomous GPU Kernel Optimization",
        fontsize=17, fontweight="bold", color=TEXT, y=0.97,
    )
    ax.set_title(
        f"{n_total} experiments  |  {n_kept} improvements  |  "
        f"18.3 -> {best_tp:.1f} TFLOPS  |  {speedup:.2f}x vs PyTorch/cuBLAS",
        fontsize=10.5, color=TEXT_SEC, pad=12, fontweight="medium",
    )

    # Axis labels
    ax.set_xlabel("Experiment #", fontsize=11.5, color=TEXT_SEC, labelpad=8)
    ax.set_ylabel("Throughput (TFLOPS)", fontsize=11.5, color=TEXT_SEC, labelpad=8)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=KEPT,
               markeredgecolor=KEPT_E, markersize=8, label='Kept (improvement)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=REVERT,
               markeredgecolor='white', markersize=6, label='Reverted (no gain)'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor=CRASH,
               markeredgecolor='#B91C1C', markersize=7, label='Crash / OOM'),
        Line2D([0], [0], color=FRONT, linewidth=2, alpha=0.65,
               label='Best so far (frontier)'),
        Line2D([0], [0], color=BASE_C, linewidth=1.5, linestyle='--',
               alpha=0.5, label='PyTorch/cuBLAS baseline'),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="lower right", fontsize=8.5, framealpha=0.95,
        edgecolor=SEDGE, facecolor=SBOX,
        borderpad=0.8, handletextpad=0.6, labelspacing=0.5,
    )
    legend.get_frame().set_linewidth(0.6)

    # Stats box
    stats_lines = [
        "matmul  4096x4096x4096  fp16",
        "NVIDIA H100 SXM  (989.5 TFLOPS peak)",
        f"Peak utilization: {best_tp/989.5*100:.1f}%",
        f"{n_total} experiments  |  {n_kept} kept  |  "
        f"{len(xs_revert)} reverted  |  {len(xs_crash)} crashes",
        f"Runtime: ~8 hours  (~5 min/experiment)",
    ]
    stats_text = "\n".join(stats_lines)
    props = dict(boxstyle="round,pad=0.6", facecolor=SBOX,
                 edgecolor=SEDGE, alpha=0.95, linewidth=0.6)
    ax.text(0.015, 0.38, stats_text, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", color=TEXT_SEC, fontfamily="monospace",
            bbox=props, zorder=7, linespacing=1.5)

    # Axis formatting
    ax.set_ylim(-5, 205)
    ax.set_xlim(-2, n_total + 2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(12.5))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    ax.tick_params(axis="both", which="major", colors=TEXT_SEC, labelsize=9.5, length=4)
    ax.tick_params(axis="both", which="minor", colors=GRID, length=2)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(GRID)
        ax.spines[spine].set_linewidth(0.8)

    # Save
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG, pad_inches=0.3)
    plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    make_plot()
