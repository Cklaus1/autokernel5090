#!/usr/bin/env python3
"""Generate AutoKernel experiment results chart."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os, shutil

# --- Data ---
experiments = [0, 21, 29, 32, 34, 36, 39, 60, 61, 62, 63, 74, 75, 76]
tflops = [15.1, 136.8, 143.9, 153.4, 155.7, 170.4, 177.5, 188.2, 196.1, 196.6, 197.9, 215.0, 290.0, 328.0]
labels = [
    "baseline", "autotune +\nL2 swizzle", "two-level\nK tiling", "persistent\nkernel",
    "persistent\n+ stages", "flat K loop", "constexpr\ngroup", "split dequant\n+ cuBLAS",
    "transposed\ndequant", "aligned\nblocks", "more\nconfigs", "dequant\ncaching",
    "FP16\naccumulate", "Triton 3.6.0\n+ BK=64"
]

THEORETICAL_PEAK = 419.0
BG_COLOR = '#0a0e1a'
GRID_COLOR = '#1a2040'
TEXT_COLOR = '#c0c8e0'
ACCENT_GLOW = '#00e5ff'

# --- Color gradient for dots: red -> gold -> cyan -> green ---
cmap = LinearSegmentedColormap.from_list('autokernel', [
    (0.0, '#ff3333'),
    (0.25, '#ff8800'),
    (0.5, '#ffd700'),
    (0.75, '#00e5ff'),
    (1.0, '#00ff88'),
])
norm_vals = np.linspace(0, 1, len(experiments))
dot_colors = [cmap(v) for v in norm_vals]

# --- Figure ---
fig, ax = plt.subplots(figsize=(19.20, 10.80), dpi=100)
fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Grid
ax.grid(True, axis='y', color=GRID_COLOR, linewidth=0.5, alpha=0.6)
ax.grid(True, axis='x', color=GRID_COLOR, linewidth=0.3, alpha=0.3)

# --- Gradient fill under curve ---
# Draw line segments with gradient color
for i in range(len(experiments) - 1):
    seg_color = cmap(norm_vals[i])
    ax.plot(experiments[i:i+2], tflops[i:i+2], color=seg_color, linewidth=2.5, alpha=0.9, zorder=3)

# Fill under curve with subtle gradient
ax.fill_between(experiments, tflops, alpha=0.08, color='#00e5ff', zorder=1)

# --- Glowing dots ---
for i, (x, y, c) in enumerate(zip(experiments, tflops, dot_colors)):
    # Outer glow
    ax.scatter(x, y, s=220, color=c, alpha=0.25, zorder=4, edgecolors='none')
    # Inner dot
    ax.scatter(x, y, s=70, color=c, alpha=1.0, zorder=5, edgecolors='white', linewidths=0.5)

# --- Theoretical peak line ---
ax.axhline(y=THEORETICAL_PEAK, color='#ff4466', linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)
ax.text(76.5, THEORETICAL_PEAK + 5, f'Theoretical Peak: {THEORETICAL_PEAK} TFLOPS',
        color='#ff4466', fontsize=11, fontweight='bold', va='bottom', ha='right',
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)])

# --- Milestone annotations ---
annotations = {
    0:  ("Baseline\n15.1 TFLOPS", (0, -55)),
    21: ("Autotune +\nL2 Swizzle\n9.1x", (0, 40)),
    32: ("Persistent\nKernel", (15, 35)),
    39: ("Constexpr\nGroup", (-5, 40)),
    60: ("Split Dequant\n+ cuBLAS", (-30, 45)),
    74: ("Dequant\nCaching", (-50, 40)),
    75: ("FP16 Accumulate\nParadigm Shift\n19.2x", (-80, 45)),
    76: ("Triton 3.6.0\n328 TFLOPS\n21.7x", (-15, -75)),
}

for idx, (exp, (label, offset)) in enumerate(annotations.items()):
    i = experiments.index(exp)
    c = dot_colors[i]
    ax.annotate(label, (exp, tflops[i]),
                textcoords='offset points', xytext=offset,
                fontsize=9, fontweight='bold', color=c, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=c, lw=1.2, alpha=0.7),
                path_effects=[pe.withStroke(linewidth=2.5, foreground=BG_COLOR)],
                zorder=6)

# --- Axes styling ---
ax.set_xlim(-3, 80)
ax.set_ylim(0, 450)
ax.set_xlabel('Experiment Number', color=TEXT_COLOR, fontsize=13, fontweight='bold', labelpad=10)
ax.set_ylabel('TFLOPS', color=TEXT_COLOR, fontsize=13, fontweight='bold', labelpad=10)
ax.tick_params(colors=TEXT_COLOR, labelsize=11)
for spine in ax.spines.values():
    spine.set_color(GRID_COLOR)
    spine.set_linewidth(0.5)

# Custom x-ticks at experiment points
ax.set_xticks(experiments)
ax.set_xticklabels([str(e) for e in experiments], fontsize=10)

# Y-ticks every 50
ax.set_yticks(range(0, 451, 50))

# --- Title ---
fig.text(0.5, 0.95, 'AutoKernel: W4A16 Quantized Matmul Optimization',
         ha='center', va='top', fontsize=22, fontweight='bold', color='white',
         path_effects=[pe.withStroke(linewidth=4, foreground=BG_COLOR)],
         fontfamily='sans-serif')

fig.text(0.5, 0.91, 'RTX 5090  —  76 experiments, 21.7x improvement',
         ha='center', va='top', fontsize=14, color=ACCENT_GLOW, alpha=0.85,
         fontfamily='sans-serif',
         path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)])

# --- Performance regions (subtle background bands) ---
regions = [
    (0, 50, '#ff3333', 'Unoptimized', 0.03),
    (50, 150, '#ff8800', 'Basic Tuning', 0.03),
    (150, 250, '#ffd700', 'Advanced Optimization', 0.03),
    (250, 350, '#00e5ff', 'Paradigm Shifts', 0.03),
    (350, 450, '#00ff88', 'Near Peak', 0.02),
]
for ymin, ymax, color, rlabel, alpha in regions:
    ax.axhspan(ymin, ymax, alpha=alpha, color=color, zorder=0)
    ax.text(79, (ymin + ymax) / 2, rlabel, fontsize=8, color=color, alpha=0.5,
            ha='right', va='center', fontstyle='italic',
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_COLOR)])

# --- Improvement arrow on right side ---
ax.annotate('', xy=(78, 328), xytext=(78, 15.1),
            arrowprops=dict(arrowstyle='<->', color='#ffd700', lw=2, alpha=0.5))
ax.text(78.5, 170, '21.7x', fontsize=14, fontweight='bold', color='#ffd700',
        alpha=0.6, rotation=90, ha='left', va='center',
        path_effects=[pe.withStroke(linewidth=3, foreground=BG_COLOR)])

plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.88])

# --- Save ---
out_primary = '/root/projects/autokernel/autokernel_results.png'
os.makedirs(os.path.dirname(out_primary), exist_ok=True)
fig.savefig(out_primary, dpi=100, facecolor=BG_COLOR, edgecolor='none',
            bbox_inches='tight', pad_inches=0.3)
print(f"Saved: {out_primary}")

# Copy to buildify-pitchdeck
out_copy = '/root/projects/remotion/buildify-pitchdeck/public/autokernel_v2_images/results_chart.png'
os.makedirs(os.path.dirname(out_copy), exist_ok=True)
shutil.copy2(out_primary, out_copy)
print(f"Copied: {out_copy}")

plt.close()
print("Done!")
