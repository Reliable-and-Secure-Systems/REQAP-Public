"""
generate_mac_chart.py
IEEE single-column ready bar chart (3.5 in wide, 300 DPI, 8pt serif fonts).
Compares Hardware-Issued MAC count Without vs With MQF-Aware Packing.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

# ── IEEE Font/Style Setup ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8.5,
    "axes.labelsize":   8,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  7.5,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
})

# ── Data ──────────────────────────────────────────────────────────────────────
models  = ["AlexNet", "VGG-11", "ResNet-18"]
no_pack = [2_764_725,   941_321,   400_064]
packed  = [2_586_229,   415_225,   228_376]

reductions = [f"-{((a-b)/a*100):.1f}%" for a, b in zip(no_pack, packed)]

# ── Figure — IEEE single column = 3.5 inches wide ────────────────────────────
fig, ax = plt.subplots(figsize=(3.5, 2.55))
fig.patch.set_facecolor("white")

x     = np.arange(len(models))
width = 0.30

COLOR_BLUE  = "#1F77B4"
COLOR_GREEN = "#2CA02C"

bars1 = ax.bar(x - width/2, no_pack, width,
               color=COLOR_BLUE, edgecolor="black", linewidth=0.5,
               label="Without Packing")
bars2 = ax.bar(x + width/2, packed,  width,
               color=COLOR_GREEN, edgecolor="black", linewidth=0.5,
               label="With Packing")

# Reduction labels above bar pairs
for bb, bp, pct in zip(bars1, bars2, reductions):
    mx = (bb.get_x() + bb.get_width() + bp.get_x()) / 2
    ty = max(bb.get_height(), bp.get_height()) * 1.03
    ax.text(mx, ty, pct, ha="center", va="bottom",
            fontsize=6.5, fontweight="bold", color="#B22222")

# Axes
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("No. of Multiplications (MAC)")
ax.set_xlabel("Neural Network Architecture")
ax.set_title(
    "MAC Count: Without Packing vs. With Packing\n"
    "(MQF Granular Quantization, R = 32-bit)",
    fontweight="bold", pad=5
)
ax.set_ylim(0, max(no_pack) * 1.26)
ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"
    )
)
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.set_axisbelow(True)

# Legend
patches = [
    mpatches.Patch(facecolor=COLOR_BLUE,  edgecolor="black", label="Without Packing"),
    mpatches.Patch(facecolor=COLOR_GREEN, edgecolor="black", label="With Packing"),
]
ax.legend(handles=patches, loc="upper right",
          framealpha=0.9, edgecolor="grey",
          handlelength=1.0, borderpad=0.4)

plt.tight_layout(pad=0.4)

out_path = (r"c:\Mubashir-BTU\Thesis\Codes\Danial\Prune_2"
            r"\systolic_sim\output\mac_comparison_chart.png")
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Chart saved → {out_path}")
