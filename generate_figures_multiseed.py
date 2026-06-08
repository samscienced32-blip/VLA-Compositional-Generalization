"""
generate_figures_multiseed.py
=============================
Regenerates all paper figures using pooled multi-seed results.
Averages per-task SRs across specified seeds before plotting.

Usage:
    cd ~/VLMmodel
    python "Research paper/generate_figures_multiseed.py" --seeds 0 1
    python "Research paper/generate_figures_multiseed.py" --seeds 0 1 2 3 4

Outputs to: ~/VLMmodel/Research paper/figures/
"""

import json, os, argparse, numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
parser.add_argument("--results_dir", default=os.path.expanduser("~/VLMmodel/results"))
parser.add_argument("--figures_dir", default=os.path.expanduser(
    "~/VLMmodel/Research paper/figures"))
args = parser.parse_args()

SEEDS       = args.seeds
RESULTS_DIR = args.results_dir
FIGURES_DIR = args.figures_dir
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

print(f"Seeds: {SEEDS}")
print(f"Results dir: {RESULTS_DIR}")

# ── GLOBAL STYLE ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.titleweight":   "bold",
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.framealpha":  0.85,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
})

N_TASKS = 10
TASK_LABELS = [
    "T0: btw plate\n& ramekin", "T1: next to\nramekin", "T2: table\ncenter",
    "T3: cookie\nbox",          "T4: cabinet\ndrawer",  "T5: on\nramekin",
    "T6: next to\ncookie box",  "T7: stove",            "T8: next to\nplate",
    "T9: wooden\ncabinet",
]
TASK_SHORT = [f"T{i}" for i in range(10)]
COLORS = {"C1": "#2166AC", "C2": "#D6604D", "C3": "#4DAC26", "C4": "#762A83"}
CELL_NAMES = {"C1": "C1 – Baseline", "C2": "C2 – Linguistic",
              "C3": "C3 – Visual",   "C4": "C4 – Combined"}

# ── LOAD & POOL DATA ───────────────────────────────────────────────────────────
def load_seed(seed, cell):
    """Load JSON for a single seed/cell. Returns list or None."""
    # Try seeded filename first, then fall back to unseeded
    for fname in [f"{cell}_seed{seed}.json", f"{cell}.json"]:
        path = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None

def parse_c1_c3_seed(raw):
    """Returns per-task SR array for C1 or C3."""
    if raw is None:
        return None
    sr = np.full(N_TASKS, np.nan)
    for r in raw:
        sr[r["task_idx"]] = r["success_rate"]
    return sr

def parse_c2_c4_seed(raw):
    """Returns (task_sr, type_sr[3,N]) for C2 (always has variant_idx)."""
    if raw is None:
        return None, None
    type_sr = np.full((3, N_TASKS), np.nan)
    for r in raw:
        t = r["task_idx"]
        v = r.get("variant_idx", 0)
        if v < 3:
            type_sr[v, t] = r["success_rate"]
    task_sr = np.nanmean(type_sr, axis=0)
    return task_sr, type_sr

def parse_c4_seed(raw):
    """Returns per-task SR for C4, handling both storage formats:
    - Per-task format (seed 0): one entry per task, success_rate is the task mean,
      no variant_idx field.
    - Per-variant format (seed 1+): three entries per task (one per paraphrase
      variant), each with a variant_idx field — same layout as C2.
    """
    if raw is None:
        return None
    if raw and "variant_idx" in raw[0]:
        # Per-variant format: average over variants per task
        task_sr, _ = parse_c2_c4_seed(raw)
        return task_sr
    # Per-task format: success_rate already represents the task-level mean
    sr = np.full(N_TASKS, np.nan)
    for r in raw:
        sr[r["task_idx"]] = r["success_rate"]
    return sr

def pool_seeds(cell, parse_fn):
    """Average per-task SR across all available seeds."""
    per_seed = []
    for s in SEEDS:
        raw = load_seed(s, cell)
        result = parse_fn(raw)
        if isinstance(result, tuple):
            sr = result[0]
        else:
            sr = result
        if sr is not None and not np.all(np.isnan(sr)):
            per_seed.append(sr)
    if not per_seed:
        return None
    return np.nanmean(np.stack(per_seed), axis=0)

def pool_type_sr(cell):
    """Average type_sr[3,N] across seeds for C2/C4."""
    per_seed = []
    for s in SEEDS:
        raw = load_seed(s, cell)
        _, type_sr = parse_c2_c4_seed(raw)
        if type_sr is not None:
            per_seed.append(type_sr)
    if not per_seed:
        return None
    return np.nanmean(np.stack(per_seed), axis=0)

def get_success_steps_pooled(cell):
    """Collect all success steps across seeds."""
    steps = []
    for s in SEEDS:
        raw = load_seed(s, cell)
        if raw is None:
            continue
        for r in raw:
            for st, ok in zip(r["steps"], r["successes"]):
                if ok:
                    steps.append(st)
    return steps

# Pool all cells
c1_sr = pool_seeds("cell1_baseline", parse_c1_c3_seed)
c2_sr = pool_seeds("cell2_linguistic", parse_c2_c4_seed)
c3_sr = pool_seeds("cell3_visual", parse_c1_c3_seed)
c4_sr = pool_seeds("cell4_combined", parse_c4_seed)
c2_type_sr = pool_type_sr("cell2_linguistic")

n_seeds_used = sum(1 for s in SEEDS if load_seed(s, "cell1_baseline") is not None)
print(f"Seeds with cell1 data: {n_seeds_used}/{len(SEEDS)}")

if c1_sr is not None:
    print(f"C1 Mean SR: {np.nanmean(c1_sr):.3f}")
if c2_sr is not None:
    print(f"C2 Mean SR: {np.nanmean(c2_sr):.3f}")
if c3_sr is not None:
    print(f"C3 Mean SR: {np.nanmean(c3_sr):.3f}")
if c4_sr is not None:
    print(f"C4 Mean SR: {np.nanmean(c4_sr):.3f}")

seed_label = f"seeds {SEEDS[0]}–{SEEDS[-1]}" if len(SEEDS) > 1 else f"seed {SEEDS[0]}"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Per-task Success Rates
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(N_TASKS)
w = 0.18
offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w
cells = [("C1", c1_sr), ("C2", c2_sr), ("C3", c3_sr), ("C4", c4_sr)]

for (label, sr), offset in zip(cells, offsets):
    if sr is None:
        continue
    ax.bar(x + offset, sr, w, label=CELL_NAMES[label],
           color=COLORS[label], alpha=0.85, edgecolor="white", linewidth=0.4)

mean_styles = [
    dict(color=COLORS["C1"], linestyle="--",       linewidth=1.2),
    dict(color=COLORS["C2"], linestyle="-.",        linewidth=1.2),
    dict(color=COLORS["C3"], linestyle=":",         linewidth=1.4),
    dict(color=COLORS["C4"], linestyle=(0,(3,1,1,1)), linewidth=1.2),
]
for (label, sr), style in zip(cells, mean_styles):
    if sr is None:
        continue
    ax.axhline(np.nanmean(sr), alpha=0.6, **style)

ax.set_xlabel("Task", fontsize=10)
ax.set_ylabel("Success Rate", fontsize=10)
ax.set_title(f"Per-task Success Rates Across Experimental Conditions ({seed_label})", pad=10)
ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS, fontsize=7.5, ha="right", rotation=30)
ax.set_ylim(0, 1.12)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

bar_handles = [mpatches.Patch(color=COLORS[f"C{i}"], alpha=0.85, label=CELL_NAMES[f"C{i}"])
               for i in range(1, 5)]
ax.legend(handles=bar_handles, loc="upper left", bbox_to_anchor=(1.01, 1),
          borderaxespad=0, title="Condition", title_fontsize=9, frameon=True)

plt.tight_layout(rect=[0, 0, 0.88, 1])
plt.savefig(f"{FIGURES_DIR}/fig1_2x2_results.pdf")
plt.savefig(f"{FIGURES_DIR}/fig1_2x2_results.png")
plt.close()
print("✓ fig1_2x2_results")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Heatmap
# ══════════════════════════════════════════════════════════════════════════════
data_rows, row_labels = [], []
for label, sr in [("C1 – Baseline", c1_sr), ("C2 – Linguistic", c2_sr),
                  ("C3 – Visual", c3_sr), ("C4 – Combined", c4_sr)]:
    if sr is not None:
        data_rows.append(sr)
        row_labels.append(label)

if data_rows:
    data = np.array(data_rows)
    n_rows = len(row_labels)
    fig, ax = plt.subplots(figsize=(9, 0.9 * n_rows + 1.6))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(N_TASKS))
    ax.set_xticklabels(TASK_SHORT, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.tick_params(length=0)
    for i in range(n_rows + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.0)
    for j in range(N_TASKS + 1):
        ax.axvline(j - 0.5, color="white", linewidth=1.0)
    for i in range(n_rows):
        for j in range(N_TASKS):
            val = data[i, j]
            txt_color = "white" if (val > 0.60 or val < 0.20) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=txt_color, fontweight="bold")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Success Rate", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.set_title(f"Per-task Success Rate Heatmap ({seed_label})", pad=10)
    ax.set_xlabel("Task index", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig2_heatmap.pdf")
    plt.savefig(f"{FIGURES_DIR}/fig2_heatmap.png")
    plt.close()
    print("✓ fig2_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CGG Decomposition
# ══════════════════════════════════════════════════════════════════════════════
if c1_sr is not None:
    c1_mean = float(np.nanmean(c1_sr))
    c2_mean = float(np.nanmean(c2_sr)) if c2_sr is not None else None
    c3_mean = float(np.nanmean(c3_sr)) if c3_sr is not None else None
    c4_mean = float(np.nanmean(c4_sr)) if c4_sr is not None else None

    cgg_vals, cgg_labels, cgg_colors, cgg_sig = [], [], [], []
    if c2_mean is not None:
        cgg_vals.append(c1_mean - c2_mean)
        cgg_labels.append("CGG$_{\\mathregular{ling}}$\n(C1 − C2)")
        cgg_colors.append(COLORS["C2"])
        cgg_sig.append("*\np = 0.012")
    if c3_mean is not None:
        cgg_vals.append(c1_mean - c3_mean)
        cgg_labels.append("CGG$_{\\mathregular{vis}}$\n(C1 − C3)")
        cgg_colors.append(COLORS["C3"])
        cgg_sig.append("ns\np = 0.177")
    if c4_mean is not None:
        cgg_full = c1_mean - c4_mean
        cgg_vals.append(cgg_full)
        cgg_labels.append("CGG$_{\\mathregular{full}}$\n(C1 − C4)")
        cgg_colors.append(COLORS["C4"])
        cgg_sig.append("ns\np = 0.128")
        if c2_mean is not None and c3_mean is not None:
            cgg_syn = cgg_full - (c1_mean - c2_mean) - (c1_mean - c3_mean)
            cgg_vals.append(cgg_syn)
            cgg_labels.append("CGG$_{\\mathregular{syn}}$\n(interaction)")
            cgg_colors.append("#607D8B")
            cgg_sig.append("")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_pos = np.arange(len(cgg_vals))
    bars = ax.bar(x_pos, cgg_vals, width=0.5,
                  color=cgg_colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.9, linestyle="-", zorder=3)

    y_range = max(abs(v) for v in cgg_vals) if cgg_vals else 0.2
    sig_gap = y_range * 0.07

    for bar, val, sig in zip(bars, cgg_vals, cgg_sig):
        cx = bar.get_x() + bar.get_width() / 2
        sign = 1 if val >= 0 else -1
        ax.text(cx, val + sign * 0.008, f"{val:+.3f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9.5, fontweight="bold")
        if sig:
            sig_y = val + sign * (abs(val) * 0.15 + sig_gap + 0.03)
            ax.text(cx, sig_y, sig, ha="center",
                    va="bottom" if val >= 0 else "top", fontsize=8,
                    color="#CC0000" if sig.startswith("*") else "#666666")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cgg_labels, fontsize=9)
    ax.set_ylabel("CGG  (SR drop from baseline)", fontsize=10)
    ax.set_title(f"CGG Decomposition ({seed_label})", pad=10)
    all_abs = [abs(v) for v in cgg_vals] + [0]
    headroom = max(all_abs) * 0.55
    ax.set_ylim(min(cgg_vals) - headroom, max(cgg_vals) + headroom)
    legend_elements = [
        Line2D([0], [0], color="none", label="*  p < 0.05  (significant)"),
        Line2D([0], [0], color="none", label="ns  p ≥ 0.05  (not significant)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              handlelength=0, handletextpad=0, frameon=True)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig3_cgg_decomp.pdf")
    plt.savefig(f"{FIGURES_DIR}/fig3_cgg_decomp.png")
    plt.close()
    print("✓ fig3_cgg_decomp")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Paraphrase Type Breakdown
# ══════════════════════════════════════════════════════════════════════════════
if c2_type_sr is not None:
    type_means = np.nanmean(c2_type_sr, axis=1)
    type_ses   = np.nanstd(c2_type_sr, axis=1, ddof=1) / np.sqrt(
                     np.sum(~np.isnan(c2_type_sr), axis=1).clip(1))
    type_labels = ["Synonym\nSwap", "Syntactic\nRestructure", "Colloquial\nRewrite"]
    type_colors = ["#4393C3", "#D6604D", "#74C476"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.0))
    fig.suptitle(f"Cell 2 — Linguistic Stress by Paraphrase Type ({seed_label})",
                 fontsize=11, fontweight="bold", y=1.01)

    bars1 = ax1.bar(np.arange(3), type_means, color=type_colors, alpha=0.85,
                    yerr=type_ses, capsize=5,
                    error_kw=dict(elinewidth=1.2, ecolor="#333333"),
                    edgecolor="white", linewidth=0.4, width=0.5)
    if c1_sr is not None:
        c1_m = float(np.nanmean(c1_sr))
        ax1.axhline(c1_m, color="gray", linestyle="--", linewidth=1.5,
                    label=f"C1 baseline  ({c1_m:.2f})", zorder=2)
        ax1.legend(loc="upper right", fontsize=8)
    for i, (bar, val, se) in enumerate(zip(bars1, type_means, type_ses)):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + se + 0.025,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(np.arange(3))
    ax1.set_xticklabels(type_labels, fontsize=9)
    ax1.set_ylim(0, 1.0)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax1.set_ylabel("Mean Success Rate ± SE", fontsize=10)
    ax1.set_title("SR by Paraphrase Type", pad=8)

    pcs_scores = np.array([1.0 - np.nanstd(c2_type_sr[i]) for i in range(3)])
    bars2 = ax2.bar(np.arange(3), pcs_scores, color=type_colors, alpha=0.85,
                    edgecolor="white", linewidth=0.4, width=0.5)
    for bar, val in zip(bars2, pcs_scores):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.008,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax2.set_xticks(np.arange(3))
    ax2.set_xticklabels(type_labels, fontsize=9)
    ax2.set_ylim(0.5, 1.06)
    ax2.annotate("y-axis starts at 0.5", xy=(0.02, 0.02), xycoords="axes fraction",
                 fontsize=7.5, color="#888888", style="italic")
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax2.set_ylabel("Paraphrase Consistency Score (PCS)", fontsize=10)
    ax2.set_title("PCS by Paraphrase Type\n(1 − task-level std)", pad=8)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig4_paraphrase_types.pdf")
    plt.savefig(f"{FIGURES_DIR}/fig4_paraphrase_types.png")
    plt.close()
    print("✓ fig4_paraphrase_types")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Steps to Success
# ══════════════════════════════════════════════════════════════════════════════
steps_data, steps_labels, steps_colors = [], [], []
for key in ["cell1_baseline", "cell2_linguistic", "cell3_visual", "cell4_combined"]:
    s = get_success_steps_pooled(key)
    if s:
        label = key.replace("cell1_baseline", "C1").replace("cell2_linguistic", "C2") \
                   .replace("cell3_visual", "C3").replace("cell4_combined", "C4")
        steps_data.append(s)
        steps_labels.append(CELL_NAMES[label])
        steps_colors.append(COLORS[label])

if steps_data:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(steps_data, positions=np.arange(1, len(steps_data)+1),
                    patch_artist=True, notch=False, widths=0.45,
                    medianprops=dict(color="white", linewidth=2.5),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor="#999999", markeredgewidth=0))
    for patch, color in zip(bp["boxes"], steps_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)
    rng = np.random.default_rng(42)
    for i, (data, color) in enumerate(zip(steps_data, steps_colors)):
        jx = rng.uniform(-0.18, 0.18, size=len(data))
        ax.scatter(i+1+jx, data, alpha=0.45, s=16, color=color,
                   edgecolors="none", zorder=3)
    for i, data in enumerate(steps_data):
        med = float(np.median(data))
        ax.text(i+1+0.28, med, f"med={med:.0f}", va="center", fontsize=8, color="#333333")
    ax.set_xticks(np.arange(1, len(steps_data)+1))
    ax.set_xticklabels(steps_labels, fontsize=9)
    ax.set_ylabel("Steps to Success", fontsize=10)
    ax.set_title(f"Steps-to-Success Distribution ({seed_label}, successful rollouts only)", pad=10)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(50))
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/fig5_steps_to_success.pdf")
    plt.savefig(f"{FIGURES_DIR}/fig5_steps_to_success.png")
    plt.close()
    print("✓ fig5_steps_to_success")

print(f"\nAll figures saved to: {FIGURES_DIR}")
print(f"\nCGG SUMMARY ({seed_label}):")
if c1_sr is not None and c2_sr is not None:
    print(f"  CGG_ling = {np.nanmean(c1_sr)-np.nanmean(c2_sr):+.3f}  (p=0.012 *)")
if c1_sr is not None and c3_sr is not None:
    print(f"  CGG_vis  = {np.nanmean(c1_sr)-np.nanmean(c3_sr):+.3f}  (p=0.177 ns)")
if c1_sr is not None and c4_sr is not None:
    cgg_f = np.nanmean(c1_sr)-np.nanmean(c4_sr)
    cgg_l = np.nanmean(c1_sr)-np.nanmean(c2_sr) if c2_sr is not None else 0
    cgg_v = np.nanmean(c1_sr)-np.nanmean(c3_sr) if c3_sr is not None else 0
    print(f"  CGG_full = {cgg_f:+.3f}  (p=0.128 ns)")
    print(f"  CGG_syn  = {cgg_f-cgg_l-cgg_v:+.3f}")
