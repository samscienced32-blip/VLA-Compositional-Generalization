"""
VLA COMPOSITIONALITY — ANALYSIS SCRIPT
=======================================
Project: Disentangling Linguistic vs. Visual Compositionality Failures in VLAs
Team: HandofGod | Sagar Kumar | BITS Pilani | CAISc 2026

Run AFTER run_experiment.py completes:
  python analyze_results.py

Outputs:
  figures/fig1_2x2_results.pdf        — grouped bar chart, 4 conditions
  figures/fig2_heatmap.pdf            — per-task success-rate heatmap
  figures/fig3_cgg_decomp.pdf         — CGG decomposition (main + synergistic)
  figures/fig4_paraphrase_types.pdf   — NEW: SR by paraphrase type (synonym/restructure/colloquial)
  figures/fig5_steps_to_success.pdf   — NEW: steps-to-success distribution across conditions
  results/summary.json                — all numbers for the paper
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

RESULTS_DIR = os.path.expanduser("~/VLMmodel/results")
FIGURES_DIR = os.path.expanduser("~/VLMmodel/figures")
Path(FIGURES_DIR).mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family"    : "serif",
    "font.size"      : 11,
    "axes.titlesize" : 13,
    "axes.labelsize" : 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi"     : 150,
    "savefig.dpi"    : 300,
    "savefig.bbox"   : "tight",
})

COLORS = {
    "C1": "#2196F3", "C2": "#FF9800", "C3": "#F44336", "C4": "#9C27B0",
    "synonym"    : "#4CAF50",
    "restructure": "#FF9800",
    "colloquial" : "#F44336",
    "cgg_ling"   : "#FF9800",
    "cgg_vis"    : "#F44336",
    "cgg_full"   : "#9C27B0",
    "cgg_syn"    : "#4CAF50",
}

PARA_TYPE_LABELS = {0: "synonym_swap", 1: "restructure", 2: "colloquial"}
PARA_DISPLAY     = {"synonym_swap": "Synonym Swap", "restructure": "Restructure", "colloquial": "Colloquial"}

# ── LOAD ──────────────────────────────────────────────────────
def load(fname):
    path = f"{RESULTS_DIR}/{fname}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}  — run run_experiment.py first")
    with open(path) as f:
        return json.load(f)

print("Loading results...")
c1_raw = load("cell1_baseline.json")
c2_raw = load("cell2_linguistic.json")
c3_raw = load("cell3_visual.json")
c4_raw = load("cell4_combined.json")

NUM_TASKS = 10

# ── AGGREGATE PER-TASK SR ─────────────────────────────────────
def per_task_sr(raw, num_tasks=NUM_TASKS):
    sr     = np.zeros(num_tasks)
    counts = np.zeros(num_tasks, dtype=int)
    for r in raw:
        sr[r["task_idx"]]     += r["success_rate"]
        counts[r["task_idx"]] += 1
    return sr / np.maximum(counts, 1)

c1_sr = per_task_sr(c1_raw)
c2_sr = per_task_sr(c2_raw)
c3_sr = per_task_sr(c3_raw)
c4_sr = per_task_sr(c4_raw)

# ── CGG METRICS ───────────────────────────────────────────────
cgg_ling_per = c1_sr - c2_sr
cgg_vis_per  = c1_sr - c3_sr
cgg_full_per = c1_sr - c4_sr
cgg_syn_per  = cgg_full_per - cgg_ling_per - cgg_vis_per

c1_mean  = float(np.mean(c1_sr))
c2_mean  = float(np.mean(c2_sr))
c3_mean  = float(np.mean(c3_sr))
c4_mean  = float(np.mean(c4_sr))
cgg_ling = float(np.mean(cgg_ling_per))
cgg_vis  = float(np.mean(cgg_vis_per))
cgg_full = float(np.mean(cgg_full_per))
cgg_syn  = float(np.mean(cgg_syn_per))

print(f"\n{'='*50}")
print(f"  Condition    Mean SR    Std")
print(f"  C1 baseline  {c1_mean:.3f}      {np.std(c1_sr):.3f}")
print(f"  C2 ling      {c2_mean:.3f}      {np.std(c2_sr):.3f}")
print(f"  C3 vis       {c3_mean:.3f}      {np.std(c3_sr):.3f}")
print(f"  C4 both      {c4_mean:.3f}      {np.std(c4_sr):.3f}")
print(f"\n  CGG_linguistic  = {cgg_ling:+.3f}")
print(f"  CGG_visual      = {cgg_vis:+.3f}")
print(f"  CGG_full        = {cgg_full:+.3f}")
print(f"  CGG_synergistic = {cgg_syn:+.3f}")

if cgg_syn > 0.05:
    syn_interp = "super-additive failure — modalities interact, joint stress is worse than sum of parts"
elif cgg_syn < -0.05:
    syn_interp = "sub-additive failure — some cross-modal compensation occurs"
else:
    syn_interp = "approximately additive failure — linguistic and visual failures are independent"
print(f"  → {syn_interp}")

# ── PCS — PARAPHRASE CONSISTENCY SCORE ────────────────────────
variant_srs = {i: [] for i in range(NUM_TASKS)}
for r in c2_raw:
    variant_srs[r["task_idx"]].append(r["success_rate"])
pcs_per_task = np.array([1.0 - np.std(variant_srs[i]) if len(variant_srs[i]) > 1 else 1.0
                          for i in range(NUM_TASKS)])
pcs_mean = float(np.mean(pcs_per_task))
print(f"\n  PCS (Paraphrase Consistency Score) = {pcs_mean:.3f}")
print(f"  (1.0 = perfectly consistent; 0.0 = completely inconsistent)")

# ── NEW: TYPE-SPECIFIC PARAPHRASE BREAKDOWN ───────────────────
# Group C2 results by paraphrase type across all tasks
type_sr = {"synonym_swap": [], "restructure": [], "colloquial": []}
for r in c2_raw:
    ptype = r.get("para_type", PARA_TYPE_LABELS.get(r.get("variant_idx", 0), "unknown"))
    if ptype in type_sr:
        type_sr[ptype].append(r["success_rate"])

type_means = {t: float(np.mean(v)) if v else 0.0 for t, v in type_sr.items()}
type_stds  = {t: float(np.std(v))  if v else 0.0 for t, v in type_sr.items()}

print(f"\n  Paraphrase Type Breakdown (NEW — not in any prior work):")
for t, m in type_means.items():
    drop = c1_mean - m
    print(f"    {PARA_DISPLAY[t]:15s}: SR={m:.3f}  drop={drop:+.3f}  std={type_stds[t]:.3f}")

# Compute PCS per type relative to baseline
pcs_by_type = {t: 1.0 - type_stds[t] for t in type_sr}
print(f"\n  PCS per paraphrase type:")
for t, p in pcs_by_type.items():
    print(f"    {PARA_DISPLAY[t]:15s}: PCS={p:.3f}")

# ── NEW: STEPS-TO-SUCCESS ANALYSIS ────────────────────────────
# For each condition, compute mean steps for SUCCESS and FAILURE separately
def steps_analysis(raw, label):
    succ_steps = []
    fail_steps = []
    for r in raw:
        steps = r.get("steps", [])
        succs = r["successes"]
        for s, st in zip(succs, steps):
            if s:
                succ_steps.append(st)
            else:
                fail_steps.append(st)
    mean_s = float(np.mean(succ_steps)) if succ_steps else float("nan")
    mean_f = float(np.mean(fail_steps)) if fail_steps else float("nan")
    print(f"  {label:15s}: success→{mean_s:.0f} steps  |  failure→{mean_f:.0f} steps")
    return succ_steps, fail_steps

print(f"\n  Steps-to-Success / Steps-at-Failure (NEW diagnostic):")
c1_ss, c1_fs = steps_analysis(c1_raw, "C1 baseline")
c2_ss, c2_fs = steps_analysis(c2_raw, "C2 ling")
c3_ss, c3_fs = steps_analysis(c3_raw, "C3 vis")
c4_ss, c4_fs = steps_analysis(c4_raw, "C4 both")

# ── STATISTICAL TESTS ─────────────────────────────────────────
print(f"\n{'='*50}")
print("  Wilcoxon signed-rank tests (per-task SR):")

def wilcoxon_report(label, a, b):
    try:
        stat, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"    C1 vs {label}: W={stat:.1f}, p={p:.4f} {sig}")
        return float(stat), float(p)
    except Exception as e:
        print(f"    C1 vs {label}: test failed ({e})")
        return None, None

w_ling_stat, w_ling_p = wilcoxon_report("C2(ling)", c1_sr, c2_sr)
w_vis_stat,  w_vis_p  = wilcoxon_report("C3(vis)",  c1_sr, c3_sr)
w_full_stat, w_full_p = wilcoxon_report("C4(full)", c1_sr, c4_sr)

# ── FIGURE 1: 2×2 grouped bar chart ───────────────────────────
print("\nGenerating figures...")
fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(NUM_TASKS)
width = 0.2
ax.bar(x - 1.5*width, c1_sr, width, label="C1: Baseline",        color=COLORS["C1"], alpha=0.9)
ax.bar(x - 0.5*width, c2_sr, width, label="C2: Ling. stress",    color=COLORS["C2"], alpha=0.9)
ax.bar(x + 0.5*width, c3_sr, width, label="C3: Vis. stress",     color=COLORS["C3"], alpha=0.9)
ax.bar(x + 1.5*width, c4_sr, width, label="C4: Full novel",      color=COLORS["C4"], alpha=0.9)
ax.set_xlabel("Task Index"); ax.set_ylabel("Success Rate")
ax.set_title("Success Rate Across 2×2 Factorial Conditions\n(LIBERO-Spatial, OpenVLA-OFT)")
ax.set_xticks(x); ax.set_xticklabels([str(i) for i in range(NUM_TASKS)])
ax.set_ylim(0, 1.15); ax.legend(loc="upper right", fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
for val, col in zip([c1_mean, c2_mean, c3_mean, c4_mean],
                     [COLORS["C1"], COLORS["C2"], COLORS["C3"], COLORS["C4"]]):
    ax.axhline(val, linestyle="--", alpha=0.35, color=col, linewidth=1)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_2x2_results.pdf")
plt.savefig(f"{FIGURES_DIR}/fig1_2x2_results.png")
print("  ✅ fig1_2x2_results.pdf")
plt.close()

# ── FIGURE 2: Heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
data_matrix = np.vstack([c1_sr, c2_sr, c3_sr, c4_sr])
im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_yticks([0,1,2,3])
ax.set_yticklabels(["C1: Baseline","C2: Ling. stress","C3: Vis. stress","C4: Full novel"])
ax.set_xticks(range(NUM_TASKS)); ax.set_xticklabels([str(i) for i in range(NUM_TASKS)])
ax.set_xlabel("Task Index")
ax.set_title("Per-Task Success Rate Heatmap")
for i in range(4):
    for j in range(NUM_TASKS):
        val = data_matrix[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=8, color="black" if 0.3 < val < 0.8 else "white")
plt.colorbar(im, ax=ax, label="Success Rate")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_heatmap.pdf")
plt.savefig(f"{FIGURES_DIR}/fig2_heatmap.png")
print("  ✅ fig2_heatmap.pdf")
plt.close()

# ── FIGURE 3: CGG Decomposition ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
cgg_labels = ["CGG\nLinguistic", "CGG\nVisual", "CGG\nFull", "CGG\nSynergistic"]
cgg_vals   = [cgg_ling, cgg_vis, cgg_full, cgg_syn]
cgg_colors = [COLORS["cgg_ling"], COLORS["cgg_vis"], COLORS["cgg_full"], COLORS["cgg_syn"]]
cgg_std    = [np.std(cgg_ling_per), np.std(cgg_vis_per),
              np.std(cgg_full_per), np.std(cgg_syn_per)]
bars = ax.bar(cgg_labels, cgg_vals, color=cgg_colors, alpha=0.85,
              yerr=cgg_std, capsize=5, edgecolor="black", linewidth=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("CGG Score (C1 SR − Condition SR)")
ax.set_title("Compositional Generalization Gap Decomposition\n"
             "(positive = drop vs baseline; CGG_syn tests interaction)")
ax.spines[["top","right"]].set_visible(False)
for bar, val, std in zip(bars, cgg_vals, cgg_std):
    ypos = val + std + 0.01 if val >= 0 else val - std - 0.025
    ax.text(bar.get_x() + bar.get_width()/2., ypos, f"{val:+.3f}",
            ha="center", va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_cgg_decomp.pdf")
plt.savefig(f"{FIGURES_DIR}/fig3_cgg_decomp.png")
print("  ✅ fig3_cgg_decomp.pdf")
plt.close()

# ── FIGURE 4: Paraphrase Type Breakdown (NEW) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: SR by paraphrase type vs baseline
ax = axes[0]
type_names   = ["Baseline\n(C1)", "Synonym\nSwap", "Restructure", "Colloquial"]
type_vals    = [c1_mean, type_means["synonym_swap"], type_means["restructure"], type_means["colloquial"]]
type_stds_v  = [np.std(c1_sr), type_stds["synonym_swap"], type_stds["restructure"], type_stds["colloquial"]]
bar_colors   = [COLORS["C1"], COLORS["synonym"], COLORS["restructure"], COLORS["colloquial"]]
bars2 = ax.bar(type_names, type_vals, color=bar_colors, alpha=0.85,
               yerr=type_stds_v, capsize=5, edgecolor="black", linewidth=0.7)
ax.set_ylim(0, 1.1); ax.set_ylabel("Mean Success Rate")
ax.set_title("Success Rate by Paraphrase Type\n(addresses evaluation gap in prior work)")
ax.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars2, type_vals):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.02, f"{val:.2f}",
            ha="center", fontsize=10)

# Right: PCS per type
ax2 = axes[1]
pcs_types  = [PARA_DISPLAY[t] for t in ["synonym_swap", "restructure", "colloquial"]]
pcs_vals_t = [pcs_by_type[t] for t in ["synonym_swap", "restructure", "colloquial"]]
pcs_colors = [COLORS["synonym"], COLORS["restructure"], COLORS["colloquial"]]
bars3 = ax2.bar(pcs_types, pcs_vals_t, color=pcs_colors, alpha=0.85,
                edgecolor="black", linewidth=0.7)
ax2.axhline(pcs_mean, color="black", linestyle="--", linewidth=1, label=f"Overall PCS={pcs_mean:.2f}")
ax2.set_ylim(0, 1.1); ax2.set_ylabel("PCS (1 − σ of SR across variants)")
ax2.set_title("Paraphrase Consistency Score (PCS)\nby Paraphrase Type")
ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)
for bar, val in zip(bars3, pcs_vals_t):
    ax2.text(bar.get_x() + bar.get_width()/2., val + 0.02, f"{val:.2f}",
             ha="center", fontsize=10)

plt.suptitle("Novel Analysis: Type-Stratified Paraphrase Evaluation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_paraphrase_types.pdf")
plt.savefig(f"{FIGURES_DIR}/fig4_paraphrase_types.png")
print("  ✅ fig4_paraphrase_types.pdf  ← NEW: type-stratified paraphrase analysis")
plt.close()

# ── FIGURE 5: Steps-to-Success Distribution (NEW) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Left: violin/box plot of steps for successes across conditions
ax = axes[0]
success_steps_all = [c1_ss, c2_ss, c3_ss, c4_ss]
labels_all        = ["C1\nBaseline", "C2\nLing.", "C3\nVis.", "C4\nBoth"]
cond_colors       = [COLORS["C1"], COLORS["C2"], COLORS["C3"], COLORS["C4"]]
valid_data  = [(d, l, c) for d, l, c in zip(success_steps_all, labels_all, cond_colors) if d]
if valid_data:
    data_v, labels_v, colors_v = zip(*valid_data)
    bp = ax.boxplot(data_v, labels=labels_v, patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], colors_v):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("Steps to Success")
    ax.set_title("Steps to Task Completion\n(successful rollouts only)")
    ax.spines[["top","right"]].set_visible(False)
else:
    ax.text(0.5, 0.5, "No successful rollouts\nto display",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title("Steps to Success (no data)")

# Right: mean steps for success vs failure per condition
ax2 = axes[1]
cond_labels  = ["C1", "C2", "C3", "C4"]
mean_s_steps = [np.mean(x) if x else 0 for x in [c1_ss, c2_ss, c3_ss, c4_ss]]
mean_f_steps = [np.mean(x) if x else 0 for x in [c1_fs, c2_fs, c3_fs, c4_fs]]
x2 = np.arange(4)
w2 = 0.35
ax2.bar(x2 - w2/2, mean_s_steps, w2, label="Success",
        color=[COLORS[c] for c in ["C1","C2","C3","C4"]], alpha=0.85, edgecolor="black")
ax2.bar(x2 + w2/2, mean_f_steps, w2, label="Failure",
        color=[COLORS[c] for c in ["C1","C2","C3","C4"]], alpha=0.4,
        edgecolor="black", hatch="//")
ax2.set_xticks(x2); ax2.set_xticklabels(cond_labels)
ax2.set_ylabel("Mean Steps"); ax2.set_ylim(0, 320)
ax2.axhline(300, color="red", linestyle="--", alpha=0.5, linewidth=1, label="Max steps (300)")
ax2.set_title("Mean Steps: Success vs Failure\n(new graded diagnostic beyond binary SR)")
ax2.legend(fontsize=9); ax2.spines[["top","right"]].set_visible(False)

plt.suptitle("Novel Analysis: Graded Effort Metric (Steps-to-Success)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_steps_to_success.pdf")
plt.savefig(f"{FIGURES_DIR}/fig5_steps_to_success.png")
print("  ✅ fig5_steps_to_success.pdf  ← NEW: graded diagnostic beyond binary success")
plt.close()

# ── SUMMARY JSON ─────────────────────────────────────────────
summary = {
    "project"   : "VLA Compositionality Failure Disentanglement",
    "model"     : "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "benchmark" : "LIBERO-Spatial",
    "seed"      : 42,
    "num_tasks" : NUM_TASKS,
    "conditions": {
        "C1_baseline" : {"mean_sr": c1_mean, "std_sr": float(np.std(c1_sr)), "per_task": c1_sr.tolist()},
        "C2_linguistic": {"mean_sr": c2_mean, "std_sr": float(np.std(c2_sr)), "per_task": c2_sr.tolist()},
        "C3_visual"   : {"mean_sr": c3_mean, "std_sr": float(np.std(c3_sr)), "per_task": c3_sr.tolist()},
        "C4_combined" : {"mean_sr": c4_mean, "std_sr": float(np.std(c4_sr)), "per_task": c4_sr.tolist()},
    },
    "cgg": {
        "CGG_linguistic" : {"mean": cgg_ling, "std": float(np.std(cgg_ling_per)), "per_task": cgg_ling_per.tolist()},
        "CGG_visual"     : {"mean": cgg_vis,  "std": float(np.std(cgg_vis_per)),  "per_task": cgg_vis_per.tolist()},
        "CGG_full"       : {"mean": cgg_full, "std": float(np.std(cgg_full_per)), "per_task": cgg_full_per.tolist()},
        "CGG_synergistic": {"mean": cgg_syn,  "std": float(np.std(cgg_syn_per)),  "per_task": cgg_syn_per.tolist(),
                            "interpretation": syn_interp},
    },
    "pcs": {
        "overall_mean": pcs_mean,
        "per_task"    : pcs_per_task.tolist(),
        "by_type"     : pcs_by_type,
    },
    "paraphrase_type_breakdown": {
        t: {"mean_sr": type_means[t], "std_sr": type_stds[t],
            "drop_from_baseline": c1_mean - type_means[t]}
        for t in ["synonym_swap", "restructure", "colloquial"]
    },
    "steps_analysis": {
        "C1": {"mean_success_steps": float(np.mean(c1_ss)) if c1_ss else None,
               "mean_failure_steps": float(np.mean(c1_fs)) if c1_fs else None},
        "C2": {"mean_success_steps": float(np.mean(c2_ss)) if c2_ss else None,
               "mean_failure_steps": float(np.mean(c2_fs)) if c2_fs else None},
        "C3": {"mean_success_steps": float(np.mean(c3_ss)) if c3_ss else None,
               "mean_failure_steps": float(np.mean(c3_fs)) if c3_fs else None},
        "C4": {"mean_success_steps": float(np.mean(c4_ss)) if c4_ss else None,
               "mean_failure_steps": float(np.mean(c4_fs)) if c4_fs else None},
    },
    "statistical_tests": {
        "C1_vs_C2_wilcoxon": {"statistic": w_ling_stat, "p_value": w_ling_p},
        "C1_vs_C3_wilcoxon": {"statistic": w_vis_stat,  "p_value": w_vis_p},
        "C1_vs_C4_wilcoxon": {"statistic": w_full_stat, "p_value": w_full_p},
    },
}

summary_path = f"{RESULTS_DIR}/summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"  ✅ summary.json → {summary_path}")

# ── PAPER-READY NUMBERS ───────────────────────────────────────
print(f"\n{'='*60}")
print("PAPER-READY NUMBERS")
print(f"{'='*60}")
print(f"""
Table 1 — Condition success rates:
  C1 (Seen/Seen):    {c1_mean:.2f} ± {np.std(c1_sr):.2f}
  C2 (Seen/Novel):   {c2_mean:.2f} ± {np.std(c2_sr):.2f}
  C3 (Novel/Seen):   {c3_mean:.2f} ± {np.std(c3_sr):.2f}
  C4 (Novel/Novel):  {c4_mean:.2f} ± {np.std(c4_sr):.2f}

Table 2 — CGG Decomposition:
  CGG_ling  = {cgg_ling:+.3f} ± {np.std(cgg_ling_per):.3f}  (p={f"{w_ling_p:.4f}" if w_ling_p else "N/A"})
  CGG_vis   = {cgg_vis:+.3f} ± {np.std(cgg_vis_per):.3f}  (p={f"{w_vis_p:.4f}"  if w_vis_p  else "N/A"})
  CGG_full  = {cgg_full:+.3f} ± {np.std(cgg_full_per):.3f}  (p={f"{w_full_p:.4f}" if w_full_p else "N/A"})
  CGG_syn   = {cgg_syn:+.3f} ± {np.std(cgg_syn_per):.3f}
  → {syn_interp}

Table 3 — Paraphrase type breakdown (novel contribution):
  Synonym swap:  SR={type_means['synonym_swap']:.2f}  drop={c1_mean-type_means['synonym_swap']:+.3f}  PCS={pcs_by_type['synonym_swap']:.2f}
  Restructure:   SR={type_means['restructure']:.2f}  drop={c1_mean-type_means['restructure']:+.3f}  PCS={pcs_by_type['restructure']:.2f}
  Colloquial:    SR={type_means['colloquial']:.2f}  drop={c1_mean-type_means['colloquial']:+.3f}  PCS={pcs_by_type['colloquial']:.2f}

Overall PCS = {pcs_mean:.3f}
""")

print("✅ ANALYSIS COMPLETE")
print(f"   5 figures → {FIGURES_DIR}/")
print(f"   Numbers   → {RESULTS_DIR}/summary.json")
