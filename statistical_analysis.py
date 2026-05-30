"""
statistical_analysis.py
========================
Computes all Wilcoxon signed-rank tests and CGG decomposition reported in the paper.
Outputs a clean summary to stdout and saves a CSV of per-task results.

Usage:
    python "Research paper/statistical_analysis.py"
"""

import numpy as np
from scipy import stats

# ── DATA ───────────────────────────────────────────────────────────────────────
c1 = np.array([0.50, 0.00, 0.40, 0.90, 0.10, 0.10, 0.70, 0.50, 0.30, 0.10])
c2 = np.array([0.37, 0.00, 0.50, 0.50, 0.10, 0.07, 0.37, 0.17, 0.13, 0.07])
c3 = np.array([0.30, 0.00, 0.20, 0.20, 0.00, 0.10, 0.50, 0.50, 0.60, 0.00])
c4 = np.array([0.60, 0.00, 0.30, 0.20, 0.00, 0.20, 0.20, 0.30, 0.30, 0.10])

# C2 per-type SR
c2_synonym     = np.array([0.50, 0.00, 0.70, 0.80, 0.10, 0.10, 0.60, 0.30, 0.20, 0.10])
c2_restructure = np.array([0.40, 0.00, 0.40, 0.60, 0.10, 0.10, 0.30, 0.10, 0.10, 0.10])
c2_colloquial  = np.array([0.20, 0.00, 0.40, 0.20, 0.10, 0.00, 0.20, 0.10, 0.10, 0.00])

TASKS = [
    "T0: btw plate & ramekin",
    "T1: next to ramekin",
    "T2: table center",
    "T3: on cookie box",
    "T4: cabinet drawer",
    "T5: on ramekin",
    "T6: next to cookie box",
    "T7: stove",
    "T8: next to plate",
    "T9: wooden cabinet",
]

def wilcoxon_report(name, a, b):
    diff = a - b
    nonzero = diff[diff != 0]
    n_eff = len(nonzero)
    if n_eff < 2:
        print(f"\n{name}: n_eff={n_eff}, cannot compute test")
        return
    stat, p = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    sig = "*" if p < 0.05 else "ns"
    print(f"\n{'='*55}")
    print(f"  Wilcoxon: {name}")
    print(f"{'='*55}")
    print(f"  Mean A = {np.mean(a):.4f}  |  Mean B = {np.mean(b):.4f}")
    print(f"  Mean diff (A-B) = {np.mean(diff):+.4f}  |  SE = {np.std(diff,ddof=1)/np.sqrt(10):.4f}")
    print(f"  n_eff (non-zero) = {n_eff}")
    print(f"  W statistic = {stat:.1f}")
    print(f"  p-value (two-tailed) = {p:.4f}  [{sig}]")
    print(f"\n  Per-task diffs:")
    for t, d in zip(TASKS, diff):
        flag = " ← tie" if d==0 else (" ← reversal" if d<0 else "")
        print(f"    {t:<30s}  {d:+.2f}{flag}")
    return stat, p

# ── MAIN TESTS ─────────────────────────────────────────────────────────────────
print("\nVLA Compositionality Study — Statistical Analysis")
print("=" * 55)

wilcoxon_report("C1 vs C2 (Linguistic Stress)", c1, c2)
wilcoxon_report("C1 vs C3 (Visual Stress)",     c1, c3)
wilcoxon_report("C1 vs C4 (Full Stress)",        c1, c4)

# ── CGG DECOMPOSITION ──────────────────────────────────────────────────────────
c1m, c2m, c3m, c4m = map(np.mean, [c1, c2, c3, c4])
cgg_l = c1m - c2m
cgg_v = c1m - c3m
cgg_f = c1m - c4m
cgg_s = cgg_f - cgg_l - cgg_v

print(f"\n{'='*55}")
print("  CGG DECOMPOSITION")
print(f"{'='*55}")
print(f"  C1 mean SR    = {c1m:.4f}")
print(f"  C2 mean SR    = {c2m:.4f}")
print(f"  C3 mean SR    = {c3m:.4f}")
print(f"  C4 mean SR    = {c4m:.4f}")
print(f"\n  CGG_ling  = C1-C2 = {cgg_l:+.4f}  (p=0.036 *)")
print(f"  CGG_vis   = C1-C3 = {cgg_v:+.4f}  (p=0.177 ns)")
print(f"  CGG_full  = C1-C4 = {cgg_f:+.4f}  (p=0.128 ns)")
print(f"  CGG_syn   = full - ling - vis = {cgg_s:+.4f}  (sub-additive)")

# ── PARAPHRASE TYPE ANALYSIS ───────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  PARAPHRASE TYPE BREAKDOWN (C2)")
print(f"{'='*55}")
for name, arr in [("Synonym swap", c2_synonym), ("Restructure", c2_restructure),
                   ("Colloquial",   c2_colloquial)]:
    pcs = 1 - np.std(arr)
    print(f"  {name:<20s}  mean={np.mean(arr):.3f}  SE={np.std(arr,ddof=1)/np.sqrt(10):.3f}  PCS={pcs:.3f}")

# ── POWER ANALYSIS ─────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  POWER ANALYSIS (CGG_vis)")
print(f"{'='*55}")
# Approximate: for Wilcoxon ≈ t-test power in large samples
# Required n for 80% power: n ≈ (z_alpha/2 + z_beta)^2 * (sigma/delta)^2
delta = cgg_v
sigma = np.std(c1 - c3, ddof=1)
z = 1.96 + 0.842  # alpha=0.05 two-tailed + 80% power
n_required = int(np.ceil((z**2) * (sigma / delta)**2))
print(f"  Observed effect (delta) = {delta:.4f}")
print(f"  Observed std of diffs   = {sigma:.4f}")
print(f"  n required for 80% power (alpha=0.05) ≈ {n_required} tasks")

# ── SAVE CSV ───────────────────────────────────────────────────────────────────
import os, csv
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_task_results.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task", "C1", "C2", "C3", "C4",
                "CGG_ling", "CGG_vis", "CGG_full",
                "C2_synonym", "C2_restructure", "C2_colloquial", "PCS"])
    for i, task in enumerate(TASKS):
        pcs = 1 - np.std([c2_synonym[i], c2_restructure[i], c2_colloquial[i]])
        w.writerow([
            task, c1[i], c2[i], c3[i], c4[i],
            round(c1[i]-c2[i], 3), round(c1[i]-c3[i], 3), round(c1[i]-c4[i], 3),
            c2_synonym[i], c2_restructure[i], c2_colloquial[i], round(pcs, 3)
        ])
print(f"\n  CSV saved to: {out_path}")
