"""
statistical_analysis_multiseed.py
==================================
Same as statistical_analysis.py but pools results across seeds 0-3.
Loads JSON result files, averages SR across seeds per task per cell,
then runs identical Wilcoxon tests and CGG decomposition.

Usage:
    # Cluster results (bfloat16, seeds 1-3):
    python "Research paper/statistical_analysis_multiseed.py" \
        --results_dir ~/openvla/results --seeds 1 2 3

    # Laptop results (8-bit, seeds 0-1):
    python "Research paper/statistical_analysis_multiseed.py" \
        --results_dir ~/VLMmodel/results --seeds 0 1

    # All seeds combined (if using same quantization):
    python "Research paper/statistical_analysis_multiseed.py" \
        --results_dir ~/openvla/results --seeds 0 1 2 3
"""

import os, sys, json, csv, argparse
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default=os.path.expanduser("~/openvla/results"))
parser.add_argument("--seeds",       type=int, nargs="+", default=[1, 2, 3])
args = parser.parse_args()

RESULTS_DIR = os.path.expanduser(args.results_dir)
SEEDS       = args.seeds

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

CELL_FILES = {
    "c1": "cell1_baseline_seed{}.json",
    "c2": "cell2_linguistic_seed{}.json",
    "c3": "cell3_visual_seed{}.json",
    "c4": "cell4_combined_seed{}.json",
}

# ── LOAD AND POOL ──────────────────────────────────────────────────────────────
def load_cell(cell_key):
    """Load SR arrays for a cell across all seeds. Returns (mean, std) per task."""
    all_seeds = []
    for seed in SEEDS:
        fname = os.path.join(RESULTS_DIR, CELL_FILES[cell_key].format(seed))
        if not os.path.exists(fname):
            print(f"  [WARNING] Missing: {fname} — skipping seed {seed}")
            continue
        with open(fname) as f:
            data = json.load(f)
        # JSON structure: list of dicts with 'sr' key, ordered by task
        entries = data if isinstance(data, list) else data.get("results", [])
        # Group by task_idx and average variants (C2/C4 have 3 entries per task)
        from collections import defaultdict
        task_srs = defaultdict(list)
        for entry in entries:
            task_srs[entry["task_idx"]].append(entry["success_rate"])
        sr_arr = np.array([np.mean(task_srs[i]) for i in sorted(task_srs.keys())])
        all_seeds.append(sr_arr)
    if not all_seeds:
        raise FileNotFoundError(f"No result files found for cell {cell_key} in {RESULTS_DIR}")
    stacked = np.stack(all_seeds, axis=0)  # shape: (n_seeds, n_tasks)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1), stacked

def load_c2_types():
    """Load C2 per-type SRs across seeds."""
    all_syn, all_res, all_col = [], [], []
    for seed in SEEDS:
        fname = os.path.join(RESULTS_DIR, CELL_FILES["c2"].format(seed))
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else data.get("results", [])
        syn_row, res_row, col_row = [], [], []
        # Group by task_idx and para_type
        from collections import defaultdict
        task_variants = defaultdict(dict)
        for entry in entries:
            task_variants[entry["task_idx"]][entry["para_type"]] = entry["success_rate"]
        for i in sorted(task_variants.keys()):
            syn_row.append(task_variants[i].get("synonym_swap", 0))
            res_row.append(task_variants[i].get("restructure",  0))
            col_row.append(task_variants[i].get("colloquial",   0))
        all_syn.append(syn_row)
        all_res.append(res_row)
        all_col.append(col_row)
    if not all_syn:
        return None, None, None
    return (np.array(all_syn).mean(axis=0),
            np.array(all_res).mean(axis=0),
            np.array(all_col).mean(axis=0))

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print(f"\nLoading results from: {RESULTS_DIR}")
print(f"Seeds: {SEEDS}")

c1_mean, c1_std, c1_stack = load_cell("c1")
c2_mean, c2_std, c2_stack = load_cell("c2")
c3_mean, c3_std, c3_stack = load_cell("c3")
c4_mean, c4_std, c4_stack = load_cell("c4")

n_seeds = c1_stack.shape[0]
print(f"Seeds loaded: {n_seeds}")

# ── WILCOXON TESTS ─────────────────────────────────────────────────────────────
def wilcoxon_report(name, a, b, a_std=None, b_std=None):
    diff = a - b
    nonzero = diff[diff != 0]
    n_eff = len(nonzero)
    if n_eff < 2:
        print(f"\n{name}: n_eff={n_eff}, cannot compute test")
        return None, None
    stat, p = stats.wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
    sig = "*" if p < 0.05 else ("†" if p < 0.10 else "ns")
    print(f"\n{'='*60}")
    print(f"  Wilcoxon: {name}")
    print(f"{'='*60}")
    print(f"  Mean A = {np.mean(a):.4f}  |  Mean B = {np.mean(b):.4f}")
    print(f"  Mean diff (A-B) = {np.mean(diff):+.4f}  |  SE = {np.std(diff,ddof=1)/np.sqrt(10):.4f}")
    print(f"  n_eff (non-zero diffs) = {n_eff}")
    print(f"  W statistic = {stat:.1f}")
    print(f"  p-value (two-tailed) = {p:.4f}  [{sig}]")
    print(f"\n  Per-task seed-averaged SRs and diffs:")
    print(f"  {'Task':<30s}  {'A (mean±std)':>14s}  {'B (mean±std)':>14s}  {'Diff':>6s}")
    for i, task in enumerate(TASKS):
        a_s = f"{a[i]:.2f}±{(a_std[i] if a_std is not None else 0):.2f}"
        b_s = f"{b[i]:.2f}±{(b_std[i] if b_std is not None else 0):.2f}"
        d   = diff[i]
        flag = " ← tie" if d==0 else (" ← reversal" if d<0 else "")
        print(f"  {task:<30s}  {a_s:>14s}  {b_s:>14s}  {d:+.2f}{flag}")
    return stat, p

print("\n\nVLA Compositionality Study — Multi-Seed Statistical Analysis")
print(f"Seeds pooled: {SEEDS}  |  n_seeds = {n_seeds}")
print("=" * 60)

wilcoxon_report("C1 vs C2 (Linguistic Stress)", c1_mean, c2_mean, c1_std, c2_std)
wilcoxon_report("C1 vs C3 (Visual Stress)",     c1_mean, c3_mean, c1_std, c3_std)
wilcoxon_report("C1 vs C4 (Full Stress)",        c1_mean, c4_mean, c1_std, c4_std)

# ── CGG DECOMPOSITION ──────────────────────────────────────────────────────────
c1m = np.mean(c1_mean)
c2m = np.mean(c2_mean)
c3m = np.mean(c3_mean)
c4m = np.mean(c4_mean)
cgg_l = c1m - c2m
cgg_v = c1m - c3m
cgg_f = c1m - c4m
cgg_s = cgg_f - cgg_l - cgg_v

# Between-seed variance on mean SR
c1_seed_means = c1_stack.mean(axis=1)
c2_seed_means = c2_stack.mean(axis=1)
c3_seed_means = c3_stack.mean(axis=1)
c4_seed_means = c4_stack.mean(axis=1)

print(f"\n{'='*60}")
print("  CGG DECOMPOSITION  (seed-averaged)")
print(f"{'='*60}")
print(f"  C1 mean SR = {c1m:.4f}  (seed std = {c1_seed_means.std(ddof=1):.4f})")
print(f"  C2 mean SR = {c2m:.4f}  (seed std = {c2_seed_means.std(ddof=1):.4f})")
print(f"  C3 mean SR = {c3m:.4f}  (seed std = {c3_seed_means.std(ddof=1):.4f})")
print(f"  C4 mean SR = {c4m:.4f}  (seed std = {c4_seed_means.std(ddof=1):.4f})")
print(f"\n  CGG_ling  = C1-C2 = {cgg_l:+.4f}")
print(f"  CGG_vis   = C1-C3 = {cgg_v:+.4f}")
print(f"  CGG_full  = C1-C4 = {cgg_f:+.4f}")
print(f"  CGG_syn   = full - ling - vis = {cgg_s:+.4f}  (sub-additive if negative)")

# ── PARAPHRASE TYPE BREAKDOWN ─────────────────────────────────────────────────
c2_syn, c2_res, c2_col = load_c2_types()
print(f"\n{'='*60}")
print("  PARAPHRASE TYPE BREAKDOWN (C2, seed-averaged)")
print(f"{'='*60}")
if c2_syn is not None:
    for name, arr in [("Synonym swap", c2_syn), ("Restructure", c2_res), ("Colloquial", c2_col)]:
        pcs = 1 - np.std(arr)
        print(f"  {name:<20s}  mean={np.mean(arr):.3f}  SE={np.std(arr,ddof=1)/np.sqrt(10):.3f}  PCS={pcs:.3f}")
else:
    print("  [Variant-level data not found in JSONs — skipping]")

# ── POWER ANALYSIS ────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  POWER ANALYSIS (CGG_vis, seed-averaged)")
print(f"{'='*60}")
delta = cgg_v
sigma = np.std(c1_mean - c3_mean, ddof=1)
z     = 1.96 + 0.842
n_required = int(np.ceil((z**2) * (sigma / delta)**2)) if delta != 0 else float("inf")
print(f"  Observed effect (delta) = {delta:.4f}")
print(f"  Observed std of diffs   = {sigma:.4f}")
print(f"  n required for 80% power (alpha=0.05) ≈ {n_required} tasks")

# ── SAVE CSV ──────────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_task_results_multiseed.csv")
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["task",
                "C1_mean", "C1_std", "C2_mean", "C2_std",
                "C3_mean", "C3_std", "C4_mean", "C4_std",
                "CGG_ling", "CGG_vis", "CGG_full"])
    for i, task in enumerate(TASKS):
        w.writerow([
            task,
            round(c1_mean[i], 3), round(c1_std[i], 3),
            round(c2_mean[i], 3), round(c2_std[i], 3),
            round(c3_mean[i], 3), round(c3_std[i], 3),
            round(c4_mean[i], 3), round(c4_std[i], 3),
            round(c1_mean[i]-c2_mean[i], 3),
            round(c1_mean[i]-c3_mean[i], 3),
            round(c1_mean[i]-c4_mean[i], 3),
        ])
print(f"\n  CSV saved → {out_path}")
print("\nDone.")
