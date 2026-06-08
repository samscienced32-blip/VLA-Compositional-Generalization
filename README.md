# Disentangling Linguistic and Visual Compositionality Failures in Vision-Language-Action Models

**CAISc 2026 Submission** — [anonymous repository](https://anonymous.4open.science/r/VLA-Compositional-Generalization)

---

## Overview

This repository contains the code, data, paraphrase annotations, and paper source for our 2×2 factorial study on compositional robustness of OpenVLA on LIBERO-Spatial.

We independently stress the **linguistic** axis (paraphrase variants) and **visual** axis (robot pose perturbation) to measure how each contributes to performance degradation, using our proposed **Compositional Generalization Gap (CGG)** decomposition.

The study runs under two precision settings to separate quantization artifacts from genuine robustness failures:
- **8-bit** (RTX 4080 Mobile, 2 seeds): primary results
- **bfloat16** (A100, 3 seeds): replication / precision comparison

---

## Repository Structure

```
├── run_experiment_openvla.py        # Main 8-bit experiment runner (RTX laptop)
├── run_experiment_cluster.py        # bfloat16 experiment runner (A100 cluster, SLURM)
├── run_spatial_s1.sh                # SLURM job script — bfloat16 seed 1
├── run_spatial_s2.sh                # SLURM job script — bfloat16 seed 2
├── run_spatial_s3.sh                # SLURM job script — bfloat16 seed 3
├── generate_figures_multiseed.py    # Regenerates all paper figures from pooled seeds
├── statistical_analysis_multiseed.py # Wilcoxon tests, CGG decomposition, PCS, power analysis
├── per_task_results_multiseed.csv   # Pooled bfloat16 per-task results (seeds 1–3, A100)
└── paraphrases.json                 # 30 human-authored paraphrase variants (10 tasks × 3 types)
```


---

## Setup

### Requirements

**8-bit (local):** NVIDIA RTX 4080 Mobile (12 GB VRAM) or equivalent, Ubuntu 22.04, CUDA 12.1, PyTorch 2.2.0, Transformers 4.40.1, BitsAndBytes 0.43.1

**bfloat16 (cluster):** NVIDIA A100 (80 GB VRAM), CUDA 12.8, SLURM

### Install

```bash
# 1. Clone LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
cd ~/LIBERO && pip install -e .

# 2. Clone OpenVLA-OFT (for libero_utils / robot_utils)
git clone https://github.com/openvla/openvla-oft.git ~/openvla-oft
cd ~/openvla-oft && pip install -e .

# 3. Install remaining deps
pip install transformers==4.40.1 accelerate bitsandbytes==0.43.1 Pillow opencv-python imageio scipy
```

> The model checkpoint (`openvla/openvla-7b-finetuned-libero-spatial`, ~14 GB) downloads automatically from HuggingFace on first run.

---

## Running the Experiment

### 8-bit (local GPU — how the paper's primary results were produced)

The paper uses two seeds (`s ∈ {0, 1}`). Seed 0 is the default (no `--seed` flag); seed 1 adds Gaussian noise seeding. Each full run takes ~20 hours on a 12 GB GPU.

```bash
# Seed 0 — all 4 cells (results saved as cell*.json)
MUJOCO_GL=egl python run_experiment_openvla.py

# Seed 1 — all 4 cells (results saved as cell*_seed1.json)
MUJOCO_GL=egl python run_experiment_openvla.py --seed 1

# Single cell only (e.g., baseline)
MUJOCO_GL=egl python run_experiment_openvla.py --cells 1

# Quick sanity check (1 task, 2 rollouts, with visualisation)
MUJOCO_GL=egl python run_experiment_openvla.py --visualize --num_tasks 1 --rollouts 2
```

All results are fully reproducible by re-running the scripts. C1/C3/C4 run 10 rollouts per task; C2 runs 30 rollouts (3 variants × 10) and reports the mean SR across variants as the task-level SR. MuJoCo runs at 20 Hz, 256×256 px, max 300 steps per rollout. Visual perturbation adds Gaussian noise ε~N(0, 0.05²I₇) (σ=0.05 rad per joint) to all 7 arm joints at initialisation.

### bfloat16 (A100 cluster — how the replication results were produced)

Three independent seeds were run on the university HPC cluster. Each SLURM script runs all 4 cells for one seed.

```bash
sbatch run_spatial_s1.sh   # bfloat16 seed 1
sbatch run_spatial_s2.sh   # bfloat16 seed 2
sbatch run_spatial_s3.sh   # bfloat16 seed 3
```

The pooled per-task summary is in `per_task_results_multiseed.csv`.

---

## Regenerating Figures

```bash
# 2-seed 8-bit figures (matches paper)
python generate_figures_multiseed.py --seeds 0 1

# Single seed only
python generate_figures_multiseed.py --seeds 0
```

---

## Statistical Analysis

```bash
python statistical_analysis_multiseed.py
```

Reproduces all Wilcoxon signed-rank tests, CGG decomposition, PCS scores, and power analysis reported in the paper.

---

## Paraphrase Annotations

`paraphrases.json` contains three paraphrase variants per task:

| Type | Description |
|------|-------------|
| `synonym_swap` | Key verbs/prepositions replaced with synonyms (e.g., "pick up" → "grab", "place" → "put") |
| `restructure` | Syntactic structure altered, semantics preserved (passive voice, relative clauses) |
| `colloquial` | Shortened, informal rewrite characteristic of natural speech |

All 30 variants were authored and validated by human researchers. Full listing in Supplementary §2.

---

## Model & Benchmark

- **Checkpoint**: `openvla/openvla-7b-finetuned-libero-spatial` — 7B-parameter VLA on a Llama-2 + DINOv2+SigLIP backbone
- **8-bit**: `load_in_8bit=True` via BitsAndBytes (peak ≈7.5 GB VRAM, RTX 4080 Mobile)
- **bfloat16**: full precision on A100 80 GB VRAM (≈28 GB peak)
- **Benchmark**: LIBERO-Spatial — 10 tabletop pick-and-place tasks where the robot locates a black bowl via a spatial relation and deposits it on a plate; task success determined by `check_success()` using object-pose thresholds from the task's `bddl` specification

---

## Results Summary

### 8-bit quantization — primary results (RTX 4080 Mobile, seeds s ∈ {0, 1})

Mean SR ± SE over tasks, averaged across both seeds (Table 1 of paper):

| Cell | Condition | Mean SR | ±SE | CGG vs C1 | Wilcoxon p |
|------|-----------|---------|-----|-----------|------------|
| C1 | Baseline | 0.400 | 0.09 | — | — |
| C2 | Linguistic Stress | 0.228 | 0.06 | +0.172 | 0.012 \* |
| C3 | Visual Stress | 0.275 | 0.06 | +0.125 | 0.177 ns |
| C4 | Full Novel | 0.220 | 0.06 | +0.180 | 0.128 ns |

**CGG_syn** = CGG_full − CGG_ling − CGG_vis = −0.117 (sub-additive failure)

Within C2, colloquial rewriting is most damaging (SR = 0.14, PCS = 0.82), synonym swap is nearly neutral (SR = 0.33, PCS = 0.73), and syntactic restructure is intermediate (SR = 0.22, PCS = 0.83). Statistical significance assessed via two-tailed Wilcoxon signed-rank test, α = 0.05, n_eff = 9 (one tie excluded for C1 vs C2 and C1 vs C3).

### bfloat16 full precision — replication (A100, seeds s ∈ {1, 2, 3})

| Config | C1 SR | CGG_ling | CGG_vis | Seeds |
|--------|-------|----------|---------|-------|
| 8-bit (RTX 4080) | 0.400 ± 0.040 | +0.172 ± 0.039 \* | +0.125 ± 0.005 ns | 2 |
| bfloat16 (A100) | 0.320 ± 0.021 | −0.023 ± 0.018 ns | +0.067 ± 0.024 ns | 3 |

The key finding is the **cross-precision asymmetry**: CGG_vis is directionally consistent (+0.125 under 8-bit; +0.067 under bfloat16), while CGG_ling reverses sign (+0.172 under 8-bit; −0.023 under bfloat16). This implicates quantization as a confounding variable for linguistic sensitivity — a consideration absent from prior VLA robustness evaluations.

---

## Citation

```bibtex
@inproceedings{anonymous2026disentangling,
  title     = {Disentangling Linguistic and Visual Compositionality Failures
               in Vision-Language-Action Models},
  author    = {Anonymous},
  booktitle = {1st Conference For AI Scientists (CAISc 2026)},
  year      = {2026}
}
```
