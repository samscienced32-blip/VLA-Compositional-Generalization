# Disentangling Linguistic and Visual Compositionality Failures in Vision-Language-Action Models

**CAISc 2026 Submission**

---

## Overview

This repository contains the code, paraphrase annotations, and paper source for our 2×2 factorial study on compositional robustness of OpenVLA on LIBERO-Spatial.

We independently stress the **linguistic** axis (paraphrase variants) and **visual** axis (robot pose perturbation) to measure how each contributes to performance degradation, using our proposed **Compositional Generalization Gap (CGG)** decomposition.

## Repository Structure

```
├── run_experiment_openvla.py   # Main experiment runner (all 4 cells)
├── analyze_results.py          # Generates figures and CGG metrics from results
├── download_openvla.py         # Downloads OpenVLA checkpoint from HuggingFace
├── paraphrases.json            # 30 human-authored paraphrase variants (10 tasks × 3 types)
├── paper.tex                   # LaTeX source
├── references.bib              # Bibliography
├── caisc_2026.sty              # CAISc 2026 style file
└── README.md
```

## Setup

### Requirements

- Python 3.10+
- NVIDIA RTX 4080 Mobile GPU or equivalent (12 GB VRAM minimum for 8-bit quantization; 16 GB+ for bfloat16)
- Ubuntu 22.04, CUDA 12.1

### Install

```bash
# 1. Clone LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ~/LIBERO
cd ~/LIBERO && pip install -e .

# 2. Clone OpenVLA (for libero_utils / robot_utils)
git clone https://github.com/openvla/openvla-oft.git ~/openvla-oft
cd ~/openvla-oft && pip install -e .

# 3. Install remaining deps
pip install transformers accelerate bitsandbytes Pillow opencv-python imageio

# 4. Download model checkpoint (~14 GB)
python download_openvla.py
```

## Running the Experiment

```bash
# All 4 cells (full run, ~20 hours on 12 GB GPU)
MUJOCO_GL=egl python run_experiment_openvla.py

# Single cell (e.g., baseline only)
MUJOCO_GL=egl python run_experiment_openvla.py --cells 1

# Quick sanity check (1 task, 2 rollouts, with visualization)
MUJOCO_GL=egl python run_experiment_openvla.py --visualize --num_tasks 1 --rollouts 2
```

Results are saved to `~/VLMmodel/results/` as JSON files per cell.

## Analyzing Results

```bash
python analyze_results.py
```

Outputs figures to `~/VLMmodel/figures/` and prints CGG decomposition.

## Paraphrase Annotations

`paraphrases.json` contains three paraphrase variants per task:

| Type | Description |
|------|-------------|
| `synonym_swap` | Key verbs/prepositions replaced with synonyms |
| `restructure` | Syntactic structure altered, semantics preserved |
| `colloquial` | Shortened, informal rewrite |

## Model

- **Checkpoint**: `openvla/openvla-7b-finetuned-libero-spatial` (HuggingFace)
- **Quantization**: 8-bit via BitsAndBytes (`load_in_8bit=True`) for 12 GB VRAM
- **Benchmark**: LIBERO-Spatial (10 tasks, pick-and-place with spatial relations)

## Results Summary

| Cell | Condition | Mean SR |
|------|-----------|---------|
| C1 | Baseline (seen visual + seen language) | 0.360 |
| C2 | Linguistic stress (novel paraphrase) | 0.227 |
| C3 | Visual stress (pose perturbation) | 0.240 |
| C4 | Full novel (both) | 0.220 |

**CGG_ling** = +0.133 (p = 0.036 *)  
**CGG_vis** = +0.120 (p = 0.177 ns)  
**CGG_full** = +0.140 | **CGG_syn** = −0.113 (sub-additive)

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
