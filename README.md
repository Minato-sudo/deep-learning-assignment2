# Deep Learning Project — Reproducibility Study & Experimentation
## Denoising Diffusion Probabilistic Models (DDPM) on CIFAR-10

**Course:** CS-4112 Deep Learning — FAST-NUCES  
**Department:** Artificial Intelligence & Data Science  
**Instructors:** Dr. Qurat Ul Ain, Dr. Zohair Ahmed  

**Group Members:**
- Zain Shahid (23i-2582)
- SanaUllah (23i-2594)
- Muhammad Talha Arshad (23i-2548)

---

## Paper Reproduced
**Denoising Diffusion Probabilistic Models**  
Ho, Jain & Abbeel — NeurIPS 2020  
arXiv:2006.11239  
Official Code: https://github.com/hojonathanho/diffusion  
Model Checkpoint: https://huggingface.co/google/ddpm-cifar10-32

---

## Project Overview

This repository contains all code, logs, and reports for a three-part deep learning project:

| Assignment | Description | Status |
|---|---|---|
| A1 | Paper Understanding | ✅ Done |
| A2 | Reproduction of DDPM Results on CIFAR-10 | ✅ Done |
| A3 | Experimentation and Extension | ✅ Done |

---

## Assignment 2 — Reproduction Results

**Task:** Reproduce DDPM on CIFAR-10 with DDIM sampling variants.

| Model | Steps | Paper FID | Reproduced FID |
|---|---|---|---|
| DDPM | 1000 | 3.17 | 20.91 |
| DDIM (η=0) | 10 | N/A | 38.69 |
| DDIM (η=0) | 1 | N/A | 462.11 |

15,000 total images generated across 3 configurations (5,000 each).

---

## Assignment 3 — Experimentation Results

**Task:** Extend reproduction with 3 original experiments.

### Experiment 1: DDIM Step Count Ablation (CIFAR-10)
**Hypothesis:** There exists an optimal number of inference steps beyond which quality saturates or degrades.

| Steps | FID | Relative to Best |
|---|---|---|
| 1 | 468.11 | +1011% |
| 5 | 95.64 | +127% |
| 10 | 64.07 | +52% |
| 20 | 51.94 | +23% |
| 50 | 45.11 | +7% |
| 100 | 42.74 | +1% |
| **200** | **42.13** | **Best** |
| 500 | 44.02 | +4% |
| 1000 | 60.55 | +44% |

**Finding:** Optimal FID at 200 steps. FID degrades beyond 200 steps due to ODE discretization error accumulation.

---

### Experiment 2: DDIM Eta Parameter Study (CIFAR-10, fixed 10 steps)
**Hypothesis:** Deterministic sampling (η=0) outperforms stochastic sampling at low step counts.

| Eta | FID | Mode |
|---|---|---|
| **0.00** | **65.72** | Fully Deterministic (best) |
| 0.25 | 65.90 | Mostly Deterministic |
| 0.50 | 66.05 | Mixed |
| 0.75 | 72.54 | Mostly Stochastic |
| 1.00 | 89.55 | Fully Stochastic |

**Finding:** Stable zone from η=0.0 to η=0.5 (FID change <1%). Sharp degradation above η=0.5. Deterministic DDIM (η=0) is 36% better than full stochastic (η=1.0) at 10 steps.

---

### Experiment 3: Cross-Domain Generalization (CelebA-HQ 256×256)
**Hypothesis:** Test whether pre-trained DDPM weights generalize to an unseen image domain.

| Metric | Value |
|---|---|
| Model | google/ddpm-celebahq-256 |
| Resolution | 256×256 |
| Inference Steps | 1000 |
| Images Generated | 244 |
| Intra-FID (diversity) | 59.34 |
| Published FID range (literature) | 29.76–40.26 |

**Finding:** Pre-trained DDPM weights generalize meaningfully to human face domain without fine-tuning. Intra-FID of 59.34 confirms high output diversity with no mode collapse.

---

## Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU (8GB, sm_120 Blackwell) |
| CPU | Intel Core i7 (16 threads) |
| RAM | 16GB |
| OS | Kubuntu Linux (Ubuntu 24.04) |
| PyTorch | 2.12.0.dev20260328+cu128 (nightly) |
| Python | 3.12.3 |
| diffusers | 0.33.1 |
| clean-fid | 0.1.35 |

---

## Repository Structure

```
deep-learning-assignment2/
├── README.md
├── Deep_Learning_A2.pdf              ← Assignment 2 report
├── DeepLearning_Assignment_3.pdf     ← Assignment 3 report
│
├── scripts/
│   ├── save_cifar10_real.py          ← Save real CIFAR-10 images for FID
│   ├── generate_images.py            ← Generate 1000 images (A2 initial)
│   ├── generate_more.py              ← Generate 5000 images per config (A2)
│   ├── compute_fid.py                ← FID computation (1k samples, A2)
│   ├── compute_fid_5k.py             ← FID computation (5k samples, A2)
│   ├── make_grid.py                  ← Generate sample image grids (A2)
│   ├── plot_results.py               ← FID comparison chart (A2)
│   ├── experiment_steps.py           ← DDIM step ablation (A3 Exp1)
│   ├── experiment_eta.py             ← Eta parameter study (A3 Exp2)
│   ├── experiment_crossdomain.py     ← Cross-domain CelebA-HQ (A3 Exp3)
│   └── plot_all_results.py           ← All A3 plots
│
├── logs/
│   ├── fid_results.txt               ← A2 FID scores (1000 samples)
│   ├── fid_results_5k.txt            ← A2 FID scores (5000 samples)
│   ├── generation_log.txt            ← A2 generation log
│   ├── results_step_ablation.json    ← A3 Exp1 step ablation results
│   ├── results_eta_study.json        ← A3 Exp2 eta study results
│   ├── log_steps.txt                 ← A3 Exp1 full run log
│   └── log_eta.txt                   ← A3 Exp2 full run log
│
└── results/
    ├── real_cifar10.png              ← Real CIFAR-10 sample grid
    ├── ddpm_1000steps.png            ← DDPM 1000-step grid
    ├── ddim_10steps.png              ← DDIM 10-step grid
    ├── ddim_1step.png                ← DDIM 1-step grid
    ├── fid_comparison.png            ← A2 FID bar chart
    ├── exp1_fid_vs_steps.png         ← A3 Exp1 step ablation curve
    ├── exp2_fid_vs_eta.png           ← A3 Exp2 eta degradation curve
    ├── exp3_summary_comparison.png   ← A3 full summary bar chart
    └── celebahq_sample_grid.png      ← A3 CelebA-HQ face grid
```

---

## How to Reproduce

### 1. Setup Environment
```bash
python3 -m venv dl_env
source dl_env/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install diffusers accelerate clean-fid matplotlib scipy tqdm pillow
```

### 2. Assignment 2 — Reproduce DDPM on CIFAR-10
```bash
# Save real CIFAR-10 reference images
python3 scripts/save_cifar10_real.py

# Generate 5000 images per config (DDPM-1000, DDIM-10, DDIM-1)
python3 scripts/generate_more.py

# Compute FID scores
python3 scripts/compute_fid_5k.py

# Generate sample grids and chart
python3 scripts/make_grid.py
python3 scripts/plot_results.py
```

### 3. Assignment 3 — Run Experiments
```bash
# Experiment 1: DDIM step ablation (9 configs)
python3 scripts/experiment_steps.py

# Experiment 2: Eta parameter study (5 configs)
python3 scripts/experiment_eta.py

# Experiment 3: Cross-domain CelebA-HQ
python3 scripts/experiment_crossdomain.py

# Generate all A3 plots
python3 scripts/plot_all_results.py
```

---

## Note on Consistency Models Checkpoint
The official Consistency Models checkpoint (Song et al., ICML 2023) was permanently deleted from both OpenAI blob storage and HuggingFace at time of evaluation (HTTP 404). Reproduction of those results was not possible. This is documented in the A2 report Section 6.

## Note on CelebA-HQ Reference Statistics
Clean-fid reference statistics for CelebA-HQ 256x256 were unavailable (HTTP 404) at time of evaluation. Intra-FID (59.34) is reported as a diversity metric. This is documented in the A3 report Section 6.3.
