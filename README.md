# Deep Learning Assignment 2 — Reproducibility Study
## Denoising Diffusion Probabilistic Models on CIFAR-10

**Course:** CS-4112 Deep Learning — FAST-NUCES  
**Group Members:**
- Zain Shahid (23i-2582)
- SanaUllah (23i-2594)
- Muhammad Talha Arshad (23i-2548)

---

## Paper Reproduced
**Denoising Diffusion Probabilistic Models**  
Ho, Jain & Abbeel — NeurIPS 2020  
[arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
Official Code: https://github.com/hojonathanho/diffusion
Model Checkpoint: https://huggingface.co/google/ddpm-cifar10-32
---

## Results Summary

| Model | Paper FID | Ours (5k samples) | Steps |
|---|---|---|---|
| DDPM | 3.17 | 20.91 | 1000 |
| DDIM | N/A | 38.69 | 10 |
| DDIM | N/A | 462.11 | 1 |

---

## Hardware
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU (8GB, sm_120 Blackwell)
- OS: Kubuntu Linux (Ubuntu 24.04)
- PyTorch: 2.12.0.dev20260328+cu128 (nightly)
- Python: 3.12.3

---

## Repository Structure
```
scripts/          # All Python scripts
  save_cifar10_real.py     # Save real CIFAR-10 images for FID
  generate_images.py       # Generate 1000 images (initial run)
  generate_more.py         # Generate 5000 images per config
  compute_fid.py           # FID computation (1k samples)
  compute_fid_5k.py        # FID computation (5k samples)
  make_grid.py             # Generate sample image grids
  plot_results.py          # Plot FID comparison chart

logs/             # Experiment logs and FID results
  fid_results.txt          # FID scores (1000 samples)
  fid_results_5k.txt       # FID scores (5000 samples)
  generation_log.txt       # Full generation log

results/          # Generated figures
  real_cifar10.png         # Real CIFAR-10 sample grid
  ddpm_1000steps.png       # DDPM 1000-step sample grid
  ddim_10steps.png         # DDIM 10-step sample grid
  ddim_1step.png           # DDIM 1-step sample grid
  fid_comparison.png       # FID comparison bar chart
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

### 2. Save Real CIFAR-10 Images
```bash
python3 scripts/save_cifar10_real.py
```

### 3. Generate Images
```bash
python3 scripts/generate_more.py
```

### 4. Compute FID
```bash
python3 scripts/compute_fid_5k.py
```

### 5. Generate Visualizations
```bash
python3 scripts/make_grid.py
python3 scripts/plot_results.py
```

---

## Note on Consistency Models Checkpoint
The official Consistency Models checkpoint (Song et al., ICML 2023) was permanently deleted from both OpenAI blob storage and HuggingFace. Reproduction of those results was not possible. See report Section 6 for full discussion.
