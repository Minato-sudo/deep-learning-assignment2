"""
Assignment 3 — Experiment 1: DDIM Step Ablation Study
=======================================================
Hypothesis: FID decreases as inference steps increase,
with diminishing returns beyond ~100 steps.

Tests: steps = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
Each configuration generates 1000 images (skips if already done).
"""

import torch
from diffusers import DDIMPipeline
import os
from PIL import Image
from cleanfid import fid
import json
import time

# ── Setup ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")
print(f"[INFO] Loading model: google/ddpm-cifar10-32 ...")

pipeline = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32").to(device)
print(f"[INFO] Model loaded successfully.\n")

real_dir = os.path.expanduser("~/deep_learning_a2/data/cifar10_real")
step_counts = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
results = {}

# ── Experiment Loop ───────────────────────────────────────────────────────────
for steps in step_counts:
    out_dir = f"outputs/ddim_steps_{steps}"
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    needed = 1000 - existing

    if needed <= 0:
        print(f"[SKIP] steps={steps:5d}: 1000 images already exist. Computing FID only...")
    else:
        print(f"[GEN]  steps={steps:5d}: Generating {needed} images (have {existing}/1000)...")
        t_start = time.time()
        idx = existing
        while idx < 1000:
            batch_size = min(32, 1000 - idx)
            with torch.no_grad():
                images = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=steps,
                    eta=0.0,           # deterministic DDIM
                    output_type="pil"
                ).images
            for img in images:
                img.save(f"{out_dir}/{idx:05d}.png")
                idx += 1
            elapsed = time.time() - t_start
            print(f"         {idx}/1000  ({elapsed:.0f}s elapsed)", end="\r")
        print(f"         {idx}/1000  Done in {time.time()-t_start:.0f}s              ")

    # Compute FID
    print(f"[FID]  steps={steps:5d}: Computing FID score ...")
    score = fid.compute_fid(real_dir, out_dir)
    results[steps] = round(score, 4)
    print(f"[FID]  steps={steps:5d}: FID = {score:.4f}\n")

# ── Save Results ──────────────────────────────────────────────────────────────
with open("results_step_ablation.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

# ── Print Final Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print(f"  EXPERIMENT 1: DDIM Step Ablation Results")
print("=" * 50)
print(f"  {'Steps':>8}  {'FID Score':>12}  {'Quality':>10}")
print("-" * 50)
for s, score in sorted(results.items()):
    quality = "Excellent" if score < 30 else ("Good" if score < 60 else ("Poor" if score < 200 else "Very Poor"))
    print(f"  {s:>8}  {score:>12.4f}  {quality:>10}")
print("=" * 50)
print("\n[DONE] Results saved to: results_step_ablation.json")
print("[DONE] All images saved to: outputs/ddim_steps_*/")
