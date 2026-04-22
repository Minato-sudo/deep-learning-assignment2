"""
Assignment 3 — Experiment 2: DDIM Eta Parameter Study
=======================================================
Hypothesis: eta=0 (fully deterministic DDIM) gives the best FID
at low step counts. As eta increases toward 1.0 (stochastic, DDPM-like),
quality degrades at fixed low step counts.

Proposed Method: Investigate the stochasticity-quality tradeoff
in DDIM sampling using eta = [0.0, 0.25, 0.5, 0.75, 1.0]
Fixed steps = 10 (low step regime where the effect is most visible)
"""

import torch
from diffusers import DDIMPipeline
import os
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
eta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
FIXED_STEPS = 10
results = {}

print(f"[INFO] Fixed inference steps: {FIXED_STEPS}")
print(f"[INFO] Testing eta values: {eta_values}\n")

# ── Experiment Loop ───────────────────────────────────────────────────────────
for eta in eta_values:
    eta_str = f"{eta:.2f}".replace('.', 'p')
    out_dir = f"outputs/ddim_eta_{eta_str}"
    os.makedirs(out_dir, exist_ok=True)

    existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
    needed = 1000 - existing

    if needed <= 0:
        print(f"[SKIP] eta={eta:.2f}: 1000 images already exist. Computing FID only...")
    else:
        print(f"[GEN]  eta={eta:.2f}: Generating {needed} images (have {existing}/1000, steps={FIXED_STEPS})...")
        t_start = time.time()
        idx = existing
        while idx < 1000:
            batch_size = min(32, 1000 - idx)
            with torch.no_grad():
                images = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=FIXED_STEPS,
                    eta=eta,
                    output_type="pil"
                ).images
            for img in images:
                img.save(f"{out_dir}/{idx:05d}.png")
                idx += 1
            elapsed = time.time() - t_start
            print(f"         {idx}/1000  ({elapsed:.0f}s elapsed)", end="\r")
        print(f"         {idx}/1000  Done in {time.time()-t_start:.0f}s              ")

    # Compute FID
    print(f"[FID]  eta={eta:.2f}: Computing FID score ...")
    score = fid.compute_fid(real_dir, out_dir)
    results[eta] = round(score, 4)
    print(f"[FID]  eta={eta:.2f}: FID = {score:.4f}\n")

# ── Save Results ──────────────────────────────────────────────────────────────
with open("results_eta_study.json", "w") as f:
    json.dump({str(k): v for k, v in results.items()}, f, indent=2)

# ── Print Final Table ─────────────────────────────────────────────────────────
best_eta = min(results, key=results.get)
print("\n" + "=" * 60)
print(f"  EXPERIMENT 2: DDIM Eta Parameter Study Results")
print(f"  (Fixed steps = {FIXED_STEPS}, CIFAR-10, 1000 samples)")
print("=" * 60)
print(f"  {'Eta':>8}  {'FID Score':>12}  {'Sampling Mode':>20}")
print("-" * 60)
modes = {0.0: "Fully Deterministic", 0.25: "Mostly Deterministic",
         0.5: "Mixed", 0.75: "Mostly Stochastic", 1.0: "Fully Stochastic (DDPM)"}
for e, score in sorted(results.items()):
    marker = " ← BEST" if e == best_eta else ""
    print(f"  {e:>8.2f}  {score:>12.4f}  {modes[e]:>20}{marker}")
print("=" * 60)
print(f"\n[DONE] Best eta = {best_eta:.2f} with FID = {results[best_eta]:.4f}")
print("[DONE] Results saved to: results_eta_study.json")
print("[DONE] All images saved to: outputs/ddim_eta_*/")
