"""
Assignment 3 — Experiment 3: Cross-Domain Evaluation (CelebA-HQ 256x256)
=========================================================================
Goal: Evaluate DDPM generalization on a completely different domain.
Dataset: CelebA-HQ 256x256 (face images) via google/ddpm-celebahq-256
This satisfies the mandatory "additional dataset" requirement.

Comparison:
- CIFAR-10 domain: small natural images (32x32, 10 classes)
- CelebA-HQ domain: high-resolution face images (256x256)
"""

import torch
from diffusers import DDPMPipeline
import os
from cleanfid import fid
import json
import time

# ── Setup ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")
print(f"[INFO] Loading model: google/ddpm-celebahq-256 ...")
print(f"[INFO] Note: This model generates 256x256 face images.")
print(f"[INFO] Batch size is small (4) due to higher memory requirements.\n")

pipeline = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(device)
print(f"[INFO] Model loaded successfully.\n")

out_dir = "outputs/celebahq_generated"
os.makedirs(out_dir, exist_ok=True)

TARGET = 500  # 500 images — sufficient for FID, feasible for 256x256
existing = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
needed = TARGET - existing

print(f"[INFO] Target: {TARGET} images | Existing: {existing} | Needed: {needed}")

# ── Image Generation ──────────────────────────────────────────────────────────
if needed > 0:
    print(f"[GEN]  Generating {needed} CelebA-HQ images (256x256, 1000 steps)...")
    print(f"[INFO] Estimated time: ~45-90 minutes depending on GPU.\n")
    t_start = time.time()
    idx = existing
    while idx < TARGET:
        batch_size = min(4, TARGET - idx)  # small batch for 256x256
        with torch.no_grad():
            images = pipeline(
                batch_size=batch_size,
                num_inference_steps=1000,
                output_type="pil"
            ).images
        for img in images:
            img.save(f"{out_dir}/{idx:05d}.png")
            idx += 1
        elapsed = time.time() - t_start
        eta_mins = ((elapsed / idx) * (TARGET - idx)) / 60 if idx > 0 else 0
        print(f"         {idx}/{TARGET}  |  {elapsed/60:.1f} min elapsed  |  ~{eta_mins:.1f} min remaining",
              end="\r")
    print(f"\n[GEN]  Done! {idx} images generated in {(time.time()-t_start)/60:.1f} minutes.")
else:
    print(f"[SKIP] {existing} images already exist. Skipping to FID computation.")

# ── Verify images ─────────────────────────────────────────────────────────────
actual_count = len([f for f in os.listdir(out_dir) if f.endswith('.png')])
print(f"\n[INFO] Verifying images in {out_dir} ...")
print(f"[INFO] Total images found: {actual_count}")

# ── FID Computation ───────────────────────────────────────────────────────────
print(f"\n[FID]  Computing FID for CelebA-HQ generated images ...")
print(f"[INFO] Using clean-fid built-in CelebA-HQ-256 reference statistics ...")

try:
    score = fid.compute_fid(
        out_dir,
        dataset_name="celebahq_256",
        dataset_res=256,
        dataset_split="train"
    )
    print(f"[FID]  CelebA-HQ FID (vs official stats): {score:.4f}")
    result = {"celebahq_fid": round(score, 4), "num_images": actual_count,
              "method": "DDPM 1000 steps", "resolution": "256x256"}

except Exception as e:
    print(f"[WARN] Built-in stats unavailable: {e}")
    print(f"[INFO] This is OK — we document cross-domain generation qualitatively.")
    print(f"[INFO] Images are saved and can be shown visually in the report.")
    score = None
    result = {"celebahq_fid": None, "num_images": actual_count,
              "method": "DDPM 1000 steps", "resolution": "256x256",
              "note": "FID stats unavailable; qualitative evaluation used"}

# ── Save Results ──────────────────────────────────────────────────────────────
with open("results_crossdomain.json", "w") as f:
    json.dump(result, f, indent=2)

# ── Final Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  EXPERIMENT 3: Cross-Domain Evaluation (CelebA-HQ 256x256)")
print("=" * 60)
print(f"  Model used      : google/ddpm-celebahq-256")
print(f"  Images generated: {actual_count}")
print(f"  Resolution      : 256 x 256 pixels")
print(f"  Inference steps : 1000 (full DDPM)")
if score is not None:
    print(f"  FID Score       : {score:.4f}")
else:
    print(f"  FID Score       : Qualitative evaluation (see images)")
print("=" * 60)
print("\n[DONE] Results saved to: results_crossdomain.json")
print("[DONE] Images saved to: outputs/celebahq_generated/")
print("[DONE] View sample images: ls outputs/celebahq_generated/ | head -5")
