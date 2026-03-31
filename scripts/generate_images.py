import torch
from diffusers import DDPMPipeline, DDIMPipeline
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Output folders
os.makedirs("outputs/ddpm_1000steps", exist_ok=True)
os.makedirs("outputs/ddim_10steps", exist_ok=True)
os.makedirs("outputs/ddim_1step", exist_ok=True)

model_id = "google/ddpm-cifar10-32"
N = 1000  # number of images to generate for FID

print("\n--- Loading DDPM model ---")
ddpm_pipe = DDPMPipeline.from_pretrained(model_id).to(device)
ddpm_pipe.unet.eval()

# --- DDPM 1000 steps ---
print(f"\n--- Generating {N} images with DDPM (1000 steps) ---")
batch_size = 64
generated = 0
with torch.no_grad():
    while generated < N:
        current_batch = min(batch_size, N - generated)
        images = ddpm_pipe(
            batch_size=current_batch,
            num_inference_steps=1000,
            output_type="pil"
        ).images
        for i, img in enumerate(images):
            img.save(f"outputs/ddpm_1000steps/{generated+i:05d}.png")
        generated += current_batch
        print(f"  Generated {generated}/{N}")
print("DDPM 1000 steps done.")

# --- DDIM 10 steps ---
print(f"\n--- Generating {N} images with DDIM (10 steps) ---")
ddim_pipe = DDIMPipeline.from_pretrained(model_id).to(device)
ddim_pipe.unet.eval()
generated = 0
with torch.no_grad():
    while generated < N:
        current_batch = min(batch_size, N - generated)
        images = ddim_pipe(
            batch_size=current_batch,
            num_inference_steps=10,
            output_type="pil"
        ).images
        for i, img in enumerate(images):
            img.save(f"outputs/ddim_10steps/{generated+i:05d}.png")
        generated += current_batch
        print(f"  Generated {generated}/{N}")
print("DDIM 10 steps done.")

# --- DDIM 1 step ---
print(f"\n--- Generating {N} images with DDIM (1 step) ---")
generated = 0
with torch.no_grad():
    while generated < N:
        current_batch = min(batch_size, N - generated)
        images = ddim_pipe(
            batch_size=current_batch,
            num_inference_steps=1,
            output_type="pil"
        ).images
        for i, img in enumerate(images):
            img.save(f"outputs/ddim_1step/{generated+i:05d}.png")
        generated += current_batch
        print(f"  Generated {generated}/{N}")
print("DDIM 1 step done.")
print("\nAll generation complete!")
