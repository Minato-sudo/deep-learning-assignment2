import torch
from diffusers import DDPMPipeline, DDIMPipeline
from PIL import Image
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

os.makedirs("outputs/ddpm_5k", exist_ok=True)
os.makedirs("outputs/ddim10_5k", exist_ok=True)
os.makedirs("outputs/ddim1_5k", exist_ok=True)

model_id = "google/ddpm-cifar10-32"
N = 5000
batch_size = 128

# DDPM 1000 steps
print(f"\n--- DDPM 1000 steps ({N} images) ---")
pipe = DDPMPipeline.from_pretrained(model_id).to(device)
generated = 0
with torch.no_grad():
    while generated < N:
        current_batch = min(batch_size, N - generated)
        images = pipe(batch_size=current_batch, num_inference_steps=1000, output_type="pil").images
        for i, img in enumerate(images):
            img.save(f"outputs/ddpm_5k/{generated+i:05d}.png")
        generated += current_batch
        print(f"  {generated}/{N}")
print("Done.")

# DDIM 10 steps
print(f"\n--- DDIM 10 steps ({N} images) ---")
pipe2 = DDIMPipeline.from_pretrained(model_id).to(device)
generated = 0
with torch.no_grad():
    while generated < N:
        current_batch = min(batch_size, N - generated)
        images = pipe2(batch_size=current_batch, num_inference_steps=10, output_type="pil").images
        for i, img in enumerate(images):
            img.save(f"outputs/ddim10_5k/{generated+i:05d}.png")
        generated += current_batch
        print(f"  {generated}/{N}")
print("Done.")

# DDIM 1 step
print(f"\n--- DDIM 1 step ({N} images) ---")
generated = 0
with torch.no_grad():
    while generated < N:
        current_batch = min(batch_size, N - generated)
        images = pipe2(batch_size=current_batch, num_inference_steps=1, output_type="pil").images
        for i, img in enumerate(images):
            img.save(f"outputs/ddim1_5k/{generated+i:05d}.png")
        generated += current_batch
        print(f"  {generated}/{N}")
print("Done.")
print("\nAll done!")
