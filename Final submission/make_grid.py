import os
import random
from PIL import Image

def make_grid(folder, title, rows=4, cols=8):
    files = sorted(os.listdir(folder))[:rows*cols]
    imgs = [Image.open(os.path.join(folder, f)).resize((64, 64)) for f in files]
    grid = Image.new('RGB', (cols*64, rows*64))
    for i, img in enumerate(imgs):
        grid.paste(img, ((i % cols)*64, (i // cols)*64))
    return grid

os.makedirs("outputs/grids", exist_ok=True)

# Real CIFAR-10
print("Making real images grid...")
grid = make_grid("data/cifar10_real", "Real CIFAR-10")
grid.save("outputs/grids/real_cifar10.png")

# DDPM 1000 steps
print("Making DDPM 1000 steps grid...")
grid = make_grid("outputs/ddpm_5k", "DDPM 1000 steps")
grid.save("outputs/grids/ddpm_1000steps.png")

# DDIM 10 steps
print("Making DDIM 10 steps grid...")
grid = make_grid("outputs/ddim10_5k", "DDIM 10 steps")
grid.save("outputs/grids/ddim_10steps.png")

# DDIM 1 step
print("Making DDIM 1 step grid...")
grid = make_grid("outputs/ddim1_5k", "DDIM 1 step")
grid.save("outputs/grids/ddim_1step.png")

print("\nAll grids saved to outputs/grids/")
print("Files:")
for f in os.listdir("outputs/grids"):
    print(f"  {f}")
