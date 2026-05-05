import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os

save_dir = os.path.expanduser("~/deep_learning_a2/data/cifar10_real")
os.makedirs(save_dir, exist_ok=True)

dataset = torchvision.datasets.CIFAR10(
    root=os.path.expanduser("~/deep_learning_a2/data"),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

print(f"Saving 10000 real CIFAR-10 images to {save_dir}...")
for i in range(10000):
    img_tensor, _ = dataset[i]
    img = transforms.ToPILImage()(img_tensor)
    img.save(os.path.join(save_dir, f"{i:05d}.png"))
    if (i+1) % 1000 == 0:
        print(f"  Saved {i+1}/10000")

print("Done!")
