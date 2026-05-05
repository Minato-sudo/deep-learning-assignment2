import torch
import os
from torchvision.utils import save_image
from model import UNet
from diffusion import GaussianDiffusionSampler

device = torch.device('cuda:0')
logdir = './logs/DDPM_Reproduction_Attempt'
out_dir = './logs/DDPM_Reproduction_Attempt/fid_samples'
os.makedirs(out_dir, exist_ok=True)

# Load Model
model = UNet(T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)
ckpt = torch.load(os.path.join(logdir, 'ckpt.pt'))
model.load_state_dict(ckpt['ema_model'])
model.to(device).eval()

sampler = GaussianDiffusionSampler(model, 1e-4, 0.02, 1000, 32, 'epsilon', 'fixedlarge').to(device)

print("Generating 256 samples for FID...")
with torch.no_grad():
    for i in range(4): # 4 batches of 64 = 256
        x_T = torch.randn((64, 3, 32, 32)).to(device)
        samples = sampler(x_T)
        samples = (samples + 1) / 2
        for j in range(64):
            save_image(samples[j], os.path.join(out_dir, f"sample_{i*64+j}.png"))
