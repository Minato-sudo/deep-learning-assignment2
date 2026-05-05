from cleanfid import fid
import os

real_dir = os.path.expanduser("~/deep_learning_a2/data/cifar10_real")

configs = [
    ("DDPM 1000 steps", "outputs/ddpm_1000steps"),
    ("DDIM 10 steps",   "outputs/ddim_10steps"),
    ("DDIM 1 step",     "outputs/ddim_1step"),
]

results = []
for name, gen_dir in configs:
    print(f"\nComputing FID: {name} ...")
    score = fid.compute_fid(real_dir, gen_dir)
    results.append((name, score))
    print(f"  FID = {score:.4f}")

print("\n" + "="*45)
print(f"{'Model':<25} {'FID Score':>10}")
print("="*45)
for name, score in results:
    print(f"{name:<25} {score:>10.4f}")
print("="*45)

# Save to log file
with open("fid_results.txt", "w") as f:
    f.write("FID Results\n")
    f.write("="*45 + "\n")
    f.write(f"{'Model':<25} {'FID Score':>10}\n")
    f.write("="*45 + "\n")
    for name, score in results:
        f.write(f"{name:<25} {score:>10.4f}\n")
    f.write("="*45 + "\n")

print("\nResults saved to fid_results.txt")
