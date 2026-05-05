from cleanfid import fid
import os

real_dir = os.path.expanduser("~/deep_learning_a2/data/cifar10_real")

configs = [
    ("DDPM 1000 steps (5k)", "outputs/ddpm_5k"),
    ("DDIM 10 steps  (5k)", "outputs/ddim10_5k"),
    ("DDIM 1 step    (5k)", "outputs/ddim1_5k"),
]

results = []
for name, gen_dir in configs:
    print(f"\nComputing FID: {name} ...")
    score = fid.compute_fid(real_dir, gen_dir)
    results.append((name, score))
    print(f"  FID = {score:.4f}")

print("\n" + "="*50)
print(f"{'Model':<30} {'FID Score':>10}")
print("="*50)
for name, score in results:
    print(f"{name:<30} {score:>10.4f}")
print("="*50)

with open("fid_results_5k.txt", "w") as f:
    f.write("FID Results (5000 samples)\n")
    f.write("="*50 + "\n")
    f.write(f"{'Model':<30} {'FID Score':>10}\n")
    f.write("="*50 + "\n")
    for name, score in results:
        f.write(f"{name:<30} {score:>10.4f}\n")
    f.write("="*50 + "\n")

print("\nSaved to fid_results_5k.txt")

