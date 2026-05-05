"""
Assignment 3 — Plot Generation: All Experiment Results
=======================================================
Generates 3 publication-quality plots from experiment results.
Reads directly from JSON files produced by experiments 1 & 2.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import os
import numpy as np

os.makedirs("outputs/grids", exist_ok=True)

# ── Load Real Experiment Data ─────────────────────────────────────────────────
with open("results_step_ablation.json") as f:
    step_raw = json.load(f)

with open("results_eta_study.json") as f:
    eta_raw = json.load(f)

step_results = {int(k): v for k, v in step_raw.items()}
eta_results  = {float(k): v for k, v in eta_raw.items()}

# ── PLOT 1: FID vs Steps ──────────────────────────────────────────────────────
steps = sorted(step_results.keys())
fids  = [step_results[s] for s in steps]
best_step = min(step_results, key=step_results.get)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(steps, fids, 'o-', color='#2196F3', linewidth=2.5,
        markersize=9, label='Our Results (DDIM, eta=0)', zorder=3)
ax.axhline(y=3.17,  color='#4CAF50', linestyle='--', linewidth=1.8,
           label='DDPM Paper FID = 3.17 (Ho et al., 2020)')
ax.axvline(x=best_step, color='#F44336', linestyle=':', linewidth=1.8,
           label=f'Optimal = {best_step} steps (FID={step_results[best_step]:.2f})')

# annotate each point
for s, f in zip(steps, fids):
    ax.annotate(f'{f:.1f}', (s, f), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=8, color='#333333')

ax.set_xscale('log')
ax.set_xlabel('Number of Inference Steps (log scale)', fontsize=12)
ax.set_ylabel('FID Score  ↓  (lower is better)', fontsize=12)
ax.set_title('Experiment 1: FID vs. DDIM Inference Steps\n'
             '(CIFAR-10, eta=0, 1000 samples per configuration)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/grids/exp1_fid_vs_steps.png', dpi=150, bbox_inches='tight')
print("Saved: exp1_fid_vs_steps.png")
plt.close()

# ── PLOT 2: FID vs Eta ────────────────────────────────────────────────────────
etas     = sorted(eta_results.keys())
fids_eta = [eta_results[e] for e in etas]
best_eta = min(eta_results, key=eta_results.get)

fig, ax = plt.subplots(figsize=(9, 5))

ax.axvspan(0.0, 0.5, alpha=0.07, color='#4CAF50', label='Stable zone (FID < 66.1)')
ax.axvspan(0.5, 1.0, alpha=0.07, color='#F44336', label='Degradation zone')
ax.plot(etas, fids_eta, 's-', color='#9C27B0', linewidth=2.5,
        markersize=10, zorder=3)

for e, f in zip(etas, fids_eta):
    ax.annotate(f'{f:.2f}', (e, f), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

ax.set_xlabel('Eta  (0 = Fully Deterministic DDIM  →  1 = Fully Stochastic / DDPM-like)',
              fontsize=11)
ax.set_ylabel('FID Score  ↓  (lower is better)', fontsize=12)
ax.set_title('Experiment 2: Effect of DDIM Eta Parameter on Image Quality\n'
             '(CIFAR-10, fixed steps=10, 1000 samples)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xticks(etas)
plt.tight_layout()
plt.savefig('outputs/grids/exp2_fid_vs_eta.png', dpi=150, bbox_inches='tight')
print("Saved: exp2_fid_vs_eta.png")
plt.close()

# ── PLOT 3: Full Comparison Bar Chart ────────────────────────────────────────
labels   = ['DDPM\n1000 steps\n(A2 Baseline)',
            'DDIM\n10 steps\n(A2 Baseline)',
            'DDIM\n1 step\n(A2 Baseline)',
            f'DDIM\n{best_step} steps\n(Exp1 Optimal)',
            'DDIM eta=0\n10 steps\n(Exp2 Best)',
            'DDIM eta=1.0\n10 steps\n(Exp2 Worst)']
fid_vals = [20.91, 38.69, 462.11,
            step_results[best_step],
            eta_results[0.0],
            eta_results[1.0]]
colors   = ['#2196F3', '#FF9800', '#F44336', '#4CAF50', '#9C27B0', '#FF5722']

fig, ax = plt.subplots(figsize=(13, 6))
bars = ax.bar(labels, fid_vals, color=colors, edgecolor='black',
              linewidth=0.6, width=0.6)

ax.set_ylabel('FID Score  ↓  (lower is better)', fontsize=12)
ax.set_title('Assignment 3 — Overall Results Summary\n'
             'A2 Baselines vs A3 Experiments (CIFAR-10, 1000 samples)',
             fontsize=13)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, fid_vals):
    ax.annotate(f'{val:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5), textcoords="offset points",
                ha='center', fontsize=9, fontweight='bold')

# legend patches
legend_items = [
    mpatches.Patch(color='#2196F3', label='A2 Baselines'),
    mpatches.Patch(color='#4CAF50', label='A3 Exp1 — Step Ablation'),
    mpatches.Patch(color='#9C27B0', label='A3 Exp2 — Eta Study (Best)'),
    mpatches.Patch(color='#FF5722', label='A3 Exp2 — Eta Study (Worst)'),
]
ax.legend(handles=legend_items, fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig('outputs/grids/exp3_summary_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: exp3_summary_comparison.png")
plt.close()

# ── Final Confirmation ────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  All 3 plots saved to: outputs/grids/")
print("=" * 55)
for fname in ['exp1_fid_vs_steps.png', 'exp2_fid_vs_eta.png',
              'exp3_summary_comparison.png']:
    path = f"outputs/grids/{fname}"
    size = os.path.getsize(path) // 1024
    print(f"  {fname}  ({size} KB)")
print("=" * 55)
print(f"\n  Key findings:")
print(f"  Exp1: Best FID at {best_step} steps = {step_results[best_step]:.4f}")
print(f"  Exp2: Best eta = {best_eta:.2f}, FID = {eta_results[best_eta]:.4f}")
print(f"  Exp2: Worst eta = 1.0, FID = {eta_results[1.0]:.4f}")
