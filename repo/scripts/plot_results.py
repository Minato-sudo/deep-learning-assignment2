import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

models = ['DDPM\n1000 steps', 'DDIM\n10 steps', 'DDIM\n1 step']
your_fid = [20.91, 38.69, 462.11]
paper_fid = [3.17, None, None]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, your_fid, width, 
               label='Our Results (5k samples)', 
               color=['#2196F3', '#FF9800', '#F44336'])

bars2 = ax.bar(x[0] + width/2, paper_fid[0], width,
               label='Paper Reported (Ho et al. 2020)',
               color='#4CAF50')

ax.set_xlabel('Model / Sampling Configuration', fontsize=12)
ax.set_ylabel('FID Score (lower is better)', fontsize=12)
ax.set_title('FID Score Comparison: DDPM vs DDIM Sampling\n(CIFAR-10, 32x32)', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(axis='y', alpha=0.3)

for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

ax.annotate('3.17',
            xy=(x[0] + width/2, paper_fid[0]),
            xytext=(0, 3), textcoords="offset points",
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/grids/fid_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved to outputs/grids/fid_comparison.png")
plt.close()
