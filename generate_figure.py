import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('outputs/domain_transfer_results.csv')
df = df[df['domain'] != 'GLOBAL']

domains = df['domain'].str.upper().tolist()
mean_poas = df['mean_poas'].tolist()
std_poas = df['std_poas'].tolist()

colors = ['#2196F3', '#FF9800', '#4CAF50']

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(domains, mean_poas, yerr=std_poas, capsize=6,
              color=colors, edgecolor='black', linewidth=0.8,
              error_kw={'linewidth': 1.2, 'ecolor': 'black'})

for bar, val in zip(bars, mean_poas):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Mean POAS Score', fontsize=12)
ax.set_xlabel('Domain', fontsize=12)
ax.set_title('Cross-Domain Intent Alignment Results', fontsize=13, fontweight='bold')
ax.set_ylim(0, max(mean_poas) + 0.06)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/cross-domain.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved outputs/cross-domain.png")
