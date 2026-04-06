import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('outputs/figures', exist_ok=True)

# ============================================================
# FIGURE 1: SYSTEM ARCHITECTURE DIAGRAM
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

boxes = [
    (0.5, 2.0, 2.0, 1.5, 'Input Prompt\n(e.g. "piano melody\nin a quiet room")', '#E3F2FD'),
    (3.0, 2.0, 2.0, 1.5, 'RAG Enhancer\nKeyword Matching\n+ Local KB', '#E8F5E9'),
    (5.5, 2.0, 2.0, 1.5, 'CLAP Text Encoder\nFrozen HTSAT-tiny\nEmbedding c', '#FFF3E0'),
    (8.0, 2.0, 2.0, 1.5, 'AudioLDM 2\nLatent Diffusion\nDDIM Sampling', '#F3E5F5'),
    (10.5, 2.0, 2.0, 1.5, 'HiFi-GAN\nDecoder\nWaveform Output', '#E0F2F1'),
    (10.5, 4.2, 2.0, 1.5, 'CLAP Audio\nEncoder\nEmbedding f_a', '#FFEBEE'),
    (8.0, 4.2, 2.0, 1.5, 'GIAF Metrics\nPOAS / CRI / CDTS\nEvaluation', '#FCE4EC'),
]

for x, y, w, h, text, color in boxes:
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='#333', linewidth=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)

arrows = [
    (2.5, 2.75, 3.0, 2.75),
    (5.0, 2.75, 5.5, 2.75),
    (7.5, 2.75, 8.0, 2.75),
    (10.0, 2.75, 10.5, 2.75),
    (11.5, 2.75, 11.5, 4.2),
    (10.5, 4.95, 10.0, 4.95),
]

for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))

ax.text(2.75, 3.1, 'Enhanced\nPrompt', ha='center', fontsize=7, style='italic', color='#1565C0')
ax.text(5.25, 3.1, 'Text\nEmbedding', ha='center', fontsize=7, style='italic', color='#1565C0')
ax.text(7.75, 3.1, 'Conditioned\nDenoising', ha='center', fontsize=7, style='italic', color='#1565C0')
ax.text(10.25, 3.1, 'Mel-Spectrogram', ha='center', fontsize=7, style='italic', color='#1565C0')
ax.text(11.8, 3.5, 'Audio\nFile', ha='center', fontsize=7, style='italic', color='#1565C0')
ax.text(10.25, 5.2, 'Audio\nEmbedding', ha='center', fontsize=7, style='italic', color='#1565C0')

ax.text(7, 5.7, 'Unified Cross-Domain Text-to-Audio Generation & Evaluation Pipeline',
        ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/system_architecture.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: system_architecture.png")

# ============================================================
# FIGURE 2: RAG KNOWLEDGE BASE VISUALIZATION
# ============================================================
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

keywords = ['guitar', 'piano', 'techno', 'speech', 'storm', 'restaurant', 'bird']
descriptors = [
    'crisp acoustic resonance,\nrich harmonics, 44.1kHz',
    'grand piano, sustained pedal,\nclassical reverb, hi-fi',
    '128 bpm, sidechain comp.,\n808 kick, sub-bass',
    'close-mic, voice-over quality,\nclear diction, noise-free',
    'low freq rumble, binaural,\nimmersive surround',
    'ambient field recording,\nstereo width, clinking',
    'outdoor field recording,\nhigh freq trills, wide stereo'
]

ax.text(6, 6.5, 'RAG Prompt Enhancement via Local Knowledge Base',
        ha='center', fontsize=14, fontweight='bold')

ax.text(1.5, 5.8, 'Input Prompt', ha='center', fontsize=11, fontweight='bold')
ax.text(1.5, 5.2, '"A cheerful pop song\nwith fast tempo\nand bright piano"',
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='#E3F2FD', edgecolor='#1565C0'))

ax.annotate('', xy=(3.5, 5.2), xytext=(2.5, 5.2),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
ax.text(3.0, 5.5, 'Keyword\nMatch', ha='center', fontsize=8, color='#1565C0', fontweight='bold')

for i, (kw, desc) in enumerate(zip(keywords, descriptors)):
    y = 4.5 - i * 0.55
    color = '#FF9800' if kw == 'piano' else '#BDBDBD'
    lw = 2.5 if kw == 'piano' else 1.0
    rect = plt.Rectangle((4.0, y - 0.2), 1.2, 0.45, facecolor=color, edgecolor='#333', linewidth=lw)
    ax.add_patch(rect)
    ax.text(4.6, y, kw, ha='center', va='center', fontsize=8, fontweight='bold' if kw == 'piano' else 'normal')
    
    ax.annotate('', xy=(6.5, y), xytext=(5.2, y),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.0))
    
    rect2 = plt.Rectangle((6.5, y - 0.2), 3.0, 0.45, facecolor='#E8F5E9', edgecolor='#333', linewidth=lw)
    ax.add_patch(rect2)
    ax.text(8.0, y, desc, ha='center', va='center', fontsize=7)

ax.text(10.5, 5.2, 'Enhanced Prompt', ha='center', fontsize=11, fontweight='bold')
ax.annotate('', xy=(10.5, 4.8), xytext=(9.5, 4.8),
            arrowprops=dict(arrowstyle='->', color='#1565C0', lw=2))
ax.text(10.5, 4.2, '"A cheerful pop song with\nfast tempo and bright piano\n| Features: grand piano,\nsustained pedal, classical\nreverb, high fidelity audio"',
        ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='#E8F5E9', edgecolor='#4CAF50'))

plt.tight_layout()
plt.savefig('outputs/figures/rag_enhancement.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: rag_enhancement.png")

# ============================================================
# FIGURE 3: CROSS-DOMAIN RESULTS (IMPROVED)
# ============================================================
import pandas as pd
df = pd.read_csv('outputs/domain_transfer_results.csv')
df = df[df['domain'] != 'GLOBAL']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

domains = df['domain'].str.upper().tolist()
mean_poas = df['mean_poas'].tolist()
std_poas = df['std_poas'].tolist()
colors = ['#2196F3', '#FF9800', '#4CAF50']

bars = axes[0].bar(domains, mean_poas, yerr=std_poas, capsize=8,
                    color=colors, edgecolor='black', linewidth=1.0,
                    error_kw={'linewidth': 1.5, 'ecolor': '#333'})

for bar, val in zip(bars, mean_poas):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

axes[0].set_ylabel('Mean POAS Score', fontsize=12)
axes[0].set_xlabel('Domain', fontsize=12)
axes[0].set_title('Per-Domain Intent Alignment', fontsize=13, fontweight='bold')
axes[0].set_ylim(0, max(mean_poas) + 0.05)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)
axes[0].grid(axis='y', alpha=0.3)

global_data = df[df['domain'] == 'GLOBAL'] if 'GLOBAL' in df['domain'].str.upper().tolist() else None
cri = 0.0891
cdts = 0.1710

axes[1].bar(['CRI\n(Robustness)', 'CDTS\n(Transfer)'], [cri, cdts],
            color=['#F44336', '#2196F3'], edgecolor='black', linewidth=1.0)
axes[1].text(0, cri + 0.003, f'{cri:.4f}', ha='center', fontsize=11, fontweight='bold')
axes[1].text(1, cdts + 0.003, f'{cdts:.4f}', ha='center', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Global Aggregate Metrics', fontsize=13, fontweight='bold')
axes[1].set_ylim(0, max(cri, cdts) + 0.03)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/cross-domain-results.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: cross-domain-results.png")

# ============================================================
# FIGURE 4: DIFFUSION PROCESS VISUALIZATION
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(14, 3))

timesteps = [0, 50, 100, 150, 200]
np.random.seed(42)
clean_signal = np.sin(np.linspace(0, 4*np.pi, 200)) * 0.5 + 0.3 * np.sin(np.linspace(0, 8*np.pi, 200))

for i, t in enumerate(timesteps):
    noise_level = t / 200.0
    noisy = clean_signal + noise_level * np.random.randn(200) * 0.5
    axes[i].plot(noisy, linewidth=0.8, color='#1565C0')
    axes[i].set_title(f't = {t}', fontsize=11, fontweight='bold')
    axes[i].set_ylim(-2, 2)
    axes[i].set_xticks([])
    axes[i].set_yticks([])
    if i == 0:
        axes[i].set_ylabel('Amplitude', fontsize=10)
    if i == 2:
        axes[i].set_xlabel('Time Steps', fontsize=10)

fig.suptitle('Forward Diffusion Process: Clean Audio to Gaussian Noise', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/figures/diffusion_process.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: diffusion_process.png")

# ============================================================
# FIGURE 5: GIAF METRICS FLOWCHART
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

ax.text(5, 7.5, 'Generative Intent Alignment Framework (GIAF)',
        ha='center', fontsize=15, fontweight='bold')

metrics = [
    (1.5, 5.5, 'POAS\n(Prompt-to-Audio Similarity)', 
     'Cosine similarity between\nCLAP text & audio embeddings',
     'Per-sample semantic\nalignment score', '#E3F2FD'),
    (5.0, 5.5, 'CRI\n(Cross-domain Robustness Index)',
     'Global std deviation of\nPOAS across all samples',
     'Measures domain-invariant\nconsistency', '#FFF3E0'),
    (8.5, 5.5, 'CDTS\n(Cross-domain Transfer Score)',
     'Global mean of all\nPOAS scores',
     'Overall cross-domain\nalignment figure of merit', '#E8F5E9'),
]

for x, y, title, formula, desc, color in metrics:
    rect = plt.Rectangle((x - 1.3, y - 1.0), 2.6, 2.5, facecolor=color, edgecolor='#333', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y + 0.8, title, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(x, y, formula, ha='center', va='center', fontsize=8, style='italic')
    ax.text(x, y - 0.7, desc, ha='center', va='center', fontsize=8)

ax.annotate('', xy=(5, 5.5), xytext=(2.8, 5.5),
            arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=1.5))
ax.annotate('', xy=(7.2, 5.5), xytext=(5, 5.5),
            arrowprops=dict(arrowstyle='<->', color='#1565C0', lw=1.5))

ax.text(5, 3.5, 'Together: POAS identifies per-sample failures,\nCRI localises domain-specific weaknesses,\nCDTS summarises global alignment',
        ha='center', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#FCE4EC', edgecolor='#E91E63'))

ax.text(5, 2.0, 'Lower CRI = More consistent across domains\nHigher CDTS = Better overall semantic fidelity',
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#E0F2F1', edgecolor='#009688'))

plt.tight_layout()
plt.savefig('outputs/figures/giaf_framework.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: giaf_framework.png")

print("\nAll figures generated in outputs/figures/")
