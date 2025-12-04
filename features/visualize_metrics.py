"""
Visualization: KL Divergence vs KS Statistic for Malicious Bots Dataset
Simple scatter plots comparing the two metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

if len(sys.argv) != 4:
    print('ERROR: missing or unexpected arguments')
    exit(-1)

# Config
# KL and KS files are CSVs that are outputs of kl_divergence.py and ks_test.py
# Output_dir is the target directory for plots
KL_FILE = sys.argv[1]
KS_FILE = sys.argv[2]
OUTPUT_DIR = sys.argv[3]

# Load data
kl_df = pd.read_csv(KL_FILE)
ks_df = pd.read_csv(KS_FILE)

# Merge on feature name
merged = pd.merge(kl_df, ks_df, on='feature', suffixes=('_kl', '_ks'))

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: KL Divergence scatter
ax1 = axes[0]
ax1.scatter(range(len(kl_df)), kl_df['kl_divergence'], 
            s=100, alpha=0.7, c='#2E86AB', edgecolors='black', linewidth=1)
ax1.set_xlabel('Feature Index (sorted by KL)', fontsize=16, fontweight='bold')
ax1.set_ylabel('KL Divergence', fontsize=16, fontweight='bold')
ax1.set_title('KL Divergence: Human vs AI (Malicious Bots)', fontsize=18, fontweight='bold')
ax1.grid(True, alpha=0.3, linewidth=1.5)
ax1.tick_params(labelsize=14)

# Plot 2: KS Statistic scatter
ax2 = axes[1]
ks_sorted = ks_df.sort_values('ks_statistic', ascending=False)
ax2.scatter(range(len(ks_sorted)), ks_sorted['ks_statistic'],
            s=100, alpha=0.7, c='#A23B72', edgecolors='black', linewidth=1)
ax2.set_xlabel('Feature Index (sorted by KS)', fontsize=16, fontweight='bold')
ax2.set_ylabel('KS Statistic', fontsize=16, fontweight='bold')
ax2.set_title('KS Statistic: Human vs AI (Malicious Bots)', fontsize=18, fontweight='bold')
ax2.grid(True, alpha=0.3, linewidth=1.5)
ax2.tick_params(labelsize=14)

plt.tight_layout()
output_path = OUTPUT_DIR / "kl_ks_scatter_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")

# Plot 3: Direct comparison scatter (KL vs KS)
fig2, ax = plt.subplots(figsize=(10, 10))
ax.scatter(merged['kl_divergence'], merged['ks_statistic'],
           s=120, alpha=0.7, c='#F18F01', edgecolors='black', linewidth=1.5)
ax.set_xlabel('KL Divergence', fontsize=18, fontweight='bold')
ax.set_ylabel('KS Statistic', fontsize=18, fontweight='bold')
ax.set_title('KL Divergence vs KS Statistic\n(Malicious Bots Dataset)', 
             fontsize=20, fontweight='bold')
ax.grid(True, alpha=0.3, linewidth=1.5)
ax.tick_params(labelsize=16)

output_path2 = OUTPUT_DIR / "kl_vs_ks_direct_comparison.png"
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path2}")

plt.show()
print("\nVisualization complete!")
