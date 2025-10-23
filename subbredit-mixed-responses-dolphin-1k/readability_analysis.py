"""
Flesch-Kincaid Readability Analysis for Dolphin AI Responses
Calculates reading grade levels across the corpus.
"""

import pandas as pd
import textstat
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

# Load data
print("Loading dataset...")
df = pd.read_csv("lm-investigation\subbredit-mixed-responses-dolphin-1k\dolph_ai_responses_1k.csv.xls")
print(f"Dataset shape: {df.shape}\n")

# Calculate Flesch-Kincaid grade level
print("Calculating Flesch-Kincaid grade levels...")
df['fk_grade'] = df['ai_response'].apply(lambda x: textstat.flesch_kincaid_grade(x) if pd.notna(x) else np.nan)
df = df.dropna(subset=['fk_grade'])

# Statistics
print("="*60)
print("READABILITY STATISTICS")
print("="*60)
print(f"Mean grade level: {df['fk_grade'].mean():.2f}")
print(f"Median grade level: {df['fk_grade'].median():.2f}")
print(f"Std deviation: {df['fk_grade'].std():.2f}")
print(f"Min: {df['fk_grade'].min():.2f}, Max: {df['fk_grade'].max():.2f}\n")

# Histogram
print("Generating histogram...")
plt.figure(figsize=(16, 12))
plt.hist(df['fk_grade'], bins=30, color='skyblue', ec='darkblue', lw=0.5, alpha=0.8)
plt.xlabel('Grade Level', fontsize=24)
plt.ylabel('Number of Responses', fontsize=24)
plt.title('Flesch-Kincaid Grade Level Distribution\nDolphin-Mistral AI Responses', 
          fontsize=32, fontweight='bold')
plt.axvline(df['fk_grade'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f"Mean: {df['fk_grade'].mean():.2f}")
plt.legend(fontsize=20)
plt.tick_params(axis='both', labelsize=20)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('flesch_kincaid_histogram.png', dpi=300, bbox_inches='tight')
print("Saved: flesch_kincaid_histogram.png")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
