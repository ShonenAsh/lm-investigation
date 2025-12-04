"""
Kolmogorov-Smirnov Test: Human vs AI Biber Features (Malicious Bots)

Tests null hypothesis for conspiracy/engagement bot dataset.
Reports K-S statistic and p-value for each Biber feature.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
import sys

# Configuration
# DATA_DIR = parent directory of datasets

if len(sys.argv) != 2:
    print("ERROR: provide data directory")
    exit(-1)

DATA_DIR = sys.argv[1]

def load_human_data():
    """Load and concatenate human Biber feature CSVs"""
    print("Loading human data (malicious bots dataset)...")
    human_files = [
        DATA_DIR / "human_flatearth_conspiracy_biber_features.csv",
        DATA_DIR / "human_pewds_engagement_biber_features.csv"
    ]
    
    dfs = []
    for f in human_files:
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total human samples: {len(combined)}")
    return combined

def load_ai_data():
    """Load and concatenate AI Biber feature CSVs"""
    print("\nLoading AI data (malicious bots dataset)...")
    ai_files = [
        DATA_DIR / "dolphin_pewds_engagement_biber_features.csv",
        DATA_DIR / "llama_flatearth_conspiracy_biber_features.csv"
    ]
    
    dfs = []
    for f in ai_files:
        if f.exists():
            df = pd.read_csv(f)
            dfs.append(df)
            print(f"  Loaded {f.name}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total AI samples: {len(combined)}")
    return combined

def calculate_ks_tests(human_df, ai_df):
    """Perform KS test for each Biber feature"""
    print("\nPerforming KS tests on Biber features...")
    
    # Get feature columns (exclude doc_id)
    feature_cols = [col for col in human_df.columns if col != 'doc_id']
    
    results = []
    
    for feature in feature_cols:
        # Get data for this feature
        human_values = human_df[feature].dropna()
        ai_values = ai_df[feature].dropna()
        
        # Perform two-sample KS test
        statistic, p_value = ks_2samp(human_values, ai_values)
        
        results.append({
            'feature': feature,
            'ks_statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05
        })
        
        sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  {feature}: KS={statistic:.4f}, p={p_value:.6f} {sig_marker}")
    
    return pd.DataFrame(results)

def main():
    print("="*70)
    print("KS TEST ANALYSIS: HUMAN vs AI BIBER FEATURES")
    print("Dataset: Malicious Bots (Conspiracy/Engagement)")
    print("="*70)
    
    # Load data
    human_df = load_human_data()
    ai_df = load_ai_data()
    
    # Calculate KS tests
    results_df = calculate_ks_tests(human_df, ai_df)
    
    # Sort by KS statistic (descending)
    results_df = results_df.sort_values('ks_statistic', ascending=False)
    
    # Save results
    output_path = Path(__file__).parent / "ks_test_malicious_bots_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("RESULTS (sorted by KS statistic, highest first):")
    print("="*70)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    print(f"Significant features (p<0.05): {results_df['significant'].sum()}/{len(results_df)}")

if __name__ == "__main__":
    main()
