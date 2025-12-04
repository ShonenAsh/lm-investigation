"""
KL Divergence Analysis: Human vs AI Biber Features (Malicious Bots Dataset)

Analyzes conspiracy/engagement bot data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import entropy
import sys

# Configuration
# DATA_DIR = parent directory of datasets

if len(sys.argv) != 2:
    print("ERROR: provide data directory")
    exit(-1)

DATA_DIR = sys.argv[1]

N_BINS = 50  # Histogram bins for discretization
EPSILON = 1e-10  # Small value to avoid log(0)

def create_probability_distribution(data, feature, bins):
    """
    Convert feature values to probability distribution using histogram
    
    Args:
        data: Series of feature values
        feature: Feature name (for range calculation)
        bins: Bin edges for histogram
    
    Returns:
        Normalized probability array (sums to 1)
    """
    # Create histogram
    counts, _ = np.histogram(data, bins=bins)
    
    # Normalize to probability distribution
    prob = counts / counts.sum()
    
    # Add epsilon to avoid zero probabilities (causes inf in KL divergence)
    prob = prob + EPSILON
    prob = prob / prob.sum()  # Re-normalize after adding epsilon
    
    return prob

def load_human_data():
    """Load and concatenate human Biber feature CSVs from malicious bots dataset"""
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
        else:
            print(f"  WARNING: {f.name} not found")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total human samples: {len(combined)}")
    return combined

def load_ai_data():
    """Load and concatenate AI Biber feature CSVs from malicious bots dataset"""
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
        else:
            print(f"  WARNING: {f.name} not found")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total AI samples: {len(combined)}")
    return combined

def calculate_kl_divergence(human_df, ai_df):
    """
    Calculate KL divergence for each Biber feature
    
    KL(P || Q) = sum(P(x) * log(P(x) / Q(x)))
    where P = human distribution, Q = AI distribution
    """
    print("\nCalculating KL divergences for Biber features...")
    
    # Get feature columns (exclude doc_id)
    feature_cols = [col for col in human_df.columns if col != 'doc_id']
    
    results = []
    
    for feature in feature_cols:
        # Get data for this feature
        human_values = human_df[feature].dropna()
        ai_values = ai_df[feature].dropna()
        
        # Determine bin edges using combined range
        combined_values = pd.concat([human_values, ai_values])
        min_val = combined_values.min()
        max_val = combined_values.max()
        
        # Create uniform bin edges
        bins = np.linspace(min_val, max_val, N_BINS + 1)
        
        # Create probability distributions
        human_prob = create_probability_distribution(human_values, feature, bins)
        ai_prob = create_probability_distribution(ai_values, feature, bins)
        
        # Calculate KL divergence using scipy
        kl_div = entropy(human_prob, ai_prob)
        
        results.append({
            'feature': feature,
            'kl_divergence': kl_div
        })
        
        print(f"  {feature}: {kl_div:.6f}")
    
    return pd.DataFrame(results)

def main():
    print("="*70)
    print("KL DIVERGENCE ANALYSIS: HUMAN vs AI BIBER FEATURES")
    print("Dataset: Malicious Bots (Conspiracy/Engagement)")
    print("="*70)
    
    # Load data
    human_df = load_human_data()
    ai_df = load_ai_data()
    
    # Calculate KL divergences
    results_df = calculate_kl_divergence(human_df, ai_df)
    
    # Sort by KL divergence (descending)
    results_df = results_df.sort_values('kl_divergence', ascending=False)
    
    # Save results
    output_path = Path(__file__).parent / "kl_divergence_malicious_bots_results.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*70)
    print("RESULTS (sorted by KL divergence, highest first):")
    print("="*70)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    main()
