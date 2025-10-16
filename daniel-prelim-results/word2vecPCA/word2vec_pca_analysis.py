"""
Word2Vec PCA Analysis for AITA Dataset
Preprocesses text (lowercase only), trains word2vec embeddings,
and performs PCA with linear and RBF kernels to identify clusters.
"""

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading dataset...")
df = pd.read_csv("../datasets/AITA_ai_smalldataset.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nLabel distribution:\n{df['Label'].value_counts()}\n")

# ============================================================================
# PREPROCESSING - Lowercase only, keep punctuation and stop words
# ============================================================================
print("Preprocessing text (lowercase only)...")

def preprocess_text(text):
    """Convert to lowercase only - keep punctuation and stop words per orders"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()

df['processed_text'] = df['Response'].apply(preprocess_text)
df['tokens'] = df['processed_text'].apply(lambda x: x.split())

print(f"Sample processed text: {df['processed_text'].iloc[0][:150]}...")
print(f"Sample tokens: {df['tokens'].iloc[0][:10]}\n")

# ============================================================================
# TRAIN WORD2VEC MODEL
# ============================================================================
print("Training Word2Vec model...")
sentences = df['tokens'].tolist()

w2v_model = Word2Vec(
    sentences=sentences,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window
    min_count=2,          # Ignore words with frequency < 2
    workers=4,            # Parallel processing
    epochs=10,            # Training epochs
    sg=0                  # CBOW algorithm (sg=1 for Skip-gram)
)

print(f"Vocabulary size: {len(w2v_model.wv)}")
print(f"Vector size: {w2v_model.wv.vector_size}\n")

# ============================================================================
# CREATE DOCUMENT EMBEDDINGS
# ============================================================================
print("Creating document embeddings via word vector averaging...")

def get_document_vector(tokens, model):
    """Average word vectors for all words in document"""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.wv.vector_size)

doc_embeddings = np.array([get_document_vector(tokens, w2v_model) for tokens in df['tokens']])
print(f"Document embeddings shape: {doc_embeddings.shape}\n")

# ============================================================================
# STANDARDIZE EMBEDDINGS
# ============================================================================
print("Standardizing embeddings...")
scaler = StandardScaler()
doc_embeddings_scaled = scaler.fit_transform(doc_embeddings)

# ============================================================================
# LINEAR PCA
# ============================================================================
print("Performing Linear PCA...")
pca_linear = PCA(n_components=2)
pca_linear_result = pca_linear.fit_transform(doc_embeddings_scaled)

print(f"Linear PCA - Explained variance ratio: {pca_linear.explained_variance_ratio_}")
print(f"Linear PCA - Total variance explained: {pca_linear.explained_variance_ratio_.sum():.4f}\n")

# ============================================================================
# RBF KERNEL PCA (Exponential Kernel)
# ============================================================================
print("Performing RBF Kernel PCA...")
pca_rbf = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
pca_rbf_result = pca_rbf.fit_transform(doc_embeddings_scaled)
print(f"RBF Kernel PCA completed - Shape: {pca_rbf_result.shape}\n")

# ============================================================================
# VISUALIZATION - LINEAR PCA
# ============================================================================
print("Generating Linear PCA visualization...")
color_map = {'Human': 'skyblue', 'Dolphin-Mistral': 'coral', 'Gemma3_4b': 'lightgreen'}
colors = df['Label'].map(color_map)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_linear_result[:, 0], pca_linear_result[:, 1], 
                     c=colors, alpha=0.6, edgecolors='black', linewidth=0.5, s=50)

# Create legend
for label, color in color_map.items():
    plt.scatter([], [], c=color, label=label, edgecolors='black', linewidth=0.5, s=50)

plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('Word2Vec PCA - Linear Kernel\nClusters by Label Type', fontsize=14, fontweight='bold')
plt.legend(title='Label', fontsize=10, title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_pca_clusters.png', dpi=300, bbox_inches='tight')
print("Saved: linear_pca_clusters.png")
plt.close()

# ============================================================================
# VISUALIZATION - RBF KERNEL PCA
# ============================================================================
print("Generating RBF Kernel PCA visualization...")

plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_rbf_result[:, 0], pca_rbf_result[:, 1], 
                     c=colors, alpha=0.6, edgecolors='black', linewidth=0.5, s=50)

# Create legend
for label, color in color_map.items():
    plt.scatter([], [], c=color, label=label, edgecolors='black', linewidth=0.5, s=50)

plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('Word2Vec PCA - RBF (Exponential) Kernel\nClusters by Label Type', fontsize=14, fontweight='bold')
plt.legend(title='Label', fontsize=10, title_fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('rbf_pca_clusters.png', dpi=300, bbox_inches='tight')
print("Saved: rbf_pca_clusters.png")
plt.close()

# ============================================================================
# STATISTICAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("CLUSTER ANALYSIS SUMMARY")
print("="*70)

for label in df['Label'].unique():
    mask = df['Label'] == label
    linear_coords = pca_linear_result[mask]
    rbf_coords = pca_rbf_result[mask]
    
    print(f"\n{label}:")
    print(f"  Count: {mask.sum()}")
    print(f"  Linear PCA - Mean: ({linear_coords[:, 0].mean():.3f}, {linear_coords[:, 1].mean():.3f})")
    print(f"  Linear PCA - Std:  ({linear_coords[:, 0].std():.3f}, {linear_coords[:, 1].std():.3f})")
    print(f"  RBF PCA - Mean:    ({rbf_coords[:, 0].mean():.3f}, {rbf_coords[:, 1].mean():.3f})")
    print(f"  RBF PCA - Std:     ({rbf_coords[:, 0].std():.3f}, {rbf_coords[:, 1].std():.3f})")

print("\n" + "="*70)
print("MISSION COMPLETE - All visualizations generated")
print("="*70)
