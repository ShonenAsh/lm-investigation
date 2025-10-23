"""
TF-IDF Analysis for Dolphin AI Responses
Analyzes 1k AI responses to find most meaningful words using TF-IDF scoring.
Removes stopwords and punctuation, visualizes top terms.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string

# Set matplotlib style to match codebase
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# LOAD DATA
# ============================================================================
print("Loading dataset...")
df = pd.read_csv("lm-investigation\subbredit-mixed-responses-dolphin-1k\dolph_ai_responses_1k.csv.xls")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ============================================================================
# PREPROCESSING - Remove punctuation
# ============================================================================
print("Preprocessing text (removing punctuation)...")

def preprocess_text(text):
    """Remove punctuation, digits, extra whitespace"""
    if pd.isna(text):
        return ""
    # Remove digits
    text = re.sub(r'\d+', ' ', text)
    # Remove punctuation
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

df['processed_text'] = df['ai_response'].apply(preprocess_text)
print(f"Sample processed text: {df['processed_text'].iloc[0][:150]}...\n")

# ============================================================================
# TF-IDF VECTORIZATION
# ============================================================================
print("Computing TF-IDF scores...")
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english',
    min_df=3,  # Word must appear in at least 3 documents
    lowercase=True
)

tfidf_matrix = tfidf_vectorizer.fit_transform(df['processed_text'])
feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
print(f"Vocabulary size: {len(feature_names)}\n")

# ============================================================================
# CALCULATE SUMMED TF-IDF SCORES ACROSS ALL DOCUMENTS
# ============================================================================
print("Calculating summed TF-IDF scores across corpus...")
# Sum TF-IDF scores for each term across all documents
summed_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()

# Create dataframe with terms and their summed scores
tfidf_scores_df = pd.DataFrame({
    'term': feature_names,
    'summed_tfidf_score': summed_scores
})

# Sort by score descending
tfidf_scores_df = tfidf_scores_df.sort_values('summed_tfidf_score', ascending=False)

# Save to CSV
tfidf_scores_df.to_csv('tfidf_scores.csv', index=False)
print(f"Saved TF-IDF scores to tfidf_scores.csv\n")

# Display top 20
print("="*60)
print("TOP 20 TERMS BY SUMMED TF-IDF SCORE")
print("="*60)
print(tfidf_scores_df.head(20).to_string(index=False))
print("\n")

# ============================================================================
# VISUALIZATION 1: TOP 20 TERMS TABLE
# ============================================================================
print("Creating top 20 terms table visualization...")
top_20 = tfidf_scores_df.head(20)

fig, ax = plt.subplots(figsize=(16, 12))
ax.axis('off')
fig.suptitle('Top 20 Terms by Summed TF-IDF Score', fontsize=32, fontweight='bold', y=0.98)

table_data = [[f"{i+1}", term, f"{score:.4f}"] 
              for i, (term, score) in enumerate(zip(top_20['term'], top_20['summed_tfidf_score']))]

table = ax.table(cellText=table_data,
                colLabels=['Rank', 'Term', 'Summed TF-IDF Score'],
                cellLoc='left',
                loc='center',
                colWidths=[0.15, 0.5, 0.35])

table.auto_set_font_size(False)
table.set_fontsize(20)
table.scale(1, 2.5)

for i in range(3):
    table[(0, i)].set_facecolor('skyblue')
    table[(0, i)].set_text_props(weight='bold', fontsize=24)

plt.tight_layout()
plt.savefig('tfidf_top_20_table.png', dpi=300, bbox_inches='tight')
print("Saved: tfidf_top_20_table.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: WORDCLOUD
# ============================================================================
print("Creating wordcloud visualization...")

# Create dictionary of terms and scores for wordcloud
word_freq_dict = dict(zip(tfidf_scores_df['term'], tfidf_scores_df['summed_tfidf_score']))

# Generate wordcloud
wordcloud = WordCloud(
    max_font_size=50,
    max_words=100,
    background_color="white",
    width=800,
    height=400
).generate_from_frequencies(word_freq_dict)

plt.figure(figsize=(16, 12))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Important Terms - TF-IDF Wordcloud (Top 100)", fontsize=32, fontweight='bold')
plt.tight_layout()
plt.savefig('tfidf_wordcloud.png', dpi=300, bbox_inches='tight')
print("Saved: tfidf_wordcloud.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"Total unique terms analyzed: {len(feature_names)}")
print(f"Documents analyzed: {len(df)}")
print(f"Terms appearing in 3+ documents: {len(feature_names)}")
print(f"\nOutputs generated:")
print("  - tfidf_scores.csv (all terms with scores)")
print("  - tfidf_top_20_table.png (top 20 visualization)")
print("  - tfidf_wordcloud.png (top 100 wordcloud)")
print("\n" + "="*60)
