"""
N-gram Probability Analysis for Dolphin AI Responses
Trains unigram, bigram, and trigram models to find highest probability sequences.
Keeps stopwords, removes punctuation.
"""

import pandas as pd
import re
import string
from collections import Counter
from gensim.models import Phrases
from gensim.models.phrases import Phraser
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid')

# Load and preprocess
print("Loading dataset...")
df = pd.read_csv("lm-investigation\subbredit-mixed-responses-dolphin-1k\dolph_ai_responses_1k.csv.xls")

def preprocess(text):
    if pd.isna(text): return ""
    text = re.sub(rf'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\s+', ' ', text).lower().strip()
    return text

df['processed'] = df['ai_response'].apply(preprocess)
tokenized = [text.split() for text in df['processed'] if text]

# Unigrams
print("\n" + "="*60)
print("UNIGRAM ANALYSIS")
print("="*60)
all_words = [word for doc in tokenized for word in doc]
unigram_counts = Counter(all_words)
top_unigrams = unigram_counts.most_common(20)
total_words = sum(unigram_counts.values())
unigram_probs = [(word, count/total_words) for word, count in top_unigrams]

print("\nTop 20 Unigrams:")
for word, prob in unigram_probs[:10]:
    print(f"  {word}: {prob:.6f}")

# Bigrams
print("\n" + "="*60)
print("BIGRAM ANALYSIS")
print("="*60)
bigram_model = Phrases(tokenized, min_count=3, threshold=1)
bigram_phraser = Phraser(bigram_model)
bigram_docs = [bigram_phraser[doc] for doc in tokenized]

bigrams = [word for doc in bigram_docs for word in doc if '_' in word]
bigram_counts = Counter(bigrams)
top_bigrams = bigram_counts.most_common(20)
total_bigrams = sum(bigram_counts.values())
bigram_probs = [(bg, count/total_bigrams) for bg, count in top_bigrams]

print("\nTop 20 Bigrams:")
for bg, prob in bigram_probs[:10]:
    print(f"  {bg}: {prob:.6f}")

# Stopword analysis for bigrams
stopwords = {'the', 'a', 'an', 'to', 'of', 'in', 'for', 'you', 'your', 'is'}
print("\nBigrams after common stopwords:")
for bg in [b for b, _ in top_bigrams]:
    parts = bg.split('_')
    if len(parts) == 2 and parts[0] in stopwords:
        print(f"  '{parts[0]}' -> '{parts[1]}'")

# Trigrams
print("\n" + "="*60)
print("TRIGRAM ANALYSIS")
print("="*60)
trigram_model = Phrases(bigram_docs, min_count=3, threshold=1)
trigram_phraser = Phraser(trigram_model)
trigram_docs = [trigram_phraser[doc] for doc in bigram_docs]

trigrams = [word for doc in trigram_docs for word in doc if word.count('_') >= 2]
trigram_counts = Counter(trigrams)
top_trigrams = trigram_counts.most_common(20)
total_trigrams = sum(trigram_counts.values())
trigram_probs = [(tg, count/total_trigrams) for tg, count in top_trigrams]

print("\nTop 20 Trigrams:")
for tg, prob in trigram_probs[:10]:
    print(f"  {tg}: {prob:.6f}")

print("\nTrigrams after common stopwords:")
for tg in [t for t, _ in top_trigrams]:
    parts = tg.split('_')
    if len(parts) >= 2 and parts[0] in stopwords:
        print(f"  '{parts[0]}' -> '{parts[1]}' -> ...")

# Create visualizations
def create_table(data, title, filename):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    table_data = [[f"{i+1}", term, f"{prob:.6f}"] for i, (term, prob) in enumerate(data)]
    table = ax.table(cellText=table_data, colLabels=['Rank', 'N-gram', 'Probability'],
                    cellLoc='left', loc='center', colWidths=[0.15, 0.55, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    for i in range(3):
        table[(0, i)].set_facecolor('skyblue')
        table[(0, i)].set_text_props(weight='bold', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

print("\nGenerating visualizations...")
create_table(unigram_probs, 'Top 20 Unigrams by Probability', 'unigram_top20.png')
create_table(bigram_probs, 'Top 20 Bigrams by Probability', 'bigram_top20.png')
create_table(trigram_probs, 'Top 20 Trigrams by Probability', 'trigram_top20.png')

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
