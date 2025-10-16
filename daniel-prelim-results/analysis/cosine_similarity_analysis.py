import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
SMALL_DATASET_PATH = '../datasets/AITA_ai_smalldataset.csv'
POSTS_DATASET_PATH = '../datasets/AITAposts.csv'

def step1_join_datasets():
    """
    Step 1: Join the two datasets on the ID field keeping the original post text 
    the same and giving the original text the label of "original poster"
    """
    print("=== Step 1: Joining Datasets ===")
    
    # Load datasets
    small_df = pd.read_csv(SMALL_DATASET_PATH)
    posts_df = pd.read_csv(POSTS_DATASET_PATH)
    
    print(f"Small dataset shape: {small_df.shape}")
    print(f"Posts dataset shape: {posts_df.shape}")
    print(f"Small dataset labels: {small_df['Label'].value_counts()}")
    
    # Prepare posts dataframe to match the structure of small_df
    posts_prepared = posts_df.copy()
    posts_prepared['Response'] = posts_prepared['OriginalText']
    posts_prepared['Label'] = 'original poster'
    posts_prepared = posts_prepared[['ConvID', 'Response', 'Label']]
    
    # Combine datasets
    combined_df = pd.concat([posts_prepared, small_df], ignore_index=True)
    
    # Drop any rows where Response is NaN
    combined_df = combined_df.dropna(subset=['Response'])
    
    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Combined dataset labels: {combined_df['Label'].value_counts()}")
    
    return combined_df

def step2_add_cosine_column(df):
    """
    Step 2: Add a new column to the dataset called "cosine_similarity"
    """
    print("\n=== Step 2: Adding Cosine Similarity Column ===")
    
    # Add the cosine_similarity column initialized with NaN
    df['cosine_similarity'] = np.nan
    
    print("Added cosine_similarity column to dataset")
    print(f"Dataset shape after adding column: {df.shape}")
    
    return df

def step3_calculate_cosine_similarity(df):
    """
    Step 3: Do a cosine similarity analysis for each reply vs its parent post, based on the convID
    """
    print("\n=== Step 3: Calculating Cosine Similarity ===")
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    
    # Group by ConvID to process each conversation
    for conv_id in df['ConvID'].unique():
        conv_data = df[df['ConvID'] == conv_id].copy()
        
        # Find the original post
        original_post = conv_data[conv_data['Label'] == 'original poster']
        if len(original_post) == 0:
            continue
            
        original_text = original_post['Response'].iloc[0]
        
        # Get all replies (non-original posts)
        replies = conv_data[conv_data['Label'] != 'original poster']
        
        if len(replies) == 0:
            continue
        
        # Prepare texts for vectorization
        texts = [original_text] + replies['Response'].tolist()
        
        try:
            # Vectorize texts
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity between original post (index 0) and each reply
            original_vector = tfidf_matrix[0:1]
            reply_vectors = tfidf_matrix[1:]
            
            similarities = cosine_similarity(original_vector, reply_vectors)[0]
            
            # Update the dataframe with similarity scores
            reply_indices = replies.index
            for i, idx in enumerate(reply_indices):
                df.loc[idx, 'cosine_similarity'] = similarities[i]
                
        except Exception as e:
            print(f"Error processing ConvID {conv_id}: {e}")
            continue
    
    # Count how many similarities were calculated
    calculated_similarities = df['cosine_similarity'].notna().sum()
    print(f"Calculated cosine similarities for {calculated_similarities} replies")
    
    return df

def step4_create_human_vs_ai_histogram(df):
    """
    Step 4: Create histogram comparing (per post) the avg cosine similarity of human replies vs AI replies
    """
    print("\n=== Step 4: Creating Human vs AI Histogram ===")
    
    # Set matplotlib style similar to your marimo notebook
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Filter to only posts that have both human and AI responses
    posts_with_ai = df[df['Label'].isin(['Gemma3_4b', 'Dolphin-Mistral'])]['ConvID'].unique()
    
    # Calculate average cosine similarity per post for each type
    human_avg_per_post = []
    ai_avg_per_post = []
    
    for conv_id in posts_with_ai:
        conv_data = df[df['ConvID'] == conv_id]
        
        # Human responses
        human_responses = conv_data[conv_data['Label'] == 'Human']
        if len(human_responses) > 0:
            human_avg = human_responses['cosine_similarity'].mean()
            human_avg_per_post.append(human_avg)
        
        # AI responses (both models combined)
        ai_responses = conv_data[conv_data['Label'].isin(['Gemma3_4b', 'Dolphin-Mistral'])]
        if len(ai_responses) > 0:
            ai_avg = ai_responses['cosine_similarity'].mean()
            ai_avg_per_post.append(ai_avg)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(human_avg_per_post, bins=15, color='skyblue', ec='darkblue', lw=0.5, alpha=0.7, label='Human')
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Number of Posts")
    plt.title("Human Replies: Avg Cosine Similarity per Post")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(ai_avg_per_post, bins=15, color='lightcoral', ec='darkred', lw=0.5, alpha=0.7, label='AI')
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Number of Posts")
    plt.title("AI Replies: Avg Cosine Similarity per Post")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("human_vs_ai_cosine_similarity.png", dpi=300)
    plt.show()
    
    print(f"Human responses - Posts analyzed: {len(human_avg_per_post)}")
    print(f"Human avg similarity - Mean: {np.mean(human_avg_per_post):.4f}, Std: {np.std(human_avg_per_post):.4f}")
    print(f"AI responses - Posts analyzed: {len(ai_avg_per_post)}")
    print(f"AI avg similarity - Mean: {np.mean(ai_avg_per_post):.4f}, Std: {np.std(ai_avg_per_post):.4f}")
    
    return human_avg_per_post, ai_avg_per_post

def step5_create_model_comparison_histogram(df):
    """
    Step 5: Create histogram comparing dolphin-mistral vs gemma cosine similarity scores per post
    """
    print("\n=== Step 5: Creating Model Comparison Histogram ===")
    
    # Calculate average cosine similarity per post for each model
    dolphin_avg_per_post = []
    gemma_avg_per_post = []
    
    # Get posts that have both model responses
    posts_with_models = df[df['Label'].isin(['Gemma3_4b', 'Dolphin-Mistral'])]['ConvID'].unique()
    
    for conv_id in posts_with_models:
        conv_data = df[df['ConvID'] == conv_id]
        
        # Dolphin-Mistral responses
        dolphin_responses = conv_data[conv_data['Label'] == 'Dolphin-Mistral']
        if len(dolphin_responses) > 0:
            dolphin_avg = dolphin_responses['cosine_similarity'].mean()
            dolphin_avg_per_post.append(dolphin_avg)
        
        # Gemma responses
        gemma_responses = conv_data[conv_data['Label'] == 'Gemma3_4b']
        if len(gemma_responses) > 0:
            gemma_avg = gemma_responses['cosine_similarity'].mean()
            gemma_avg_per_post.append(gemma_avg)
    
    # Create histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(dolphin_avg_per_post, bins=15, color='lightgreen', ec='darkgreen', lw=0.5, alpha=0.7, label='Dolphin-Mistral')
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Number of Posts")
    plt.title("Dolphin-Mistral: Avg Cosine Similarity per Post")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(gemma_avg_per_post, bins=15, color='orange', ec='darkorange', lw=0.5, alpha=0.7, label='Gemma3_4b')
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Number of Posts")
    plt.title("Gemma3_4b: Avg Cosine Similarity per Post")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("dolphin_vs_gemma_cosine_similarity.png", dpi=300)
    plt.show()
    
    print(f"Dolphin-Mistral - Posts analyzed: {len(dolphin_avg_per_post)}")
    print(f"Dolphin avg similarity - Mean: {np.mean(dolphin_avg_per_post):.4f}, Std: {np.std(dolphin_avg_per_post):.4f}")
    print(f"Gemma3_4b - Posts analyzed: {len(gemma_avg_per_post)}")
    print(f"Gemma avg similarity - Mean: {np.mean(gemma_avg_per_post):.4f}, Std: {np.std(gemma_avg_per_post):.4f}")
    
    return dolphin_avg_per_post, gemma_avg_per_post

if __name__ == "__main__":
    # Step 1: Join datasets
    combined_df = step1_join_datasets()
    
    # Step 2: Add cosine similarity column
    combined_df = step2_add_cosine_column(combined_df)
    
    # Step 3: Calculate cosine similarities
    combined_df = step3_calculate_cosine_similarity(combined_df)
    
    # Display sample results
    print("\n=== Sample Results ===")
    sample_with_similarity = combined_df[combined_df['cosine_similarity'].notna()].head()
    print(sample_with_similarity[['ConvID', 'Label', 'cosine_similarity']])
    
    print(f"\nSimilarity statistics:")
    print(combined_df['cosine_similarity'].describe())
    
    # Step 4: Create human vs AI histogram
    human_avg, ai_avg = step4_create_human_vs_ai_histogram(combined_df)
    
    # Step 5: Create model comparison histogram
    dolphin_avg, gemma_avg = step5_create_model_comparison_histogram(combined_df)
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- human_vs_ai_cosine_similarity.png")
    print("- dolphin_vs_gemma_cosine_similarity.png")