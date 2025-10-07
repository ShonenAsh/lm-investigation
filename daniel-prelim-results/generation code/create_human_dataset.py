"""
Create Human-Only Dataset from Cornell ConvoKit

This script creates two normalized datasets:
1. Posts dataset: ConvID, Title, OriginalText
2. Responses dataset: ConvID, Response, Label

This normalized approach eliminates redundancy and enables efficient SQL-style queries.

Based on ConvoKit documentation:
- Each conversation has a title and the original post text
- Utterances include both the original post and all comments
- Original post: conversation_id == utterance_id
- Comments: conversation_id != utterance_id
"""

import os
import sys
import logging
import pandas as pd
from typing import List, Dict
from convokit import Corpus, download
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('human_dataset_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class HumanDatasetCreator:
    """Create human-only dataset from ConvoKit"""
    
    def __init__(self, subreddit: str = "subreddit-AmItheAsshole", max_conversations: int = 1000):
        self.subreddit = subreddit
        self.max_conversations = max_conversations
        
    def load_dataset(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load ConvoKit dataset"""
        logger.info(f"Loading {self.subreddit} corpus...")
        
        try:
            corpus = Corpus(filename=download(self.subreddit))
            logger.info("Corpus loaded successfully")
            corpus.print_summary_stats()
            
            # Get conversations and utterances
            conv_df = corpus.get_conversations_dataframe()
            utt_df = corpus.get_utterances_dataframe()
            
            logger.info(f"Loaded {len(conv_df)} conversations and {len(utt_df)} utterances")
            return conv_df, utt_df
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def create_normalized_datasets(self, conv_df: pd.DataFrame, utt_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create normalized posts and responses datasets"""
        logger.info("Creating normalized posts and responses datasets...")
        
        posts_data = []
        responses_data = []
        processed_conversations = 0
        
        # Limit conversations if specified
        conversations_to_process = conv_df.head(self.max_conversations) if self.max_conversations else conv_df
        
        for conv_id, conv_row in conversations_to_process.iterrows():
            try:
                # Get conversation title and original post
                title = conv_row.get('title', '')
                
                # Find the original post (where conversation_id == utterance_id)
                original_post_utterance = utt_df[utt_df.index == conv_id]
                
                if len(original_post_utterance) == 0:
                    logger.warning(f"No original post found for conversation {conv_id}")
                    continue
                
                original_post_text = original_post_utterance['text'].iloc[0]
                
                # Skip if original post is too short
                if len(original_post_text.strip()) < 20:
                    continue
                
                # Add to posts dataset
                posts_data.append({
                    'ConvID': conv_id,
                    'Title': title[:500],  # Limit title length
                    'OriginalText': original_post_text[:2000]  # Limit post length
                })
                
                # Get all comments/responses for this conversation (excluding the original post)
                comments = utt_df[
                    (utt_df['conversation_id'] == conv_id) &
                    (utt_df.index != conv_id)  # Exclude the original post
                ]['text'].tolist()
                
                # Filter valid comments and add to responses dataset
                valid_comments = [
                    comment for comment in comments
                    if comment and len(comment.strip()) > 10 and not comment.strip().startswith('[deleted]')
                ]
                
                for comment in valid_comments:
                    responses_data.append({
                        'ConvID': conv_id,
                        'Response': comment[:1500],  # Limit response length
                        'Label': 'Human'
                    })
                
                processed_conversations += 1
                
                if processed_conversations % 100 == 0:
                    logger.info(f"Processed {processed_conversations} conversations")
                    logger.info(f"  Posts: {len(posts_data)}, Responses: {len(responses_data)}")
                    
            except Exception as e:
                logger.warning(f"Error processing conversation {conv_id}: {e}")
                continue
        
        posts_df = pd.DataFrame(posts_data)
        responses_df = pd.DataFrame(responses_data)
        
        logger.info(f"Created normalized datasets:")
        logger.info(f"  Posts: {len(posts_df)} original posts")
        logger.info(f"  Responses: {len(responses_df)} human responses")
        logger.info(f"  From {processed_conversations} conversations")
        
        return posts_df, responses_df
    
    def save_datasets(self, posts_df: pd.DataFrame, responses_df: pd.DataFrame, base_filename: str = None) -> tuple[str, str]:
        """Save the normalized datasets"""
        if base_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            posts_filename = f"posts_dataset_{timestamp}.csv"
            responses_filename = f"responses_dataset_{timestamp}.csv"
        else:
            posts_filename = f"posts_{base_filename}"
            responses_filename = f"responses_{base_filename}"
        
        # Clean posts dataset
        posts_clean = posts_df.dropna(subset=['OriginalText'])
        posts_clean = posts_clean[posts_clean['OriginalText'].str.strip() != '']
        
        # Clean responses dataset
        responses_clean = responses_df.dropna(subset=['Response'])
        responses_clean = responses_clean[responses_clean['Response'].str.strip() != '']
        
        # Save both datasets
        posts_clean.to_csv(posts_filename, index=False)
        responses_clean.to_csv(responses_filename, index=False)
        
        logger.info(f"Posts dataset saved to {posts_filename}")
        logger.info(f"Responses dataset saved to {responses_filename}")
        
        # Log statistics
        logger.info(f"Final datasets:")
        logger.info(f"  Posts: {len(posts_clean)} original posts")
        logger.info(f"  Responses: {len(responses_clean)} human responses")
        logger.info(f"  Unique conversations: {posts_clean['ConvID'].nunique()}")
        
        return posts_filename, responses_filename
    
    def create_datasets(self, base_filename: str = None) -> tuple[str, str]:
        """Main method to create the normalized datasets"""
        try:
            logger.info("Starting normalized dataset creation...")
            
            # Load dataset
            conv_df, utt_df = self.load_dataset()
            
            # Create normalized datasets
            posts_df, responses_df = self.create_normalized_datasets(conv_df, utt_df)
            
            if len(posts_df) == 0:
                raise ValueError("No valid posts found in the dataset")
            
            if len(responses_df) == 0:
                raise ValueError("No valid responses found in the dataset")
            
            # Save datasets
            posts_file, responses_file = self.save_datasets(posts_df, responses_df, base_filename)
            
            logger.info("Normalized dataset creation completed successfully!")
            return posts_file, responses_file
            
        except Exception as e:
            logger.error(f"Dataset creation failed: {e}")
            raise

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Create human-only dataset from Cornell ConvoKit")
    parser.add_argument("--subreddit", default="subreddit-AmItheAsshole", 
                       help="Subreddit corpus to use (default: subreddit-AmItheAsshole)")
    parser.add_argument("--max-conversations", type=int, default=1000,
                       help="Maximum number of conversations to process (default: 1000)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output filename (default: auto-generated with timestamp)")
    
    args = parser.parse_args()
    
    try:
        # Create dataset creator instance
        creator = HumanDatasetCreator(
            subreddit=args.subreddit,
            max_conversations=args.max_conversations
        )
        
        # Create datasets
        posts_file, responses_file = creator.create_datasets(args.output)
        
        print(f"\n✅ Success! Normalized datasets created:")
        print(f"  Posts: {posts_file}")
        print(f"  Responses: {responses_file}")
        print(f"\nDataset formats:")
        print("Posts dataset:")
        print("- ConvID: Conversation ID")
        print("- Title: Post title")
        print("- OriginalText: Original post text")
        print("\nResponses dataset:")
        print("- ConvID: Conversation ID (foreign key)")
        print("- Response: Human comment/response text")
        print("- Label: 'Human' for all entries")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()