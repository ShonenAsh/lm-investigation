"""
AI Response Augmentation Script

This script augments an existing responses dataset with AI-generated responses from Ollama models.
It reads the posts dataset, generates responses using Gemma3:4b and Dolphin-Mistral models,
and appends them to the responses dataset.

For each original post, it generates:
- 3 responses from Gemma3:4b (labeled as "Gemma3_4b")
- 3 responses from Dolphin-Mistral (labeled as "Dolphin-Mistral")
"""

import os
import sys
import json
import time
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ai_augmentation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Test immediate output
print("🚀 AI Augmentation Script Starting...", flush=True)

class OllamaClient:
    """Client for interacting with locally hosted Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 120  # 2 minute timeout
        
    def check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return False
    
    def list_models(self) -> List[str]:
        """List available models"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def generate_response(self, model: str, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Generate response from specified model"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.8,  # Slightly higher for more variety
                        "top_p": 0.9
                    }
                }
                
                response = self.session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '').strip()
                else:
                    logger.warning(f"Attempt {attempt + 1}: HTTP {response.status_code} for model {model}")
                    
            except requests.exceptions.RequestException as e:
                logger.warning(f"Attempt {attempt + 1}: Request failed for model {model}: {e}")
                
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"Failed to generate response from {model} after {max_retries} attempts")
        return None

class AIResponseAugmenter:
    """Augment responses dataset with AI-generated responses"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434", max_posts: int = None, responses_per_model: int = 3):
        self.ollama_client = OllamaClient(ollama_url)
        self.models = {
            'gemma3': 'gemma3:4b',
            'mistral': 'dolphin-mistral:latest'
        }
        self.responses_per_model = responses_per_model
        self.max_posts = max_posts
        
    def load_datasets(self, posts_file: str, responses_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load posts and responses datasets"""
        logger.info(f"Loading datasets...")
        
        try:
            posts_df = pd.read_csv(posts_file)
            responses_df = pd.read_csv(responses_file)
            
            logger.info(f"Loaded {len(posts_df)} posts and {len(responses_df)} existing responses")
            return posts_df, responses_df
            
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            raise
    
    def create_prompt(self, title: str, original_text: str) -> str:
        """Create a prompt for AI models based on the original post"""
        combined_post = f"{title}\n\n{original_text}".strip()
        
        return f"""You are responding to a Reddit post. Please provide a thoughtful, helpful response that addresses the main points of the post. Keep your response conversational, under 300 words, and appropriate for the context.

Post: {combined_post}

Response:"""
    
    def generate_ai_responses(self, posts_df: pd.DataFrame) -> List[Dict]:
        """Generate AI responses for all posts"""
        logger.info("Starting AI response generation...")
        
        # Check Ollama connection
        if not self.ollama_client.check_connection():
            raise ConnectionError("Cannot connect to Ollama server. Please ensure it's running.")
        
        # Check available models
        available_models = self.ollama_client.list_models()
        logger.info(f"Available models: {available_models}")
        
        # Verify required models are available
        missing_models = []
        for model_name, model_id in self.models.items():
            if not any(model_id in available for available in available_models):
                missing_models.append(model_id)
        
        if missing_models:
            logger.warning(f"Missing models: {missing_models}")
            logger.info("You may need to pull these models using: ollama pull <model_name>")
        
        # Limit posts if max_posts is specified
        if self.max_posts and self.max_posts < len(posts_df):
            posts_df = posts_df.head(self.max_posts)
            logger.info(f"Limited to first {self.max_posts} posts")
        
        new_responses = []
        total_posts = len(posts_df)
        total_responses_to_generate = total_posts * len(self.models) * self.responses_per_model
        responses_generated = 0
        start_time = time.time()
        
        logger.info(f"Will generate {total_responses_to_generate} total AI responses ({total_posts} posts × {len(self.models)} models × {self.responses_per_model} responses each)")
        
        for idx, row in posts_df.iterrows():
            try:
                conv_id = row['ConvID']
                title = row['Title']
                original_text = row['OriginalText']
                
                logger.info(f"Processing post {idx + 1}/{total_posts} (ConvID: {conv_id})")
                
                # Create prompt
                prompt = self.create_prompt(title, original_text)
                
                # Generate responses from each model
                for model_name, model_id in self.models.items():
                    label = "Gemma3_4b" if model_name == 'gemma3' else "Dolphin-Mistral"
                    
                    for response_num in range(self.responses_per_model):
                        response_start_time = time.time()
                        
                        ai_response = self.ollama_client.generate_response(model_id, prompt)
                        
                        if ai_response:
                            new_responses.append({
                                'ConvID': conv_id,
                                'Response': ai_response,
                                'Label': label
                            })
                        else:
                            logger.warning(f"Failed to generate {label} response {response_num + 1} for ConvID {conv_id}")
                            new_responses.append({
                                'ConvID': conv_id,
                                'Response': f"Error: Failed to generate {label} response",
                                'Label': label
                            })
                        
                        responses_generated += 1
                        
                        # Calculate and display real-time progress after each response
                        elapsed_total = time.time() - start_time
                        avg_time_per_response = elapsed_total / responses_generated
                        responses_remaining = total_responses_to_generate - responses_generated
                        estimated_time_remaining = avg_time_per_response * responses_remaining
                        
                        # Format time as minutes:seconds
                        minutes = int(estimated_time_remaining // 60)
                        seconds = int(estimated_time_remaining % 60)
                        
                        # Force immediate output with explicit flush and unbuffered print
                        progress_msg = f"{responses_generated}/{total_responses_to_generate} responses, est. time {minutes}:{seconds:02d}"
                        print(f"\r{progress_msg}", end="", flush=True)
                        sys.stdout.flush()  # Extra flush to ensure immediate display
                
                # Save intermediate results every 10 posts
                if (idx + 1) % 10 == 0:
                    print()  # New line after progress indicator
                    sys.stdout.flush()
                    logger.info(f"Saving intermediate results after {idx + 1} posts...")
                    self.save_intermediate_responses(new_responses, idx + 1)
                
            except Exception as e:
                logger.error(f"Error processing post {conv_id}: {e}")
                continue
        
        print()  # Final new line after progress indicator
        sys.stdout.flush()
        logger.info(f"AI response generation completed. Generated {len(new_responses)} new responses")
        return new_responses
    
    def save_intermediate_responses(self, new_responses: List[Dict], processed_count: int):
        """Save intermediate AI responses"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ai_responses_intermediate_{processed_count}_{timestamp}.csv"
        
        df = pd.DataFrame(new_responses)
        df.to_csv(filename, index=False)
        logger.info(f"Intermediate AI responses saved to {filename}")
    
    def augment_responses_dataset(self, posts_file: str, responses_file: str, output_file: str = None) -> str:
        """Main method to augment the responses dataset"""
        try:
            logger.info("Starting AI response augmentation...")
            
            # Load datasets
            posts_df, responses_df = self.load_datasets(posts_file, responses_file)
            
            # Generate AI responses
            new_responses = self.generate_ai_responses(posts_df)
            
            if len(new_responses) == 0:
                raise ValueError("No AI responses were generated")
            
            # Combine with existing responses
            new_responses_df = pd.DataFrame(new_responses)
            augmented_df = pd.concat([responses_df, new_responses_df], ignore_index=True)
            
            # Save augmented dataset
            if output_file is None:
                output_file = "augmented_data.csv"
            
            augmented_df.to_csv(output_file, index=False)
            logger.info(f"Augmented responses dataset saved to {output_file}")
            
            # Log statistics
            original_count = len(responses_df)
            new_count = len(new_responses_df)
            total_count = len(augmented_df)
            
            logger.info(f"Dataset statistics:")
            logger.info(f"  Original responses: {original_count}")
            logger.info(f"  New AI responses: {new_count}")
            logger.info(f"  Total responses: {total_count}")
            
            # Count by label
            label_counts = augmented_df['Label'].value_counts()
            for label, count in label_counts.items():
                logger.info(f"  {label}: {count} responses")
            
            logger.info("AI response augmentation completed successfully!")
            return output_file
            
        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            raise

def main():
    """Main function with configurable parameters"""
    # Configuration - easily adjustable
    MAX_POSTS = 50  # Set to None for all posts, or specific number to limit
    RESPONSES_PER_MODEL = 1  # Number of responses per model per post
    
    # File paths
    posts_file = "daniel-prelim-results\datasets\AITAposts.csv"
    responses_file = "daniel-prelim-results\datasets\AITAresponses.csv"
    output_file = "daniel-prelim-results\datasets\AITA_ai_smalldataset.csv"
    ollama_url = "http://localhost:11434"
    
    try:
        # Verify input files exist
        if not os.path.exists(posts_file):
            raise FileNotFoundError(f"Posts file not found: {posts_file}")
        if not os.path.exists(responses_file):
            raise FileNotFoundError(f"Responses file not found: {responses_file}")
        
        logger.info(f"Using posts file: {posts_file}")
        logger.info(f"Using responses file: {responses_file}")
        
        # Create augmenter instance with configuration
        augmenter = AIResponseAugmenter(
            ollama_url=ollama_url,
            max_posts=MAX_POSTS,
            responses_per_model=RESPONSES_PER_MODEL
        )
        
        logger.info(f"Configuration: {MAX_POSTS or 'ALL'} posts, {RESPONSES_PER_MODEL} responses per model")
        
        # Run augmentation
        output_file = augmenter.augment_responses_dataset(
            posts_file, responses_file, output_file
        )
        
        print(f"\n✅ Success! Augmented responses dataset saved to: {output_file}")
        print(f"\nGenerated responses:")
        print("- 3 responses per post from Gemma3:4b (labeled as 'Gemma3_4b')")
        print("- 3 responses per post from Dolphin-Mistral (labeled as 'Dolphin-Mistral')")
        print("\nPrerequisites:")
        print("1. Ensure Ollama is running: ollama serve")
        print("2. Ensure models are available: ollama pull gemma3:4b && ollama pull dolphin-mistral:latest")
        print("3. Run: python augment_with_ai.py")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()