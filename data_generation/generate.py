import pandas as pd
from convokit import Corpus
from ollama import chat
from tqdm import tqdm
import json
import os
import logging
import sys

# Stdout goes to sbatch job.out file

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s : %(levelname)s : %(message)s'
)

# Get a logger instance
logger = logging.getLogger(__name__)

START_OFFSET = -5000
MODEL = "dolphin-mistral:7b"
BASE_PROMPT = "engagement-farmer"
SUBREDDIT = "pewdiepie"
OUT_CSV_NAME = "dolphin_engagement_farmer_dataset.csv"
USER_DIR = os.environ['USER']
logger.info(f'User: {USER_DIR}')


def get_base_prompt(base_prompt, path=f"/home/{USER_DIR}/workspace/prompts.json"):
    json_data = ""
    try:
        with open(path, 'r') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        logger.error("Error: The file 'your_file.json' was not found.")
    except json.JSONDecodeError:
        logger.error("Error: Failed to decode JSON from the file. Check for malformed JSON.")
    bot_prompt = json_data[base_prompt]
    logger.info(f"BASE_PROMPT:\n {bot_prompt}")
    return bot_prompt


def generate_resp(prompt, base_prompt, subreddit=SUBREDDIT):
    response = chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"{base_prompt} \nFollowing is a post from r/{subreddit}",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        think=False,
    )
    return response["message"]["content"]


def process_reqs(prompt_list, base_prompt):
    results = []
    for prompt in tqdm(prompt_list, total=len(prompt_list)):
        result = generate_resp(prompt, base_prompt)
        results.append(result)
    return results


"""
Conversations dataframe column names
    vectors, meta.title, meta.num_comments, meta.domain, meta.timestamp, 
    meta.subreddit, meta.gilded, meta.gildings, meta.stickied, meta.author_flair_text

Utterances dataframe column names
    timestamp, text, speaker, reply_to, conversation_id,
       meta.score, meta.top_level_comment, meta.retrieved_on,
       meta.gilded, meta.gildings, meta.subreddit, meta.stickied',
       meta.permalink, meta.author_flair_text, vectors
"""
def prepare_prompts(subreddit=SUBREDDIT):
    corpus = Corpus(f"/home/{USER_DIR}/.convokit/saved-corpora/subreddit-{subreddit}")
    logger.info(corpus.print_summary_stats())
    conv_df = corpus.get_conversations_dataframe()
    utt_df = corpus.get_utterances_dataframe()
    prompt_list = []
    id_list = []
    logger.info(f"Posts: {START_OFFSET} - End")
    for row in conv_df.iloc[START_OFFSET:].itertuples():
        og_utt_text = row[2] + " " + utt_df[utt_df.index == row.Index]["text"].iloc[0]
        prompt_list.append(og_utt_text)
        id_list.append(row.Index)
    return id_list, prompt_list


def main():
    base_prompt = get_base_prompt(BASE_PROMPT)
    ids, prompt_list = prepare_prompts()

    results = process_reqs(prompt_list, base_prompt)
    df_data = {"id": ids, "ai_response": results}
    temp_df = pd.DataFrame(df_data)
    temp_df.set_index('id', inplace=True)
    temp_df.to_csv(OUT_CSV_NAME)
    logger.info("done")


if __name__ == "__main__":
    main()

