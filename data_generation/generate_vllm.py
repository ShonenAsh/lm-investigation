# This file was not used for the final responses generation
# File is retained for future use

import os

import pandas as pd
from convokit import Corpus
from dotenv import load_dotenv
from vllm import LLM, SamplingParams

load_dotenv()

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

START_OFFSET = 5000
N_POSTS = 10
OFFLINE_MODEL_PATH = "/scratch/magadum.a/hf/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/"
OUT_CSV_NAME = "test.csv"


def process_reqs(prompts, model_path=OFFLINE_MODEL_PATH, max_concurrent=50):
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,  # no. of GPUs is 1
        max_num_seqs=max_concurrent,  # batch size/concurrency
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4096)
    formatted_prompts = [

        f"Respond like a reddit user of r/Cornell\n{prompt}"
        for prompt in prompts
    ]

    outputs = llm.generate(
        prompts=formatted_prompts, sampling_params=sampling_params, use_tqdm=True
    )
    results = [output.outputs[0].text for output in outputs]
    print(results)
    return results

"""
Conversations dataframe column names
    vectors, meta.title, meta.num_comments, meta.domain, meta.timestamp, 
    meta.subreddit, meta.gilded, meta.gildings, meta.stickied, meta.author_flair_text

Utterances dataframe column names
    timestamp, text speaker, reply_to, conversation_id,
       meta.score, meta.top_level_comment, meta.retrieved_on,
       meta.gilded, meta.gildings, meta.subreddit, meta.stickied',
       meta.permalink, meta.author_flair_text, vectors
"""
def prepare_prompts():
    corpus = Corpus("/home/magadum.a/.convokit/saved-corpora/subreddit-Cornell")
    print(corpus.print_summary_stats())
    conv_df = corpus.get_conversations_dataframe()
    utt_df = corpus.get_utterances_dataframe()
    prompt_list = []
    id_list = []
    print(f"Posts: {START_OFFSET} - {START_OFFSET + N_POSTS}")
    for row in conv_df.iloc[START_OFFSET: START_OFFSET + N_POSTS].itertuples():
        og_utt_text = row[2] + " " + utt_df[utt_df.index == row.Index]["text"].iloc[0]
        prompt_list.append(og_utt_text)
        id_list.append(row.Index)
    return id_list, prompt_list


ids, prompt_list = prepare_prompts()
results = process_reqs(prompt_list)
df_data = {"id": ids, "ai_response": results}
temp_df = pd.DataFrame(df_data)
temp_df.set_index('id')
temp_df.to_csv(OUT_CSV_NAME)
print("done")
