from vllm import LLM, SamplingParams
from tqdm import tqdm
import convokit
from convokit import Corpus, download
import pandas as pd

#MODEL_NAME = 'cognitivecomputations/dolphin-2.6-mistral-7b'
#Q_MODEL_NAME = 'TheBloke/dolphin-2.6-mistral-7B-GPTQ'
Q_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
N_POSTS = 5000

def process_all_chats(prompts, model=Q_MODEL_NAME, max_concurrent=1000):
    llm = LLM(
        model=model,
#        quantization='gptq',
        tensor_parallel_size=1,  # no. of GPUs is 1
	    max_num_seqs=max_concurrent,  # batch size/concurrency
        gpu_memory_utilization=0.9
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=32000
    )

    formatted_prompts = [
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        for prompt in prompts
    ]

    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    results = [output.outputs[0].text for output in outputs]
    return results


def prepare_prompts():
    corpus = Corpus(download("subreddit-Cornell"))
    print(corpus.print_summary_stats())
    conv_df = corpus.get_conversations_dataframe()
    utt_df = corpus.get_utterances_dataframe()
    prompt_list = []
    for row in conv_df[:N_POSTS].itertuples():
        og_utt_text = row[2] + " " + utt_df[utt_df.index == row.Index]["text"].iloc[0]
        prompt_list.append(og_utt_text)
    # prompt_list = conv_df["meta.title"][:5000].values.flatten().tolist()
    return prompt_list


prompt_list = prepare_prompts()
results = process_all_chats(prompt_list)
temp_df = pd.DataFrame(results)
temp_df.columns = ['ai_response']
temp_df.to_csv("./llama3_2_temp_ai_responses_5k.csv")
print("done")
