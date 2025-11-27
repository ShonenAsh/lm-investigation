import pandas as pd
import polars as pl
import pybiber as pb
import spacy as sp
from convokit import Corpus, download

START_OFFSET = 0
N_POSTS = 5000

# Extract subreddit prompts (only for human/original data)
def prepare_prompts(subreddit):
    corpus = Corpus(download(f"subreddit-{subreddit}"))
    print(corpus.print_summary_stats())
    conv_df = corpus.get_conversations_dataframe()
    utt_df = corpus.get_utterances_dataframe()
    prompt_list = []
    id_list = []
    print(f"Posts: {START_OFFSET} - END")
    for row in conv_df.iloc[START_OFFSET: START_OFFSET + N_POSTS].itertuples():
        og_utt_text = row[2] + " " + utt_df[utt_df.index == row.Index]["text"].iloc[0]
        prompt_list.append(og_utt_text)
        id_list.append(row.Index)

    #for row in utt_df.iloc[START_OFFSET:].itertuples():
    #    prompt_list.append(row.text)
    #    id_list.append(row.Index)
    return id_list, prompt_list

# Apply pybiber's processing to obtain 67 Biber features
def calculate_biber_features(corpus_df: pl.DataFrame) -> pl.DataFrame:
    nlp_model = sp.load("en_core_web_sm", disable=["ner"])
    pb_processor = pb.CorpusProcessor()
    tokens_df = pb_processor.process_corpus(corpus_df, nlp_model)
    features_df = pb.biber(tokens_df)

    return features_df

# Helper function to rename columns for PyBibr processing
# Also drops any additional columns if provided (required).
def prep_columns_pybiber(
    corpus_df: pl.DataFrame | pd.DataFrame, id_col: str, text_col: str
) -> pl.DataFrame:
    if isinstance(corpus_df, pd.DataFrame):
        corpus_df = corpus_df.reset_index()
        corpus_df = pl.from_pandas(corpus_df)

    if id_col not in corpus_df.columns or text_col not in corpus_df.columns:
        raise Exception(f"One or more columns are missing: {id_col}, {text_col}")

    corpus_df = corpus_df.rename({id_col: "doc_id", text_col: "text"})
    corpus_df = corpus_df.with_columns(pl.col("doc_id").cast(str))
    # pybiber only accepts these two columns
    corpus_df = corpus_df.select(["doc_id", "text"])
    return corpus_df


def main():
    #corpus = Corpus(download("subreddit-PewdiepieSubmissions"))
    #utt_df: pd.DataFrame = corpus.get_utterances_dataframe().iloc[:5000]
    
    ids, prompt_lists = prepare_prompts("flatearth")
    utt_df = pl.DataFrame({"id": ids, "text": prompt_lists})

    #utt_df = pl.read_csv('./data/p2/human_pewds_engagement.csv')
    df = prep_columns_pybiber(utt_df, "id", "text")
    print(df.head())
    res = calculate_biber_features(df)
    res.write_csv("./data/p2/human_flatearth_biber_features.csv")


if __name__ == "__main__":
    main()
