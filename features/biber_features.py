import sys
import pandas as pd
import polars as pl
import pybiber as pb
import spacy as sp
from convokit import Corpus, download

# Last 5000 prefered because the deleted posts are in the beginning
START_OFFSET = -5000
N_POSTS = 5000 

def prep_prompts(subreddit, only_utterances=False):
    """ Extracts post titles and text by performing a join 
        on conversations and utterances from a convokit subreddit
    
    Keyword arguments:
        only_utterances -- returns text only from utterances dataframe
    """
    corpus = Corpus(download(f"subreddit-{subreddit}"))
    print(corpus.print_summary_stats())
    conv_df = corpus.get_conversations_dataframe()
    utt_df = corpus.get_utterances_dataframe()
    prompt_list = []
    id_list = []

    if only_utterances:
        print(f"Utterances: {START_OFFSET} - END")
        for row in utt_df.iloc[START_OFFSET:].itertuples():
            prompt_list.append(row.text)
            id_list.append(row.Index)
    else:
        print(f"Posts: {START_OFFSET} - END")
        for row in conv_df.iloc[START_OFFSET:].itertuples():
            og_utt_text = row[2] + " " + utt_df[utt_df.index == row.Index]["text"].iloc[0]
            prompt_list.append(og_utt_text)
            id_list.append(row.Index)

    return id_list, prompt_list

def calculate_biber_features(corpus_df: pl.DataFrame) -> pl.DataFrame:
    """Apply pybiber's processing and returns 67 Biber features"""

    nlp_model = sp.load("en_core_web_sm", disable=["ner"])
    pb_processor = pb.CorpusProcessor()
    tokens_df = pb_processor.process_corpus(corpus_df, nlp_model)
    features_df = pb.biber(tokens_df)

    return features_df

def prep_columns_pybiber(
    corpus_df: pl.DataFrame | pd.DataFrame, id_col: str, text_col: str
) -> pl.DataFrame:
    """ Helper function to rename columns for PyBibr processing
        Also drops any additional columns if provided (required).
    """

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
    if len(sys.argv) != 3:
        print("Usage $python biber_features.py <convokit-subreddit | text_input.csv> <biber_features_output.csv>")
        exit(-1)
    
    args = sys.argv
    
    if ".csv" in args[1]:
        utt_df = pl.read_csv(args[1])
        df = prep_columns_pybiber(utt_df, "id", "ai_response")
    else:
        ids, prompt_lists = prep_prompts(args[1])
        utt_df = pl.DataFrame({"id": ids, "text": prompt_lists})
        df = prep_columns_pybiber(utt_df, "id", "text")

    print(df.head())
    res = calculate_biber_features(df)
    res.write_csv(args[2])
    print(f"Successfully written biber features to {args[2]}")


if __name__ == "__main__":
    main()
