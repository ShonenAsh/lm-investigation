import pandas as pd
import polars as pl
import pybiber as pb
import spacy as sp
from convokit import Corpus, download


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
    corpus = Corpus(download("subreddit-AmItheAsshole"))
    utt_df: pd.DataFrame = corpus.get_utterances_dataframe().iloc[:10000]
    df = prep_columns_pybiber(utt_df, "id", "text")
    print(df.head())
    res = calculate_biber_features(df)
    res.write_csv("./data/human_AmItheAsshole_biber_features_10k.csv")


if __name__ == "__main__":
    main()
