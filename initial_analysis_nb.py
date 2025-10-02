import marimo

__generated_with = "0.16.3"
app = marimo.App(width="full")


@app.cell
def _():
    from convokit import Corpus
    import pandas as pd
    import marimo as mo
    return Corpus, mo, pd


@app.cell
def _(Corpus):
    corpus = Corpus("/home/shonenash/.convokit/saved-corpora/subreddit-Cornell")
    corpus.print_summary_stats()
    utt_df = corpus.get_utterances_dataframe()
    return corpus, utt_df


@app.cell
def _(utt_df):
    utt_df.columns
    return


@app.cell
def _(utt_df):
    utt_df.head()
    return


@app.cell
def _(corpus):
    conv_df = corpus.get_conversations_dataframe()
    conv_df.head()
    return (conv_df,)


@app.cell
def _():
    import re
    import string
    # Preprocessing
    def char_level_preprocess(doc):
        doc = re.sub(r'(\d+)', ' ', doc)
        doc = re.sub(r'(\s+)', ' ', doc)
        doc = re.sub(rf'[{re.escape(string.punctuation)}]', '', doc)
        doc = doc.lower()
        doc = doc.strip()
        return doc

    def apply_preprocess(doc, min_word_len=3):
        doc = char_level_preprocess(doc)
        return " ".join([word for word in doc.split() if len(word) >= min_word_len])

    return (apply_preprocess,)


@app.cell
def _(apply_preprocess, conv_df, pd, utt_df):
    merged_list = []
    for row_tuple in conv_df.head(59).itertuples():
        joined_df = utt_df[ utt_df.conversation_id == row_tuple.Index]['text']
        merged_list.append(
            {"id": row_tuple.Index, "text": apply_preprocess(row_tuple[2], min_word_len=1), "responses": apply_preprocess(" ".join(joined_df.values.flatten().tolist()), min_word_len=1)}
        )
    merged_df = pd.DataFrame(merged_list)
    return (merged_df,)


@app.cell
def _(merged_df):
    merged_df.head()
    return


@app.cell
def _(mo):
    mo.md(r"""#### AI text preprocessing""")
    return


@app.cell
def _(pd):
    ai_df = pd.read_csv("./cornell_ai_responses.csv")
    ai_df.columns
    return (ai_df,)


@app.cell
def _(ai_df, apply_preprocess, pd):
    ai_list = []
    for ai_tuple in ai_df.itertuples():
        ai_list.append({"responses": apply_preprocess(" ".join(ai_tuple[2:]), min_word_len=1)})

    ai_responses_df = pd.DataFrame(ai_list)
    ai_responses_df.head()
    return (ai_responses_df,)


@app.cell
def _(merged_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from scipy.spatial.distance import cdist
    import numpy as np

    tfidf_vec = TfidfVectorizer()
    X_1 = tfidf_vec.fit_transform(merged_df["responses"].values.tolist())
    y_1 = tfidf_vec.transform(merged_df["text"])
    return X_1, cdist, np, tfidf_vec, y_1


@app.cell
def _(y_1):
    y_1.shape
    return


@app.cell
def _(ai_responses_df, tfidf_vec):
    X_2 = tfidf_vec.transform(ai_responses_df["responses"].values.tolist())
    X_2.todense().shape
    return (X_2,)


@app.cell
def _(X_1, X_2, cdist, np, y_1):
    post_human_sim = np.squeeze(np.array([1 - cdist(XA=X_1.todense()[i], XB=y_1.todense()[i])[0] for i in range(X_1.shape[0])]), axis=None)
    post_ai_sim = np.squeeze(np.array([1 - cdist(XA=X_2.todense()[i], XB=y_1.todense()[i])[0] for i in range(X_1.shape[0])]), axis=None)
    return post_ai_sim, post_human_sim


@app.cell
def _(post_human_sim):
    post_human_sim
    return


@app.cell
def _(post_human_sim):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    plt.style.use('seaborn-v0_8-whitegrid')

    plt.hist(post_human_sim, color='skyblue', ec='darkblue', lw=0.5)
    plt.xlabel("Cosine similarity")
    plt.ylabel("# of Reddit posts")
    plt.title("Cosine similarity between human comments and Reddit posts")
    # plt.show()
    plt.savefig("image_1.png", dpi=300)
    return WordCloud, plt


@app.cell
def _(plt, post_ai_sim):
    plt.hist(post_ai_sim, color='skyblue', ec='darkblue', lw=0.5)
    plt.xlabel("Cosine similarity")
    plt.ylabel("# of Reddit posts")
    plt.title("Cosine similarity between AI comments and Reddit posts")
    # plt.show()
    plt.savefig("image_2.png", dpi=300)
    return


@app.cell
def _(ai_responses_df):
    all_ai_words = " ".join(ai_responses_df["responses"].values.tolist())
    all_ai_words
    return (all_ai_words,)


@app.cell
def _(merged_df):
    all_human_words = " ".join(merged_df["responses"].values.tolist())
    all_human_words
    return (all_human_words,)


@app.cell
def _(WordCloud, all_ai_words, plt):
    ai_wc = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(all_ai_words)
    plt.imshow(ai_wc)
    plt.axis("off")
    plt.title("AI responses: Top 100 words")
    plt.savefig("image_ai_wc.png")
    return


@app.cell
def _(WordCloud, all_human_words, plt):
    human_wc = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(all_human_words)
    plt.imshow(human_wc)
    plt.axis("off")
    plt.title("Human responses: Top 100 words")
    plt.savefig("image_human_wc.png")
    return


if __name__ == "__main__":
    app.run()
