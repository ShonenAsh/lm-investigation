import marimo

__generated_with = "0.16.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    from ollama import chat
    from ollama import ChatResponse
    from convokit import Corpus, download
    import pandas as pd
    import numpy as np
    import ollama
    import matplotlib.pyplot as plt
    import plotly.express as px
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    return Corpus, PCA, TSNE, download, mo, np, ollama, pd, plt


@app.cell
def _(Corpus, download):
    corpus = Corpus(filename=download("subreddit-Cornell"))
    corpus.print_summary_stats()
    conv_df = corpus.get_conversations_dataframe()
    utt_df = corpus.get_utterances_dataframe()
    return conv_df, utt_df


@app.cell
def _(conv_df):
    conv_df[:100]
    return


@app.cell
def _():
    # def create_dataset():
    #     merged_list = []
    #     for row in conv_df[:100].itertuples():
    #         response: ChatResponse = chat(model='llama3.2', messages=[{
    #                 'role': 'system',
    #                 'content': "Respond like a reddit user of r/Cornell, don't use reasoning tokens"
    #             },
    #             {
    #                 'role': 'user',
    #                 'content': row[2],
    #             }])
    #             joined_df = utt_df[ utt_df.conversation_id == row.Index]['text']
    #         merged_list.append({"id": row.Index, "prompt": row[2],"ai_resp": response.message.content, "human_resp": " ".join(joined_df.values.flatten().tolist())})
    #     merged_df = pd.DataFrame(merged_list)
    #     return merged_df
    return


@app.cell
def _(pd):
    merged_df = pd.read_csv("../llama3_2_aug_dataset.csv", index_col="id")
    return (merged_df,)


@app.cell
def _(merged_df):
    clean_df = merged_df.fillna(" ")
    clean_df
    return (clean_df,)


@app.cell
def _(conv_df, merged_df, utt_df):
    avg_list = []
    for _row in conv_df[:100].itertuples():
        avg_list.append(utt_df[ utt_df.conversation_id == _row.Index]['text'].str.len().mean())

    print(f"Avg length of human responses: {sum(avg_list)/ len(avg_list)}")
    print(f"Avg length of AI responses: {merged_df['ai_resp'].str.len().mean()}")
    return


@app.cell
def _(clean_df):
    clean_df
    return


@app.cell
def _(np, ollama):
    def get_embeddings(texts, model='nomic-embed-text'):
        _embeddings = []
        for text in texts:
            response = ollama.embeddings(model=model, prompt=text)
            _embeddings.append(response['embedding'])
        return np.array(_embeddings)
    return (get_embeddings,)


@app.cell
def _(clean_df, get_embeddings):
    all_texts = []
    labels = []
    types = []

    for _idx, row in clean_df.iterrows():
        all_texts.extend([row['text'], row['ai_resp'], row['human_resp']])
        labels.extend([f"Q{_idx}", f"Q{_idx}", f"Q{_idx}"])
        types.extend(['Prompt', 'AI', 'Human'])

    # print(all_texts)
    embeddings = get_embeddings(all_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings, types


@app.cell
def _(PCA, TSNE, embeddings, np, plt, types):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['text.color'] = 'black'

    colors = {'Prompt': 'blue', 'AI': 'red', 'Human': 'green'}
    markers = {'Prompt': 's', 'AI': 'o', 'Human': '^'}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    for _type_label in ['Prompt', 'AI', 'Human']:
        _mask = np.array(types) == _type_label
        ax1.scatter(
            embeddings_2d[_mask, 0],
            embeddings_2d[_mask, 1],
            c=colors[_type_label],
            marker=markers[_type_label],
            s=70,
            alpha=0.7,
            label=_type_label,
            edgecolors='black',
            linewidth=0.5
        )

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    ax1.set_title('PCA Visualization: Prompts vs AI vs Human Responses\n(Nomic Embeddings)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Print variance explained
    print(f"\nVariance explained by PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"Variance explained by PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"Total variance explained: {pca.explained_variance_ratio_[:2].sum():.2%}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=25)
    embeddings_tsne = tsne.fit_transform(embeddings)

    for _type_label in ['Prompt', 'AI', 'Human']:
        _mask = np.array(types) == _type_label
        ax2.scatter(
            embeddings_tsne[_mask, 0],
            embeddings_tsne[_mask, 1],
            c=colors[_type_label],
            marker=markers[_type_label],
            s=70,
            alpha=0.7,
            label=_type_label,
            edgecolors='black',
            linewidth=0.5
        )

    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('t-SNE Visualization: Prompts vs AI vs Human Responses\n(Nomic Embeddings)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _():
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    return go, make_subplots


@app.cell
def _():
    # pca_3d = PCA(n_components=3)
    # embeddings_3d = pca_3d.fit_transform(embeddings)
    # # print(embeddings_3d.shape)
    # fig_3d = make_subplots(rows=1, cols=2, subplot_titles=("PCA 3D", "t-SNE 3D"))

    # fig_3d.add_trace(pgo.Scatter3d(x=embeddings_3d[:,0], y=embeddings_3d[:,1], z=embeddings_3d[:,2], color=types, opacity=0.8), row=1, col=1)


    # tsne_3d = TSNE(n_components=3, perplexity=15)
    # embeddings_tsne_3d = tsne_3d.fit_transform(embeddings)

    # fig_3d.add_trace(pgo.Scatter3d(x=embeddings_tsne_3d[:,0], y=embeddings_tsne_3d[:,1], z=embeddings_tsne_3d[:,2], color=types, opacity=0.8), row=1, col=1)

    # fig_3d.update_layout(template='plotly_white', scene=dict(
    #         xaxis_title='Dim1',
    #         yaxis_title='Dim2',
    #         zaxis_title='Dim3'
    #     ))

    # fig_3d.show()
    return


@app.cell
def _(PCA, TSNE, embeddings, go, make_subplots, np, types):

    pca_3d = PCA(n_components=3)
    embeddings_3d = pca_3d.fit_transform(embeddings)

    # Create subplots with 3D specs
    fig_3d = make_subplots(
        rows=1, cols=2,
        subplot_titles=("PCA 3D", "t-SNE 3D"),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
    )

    # Color mapping
    color_map = {'Prompt': 'blue', 'AI': 'red', 'Human': 'green'}
    marker_map = {'Prompt': 'square', 'AI': 'circle', 'Human': 'diamond'}

    # Add PCA traces
    for type_label in ['Prompt', 'AI', 'Human']:
        mask = np.array(types) == type_label
        fig_3d.add_trace(
            go.Scatter3d(
                x=embeddings_3d[mask, 0],
                y=embeddings_3d[mask, 1],
                z=embeddings_3d[mask, 2],
                mode='markers',
                name=type_label,
                marker=dict(
                    size=6,
                    color=color_map[type_label],
                    symbol=marker_map[type_label],
                    opacity=0.8
                    # line=dict(color='black', width=0.5)
                ),
                legendgroup=type_label,
                showlegend=True
            ),
            row=1, col=1
        )

    # Compute t-SNE 3D
    tsne_3d = TSNE(n_components=3, perplexity=15, random_state=42)
    embeddings_tsne_3d = tsne_3d.fit_transform(embeddings)

    # Add t-SNE traces
    for type_label in ['Prompt', 'AI', 'Human']:
        mask = np.array(types) == type_label
        fig_3d.add_trace(
            go.Scatter3d(
                x=embeddings_tsne_3d[mask, 0],
                y=embeddings_tsne_3d[mask, 1],
                z=embeddings_tsne_3d[mask, 2],
                mode='markers',
                name=type_label,
                marker=dict(
                    size=6,
                    color=color_map[type_label],
                    symbol=marker_map[type_label],
                    opacity=0.8
                    # line=dict(color='black', width=0.5)
                ),
                legendgroup=type_label,
                showlegend=False  # Don't duplicate legend
            ),
            row=1, col=2
        )

    # Update layout
    fig_3d.update_layout(
        template='plotly_white',
        height=600,
        width=1200,
        title_text="3D Embeddings Visualization: PCA vs t-SNE"
    )

    # Update 3D scene for both subplots
    fig_3d.update_scenes(
        xaxis_title='Dimension 1',
        yaxis_title='Dimension 2',
        zaxis_title='Dimension 3'
    )

    fig_3d.show()

    # Print variance explained for PCA
    print(f"\nPCA 3D Variance explained:")
    print(f"PC1: {pca_3d.explained_variance_ratio_[0]:.2%}")
    print(f"PC2: {pca_3d.explained_variance_ratio_[1]:.2%}")
    print(f"PC3: {pca_3d.explained_variance_ratio_[2]:.2%}")
    print(f"Total: {pca_3d.explained_variance_ratio_[:3].sum():.2%}")
    return


@app.cell
def _(mo):
    mo.md(r"""###TF-IDF Analysis""")
    return


@app.cell
def _(clean_df):
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

    def apply_preprocess(doc, min_word_len=1):
        doc = char_level_preprocess(doc)
        return " ".join([word for word in doc.split() if len(word) >= min_word_len])

    clean_df['ai_resp'] = clean_df['ai_resp'].map(apply_preprocess)
    clean_df['human_resp'] = clean_df['human_resp'].map(apply_preprocess)
    clean_df['text'] = clean_df['text'].map(apply_preprocess)
    return


@app.cell
def _(clean_df):
    clean_df
    return


@app.cell
def _(clean_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    from scipy.spatial.distance import cdist

    tfidf_vec = TfidfVectorizer()
    tfidf_vec.fit(clean_df["text"].values.tolist() + clean_df["human_resp"].values.tolist() + clean_df["ai_resp"].values.tolist())
    y_1 = tfidf_vec.transform(clean_df["text"].values.tolist())

    X_1 = tfidf_vec.transform(clean_df["human_resp"].values.tolist())
    X_2 = tfidf_vec.transform(clean_df["ai_resp"].values.tolist())
    X_1.shape
    return X_1, X_2, cdist, y_1


@app.cell
def _(X_1, X_2, cdist, np, y_1):
    post_human_sim = np.squeeze(np.array([1 - cdist(XA=X_1.todense()[i], XB=y_1.todense()[i])[0] for i in range(X_1.shape[0])]), axis=None)
    post_ai_sim = np.squeeze(np.array([1 - cdist(XA=X_2.todense()[i], XB=y_1.todense()[i])[0] for i in range(X_1.shape[0])]), axis=None)
    return post_ai_sim, post_human_sim


@app.cell
def _(plt, post_ai_sim, post_human_sim):
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.subplot(1, 2, 1)
    plt.hist(post_human_sim, bins=15, color='lightblue', ec='darkblue', lw=0.5, alpha=0.7, label='Human')
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Number of Posts")
    plt.title("Human Replies: Avg Cosine Similarity per Post")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(post_ai_sim, bins=15, color='indianred', ec='red', lw=0.5, alpha=0.7, label='Llama3.2')
    plt.xlabel("Average Cosine Similarity")
    plt.ylabel("Number of Posts")
    plt.title("Llama3.2 Replies: Avg Cosine Similarity per Post")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return


@app.cell
def _(plt, post_ai_sim):
    plt.hist(post_ai_sim, color='skyblue', ec='darkblue', lw=0.5)
    plt.xlabel("Cosine similarity")
    plt.ylabel("# of Reddit posts")
    plt.title("Cosine similarity between AI responses and Reddit posts")
    plt.show()
    return


if __name__ == "__main__":
    app.run()
