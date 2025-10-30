# Language Model Investigation [WIP]
DS 5500 - Data Science Capstone Project

## Summary

Large Language Models have taken the world by storm and are particularly prevalent in public online spaces such as social media. LLMs and their cousins, Small Language Models, are particularly prevalent in spam dedicated to harmful online discourse such as in comment sections of politically inflammatory topics. We propose an exploratory analysis to view public datasets of user generated content and compare user generated responses to synthetic AI generated responses. To achieve this, we conduct a statistical analysis to find meaningful differentiation patterns and potentially find exploits that help us quickly identify synthetic posts and the popular language model architectures behind them.

## Stylometric Analysis
- AI (Llama3.2) vs human text, stylometric analysis on r/Cornell posts
<img width="2048" height="1016" alt="image" src="https://github.com/user-attachments/assets/15aa8bff-2cd9-4fbc-a0c0-12e675bb9917" />

- AI (Dolphin-Mistral) vs human text, stylometric analysis on r/AmItheAsshole posts
<img width="1589" height="789" alt="image" src="https://github.com/user-attachments/assets/6637562b-9f99-4636-84ab-56916147e6e4" />

- AI (Dolphin-Mistral) vs human text - reading grade analysis on r/AmItheAsshole posts
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/6fa8c650-a946-467a-9e76-c112ad30e384" />

Our current findgings are summarized in the figures above. Token-type ratio (or Type-token ratio) is the amount of unique token relative to the total number of tokens in a text document. Similarly lemma diversity is the amount of unique lemmas in a document relative to the total number. The other density graphs describe the distributions of various parts of speech in the two text corpora.

## Observations
-  We see that AI's distributions are unlike what humans' typically type in an online discussion board like reddit.
-  AI-generated text is generally harder to read for an average reader in the US (based on Flesch-Kincade reading scale).
-  These results are strictly applicable to reddit posts as LLMs are generally reinforced and fine-tuned to write like other more well-written human literature (scientic articles, books etc.) unlike Reddit.
-  These small yet many differences make Small Language Models stand out in a human discussion forum setting.
-  SLMs/LLMs may mimic human distributions in other setting (e.g. it's common to see single word responses like 'LMAO' on reddit but not in homework assignments, wikipedia, etc.)
