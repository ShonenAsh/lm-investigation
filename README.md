# Language Model Investigation
DS 5500 - Data Science Capstone Project

### Summary

Large Language Models have taken the world by storm and are particularly prevalent in public online spaces such as social media. LLMs and their cousins, Small Language Models, are particularly prevalent in spam dedicated to harmful online discourse such as in comment sections of politically inflammatory topics. We propose an exploratory analysis to view public datasets of user generated content and compare user generated responses to synthetic AI generated responses. To achieve this, we conduct a statistical analysis to find meaningful differentiation patterns and potentially find exploits that help us quickly identify synthetic posts and the popular language model architectures behind them.


### Initial Analysis

#### Word frequency cosine-similarity

Human Reponses           |  AI Responses
:-------------------------:|:-------------------------:
<img width="600" height="480" alt="image_1" src="https://github.com/user-attachments/assets/2a89b70d-3ba6-4c1e-affd-ae381c782e91" />  |  <img width="600" height="480" alt="image_2" src="https://github.com/user-attachments/assets/e7207f10-5e6b-4bdc-b153-4ca13f5ad450" />

AI responses have an interesting pattern to them when it comes to the similarity between their and the post titles’ word distribution. Human responses show diversity in their distribution meanwhile AI responses are mostly dissimilar.
