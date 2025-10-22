import pandas
import textstat
import math
import matplotlib.pyplot as plt
import numpy as np

data = pandas.read_csv('lm-investigation\daniel-prelim-results\datasets\AITA_ai_smalldataset.csv')

## Both formulas here use average sentence length and average syllable count of a word as the difficulty parameters, with scaling modified for different contexts.

#this is a 0-100 score, 100 being easy to read for a middle schooler and 0 being impossible to read.
data['flesch_score'] = data['Response'].apply(lambda x: textstat.flesch_reading_ease(x))
#this formula outputs a rough approximation of the grade level of a text.
data['flesch_kincaid_score'] = data['Response'].apply(lambda x: textstat.flesch_kincaid_grade(x))
#Gunning Fog Index - estimates years of formal education needed to understand text

flesch_avgs = {
    'all_avg' : data['flesch_score'].mean(),
    'human_avg' : data.loc[data['Label'] == 'Human', 'flesch_score'].mean(),
    'dolphin_avg' : data.loc[data['Label'] == 'Dolphin-Mistral', 'flesch_score'].mean(),
    'gemma_avg' : data.loc[data['Label'] == 'Gemma3_4b', 'flesch_score'].mean()
}
flesch_kincaid_avgs = {
    'all_avg' : data['flesch_kincaid_score'].mean(),
    'human_avg' : data.loc[data['Label'] == 'Human', 'flesch_kincaid_score'].mean(),
    'dolphin_avg' : data.loc[data['Label'] == 'Dolphin-Mistral', 'flesch_kincaid_score'].mean(),
    'gemma_avg' : data.loc[data['Label'] == 'Gemma3_4b', 'flesch_kincaid_score'].mean()
}

print("Flesch Reading Ease (higher = easier to read):", flesch_avgs)
print("Flesch-Kincaid Grade Level (grade level):", flesch_kincaid_avgs)

# Create bar graph for Flesch-Kincaid grade level
author_types = ['Human', 'Dolphin-Mistral', 'Gemma3_4b']
grade_levels = [flesch_kincaid_avgs['human_avg'],
                flesch_kincaid_avgs['dolphin_avg'],
                flesch_kincaid_avgs['gemma_avg']]

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
bars = ax.bar(author_types, grade_levels, width=0.6, color=['#3498db', '#2ecc71', '#e74c3c'])

# Add data labels on top of bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}', ha='center', va='bottom')

# Add labels, title, and customize appearance
ax.set_xlabel('Author Type', fontsize=12)
ax.set_ylabel('Grade Level', fontsize=12)
ax.set_title('Average Grade Levels by Author Type for AITA Posts', fontsize=14, fontweight='bold')
ax.set_ylim(bottom=0)  # Ensure y-axis starts at 0
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('lm-investigation/daniel-prelim-results/readability_scores/grade_levels_by_author.png')
plt.show()
