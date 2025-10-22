import pandas as pd
import textstat
import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = pd.read_csv('.\lm-investigation\daniel-prelim-results\datasets\AITA_ai_smalldataset.csv')

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

# Set the seaborn style to match other visualizations
plt.style.use('seaborn-v0_8-whitegrid')

# Create a DataFrame for seaborn
plot_data = pd.DataFrame({
    'Author Type': author_types,
    'Grade Level': grade_levels
})

# Set up the figure and axis
plt.figure(figsize=(12, 8))

# Define colors to match other visualizations
color_map = {'Human': 'skyblue', 'Dolphin-Mistral': 'coral', 'Gemma3_4b': 'lightgreen'}
colors = [color_map[author] for author in author_types]

# Create the bar plot with seaborn
bars = sns.barplot(x='Author Type', y='Grade Level', data=plot_data, palette=colors,
                  edgecolor='black', linewidth=0.5, alpha=0.8)

# Add data labels on top of bars
for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f}', ha='center', va='bottom', fontsize=11)

# Add labels, title, and customize appearance
plt.xlabel('Author Type', fontsize=12)
plt.ylabel('Grade Level', fontsize=12)
plt.title('Average Grade Levels by Author Type for AITA Posts', fontsize=14, fontweight='bold')
plt.ylim(bottom=0)  # Ensure y-axis starts at 0
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Save the figure
plt.tight_layout()
plt.savefig('lm-investigation/daniel-prelim-results/readability_scores/grade_levels_by_author.png', dpi=300, bbox_inches='tight')
plt.show()
