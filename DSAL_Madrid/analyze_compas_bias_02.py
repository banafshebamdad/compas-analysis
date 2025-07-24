#
# Created on Thu Jul 24 2025 08:34:41 CET
#
# Banafshe Bamdad
#
# analyze_compas_bias_02.py
#
"""
This script analyzes the COMPAS dataset for potential bias in the algorithm's risk predictions,
specifically by computing false positive and false negative rates across demographic groups
defined by race and sex.

The main steps include:
    1. Loading and filtering the dataset.
    2. Defining high risk based on a decile score threshold (≥ 5).
    3. Calculating group-wise false positive and false negative rates.
    4. Saving the summary statistics as a CSV file.
    5. Visualizing the error rates using bar plots grouped by race and sex.

Outputs:
- CSV file with bias summary: analyze_compas_bias_plot/bias_summary_by_race_sex.csv
- Bar plot for false positive rates: analyze_compas_bias_plot/false_positive_rate_by_race_sex.png
- Bar plot for false negative rates: analyze_compas_bias_plot/false_negative_rate_by_race_sex.png

Usage:
    $ python analyze_compas_bias_02.py
    
    NOTE: Ensure the file compas-scores-two-years.csv` is in the same directory.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("compas-scores-two-years.csv")

output_dir = "analyze_compas_bias_plot"
os.makedirs(output_dir, exist_ok=True)

df = df.dropna(subset=['race', 'sex', 'age', 'priors_count', 'decile_score', 'two_year_recid'])

df = df[df['score_text'].isin(['Low', 'Medium', 'High'])]

threshold = 5
df['high_risk'] = df['decile_score'] >= threshold
df['recid'] = df['two_year_recid'] == 1

# Group by race and sex
group_stats = []

for (race, sex), group in df.groupby(['race', 'sex']):
    total = len(group)
    if total == 0:
        continue
    fp = ((group['high_risk'] == True) & (group['recid'] == False)).sum()
    fn = ((group['high_risk'] == False) & (group['recid'] == True)).sum()
    fp_rate = fp / total
    fn_rate = fn / total
    group_stats.append({
        'race': race,
        'sex': sex,
        'false_positive_rate': fp_rate,
        'false_negative_rate': fn_rate
    })

bias_df = pd.DataFrame(group_stats)

plt.figure(figsize=(12, 6))
sns.barplot(data=bias_df, x='race', y='false_positive_rate', hue='sex')
plt.title("False Positive Rate by Race and Sex (decile_score ≥ 5)")
plt.ylabel("False Positive Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "false_positive_rate_by_race_sex.png"))
plt.close()

plt.figure(figsize=(12, 6))
sns.barplot(data=bias_df, x='race', y='false_negative_rate', hue='sex')
plt.title("False Negative Rate by Race and Sex (decile_score < 5)")
plt.ylabel("False Negative Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "false_negative_rate_by_race_sex.png"))
plt.close()

bias_df.to_csv("analyze_compas_bias_plot/bias_summary_by_race_sex.csv", index=False)
print("\nBias summary saved to: analyze_compas_bias_plot/bias_summary_by_race_sex.csv")
print("\n--- Bias Summary Table ---")
print(bias_df)
