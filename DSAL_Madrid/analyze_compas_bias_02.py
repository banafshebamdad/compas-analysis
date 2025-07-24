#
# Created on Thu Jul 24 2025 08:34:41 CET
#
# Banafshe Bamdad
#
# analyze_compas_bias_02.py
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("compas-scores-two-years.csv")

# Create output directory
output_dir = "analyze_compas_bias_plot"
os.makedirs(output_dir, exist_ok=True)

# Drop rows with missing key features
df = df.dropna(subset=['race', 'sex', 'age', 'priors_count', 'decile_score', 'two_year_recid'])

# Filter to valid score_text categories
df = df[df['score_text'].isin(['Low', 'Medium', 'High'])]

# Define high-risk threshold
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

# Convert to DataFrame
bias_df = pd.DataFrame(group_stats)

# Plot False Positive Rate
plt.figure(figsize=(12, 6))
sns.barplot(data=bias_df, x='race', y='false_positive_rate', hue='sex')
plt.title("False Positive Rate by Race and Sex (decile_score ≥ 5)")
plt.ylabel("False Positive Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "false_positive_rate_by_race_sex.png"))
plt.close()

# Plot False Negative Rate
plt.figure(figsize=(12, 6))
sns.barplot(data=bias_df, x='race', y='false_negative_rate', hue='sex')
plt.title("False Negative Rate by Race and Sex (decile_score < 5)")
plt.ylabel("False Negative Rate")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "false_negative_rate_by_race_sex.png"))
plt.close()

# Save bias summary as CSV
bias_df.to_csv("analyze_compas_bias_plot/bias_summary_by_race_sex.csv", index=False)
print("\nBias summary saved to: analyze_compas_bias_plot/bias_summary_by_race_sex.csv")
print("\n--- Bias Summary Table ---")
print(bias_df)
