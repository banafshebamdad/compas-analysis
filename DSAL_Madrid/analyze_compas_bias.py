#
# Created on Thu Jul 24 2025 08:12:26 CET
#
# Banafshe Bamdad
#
# analyze_compas_bias.py
#
"""
This script analyzes the COMPAS recidivism dataset for potential racial and gender bias.
It performs the following tasks:

1. Loads and cleans the dataset.
2. Selects relevant features for analysis.
3. Performs descriptive statistical analysis grouped by race and sex.
4. Computes false positive and false negative rates for each racial group and (race + sex) group.
5. Visualizes score distributions and recidivism rates by race and gender.
6. Trains a logistic regression model and displays feature importance.

All visualizations are saved in the 'analyze_compas_bias_plot/' folder.

Dataset: compas-scores-two-years.csv
Source: ProPublica (https://github.com/propublica/compas-analysis)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

os.makedirs("analyze_compas_bias_plot", exist_ok=True)

df = pd.read_csv("compas-scores-two-years.csv")

columns_to_drop = ['id', 'name', 'first', 'last', 'compas_screening_date', 'dob',
                   'c_case_number', 'c_offense_date', 'c_arrest_date', 'r_case_number',
                   'r_offense_date', 'vr_case_number', 'vr_offense_date',
                   'start', 'end', 'event']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

df = df.dropna(subset=['race', 'sex', 'age', 'priors_count', 'decile_score', 'two_year_recid'])

df = df[df['score_text'].isin(['Low', 'Medium', 'High'])]

print("\n--- Recidivism rates by race ---")
print(df.groupby('race')['two_year_recid'].mean())

print("\n--- Average decile score by race ---")
print(df.groupby('race')['decile_score'].mean())

print("\n--- False positive and negative rates by race ---")
for race in df['race'].unique():
    subset = df[df['race'] == race]
    high_risk = subset['score_text'].isin(['High', 'Medium'])
    actual = subset['two_year_recid']

    fp = ((high_risk == True) & (actual == 0)).sum()
    fn = ((high_risk == False) & (actual == 1)).sum()
    total = len(subset)

    print(f"\nRace: {race}")
    print(f"False positive rate: {fp / total:.2f}")
    print(f"False negative rate: {fn / total:.2f}")

print("\n--- False positive and negative rates by Race + Sex ---")
grouped = df.groupby(['race', 'sex'])
for (race, sex), subset in grouped:
    high_risk = subset['score_text'].isin(['High', 'Medium'])
    actual = subset['two_year_recid']

    fp = ((high_risk == True) & (actual == 0)).sum()
    fn = ((high_risk == False) & (actual == 1)).sum()
    total = len(subset)

    print(f"\nGroup: {race} + {sex}")
    print(f"False Positive Rate: {fp / total:.2f}")
    print(f"False Negative Rate: {fn / total:.2f}")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='race', y='decile_score')
plt.title("Decile score distribution by race")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("analyze_compas_bias_plot/decile_score_by_race.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='sex', y='decile_score')
plt.title("Decile Score Distribution by Sex")
plt.tight_layout()
plt.savefig("analyze_compas_bias_plot/decile_score_by_sex.png")
plt.close()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='race', y='decile_score', hue='sex')
plt.title("Decile Score by Race and Sex")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("analyze_compas_bias_plot/decile_score_by_race_and_sex.png")
plt.close()

# Logistic Regression Model
features = ['age', 'sex', 'race', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'c_charge_degree']
df_model = df[features + ['two_year_recid']].copy()

df_model['sex'] = df_model['sex'].astype('category').cat.codes
df_model['race'] = df_model['race'].astype('category').cat.codes
df_model['c_charge_degree'] = df_model['c_charge_degree'].astype('category').cat.codes

X = df_model.drop(columns='two_year_recid')
y = df_model['two_year_recid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n--- Classification report ---")
print(classification_report(y_test, y_pred))

print("\n--- Feature importances (Logistic coefficients) ---")
for name, coef in zip(X.columns, model.coef_[0]):
    print(f"{name}: {coef:.4f}")
