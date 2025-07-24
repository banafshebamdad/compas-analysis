#
# Created on Thu Jul 24 2025 17:03:44 CET
#
# Banafshe Bamdad
#
# reweighing_compas_analysis_xgboost_df_cleaned.py
#

"""
This script evaluates and mitigates algorithmic bias in the COMPAS recidivism dataset
using XGBoost and the AIF360 fairness toolkit. It operates on a cleaned version of the
dataset with one-hot encoded categorical variables, which are transformed into binary
indicators for protected attributes.

Steps performed:
    1. Loads and preprocesses the cleaned COMPAS dataset.
    2. Converts one-hot encoded race and sex attributes into single binary columns.
    3. Trains an XGBoost classifier on the original (biased) dataset and evaluates its fairness.
    4. Applies the Reweighing algorithm (a pre-processing bias mitigation technique).
    5. Trains an XGBoost classifier on the reweighed dataset and re-evaluates fairness.
    6. Computes and compares fairness metrics before and after reweighing:
        - Accuracy
        - Statistical Parity Difference
        - Disparate Impact
        - Equal Opportunity Difference
        - Average Odds Difference
    7. Saves bar plots visualizing the fairness metrics to the 'analyze_compas_bias_plot' directory.

"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# Load cleaned dataset
df = pd.read_csv("df_cleaned.csv")

# Reconstruct categorical variables from one-hot encoding
df['sex'] = df['sex_Male'].apply(lambda x: 1 if x == 1 else 0)  # Male = 1, Female = 0
df['race'] = df['race_African-American'].apply(lambda x: 1 if x == 1 else 0)  # African-American = 1, others = 0

# Drop redundant columns
df = df.drop(columns=[
    'sex_Female', 'sex_Male',
    'race_African-American', 'race_Asian', 'race_Caucasian',
    'race_Hispanic', 'race_Native American', 'race_Other',
    'priors_count.1', 'in_minus_offense', 'out_minus_in',
    'c_charge_degree_F', 'c_charge_degree_M'
])

# Define features and labels
features = ['age', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'sex']
target = 'two_year_recid'
protected = 'race'

# Prepare AIF360 dataset
dataset = BinaryLabelDataset(
    favorable_label=0,
    unfavorable_label=1,
    df=df[features + [target, protected]],
    label_names=[target],
    protected_attribute_names=[protected]
)

train, test = dataset.split([0.7], shuffle=True)
X_train = train.features
y_train = train.labels.ravel()
X_test = test.features
y_test = test.labels.ravel()

# Train XGBoost (before reweighing)
clf_orig = XGBClassifier(eval_metric='logloss', random_state=42)
clf_orig.fit(X_train, y_train)
y_pred_orig = clf_orig.predict(X_test)

test_pred_orig = test.copy()
test_pred_orig.labels = y_pred_orig.reshape(-1, 1)

metric_orig = ClassificationMetric(test, test_pred_orig,
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)

print("\n--- Before Reweighing (XGBoost) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_orig):.3f}")
print(f"Statistical parity difference: {metric_orig.statistical_parity_difference():.3f}")
print(f"Disparate impact: {metric_orig.disparate_impact():.3f}")
print(f"Equal opportunity difference: {metric_orig.equal_opportunity_difference():.3f}")
print(f"Average odds difference: {metric_orig.average_odds_difference():.3f}")

# Apply reweighing
RW = Reweighing(unprivileged_groups=[{protected: 1}], privileged_groups=[{protected: 0}])
RW.fit(train)
train_rw = RW.transform(train)

# Train XGBoost (after reweighing)
clf_rw = XGBClassifier(eval_metric='logloss', random_state=42)
clf_rw.fit(train_rw.features, train_rw.labels.ravel(), sample_weight=train_rw.instance_weights)
y_pred_rw = clf_rw.predict(X_test)

test_pred_rw = test.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_rw = ClassificationMetric(test, test_pred_rw,
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)

print("\n--- After Reweighing (XGBoost) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rw):.3f}")
print(f"Statistical parity difference: {metric_rw.statistical_parity_difference():.3f}")
print(f"Disparate impact: {metric_rw.disparate_impact():.3f}")
print(f"Equal opportunity difference: {metric_rw.equal_opportunity_difference():.3f}")
print(f"Average odds difference: {metric_rw.average_odds_difference():.3f}")

os.makedirs("analyze_compas_bias_plot", exist_ok=True)

metrics = {
    "Statistical Parity Difference": (metric_orig.statistical_parity_difference(), metric_rw.statistical_parity_difference(), 0),
    "Disparate Impact": (metric_orig.disparate_impact(), metric_rw.disparate_impact(), 1.0),
    "Equal Opportunity Difference": (metric_orig.equal_opportunity_difference(), metric_rw.equal_opportunity_difference(), 0),
    "Average Odds Difference": (metric_orig.average_odds_difference(), metric_rw.average_odds_difference(), 0),
}

for metric_name, (before, after, ref_line) in metrics.items():
    plt.figure(figsize=(6, 5))
    bars = plt.bar(['Before Reweighing', 'After Reweighing'], [before, after],
                   color=['tomato', 'seagreen'])
    plt.axhline(ref_line, color='black', linestyle='--')
    plt.title(f"{metric_name} (XGBoost)")
    plt.ylabel("Value")

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.01 if height > 0 else height - 0.05,
                 f"{height:.3f}", ha='center', va='bottom' if height > 0 else 'top')

    plt.tight_layout()
    filename = metric_name.lower().replace(" ", "_").replace(".", "") + "_xgboost.png"
    plt.savefig(f"analyze_compas_bias_plot/{filename}")
