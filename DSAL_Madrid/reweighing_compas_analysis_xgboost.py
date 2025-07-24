#
# Created on Thu Jul 24 2025 16:22:59 CET
#
# Banafshe Bamdad
#
# reweighing_compas_analysis_xgboost.py
#

"""
This script evaluates and mitigates racial bias in the COMPAS recidivism prediction dataset
using the AIF360 fairness toolkit and XGBoost classifier.

It focuses on two demographic groups:
- African-American (unprivileged group)
- Caucasian and others (privileged group)

Main steps:
1. Load the original 'compas-scores-two-years.csv' dataset and preprocess it.
2. Encode 'race' and 'sex' for fairness evaluation (African-American = 1, others = 0).
3. Train an XGBoost classifier on the original (biased) data.
4. Evaluate fairness using:
   - Accuracy
   - Statistical Parity Difference
   - Disparate Impact
5. Apply the Reweighing algorithm to mitigate bias before model training.
6. Retrain the classifier using reweighted data and re-evaluate the metrics.
7. Generate and save bar plots visualizing the fairness metrics before and after mitigation.

Fairness Metrics:
- Statistical Parity Difference: Measures difference in positive outcome rates between groups.
- Disparate Impact: Ratio of positive outcome rates. A value near 1.0 indicates fairness.

The generated plots are saved in the 'analyze_compas_bias_plot/' directory.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

df = pd.read_csv("compas-scores-two-years.csv")
df = df.dropna(subset=['race', 'sex', 'age', 'priors_count', 'decile_score', 'two_year_recid'])
df = df[df['score_text'].isin(['Low', 'Medium', 'High'])]

features = ['sex', 'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
target = 'two_year_recid'
protected = 'race'

df['sex'] = df['sex'].astype('category').cat.codes  # Male=1, Female=0
df['race'] = df['race'].apply(lambda r: 1 if r == 'African-American' else 0)  # Unprivileged: 1, Privileged: 0

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

# Train original XGBoost model
clf_orig = XGBClassifier(eval_metric='logloss', random_state=42)
clf_orig.fit(X_train, y_train)
y_pred_orig = clf_orig.predict(X_test)

test_pred_orig = test.copy()
test_pred_orig.labels = y_pred_orig.reshape(-1, 1)

metric_orig = ClassificationMetric(test, test_pred_orig,
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)

print("\n--- Before reweighing (XGBoost) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_orig):.3f}")
print(f"Statistical parity difference: {metric_orig.statistical_parity_difference():.3f}")
print(f"Disparate impact: {metric_orig.disparate_impact():.3f}")

# Apply reweighing
RW = Reweighing(
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)
RW.fit(train)
train_rw = RW.transform(train)

# Train XGBoost model with reweighing weights
clf_rw = XGBClassifier(eval_metric='logloss', random_state=42)

clf_rw.fit(train_rw.features, train_rw.labels.ravel(), sample_weight=train_rw.instance_weights)
y_pred_rw = clf_rw.predict(X_test)

test_pred_rw = test.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_rw = ClassificationMetric(test, test_pred_rw,
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)

print("\n--- After reweighing (XGBoost) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rw):.3f}")
print(f"Statistical parity difference: {metric_rw.statistical_parity_difference():.3f}")
print(f"Disparate impact: {metric_rw.disparate_impact():.3f}")

# Plotting
parity_diff_before = metric_orig.statistical_parity_difference()
parity_diff_after = metric_rw.statistical_parity_difference()
disparate_impact_before = metric_orig.disparate_impact()
disparate_impact_after = metric_rw.disparate_impact()

# Plot statistical parity difference
plt.figure(figsize=(6, 5))
bars = plt.bar(['Before Reweighing', 'After Reweighing'],
               [parity_diff_before, parity_diff_after],
               color=['tomato', 'seagreen'])
plt.axhline(0, color='black', linestyle='--')
plt.title("Statistical Parity Difference (XGBoost)")
plt.ylabel("Value")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01 if height > 0 else height - 0.05,
             f"{height:.3f}", ha='center', va='bottom' if height > 0 else 'top')
plt.tight_layout()
plt.savefig("analyze_compas_bias_plot/statistical_parity_difference_xgboost.png")
# plt.show()

# Plot disparate impact
plt.figure(figsize=(6, 5))
bars = plt.bar(['Before Reweighing', 'After Reweighing'],
               [disparate_impact_before, disparate_impact_after],
               color=['dodgerblue', 'mediumorchid'])
plt.axhline(1.0, color='black', linestyle='--')
plt.title("Disparate Impact (XGBoost)")
plt.ylabel("Value")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.3f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("analyze_compas_bias_plot/disparate_impact_xgboost.png")
# plt.show()
