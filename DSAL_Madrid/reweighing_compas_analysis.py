#
# Created on Thu Jul 24 2025 10:57:23 CET
#
# Banafshe Bamdad
#
# reweighing_compas_analysis.py
#
"""
This script applies the Reweighing algorithm from AIF360 to the COMPAS dataset
to mitigate bias with respect to the 'race' attribute. It trains a logistic regression
model before and after reweighing, and compares fairness metrics such as
statistical parity difference and disparate impact.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("compas-scores-two-years.csv")

df = df.dropna(subset=['race', 'sex', 'age', 'priors_count', 'decile_score', 'two_year_recid']) # Drop rows with missing values

df = df[df['score_text'].isin(['Low', 'Medium', 'High'])] # Filter to valid score_text categories

features = ['sex', 'age', 'priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count']
target = 'two_year_recid'
protected = 'race'

# Encode categorical features
df['sex'] = df['sex'].astype('category').cat.codes  # Male=1, Female=0
df['race'] = df['race'].apply(lambda r: 1 if r == 'African-American' else 0)  # Unprivileged: 1, Privileged: 0

dataset = BinaryLabelDataset(
    favorable_label=0,  # Not recidivated
    unfavorable_label=1,  # Recidivated
    df=df[features + [target, protected]],
    label_names=[target],
    protected_attribute_names=[protected]
)

train, test = dataset.split([0.7], shuffle=True)

# Train logistic regression model on original (biased) data
X_train = train.features
y_train = train.labels.ravel()
X_test = test.features
y_test = test.labels.ravel()

clf_orig = LogisticRegression(max_iter=1000)
clf_orig.fit(X_train, y_train)
y_pred_orig = clf_orig.predict(X_test)

# Convert prediction to AIF360 format
test_pred_orig = test.copy()
test_pred_orig.labels = y_pred_orig.reshape(-1, 1)

# Evaluate fairness before reweighing
metric_orig = ClassificationMetric(test, test_pred_orig,
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)

print("\n=== Before reweighing ===")
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

# Train model with sample weights
clf_rw = LogisticRegression(max_iter=1000)
clf_rw.fit(train_rw.features, train_rw.labels.ravel(), sample_weight=train_rw.instance_weights)

# Predict and evaluate
y_pred_rw = clf_rw.predict(X_test)

test_pred_rw = test.copy()
test_pred_rw.labels = y_pred_rw.reshape(-1, 1)

metric_rw = ClassificationMetric(test, test_pred_rw,
    unprivileged_groups=[{protected: 1}],
    privileged_groups=[{protected: 0}]
)

print("\n=== After reweighing ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rw):.3f}")
print(f"Statistical parity difference: {metric_rw.statistical_parity_difference():.3f}")
print(f"Disparate impact: {metric_rw.disparate_impact():.3f}")

# Collect metrics
parity_diff_before = metric_orig.statistical_parity_difference()
parity_diff_after = metric_rw.statistical_parity_difference()

disparate_impact_before = metric_orig.disparate_impact()
disparate_impact_after = metric_rw.disparate_impact()

# Statistical Parity Difference Plot
plt.figure(figsize=(6, 5))
bars = plt.bar(['Before reweighing', 'After reweighing'],
               [parity_diff_before, parity_diff_after],
               color=['tomato', 'seagreen'])
plt.axhline(0, color='black', linestyle='--')
plt.title("Statistical Parity Difference")
plt.ylabel("Value")

# Annotate values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01 if height > 0 else height - 0.05,
             f"{height:.3f}", ha='center', va='bottom' if height > 0 else 'top')
plt.tight_layout()
plt.savefig("analyze_compas_bias_plot/statistical_parity_difference_before_after.png")
plt.show()

# Disparate impact plot
plt.figure(figsize=(6, 5))
bars = plt.bar(['Before Reweighing', 'After Reweighing'],
               [disparate_impact_before, disparate_impact_after],
               color=['dodgerblue', 'mediumorchid'])
plt.axhline(1.0, color='black', linestyle='--')  # ideal disparate impact
plt.title("Disparate Impact")
plt.ylabel("Value")

# Annotate values
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
             f"{height:.3f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("disparate_impact_before_after.png")
plt.show()
