> **This repository is a fork of the original ProPublica COMPAS analysis repository.**  
> The original project and dataset are available here: [https://github.com/propublica/compas-analysis](https://github.com/propublica/compas-analysis)

> Source code for **Group 3**'s project in the Una Europa Challenge: Data Science and AI for Social Welfare is available here: https://github.com/banafshebamdad/compas-analysis/tree/master/DSAL_Madrid


# Bias Analysis Report on the COMPAS Dataset - Group 3

## Summary
This report investigates potential bias in the COMPAS recidivism risk assessment algorithm using the publicly available dataset from ProPublica. The focus is on whether the model disproportionately classifies certain demographic groups, especially by race and gender, as high risk, even when their actual recidivism behavior does not justify such classifications. Our findings reveal patterns that strongly suggest the presence of racial and gender bias in the COMPAS system.

## Key Findings

### 1. Disparities in Risk Scores
- African-American individuals tend to receive higher decile scores compared to other groups.
- In contrast, Caucasian individuals tend to receive lower scores, even when their actual recidivism rates are similar.

### 2. False Positive Rates (FPR)
- African-American females had a false positive rate of 25.2%, meaning over a quarter were wrongly labeled high risk while they did not reoffend.
- This contrasts with Caucasian females, who had an FPR of only 19.6%.
- Such disparities suggest the algorithm may overestimate risk for African-Americans.

### 3. False Negative Rates (FNR)
- For Caucasian females, the false negative rate was 15.2%, while for African-American males, it was 15.0%, indicating potentially underestimating risk for certain groups.
- Some minority groups, such as Asian females, had very small sample sizes, but still experienced an FNR of 50%, highlighting instability.

## Visual Evidence

### Decile Score Distribution by Race and Sex
This plot shows that African-American individuals, especially males, tend to receive higher scores:
![Decile Score by Race and Sex](https://github.com/banafshebamdad/compas-analysis/raw/master/DSAL_Madrid/analyze_compas_bias_plot/decile_score_by_race_and_sex.png)

### False Positive Rate by Race and Sex
Certain groups are disproportionately predicted to reoffend despite not doing so.
![False Positive Rate](https://github.com/banafshebamdad/compas-analysis/raw/master/DSAL_Madrid/analyze_compas_bias_plot/false_positive_rate_by_race_sex.png)

### False Negative Rate by Race and Sex
Some groups receive lower risk scores even when they did reoffend:
![False Negative Rate](https://github.com/banafshebamdad/compas-analysis/raw/master/DSAL_Madrid/analyze_compas_bias_plot/false_negative_rate_by_race_sex.png)

## Conclusion

The analysis demonstrates that the COMPAS algorithm exhibits disparate impact across racial and gender lines. This violates principles of fairness in algorithmic decision-making. While COMPAS may not explicitly use race as an input, its use of correlated variables appears to produce racially biased outcomes. These findings underscore the need for greater transparency, accountability, and fairness auditing in criminal justice algorithms.

