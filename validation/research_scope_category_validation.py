"""
Category-specific validation for "Research Scope, Novelty, and Generalization"
(Reviewer 2, point #5).

No new data needed: amostra_500_audited.xlsx (the same 500-sentence audit used for
the overall accuracy/F1/confusion matrix) already contains labelled rows for this
category. Extracting them reproduces the SI's reported Precision 0.97 / Recall 1.00
for this category exactly, and provides representative extracted phrases (2-5 word
LLM labels, not raw WoS sentences -- see the challenge-extraction methodology in the
SI) to answer the reviewer's request for illustrative examples.
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

CATEGORY = "Research Scope, Novelty, and Generalization"

df = pd.read_excel("amostra_500_audited.xlsx")
y_true = (df["Correct Category (Ground Truth)"] == CATEGORY).astype(int)
y_pred = (df["Challenge Category (AI refined)"] == CATEGORY).astype(int)

print(f"Category: {CATEGORY}")
print(f"Rows involving this category (ground truth or predicted): "
      f"{((df['Correct Category (Ground Truth)']==CATEGORY) | (df['Challenge Category (AI refined)']==CATEGORY)).sum()}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1:        {f1_score(y_true, y_pred):.3f}")

print("\nRepresentative extracted phrases (ground truth = this category):")
sample = df.loc[df["Correct Category (Ground Truth)"] == CATEGORY, "challenge"].head(20)
for phrase in sample:
    print(f"  - {phrase}")
