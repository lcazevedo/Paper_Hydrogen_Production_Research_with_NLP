"""
Reconstruction of the Stage-2 challenge classifier (Reviewer 2, point #3).

STATUS: this is NOT the original script. The script and trained weights that produced
the published 206,533 classified challenge sentences were lost and could not be
recovered. This file trains a NEW classifier with the SAME described architecture
(TF-IDF + Logistic Regression, hyperparameters chosen by GridSearchCV) on the one
artifact that survived: the label-to-category mapping used to name the macro-themes
(challenges_classified_v3_optimized.xlsx, 6,000 unique LLM-extracted short labels,
2-5 words each -- never raw title/abstract text, so no WoS copyright restriction
applies to this input).

It exists to show that the described two-step architecture (zero-shot LLM extraction
-> label clustering -> supervised classifier to scale to unlabelled phrases) is
technically sound and reproducible in spirit, not to certify the exact numbers
originally reported. Report results as: "a documented reconstruction with equivalent
architecture, not a recovery of the original weights."

Input: challenges_classified_v3_optimized.xlsx (columns: challenge_grouped,
"number of mentions", "Challenge Category (AI refined)").
Output: 5-fold cross-validated accuracy/F1 (unweighted per unique label, and weighted
by "number of mentions" to approximate sentence-level performance), classification
report, confusion matrix, and out-of-fold predictions for audit.
"""
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

INPUT_XLSX = Path(__file__).parent / "challenges_classified_v3_optimized.xlsx"
OUTPUT_CSV = Path(__file__).parent / "challenge_stage2_classifier_cv_predictions.csv"

df = pd.read_excel(INPUT_XLSX).rename(columns={
    "challenge_grouped": "label_text",
    "number of mentions": "count",
    "Challenge Category (AI refined)": "group",
}).dropna(subset=["label_text", "group"])

X = df["label_text"].astype(str).values
y = df["group"].astype(str).values
weights = df["count"].astype(float).values

pipe = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__max_features": [None, 5000],
    "clf__C": [0.1, 1.0, 3.0, 10.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Training rows (unique labels): {len(df)} | classes: {df['group'].nunique()}")
print("Running GridSearchCV (5-fold, macro-F1)...")
gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=cv, n_jobs=-1)
gs.fit(X, y)
print(f"Best params: {gs.best_params_}")
print(f"Best CV macro-F1: {gs.best_score_:.4f}")

y_pred = cross_val_predict(gs.best_estimator_, X, y, cv=cv, n_jobs=-1)

print(f"\nUnweighted (per unique label)    -> accuracy: {accuracy_score(y, y_pred):.4f} | "
      f"macro-F1: {f1_score(y, y_pred, average='macro'):.4f}")
print(f"Weighted by 'number of mentions' -> accuracy: {accuracy_score(y, y_pred, sample_weight=weights):.4f} | "
      f"macro-F1: {f1_score(y, y_pred, average='macro', sample_weight=weights):.4f}")

print("\nClassification report (unweighted, per unique label):")
print(classification_report(y, y_pred, digits=3))

labels_sorted = sorted(set(y))
print("\nConfusion matrix (rows=true, cols=predicted):")
print(pd.DataFrame(confusion_matrix(y, y_pred, labels=labels_sorted), index=labels_sorted, columns=labels_sorted))

df.assign(predicted_group=y_pred).to_csv(OUTPUT_CSV, index=False)
print(f"\nOut-of-fold predictions saved to {OUTPUT_CSV}")
