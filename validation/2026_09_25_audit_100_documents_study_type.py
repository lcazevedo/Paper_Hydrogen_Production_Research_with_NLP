"""
Experimental vs. computational study-type audit (Reviewer 2, point #6).

Uses audit_100_documents_labels_only.csv (predicted_topic_2, audit), a copyright-safe
derivative of the original audit spreadsheet with the title/abstract columns removed
(that original file also cannot be redistributed under the Web of Science licensing
restriction described in the main README).

Reproduced result (matches Reviewer 2's own figures exactly):
  Overall error rate: 27.0%
  Hybrid studies misclassified: 26 (23 as Experimental, 0 as Computational)
  Pure Experimental misclassified as Computational: 0
  Pure Computational misclassified as Experimental: 1
"""
import pandas as pd

df = pd.read_csv("audit_100_documents_labels_only.csv")

col_pred, col_true = "predicted_topic_2", "audit"
df["pred_clean"] = df[col_pred].astype(str).str.lower().str.strip()
df["true_clean"] = df[col_true].astype(str).str.lower().str.strip()

label_map = {
    "experimental studies": "experimental",
    "computational and simulation studies": "computacional",
}
df["pred_mapped"] = df["pred_clean"].map(label_map)

errors = df[df["pred_mapped"] != df["true_clean"]]
error_rate = len(errors) / len(df) * 100

print(f"Total studies evaluated: {len(df)}")
print(f"Overall error rate: {error_rate:.1f}%\n")

hybrid_errors = errors[errors["true_clean"] == "híbrida"]
print("--- Failure mode breakdown ---")
print(f"Hybrid studies incorrectly classified: {len(hybrid_errors)}")
print(f"  - as Experimental: {len(hybrid_errors[hybrid_errors['pred_mapped'] == 'experimental'])}")
print(f"  - as Computational: {len(hybrid_errors[hybrid_errors['pred_mapped'] == 'computacional'])}\n")

exp_errors = errors[errors["true_clean"] == "experimental"]
print(f"Pure Experimental misclassified as Computational: {len(exp_errors)}")

comp_errors = errors[errors["true_clean"] == "computacional"]
print(f"Pure Computational misclassified as Experimental: {len(comp_errors)}")
