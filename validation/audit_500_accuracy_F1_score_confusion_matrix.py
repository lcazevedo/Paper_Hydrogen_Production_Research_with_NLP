import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load the audited data file (Ground Truth vs Prediction)
# Ensure the filename matches the provided spreadsheet
df = pd.read_excel("amostra_500_audited.xlsx")

# Define validation columns
y_true = df["Correct Category (Ground Truth)"]
y_pred = df["Challenge Category (AI refined)"]

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average='macro')

print(f"Overall Accuracy: {acc:.3f}")
print(f"Macro-averaged F1 Score: {f1_macro:.3f}\n")

# Count and display errors
errors = df[df["Correct Category (Ground Truth)"] != df["Challenge Category (AI refined)"]]
print(f"Total Errors: {len(errors)}")

# Generate Confusion Matrix
labels = sorted(df["Correct Category (Ground Truth)"].unique().tolist())
cm_df = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels)

print("\nConfusion Matrix:")
print(cm_df)