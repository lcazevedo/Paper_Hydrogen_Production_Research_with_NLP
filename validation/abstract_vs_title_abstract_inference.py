"""
Quantifies the effect of the training/inference input mismatch found in
classify_relevant_setfit.ipynb (training and validation use title+abstract; the
global-inference step on the 40,467 candidates used the abstract field alone).

Requires rel_nrel.csv and papers40k_cluster1_v2.xlsx (not deposited, raw WoS text --
see the main README) and the SetFit checkpoint trained on the GroupShuffleSplit
train partition (2,969 records; never exposed to the 1,004-record held-out test set
used here, so this is a fair, non-leaked comparison).

Reproduced result: evaluating the SAME held-out test set (n=1,004) with the two
inputs gives a 0.5-percentage-point accuracy difference:
  title+abstract (matches training): accuracy 0.9153 | macro F1 0.9013 | [[270,22],[63,649]]
  abstract-only  (matches what was actually deployed): accuracy 0.9104 | macro F1 0.8944 | [[262,30],[60,652]]

This indicates the classifier's validated performance is representative of the
input actually used to build the corpus, so we did not rebuild the corpus on this
basis (see REPRODUCIBILITY.md for the fuller discussion and the suggested SI wording).
"""
import zipfile
from pathlib import Path

import pandas as pd
from setfit import SetFitModel
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

REL_NREL_CSV = "rel_nrel.csv"                       # not deposited (raw text)
CLUSTER_MAP_XLSX = "papers40k_cluster1_v2.xlsx"     # not deposited (raw text)
CHECKPOINT_NAME = "rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit"
CHECKPOINT_ZIP = Path(__file__).parent / f"{CHECKPOINT_NAME}.zip"
CHECKPOINT_DIR = Path(__file__).parent / CHECKPOINT_NAME

if not CHECKPOINT_DIR.exists():
    print(f"Extracting {CHECKPOINT_ZIP.name}...")
    with zipfile.ZipFile(CHECKPOINT_ZIP) as zf:
        zf.extractall(Path(__file__).parent)

print("Loading the group-aware held-out checkpoint...")
model = SetFitModel.from_pretrained(str(CHECKPOINT_DIR))

print("Rebuilding the exact group-aware split...")
xls = pd.ExcelFile(CLUSTER_MAP_XLSX)
cluster_frames = []
for sn in xls.sheet_names:
    df_sheet = pd.read_excel(xls, sn)
    if "doi" in df_sheet.columns:
        cluster_frames.append(df_sheet[["doi"]].assign(cluster_id=sn))
df_clusters = pd.concat(cluster_frames, ignore_index=True).drop_duplicates(subset=["doi"])

df = pd.read_csv(REL_NREL_CSV)
df["abstract_only_text"] = df["abstract"].fillna("")
df["title_abstract_text"] = df["title"].fillna("") + ".\n" + df["abstract"].fillna("")
df = df.merge(df_clusters, on="doi", how="inner")

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
_, test_idx = next(gss.split(df, groups=df["cluster_id"]))
df_test = df.iloc[test_idx].reset_index(drop=True)
true = df_test["label"].tolist()
labels_sorted = sorted(set(true))

for name, col in [("TITLE+ABSTRACT (matches training)", "title_abstract_text"),
                   ("ABSTRACT-ONLY (matches what was deployed)", "abstract_only_text")]:
    preds = model.predict(df_test[col].tolist())
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(true, preds):.4f} | Macro F1: {f1_score(true, preds, average='macro'):.4f}")
    print(confusion_matrix(true, preds, labels=labels_sorted))
