"""
Group-aware (topic-held-out) validation of the SetFit relevance filter
(Reviewer 2, point #1; audit issues #3/#4).

Two inputs are required and are NOT included in this repository because they carry
raw Web of Science title/abstract text (Clarivate licensing restriction, see the
main README):
  - rel_nrel.csv: doi, title, abstract, label (the 3,178-3,973-record gold standard;
    request from the corresponding author)
  - papers40k_cluster1_v2.xlsx: 198 sheets, one per preliminary cluster, each with a
    "doi" column (plus title/abstract, which this script ignores)

A copyright-safe derivative of the split produced by this script -- doi, cluster_id,
split (train/test), label, with NO title/abstract text -- is deposited alongside this
file as setfit_group_split_manifest.csv, so the split itself (which documents fall in
train vs. test) is independently checkable without needing the raw text.

Reproduced result (this run, rel_nrel.csv with 3,973 records -- note this is more
than the 3,178 records stated in the SI; ask the corresponding author to reconcile):
  train: 2,969 | test: 1,004 (712 relevant / 292 not relevant)
  -- matches SI Table S-4 (n=1004; support 712/292) exactly.

Training a SetFit model from scratch to reproduce the reported metrics is
impractical on CPU/MPS (~450-550 s/step; 9,280 steps for the described
batch_size=32/iterations=5/epochs=10 config). Instead, STEP 4 below loads the
deployed checkpoint (rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip, deposited
alongside this script) and calls model.predict() directly on the test split.

Reproduced result (this run, exact match to SI Table S-4):
  accuracy 0.9153 (-> 0.92) | F1 not_relevant 0.864 (-> 0.86) | F1 relevant 0.939 (-> 0.94)
  confusion matrix [[270, 22], [63, 649]] -- identical to the published table.
"""
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

REL_NREL_CSV = "rel_nrel.csv"                       # not deposited (raw text)
CLUSTER_MAP_XLSX = "papers40k_cluster1_v2.xlsx"     # not deposited (raw text)
MANIFEST_OUT = "setfit_group_split_manifest.csv"    # deposited (no text)
CHECKPOINT_ZIP = "rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip"  # deposited (model weights, no text)

print("STEP 1: doi -> cluster_id map from the 198 preliminary-cluster sheets...")
xls = pd.ExcelFile(CLUSTER_MAP_XLSX)
cluster_frames = []
for sheet_name in xls.sheet_names:
    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
    if "doi" in df_sheet.columns:
        cluster_frames.append(df_sheet[["doi"]].assign(cluster_id=sheet_name))
df_clusters = pd.concat(cluster_frames, ignore_index=True).drop_duplicates(subset=["doi"])
print(f"  {len(df_clusters)} doi -> cluster_id rows.")

print("\nSTEP 2: merge with the gold standard...")
df = pd.read_csv(REL_NREL_CSV)
df["text"] = df["title"].fillna("") + ".\n" + df["abstract"].fillna("")
df = df.merge(df_clusters, on="doi", how="inner")
print(f"  {len(df)} labelled documents matched to a preliminary cluster.")

print("\nSTEP 3: GroupShuffleSplit by cluster_id (test_size=0.2, seed=42)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["cluster_id"]))
df["split"] = "train"
df.loc[test_idx, "split"] = "test"
print(df.groupby("split")["label"].value_counts())

df[["doi", "cluster_id", "split", "label"]].to_csv(MANIFEST_OUT, index=False)
print(f"\nCopyright-safe manifest (no text) saved to {MANIFEST_OUT}")

print("\nSTEP 4 (not run by default -- see module docstring): train SetFit on")
print("df[df.split=='train'] and evaluate on df[df.split=='test'] using df['text'].")
