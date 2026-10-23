"""
Checks that the curated label->category dictionary (challenges_classified_v1.csv,
the mapping applied by pipeline/6_group_challenges.py) accounts for every categorized
sentence-level challenge assignment in the deposited dataset.

The deposited dataset (data/final_corpus_data.csv.zip) carries each sentence's final
category ("challenge_group") but not its extracted label text, so the per-sentence join
cannot be rerun here. The check is done on the aggregate instead: the dictionary lists
how many sentences carry each label ("frequency"); summing those counts by category must
reproduce the deposited category totals exactly, and no lowercased/stripped label (the
normalization 6_group_challenges.py applies) may map to two categories.

Result: 47,878 distinct normalized labels; frequencies sum to 206,533; the 12 category
totals match the deposited dataset exactly; 0 conflicting duplicate labels.
"""
import ast
import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
DICT_CSV = HERE / "challenges_classified_v1.csv"
CORPUS_ZIP = HERE.parent / "data" / "final_corpus_data.csv.zip"

d = pd.read_csv(DICT_CSV)
label_col, freq_col, cat_col = d.columns[0], d.columns[1], d.columns[2]
d["key"] = d[label_col].str.lower().str.strip()

print(f"Dictionary rows: {len(d):,} | distinct normalized labels: {d['key'].nunique():,}")
conflicts = int((d.groupby("key")[cat_col].nunique() > 1).sum())
print(f"Normalized labels mapped to more than one category: {conflicts}")

from_dict = d.groupby(cat_col)[freq_col].sum()
print(f"Sentences accounted for by the dictionary: {from_dict.sum():,}")

with zipfile.ZipFile(CORPUS_ZIP) as z:
    name = z.namelist()[0]
    corpus = pd.read_csv(z.open(name), usecols=["challenge_group"])
sentences = corpus["challenge_group"].apply(ast.literal_eval).explode()
deposited = sentences[sentences != "Uncategorized"].value_counts()
print(f"Categorized sentence assignments in the deposited dataset: {deposited.sum():,}")

cmp = pd.DataFrame({"dictionary": from_dict, "deposited": deposited}).fillna(0).astype(int)
cmp["match"] = cmp["dictionary"] == cmp["deposited"]
print(cmp.sort_values("deposited", ascending=False).to_string())
print("\nAll 12 category totals match:", bool(cmp["match"].all()))
