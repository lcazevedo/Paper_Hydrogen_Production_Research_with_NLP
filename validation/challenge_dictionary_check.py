"""
Checks the curated label->category dictionary (data/challenges_classified_v1.csv, the mapping
applied by pipeline/6_group_challenges.py) against the deposited sentence-level table
(data/df_data_phrases.csv.zip: one row per abstract sentence with its extracted challenge label
and final category; no title/abstract text).

Per-sentence checks:
  1. applying the dictionary (labels lowercased and stripped, as 6_group_challenges.py does) to every
     sentence's challenge_label reproduces its challenge_group exactly (0 mismatches);
  2. every categorized sentence's label is in the dictionary;
  3. every "Uncategorized" sentence carries the LLM label "No Challenge" (not in the dictionary).
Aggregate checks:
  4. the dictionary's per-category sentence counts ("frequency") match the category totals of
     data/final_corpus_data.csv.zip for all 12 categories;
  5. no normalized label maps to more than one category.

Result: 223,180 sentences / 32,114 documents; 0 mismatches; 206,533 categorized, 16,647 "No Challenge".
"""
import ast
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent.parent / "data"
d = pd.read_csv(DATA / "challenges_classified_v1.csv")
label_col, freq_col, cat_col = d.columns[0], d.columns[1], d.columns[2]
d["key"] = d[label_col].str.lower().str.strip()
mapping = dict(zip(d["key"], d[cat_col]))

print(f"Dictionary rows: {len(d):,} | distinct normalized labels: {d['key'].nunique():,}")
conflicts = int((d.groupby("key")[cat_col].nunique() > 1).sum())
print(f"Normalized labels mapped to more than one category: {conflicts}")

with zipfile.ZipFile(DATA / "df_data_phrases.csv.zip") as z:
    s = pd.read_csv(z.open(z.namelist()[0]), usecols=["doi_phrase", "doi_original", "challenge_label", "challenge_group"])
s["key"] = s["challenge_label"].astype(str).str.lower().str.strip()
s["dict_category"] = s["key"].map(mapping).fillna("Uncategorized")
unc = s["challenge_group"] == "Uncategorized"

print(f"\nSentences: {len(s):,} | documents: {s['doi_original'].nunique():,}")
print(f"Per-sentence mismatches (dictionary vs. challenge_group): {int((s['dict_category'] != s['challenge_group']).sum())}")
print(f"Categorized: {int((~unc).sum()):,} (label in dictionary: {int(s.loc[~unc, 'key'].isin(mapping).sum()):,})")
print(f"Uncategorized: {int(unc.sum()):,} | labels: {s.loc[unc, 'challenge_label'].value_counts().to_dict()}")
print(f"Uncategorized sentences whose label is in the dictionary: {int(s.loc[unc, 'key'].isin(mapping).sum())}")

from_dict = d.groupby(cat_col)[freq_col].sum()
with zipfile.ZipFile(DATA / "final_corpus_data.csv.zip") as z:
    corpus = pd.read_csv(z.open(z.namelist()[0]), usecols=["challenge_group"])
sentences = corpus["challenge_group"].apply(ast.literal_eval).explode()
deposited = sentences[sentences != "Uncategorized"].value_counts()
cmp = pd.DataFrame({"dictionary": from_dict, "deposited": deposited}).fillna(0).astype(int)
cmp["match"] = cmp["dictionary"] == cmp["deposited"]
print("\nCategory totals, dictionary frequencies vs. final_corpus_data.csv:")
print(cmp.sort_values("deposited", ascending=False).to_string())
print("\nAll 12 category totals match:", bool(cmp["match"].all()))
