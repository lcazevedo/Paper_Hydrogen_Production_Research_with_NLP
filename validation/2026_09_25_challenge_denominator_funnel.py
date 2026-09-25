"""
Reproduces the sentence-level denominator funnel for the 12 challenge categories
(Reviewer 2, point #4, JFUE-D-26-07591R1).

Input: final_corpus_data.csv (inside final_corpus_data.csv.zip), field "challenge_group",
one list of category strings per document (one entry per title/abstract sentence).

Reports, from the same public data file already in this repository:
  - total sentence-level challenge assignments (N_total)
  - "Uncategorized" assignments (N_uncategorized)
  - categorized assignments, i.e. the denominator used for Figure 6 (N_categorized)
  - each category's share of N_categorized (as printed in the manuscript)
  - each category's share of N_total (i.e., including Uncategorized in the denominator)
"""
import ast
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

DATA_ZIP = Path(__file__).parent.parent / "data" / "final_corpus_data.csv.zip"

with zipfile.ZipFile(DATA_ZIP) as zf:
    with zf.open("final_corpus_data.csv") as f:
        df = pd.read_csv(f)

df["challenge_group"] = df["challenge_group"].apply(ast.literal_eval)
all_labels = [label for row in df["challenge_group"] for label in row]
counts = Counter(all_labels)

n_total = len(all_labels)
n_uncategorized = counts["Uncategorized"]
n_categorized = n_total - n_uncategorized

print(f"Documents:                          {len(df):,}")
print(f"Total sentence-level assignments:   {n_total:,}")
print(f"  Uncategorized:                    {n_uncategorized:,} ({n_uncategorized / n_total:.2%} of total)")
print(f"  Categorized (Figure 6 denominator):{n_categorized:,} ({n_categorized / n_total:.2%} of total)")
print()
print(f"{'Category':<50}{'n':>8}{'% of categorized':>20}{'% of total':>14}")
for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    if name == "Uncategorized":
        continue
    print(f"{name:<50}{n:>8}{n / n_categorized:>19.2%}{n / n_total:>14.2%}")

top3 = [n for name, n in counts.most_common(4) if name != "Uncategorized"][:3]
print()
print(f"Top-3 categories, share of categorized (as reported, e.g. 53.3%): {sum(top3) / n_categorized:.2%}")
print(f"Top-3 categories, share of total incl. Uncategorized:             {sum(top3) / n_total:.2%}")
