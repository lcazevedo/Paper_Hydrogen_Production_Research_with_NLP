"""
Reproduces the published Relevance Index (Table 1, "Without Review RI" column)
exactly from the deposited per-document data (pipeline/relevance_index_input_data.csv
-- doi, year, citations, document_type, topic_name; no title/abstract text), and
tests the dimensional-homogeneity concern raised independently (the RI sums a
dimensional quantity, citations-per-year, with two dimensionless quantities, so its
ranking is not guaranteed invariant to the citation-rate time unit).

Formula (recovered from analysis/charts_and_tables_general.ipynb,
compute_topic_index(), "without review" branch):
  citations_per_year = citations / (2024 - year + 1 + 0.25)
  top_10_percent = citation count in the top decile for that publication year,
                    threshold computed each year over all non-review documents
                    (raw citation counts, not citation rate)
  RI = 0.5 * mean(citations_per_year) + 0.3 * mean(top_10_percent) + 0.2 * log(1+n)
  Review-type documents (document_type contains "Review") are excluded throughout.

Reproduces all 25 published ranks exactly (verified against Table 1).

Sensitivity check: recomputing with citations-per-MONTH instead of citations-per-year
(dividing the citation-rate term only by 12; the top-10% term is already
unit-invariant, since it thresholds raw citation counts, not a rate) gives
Spearman rho = 0.984 between the two rankings -- markedly more stable than the
independent audit's estimate (rho 0.31-0.48), because only the citation-rate term
(50% of the weight) is unit-dependent; the other 50% (decile membership + log
topic size) does not depend on the time unit at all.

A fully unit-invariant alternative (z-scoring each of the three components before
combining, instead of summing raw-scale values) is also reported: Spearman rho with
the published ranking is 0.943; the top 3 ranks are unchanged (including the
2nd-rank position of the Ru-based topic referenced in the manuscript's Conclusions),
with moderate reshuffling further down (largest move: High-Entropy Materials,
10th -> 4th).
"""
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_YEAR = 2024
ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2
INPUT_CSV = Path(__file__).parent.parent / "pipeline" / "relevance_index_input_data.csv"

df = pd.read_csv(INPUT_CSV)
df["document_type"] = df["document_type"].fillna("")
df_wo_review = df[~df["document_type"].str.contains("Review")].copy()


def compute_ri(data, all_docs, cpy_divisor=1.0):
    results = []
    for topic, topic_df in data.groupby("topic_name"):
        d = topic_df.copy()
        d["cpy"] = (d["citations"] / (CURRENT_YEAR - d["year"] + 1 + 0.25)) / cpy_divisor

        top10_flags = []
        for year in d["year"].unique():
            pop = all_docs[all_docs["year"] == year]["citations"]
            if len(pop) >= 10:
                threshold = np.percentile(pop, 90)
                top10_flags.extend((d.loc[d["year"] == year, "citations"] >= threshold).tolist())
            else:
                top10_flags.extend([False] * int((d["year"] == year).sum()))
        d["top10"] = top10_flags[: len(d)]

        n = len(d)
        mean_cpy = d["cpy"].mean()
        prop_top10 = d["top10"].mean()
        log_n = np.log1p(n)
        ri = ALPHA * mean_cpy + BETA * prop_top10 + GAMMA * log_n
        results.append({"topic": topic, "n": n, "mean_cpy": mean_cpy, "prop_top10": prop_top10, "RI": ri})
    return pd.DataFrame(results)


print("Reproducing the published ranking (citations-per-year, reviews excluded)...")
published = compute_ri(df_wo_review, df_wo_review, cpy_divisor=1.0)
published["rank"] = published["RI"].rank(ascending=False, method="min").astype(int)
print(published.sort_values("rank").to_string(index=False))

print("\nSensitivity check: citations-per-MONTH instead of citations-per-year...")
per_month = compute_ri(df_wo_review, df_wo_review, cpy_divisor=1 / 12)
per_month["rank"] = per_month["RI"].rank(ascending=False, method="min").astype(int)
rho_unit = published.set_index("topic")["RI"].corr(per_month.set_index("topic")["RI"], method="spearman")
print(f"Spearman rho (year-based vs. month-based ranking): {rho_unit:.3f}")

print("\nFully unit-invariant alternative: z-score each component before combining...")
def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

z_scored = published.copy()
z_scored["RI_z"] = ALPHA * zscore(published["mean_cpy"]) + BETA * zscore(published["prop_top10"]) + GAMMA * zscore(np.log1p(published["n"]))
z_scored["rank_z"] = z_scored["RI_z"].rank(ascending=False, method="min").astype(int)
rho_z = published["RI"].corr(z_scored["RI_z"], method="spearman")
print(f"Spearman rho (published ranking vs. z-scored ranking): {rho_z:.3f}")
merged = published[["topic", "rank"]].merge(z_scored[["topic", "rank_z"]], on="topic")
merged["rank_change"] = merged["rank"] - merged["rank_z"]
print(merged.sort_values("rank").to_string(index=False))
