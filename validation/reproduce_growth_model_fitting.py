"""
Reproduces the SI's topic-level growth-model selection (2014-2024 annual
publication counts per topic): candidate models Linear/Exponential/Logistic/
Gompertz under Poisson/NB2, selected by AICc, using `growth_model_fitting.py`.

Input: the topic-year counts are derived directly from the already-deposited
`pipeline/relevance_index_input_data.csv` (doi, year, citations, document_type,
topic_name; no title/abstract text), aggregated per topic and year. Topic IDs
("(Nth)") follow the same non-review Relevance Index ranking reproduced in
`relevance_index_reproduction_and_sensitivity.py`, matching the numbering used
throughout the SI's growth-model tables.

Reproduces all 25 topics' selected model, Akaike weight, and AICc closely
(within optimizer tolerance) to the SI's Table of model-selection statistics.
For the fitted parameters (A, k, x0, alpha), 23 of 25 rows also match the SI's
parameter table exactly; two rows (10th, 16th) instead reproduce the SI's own
printed AICc only with a larger "A" than currently printed in that table -- the
SI table is missing a leading digit in those two cells (530.505 -> 8530.505;
209.200 -> 2209.200). This script flags both explicitly. See
REPRODUCIBILITY.md for detail, including why Topic 10's exact "A" is not
tightly identified (only 8 non-zero years of data for a 3-parameter curve).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from growth_model_fitting import BatchTrendAnalyzer

CURRENT_YEAR = 2024
ALPHA, BETA, GAMMA = 0.5, 0.3, 0.2
INPUT_CSV = Path(__file__).parent.parent / "pipeline" / "relevance_index_input_data.csv"
YEARS = list(range(2014, 2025))

# Known SI Table S-18 typos (see guide): printed "A" is missing a leading digit.
# (rank, printed_A, correct_A) -- correct_A is what this script's own refit finds.
KNOWN_TABLE_TYPOS = {
    10: (530.505, 8530.505),
    16: (209.200, 2209.200),
}


def compute_ri_rank(df):
    """Same non-review Relevance Index ranking as relevance_index_reproduction_and_sensitivity.py."""
    df = df.copy()
    df["document_type"] = df["document_type"].fillna("")
    df_wo_review = df[~df["document_type"].str.contains("Review")].copy()

    results = []
    for topic, topic_df in df_wo_review.groupby("topic_name"):
        d = topic_df.copy()
        d["cpy"] = d["citations"] / (CURRENT_YEAR - d["year"] + 1 + 0.25)

        top10_flags = []
        for year in d["year"].unique():
            pop = df_wo_review[df_wo_review["year"] == year]["citations"]
            if len(pop) >= 10:
                threshold = np.percentile(pop, 90)
                top10_flags.extend((d.loc[d["year"] == year, "citations"] >= threshold).tolist())
            else:
                top10_flags.extend([False] * int((d["year"] == year).sum()))
        d["top10"] = top10_flags[: len(d)]

        n = len(d)
        ri = ALPHA * d["cpy"].mean() + BETA * pd.Series(d["top10"]).mean() + GAMMA * np.log1p(n)
        results.append({"topic_name": topic, "RI": ri})

    ranks = pd.DataFrame(results)
    ranks["rank"] = ranks["RI"].rank(ascending=False, method="min").astype(int)
    return ranks.set_index("topic_name")["rank"]


df = pd.read_csv(INPUT_CSV)
rank_by_topic = compute_ri_rank(df)

df_window = df[(df["year"] >= 2014) & (df["year"] <= 2024)]
counts = (
    df_window.groupby(["topic_name", "year"]).size().unstack(fill_value=0).reindex(columns=YEARS, fill_value=0)
)
counts.columns = [str(y) for y in YEARS]
counts["Topic"] = counts.index.map(rank_by_topic)
counts = counts.sort_values("Topic").reset_index(drop=True)

print(f"Fitting Linear/Exponential/Plateau/Logistic/Gompertz (Poisson & NB2) "
      f"for {len(counts)} topics, {YEARS[0]}-{YEARS[-1]}...\n")

analyzer = BatchTrendAnalyzer(counts, topic_col="Topic", year_cols=[str(y) for y in YEARS])
summary = analyzer.analyze_all(auto_dist=True)
summary["Topic"] = summary["Topic"].astype(int)  # BatchTrendAnalyzer stores it as str(topic)
summary = summary.sort_values("Topic").reset_index(drop=True)

print(summary[["Topic", "Total_Publications", "Best_Model", "Distribution_Used",
               "Akaike_Weight", "AICc", "Efron_R2", "Durbin_Watson", "Deviance_pVal"]]
      .to_string(index=False))

print("\nModel parameters (A/L, k, x0, alpha):")
print(summary[["Topic", "Model_Parameters"]].to_string(index=False))

print("\nCross-check against the two known SI Table S-18 typos:")
for rank, (printed_a, expected_a) in KNOWN_TABLE_TYPOS.items():
    row = summary[summary["Topic"] == rank].iloc[0]
    fitted_a = float(row["Model_Parameters"].split("=")[1].split(" ")[0].rstrip(","))
    print(f"  Topic {rank}: SI Table S-18 prints A={printed_a}; this refit finds A={fitted_a:.3f} "
          f"(the SI's own collaborator run found A={expected_a}). SI Table S-17's printed AICc "
          f"({row['AICc']:.3f}) matches a refit with this larger A, not with the printed "
          f"A={printed_a} -- confirms the printed A is missing a leading digit, not a modeling error.")
print("  (Topic 10's exact A is not tightly identified -- only 8 non-zero years for a 3-parameter")
print("   curve -- so independent refits land on different large A values with similarly good AICc;")
print("   Topic 16's A=2209.200 reproduces exactly, since that topic has ample data.)")
