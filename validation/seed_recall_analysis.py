"""
Seed-set recall analysis (audit issue #5; correction plan A6).

Checks how many of the 476 expert-nominated seed documents (used to build the
Boolean search query) were actually retrieved by that query, and how many survived
relevance filtering into the final corpus -- broken down by study type. No new data
needed: all three inputs are already in data/.
"""
import pandas as pd

SEEDS_CSV = "../data/seed_papers_extracted_v2.csv"
INITIAL_IDS_CSV = "../data/initial_corpus_IDs.csv"      # 40,467 retrieved records
FINAL_IDS_CSV = "../data/final_corpus_IDs.csv"          # 32,114 final corpus records

seeds = pd.read_csv(SEEDS_CSV)
seeds["doi_norm"] = seeds["doi_from_filename"].astype(str).str.lower().str.strip()
seeds = seeds.drop_duplicates(subset=["doi_norm"])
print(f"Distinct seed DOIs: {len(seeds)}")
print(seeds["category"].value_counts())

initial = pd.read_csv(INITIAL_IDS_CSV)
initial_dois = set(initial["doi"].astype(str).str.lower().str.strip())

final = pd.read_csv(FINAL_IDS_CSV)
final_dois = set(final["doi"].astype(str).str.lower().str.strip())

seeds["retrieved"] = seeds["doi_norm"].isin(initial_dois)
seeds["retained"] = seeds["doi_norm"].isin(final_dois)

n = len(seeds)
n_retrieved = seeds["retrieved"].sum()
n_retained = seeds["retained"].sum()
print(f"\nOverall: retrieved {n_retrieved}/{n} ({n_retrieved/n:.1%}) | "
      f"retained {n_retained}/{n} ({n_retained/n:.1%})")

print("\nBy category:")
for cat, sub in seeds.groupby("category"):
    r = sub["retrieved"].sum()
    k = sub["retained"].sum()
    m = len(sub)
    print(f"  {cat}: retrieved {r}/{m} ({r/m:.1%}) | retained {k}/{m} ({k/m:.1%})")

n_retrieved_not_retained = (seeds["retrieved"] & ~seeds["retained"]).sum()
print(f"\nRetrieved but rejected by the relevance filter: {n_retrieved_not_retained} "
      f"({n_retrieved_not_retained / n_retrieved:.1%} of retrieved)")
