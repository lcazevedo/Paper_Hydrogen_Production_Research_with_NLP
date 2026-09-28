"""
Resolves audit issue #26: seed_papers_extracted_v2.csv has 476 rows but only 473
distinct documents (one filename typo produces a near-duplicate pair twice), and 10
rows have a DOI extracted from the filename that disagrees with the DOI extracted
from the PDF's full text (doi_from_pdf2doi).

Resolution rule: prefer doi_from_filename. Every seed PDF in this collection is
named after its own DOI (dots replaced with underscores), so the filename encodes
the DOI that was assigned when the file was saved -- a deliberate identifier, not a
heuristic guess. doi_from_pdf2doi is a full-text search tool that can match a
different DOI mentioned inside the PDF (a citation, or -- confirmed for one of the
10 conflicts below -- the "Preface" of the same journal issue) rather than the
paper's own DOI. Where doi_from_filename is missing (one row: the filename itself
has a data-entry typo, "110.1016..." instead of "10.1016...", so DOI extraction
from it failed), doi_from_pdf2doi is used instead.

Verified against Crossref (https://api.crossref.org/works/{doi}) for all 10
conflicting rows on 2026-09-28:
  - 10.1039/B803857K (filename) is "Hydrogen evolution on nano-particulate
    transition metal sulfides" (Faraday Discuss.); 10.1039/b814058h (pdf2doi) is
    literally the issue's "Preface" -- confirms doi_from_pdf2doi can pick up an
    unrelated DOI from the same PDF/issue.
  - 10.1126/science.1141483 (filename) is Jaramillo et al.'s "Identification of
    Active Edge Sites for Electrochemical H2 Evolution from MoS2 Nanocatalysts"
    (Science, 2007) -- a canonical paper for this exact research area, strongly
    supporting the filename-DOI as the intended seed.
  - The other 8 conflicts resolve to two different, both topically-plausible
    papers under either DOI; filename-DOI is used for consistency with the two
    unambiguous cases above.

Resolving all 10 this way and de-duplicating by the resolved DOI gives exactly
473 distinct documents (344 experimental, 129 theoretical) -- matching the
independent count the audit itself derived from the raw DOI fields.
"""
import pandas as pd

df = pd.read_csv("../data/seed_papers_extracted_v2.csv")

df["doi_from_filename_norm"] = df["doi_from_filename"].astype(str).str.strip().str.lower()
missing_filename_doi = df["doi_from_filename"].isna() | (df["doi_from_filename_norm"] == "nan")
df["resolved_doi"] = df["doi_from_filename"]
df.loc[missing_filename_doi, "resolved_doi"] = df.loc[missing_filename_doi, "doi_from_pdf2doi"]
df["resolved_doi_norm"] = df["resolved_doi"].astype(str).str.strip().str.lower()

before = len(df)
resolved = df.drop_duplicates(subset="resolved_doi_norm", keep="first").copy()
print(f"{before} rows -> {len(resolved)} distinct documents after DOI resolution and de-duplication")
print(resolved["category"].value_counts().to_string())
assert resolved["resolved_doi_norm"].nunique() == len(resolved), "resolved DOIs are not unique"

out = resolved[["category", "filename", "resolved_doi"]].rename(columns={"resolved_doi": "doi"})
out.to_csv("../data/seed_papers_resolved.csv", index=False)
print(f"\nSaved ../data/seed_papers_resolved.csv ({len(out)} rows)")
