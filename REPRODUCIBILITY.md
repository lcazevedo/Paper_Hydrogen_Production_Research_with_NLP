# Reproducibility notes

This file states what is independently reproducible from the data and code in this
repository, and is explicit about what is not (and why).

## TL;DR

Most of the core pipeline claims are **fully reproduced** from the deposited data
alone: the relevance-filter validation (split, accuracy, F1, confusion matrix), the
challenge-category denominator, the "Research Scope" category validation, the
experimental/computational study-type audit, and the topic-model sizes (Table 1),
using the original saved topic model. One component — a supervised classifier used
to scale challenge classification beyond a curated label dictionary — could not be
recovered; a documented **reconstruction** with an equivalent architecture is
deposited instead and clearly marked as such, alongside the deterministic grouping
step that was recovered. We also identified and resolved an input-consistency issue
in the relevance classifier, and a gold-standard-size discrepancy that turned out to
be a target-vs-realized split-size artifact rather than a data mismatch (both
documented below).

## Quick verification

These reproduce a published number exactly, from data already in this repository,
with no non-deposited inputs needed. From the repository root:

```bash
cd validation
python3 challenge_denominator_funnel.py
# -> Categorized (Figure 6 denominator): 206,533 (92.54% of total); top-3 share 53.25% (-> 53.3%)

python3 audit_500_accuracy_F1_score_confusion_matrix.py
# -> Overall Accuracy: 0.934, Macro-averaged F1 Score: 0.907  (matches SI Table S-7)

python3 research_scope_category_validation.py
# -> Precision: 0.967, Recall: 1.000  (matches SI: 0.97 / 1.00)

python3 audit_100_documents_study_type.py
# -> Overall error rate: 27.0%; hybrid studies misclassified: 26 (23 as Experimental)

python3 reproduce_table1_from_original_model.py
# -> reproduces Table 1's topic sizes from the original BERTopic/EVoC model

python3 relevance_index_reproduction_and_sensitivity.py
# -> reproduces Table 1's Relevance Index ranking exactly, then reports the
#    citation-rate-unit sensitivity check (rho=0.984) and a unit-invariant
#    z-scored alternative (rho=0.943 vs. the published ranking)

python3 reproduce_growth_model_fitting.py
# -> refits all 25 topics' 2014-2024 growth models (Linear/Exponential/Logistic/
#    Gompertz, Poisson/NB2, selected by AICc) and reproduces the SI's selected
#    model, Akaike weight, and AICc for every topic
```

The following also reproduce exactly but need the deposited checkpoint
(`rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip`, unzips automatically) plus two
raw-text files available on request from the corresponding author
(`rel_nrel.csv`, `papers40k_cluster1_v2.xlsx` — not deposited, see below):

```bash
python3 setfit_group_aware_validation.py
# -> train: 2,969 | test: 1,004 (712 relevant / 292 not relevant)  (matches SI Table S-4)

python3 abstract_vs_title_abstract_inference.py
# -> title+abstract: accuracy 0.9153, [[270,22],[63,649]]  (matches SI Table S-4 exactly)
# -> abstract-only:  accuracy 0.9104, [[262,30],[60,652]]  (0.5-pp difference)
```

## Why some inputs are not in this repository

Web of Science (Clarivate) title/abstract text cannot be redistributed under the
database's terms of business (see the main README). Any file that would carry raw
title/abstract text for more than a small, individually-quoted illustrative excerpt
is therefore **not** deposited here, even where the corresponding code is. This
affects the SetFit gold standard (`rel_nrel.csv`) and the preliminary-cluster map
(`papers40k_cluster1_v2.xlsx`, which also carries title/abstract columns). Where
possible, a text-free derivative (DOI + numeric/categorical fields only) is deposited
instead, so the parts of the pipeline that do not require reading the text — split
composition, category tallies, group assignments, topic assignment — remain
independently checkable.

## Verification status by component

| Component | Status | Where |
|---|---|---|
| SetFit relevance-filter `GroupShuffleSplit` validation | **Fully reproduced.** Split reproduces exactly (train 2,969 / test 1,004; 712 relevant, 292 not relevant). The deposited checkpoint reproduces the published accuracy (0.9153→0.92), F1 (0.94/0.86), and confusion matrix ([[270,22],[63,649]]) exactly. | `validation/setfit_group_aware_validation.py`, `validation/setfit_group_split_manifest.csv`, `validation/rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip` |
| Topic-model sizes and stability | **Fully reproduced.** The original saved topic model reproduces Table 1 almost exactly (~0.1-0.3% differences, matching the 109 metadata-excluded documents). | `pipeline/model_pred/`, `pipeline/topic_assignment_original_model.csv`, `validation/reproduce_table1_from_original_model.py`; independent re-fit sensitivity check in `validation/evoc_clustering_sensitivity_analysis.py` |
| Challenge-classification pipeline: grouping step + 500-sentence audit + supervised classifier | Grouping step (`6_group_challenges.py`) recovered and deposited (deterministic label→group merge). The supervised classifier's script/weights were **not recoverable**; a documented *reconstruction* with an equivalent architecture is deposited instead, clearly marked as such. The 500-sentence audit fully reproduces. | `pipeline/6_group_challenges.py`, `validation/challenge_stage2_classifier_reconstruction.py`, `validation/audit_500_accuracy_F1_score_confusion_matrix.py`, `validation/amostra_500_audited.xlsx` |
| Challenge-category denominator | **Fully reproduced** from the already-public `final_corpus_data.csv`. | `validation/challenge_denominator_funnel.py` |
| "Research Scope" category sample and precision/recall | **Fully reproduced**, no new data needed. | `validation/research_scope_category_validation.py` |
| Experimental/computational study-type audit (100 documents) | **Fully reproduced**: 27.0% overall error rate; 26 of the errors are hybrid experimental/computational studies (23 misclassified as Experimental). | `validation/audit_100_documents_study_type.py`, `validation/audit_100_documents_labels_only.csv` |
| Seed-set recall | **Reproduced**, with a data-quality fix applied first (see notes below): the raw seed file has 476 rows but only 473 distinct documents. After resolving this: 473 distinct seeds (344 experimental, 129 theoretical); Experimental 257/344 (74.7%) retrieved, 240/344 (69.8%) retained; Theoretical 72/129 (55.8%) retrieved, 68/129 (52.7%) retained. | `validation/resolve_seed_doi_conflicts.py`, `validation/seed_recall_analysis.py`, `data/seed_papers_resolved.csv` |
| Relevance-filter input consistency (training vs. deployment) | **Resolved.** The SI described the wrong input for global inference; corrected to describe the input actually deployed, with a validation check confirming equivalent performance (0.9153 vs. 0.9104 accuracy on the same held-out test set). | `validation/abstract_vs_title_abstract_inference.py` |
| Gold-standard size (SI text vs. deposited file) | **Resolved.** The SI's stated size (3,178 records) is the *target* size of the `GroupShuffleSplit`'s 80% training partition (`test_size=0.2`, applied to the full 3,973-record gold standard: 0.8×3,973=3,178.4), not a different dataset. Because the split must keep each preliminary cluster whole, the realized partition sizes differ from the 80/20 target — the deposited split (`random_state=42`) comes out to 2,969/1,004 (74.7%/25.3%) rather than ≈3,178/795. Confirmed with the corresponding author. | `validation/setfit_group_aware_validation.py` |
| Relevance Index dimensional-homogeneity check | **Reproduced and tested.** The published ranking (Table 1) reproduces exactly from the deposited per-document data. The index sums a dimensional quantity (citations per year) with two dimensionless ones (top-decile membership, log topic size), so nothing guarantees its ranking is invariant to the citation-rate time unit; we tested this directly. Switching citations-per-year to citations-per-month gives Spearman ρ=0.984 with the published ranking: only the citation-rate term (50% of the weight) is unit-dependent, since decile membership is thresholded on raw citation counts, not a rate, and log topic size does not involve citations at all. A fully unit-invariant alternative (z-scoring each component before combining) correlates at ρ=0.943 with the published ranking; the top 3 ranks, including the 2nd-rank position referenced in the Conclusions, are unchanged. | `pipeline/relevance_index_input_data.csv`, `validation/relevance_index_reproduction_and_sensitivity.py` |
| Topic-level growth-model selection (2014-2024) | **Fully reproduced.** Refitting all 25 topics' annual counts (Linear/Exponential/Logistic/Gompertz, Poisson/NB2, selected by corrected AIC) reproduces the SI's selected model, Akaike weight, and AICc for every topic. Two of the SI's fitted-parameter table cells have a typo (a missing leading digit in the "A" column, see notes below); everything else, including the model-selection table, is consistent. | `pipeline/relevance_index_input_data.csv`, `validation/growth_model_fitting.py`, `validation/reproduce_growth_model_fitting.py` |

## Notes on specific artifacts

**SetFit checkpoint.** `rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip` is trained
on the `GroupShuffleSplit` train partition only (2,969 records; its model card
confirms 1,397 not_relevant + 1,572 relevant), so it was never exposed to the
1,004-record held-out test set used above — a fair evaluation. Retraining from
scratch was not attempted here: on Apple-Silicon (MPS) the described configuration
(9,280 steps) runs at roughly 450-550 s/step, impractical on that hardware.

**Challenge classifier.** `validation/challenge_stage2_classifier_reconstruction.py`
is an explicitly-labelled reconstruction, not the original code: TF-IDF + Logistic
Regression (hyperparameters chosen by 5-fold `GridSearchCV`) trained on the deposited
label→category dictionary (`challenges_classified_v3_optimized.xlsx`). 5-fold CV
accuracy 0.922 (unweighted per unique label) / 0.969 (weighted by frequency). This is
documented explicitly as a reconstruction, not recovered code.

**Seed-file DOI resolution.** The raw seed file (`data/seed_papers_extracted_v2.csv`)
has 476 rows carrying two DOI fields per document, extracted two different ways
(from the filename, and from the PDF's full text via a DOI-lookup tool). These
476 rows correspond to only 473 distinct documents: one filename typo creates a
near-duplicate pair, and 10 rows have the two DOI fields disagreeing. Each seed PDF
in this collection is deliberately named after its own DOI, so the filename-derived
DOI is treated as authoritative, with the PDF-text-derived DOI used only as a
fallback for the one row where the filename itself has a typo. This rule was checked
against public Crossref metadata (`api.crossref.org/works/{doi}`) for all 10
disagreements: in one case the PDF-text-derived DOI resolves to the journal issue's
"Preface" rather than to an article — confirming that full-text DOI lookup can pick
up an unrelated DOI mentioned elsewhere in the same PDF — while the corresponding
filename-derived DOI resolves to a genuine, topically-relevant article. Applying
this rule and de-duplicating gives exactly 473 distinct seed documents (344
experimental, 129 theoretical). Script: `validation/resolve_seed_doi_conflicts.py`;
output: `data/seed_papers_resolved.csv` (doi, filename, category — no title/abstract
text).

**Relevance Index: computed-but-unused term.** The Relevance Index computation also
derives a per-document z-score and a year-level mean/standard deviation, but these
do not appear in the composite index formula (0.5·mean_CPY + 0.3·prop_top10 +
0.2·log(1+n) — see `validation/relevance_index_reproduction_and_sensitivity.py`).
This is intentional and not a bug in the reproduction: the reproduced ranking
matches Table 1 exactly using only the three terms in the formula, without the
z-score.

**Growth-model fitting parameters: two SI table cells missing a leading digit.**
Refitting the 2014-2024 annual counts for all 25 topics reproduces the model-selection
table (best model, Akaike weight, AICc) exactly for every topic. Cross-checking against
the SI's separate table of fitted parameters (A/L, k, x0, alpha) shows 23 of 25 rows
match exactly; two do not: the SI table's asymptote column reads 530.505 for one topic
and 209.200 for another, while a refit that reproduces that same table's own AICc for
those two rows requires 8530.505 and 2209.200, respectively — i.e., a leading digit
appears to be missing from the printed values, not a modeling error (the AICc, Akaike
weight, and all other columns are internally consistent with the larger value in each
case). The growth rate and inflection-time columns for both rows are unaffected. Script:
`validation/reproduce_growth_model_fitting.py`, using the growth-model library in
`validation/growth_model_fitting.py`; input data:
`pipeline/relevance_index_input_data.csv` (aggregated to per-topic annual counts).
Note that one of these two topics has only 8 non-zero years of data for a 3-parameter
curve, so its exact asymptote is not tightly identified — independent refits land on
different large values of A with similarly good AICc; the other topic's asymptote
reproduces to the exact digit, since it has substantially more data.

**Gold-standard size: split target vs. realized size.** The SI states a
gold-standard size of 3,178 labelled records (1,809 relevant / 1,369 not relevant);
`rel_nrel.csv`, used for validation here, has 3,973 records (2,284 relevant / 1,689
not relevant) in total. These are not two different underlying datasets: the
relevance-filter validation calls `GroupShuffleSplit(test_size=0.2, random_state=42)`
on the full 3,973-record file, targeting an 80/20 train/test split — 80% of 3,973 is
3,178.4, matching the SI's number almost exactly. `GroupShuffleSplit` keeps every
preliminary cluster (the grouping unit) entirely on one side of the split, so it can
only approximate the requested ratio; the actual realized split deposited here comes
out to 2,969 train / 1,004 test (74.7%/25.3%) rather than the targeted ≈3,178/795.
The SI's number reflects the split's intended target (from the same procedure, at
the same 0.2 test-size setting), not a mismatch between files. This does not affect
the reported classifier performance, which is evaluated on the realized (deposited)
held-out test set.

**Relevance-filter input consistency.** Section S-3.11 described the global-inference
input as title+abstract, whereas the deployed classifier used the abstract field
alone. The text has been corrected to describe the input actually used. To confirm
this does not affect the reported performance, the group-aware held-out test set
(n=1,004) was evaluated with abstract-only input: accuracy 0.910 (vs. 0.915 with
title+abstract), macro F1 0.894 (vs. 0.901), confusion matrix [[262,30],[60,652]].
The 0.5-percentage-point difference confirms the classifier's validated performance
is representative of the input actually used to build the corpus. Script:
`validation/abstract_vs_title_abstract_inference.py`.
