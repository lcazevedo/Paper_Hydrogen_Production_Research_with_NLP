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
in the relevance classifier (documented below). One small, unresolved numeric
discrepancy is listed below rather than left for a reader to find.

## Quick verification

These reproduce a published number exactly, from data already in this repository,
with no non-deposited inputs needed. From the repository root:

```bash
cd validation
python3 2026_09_25_challenge_denominator_funnel.py
# -> Categorized (Figure 6 denominator): 206,533 (92.54% of total); top-3 share 53.25% (-> 53.3%)

python3 2026_09_25_audit_500_accuracy_F1_score_confusion_matrix.py
# -> Overall Accuracy: 0.934, Macro-averaged F1 Score: 0.907  (matches SI Table S-7)

python3 2026_09_25_research_scope_category_validation.py
# -> Precision: 0.967, Recall: 1.000  (matches SI: 0.97 / 1.00)

python3 2026_09_25_audit_100_documents_study_type.py
# -> Overall error rate: 27.0%; hybrid studies misclassified: 26 (23 as Experimental)

python3 2026_09_25_reproduce_table1_from_original_model.py
# -> reproduces Table 1's topic sizes from the original BERTopic/EVoC model
```

The following also reproduce exactly but need the deposited checkpoint
(`rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip`, unzips automatically) plus two
raw-text files available on request from the corresponding author
(`rel_nrel.csv`, `papers40k_cluster1_v2.xlsx` — not deposited, see below):

```bash
python3 2026_09_25_setfit_group_aware_validation.py
# -> train: 2,969 | test: 1,004 (712 relevant / 292 not relevant)  (matches SI Table S-4)

python3 2026_09_25_abstract_vs_title_abstract_inference.py
# -> title+abstract: accuracy 0.9153, [[270,22],[63,649]]  (matches SI Table S-4 exactly)
# -> abstract-only:  accuracy 0.9104, [[262,30],[60,652]]  (0.5-pp difference)
```

## Known open discrepancy

**Gold-standard size.** The SI states 3,178 labelled records (1,809 relevant / 1,369
not relevant). The `rel_nrel.csv` file used for validation here has 3,973 records
(2,284 relevant / 1,689 not relevant). The `GroupShuffleSplit` on the 3,973-record
file reproduces the published test-set composition exactly (n=1,004; 712/292), so
this looks like a stale count in the SI text rather than a different underlying file
— not yet reconciled with the corresponding author.

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
| SetFit relevance-filter `GroupShuffleSplit` validation | **Fully reproduced.** Split reproduces exactly (train 2,969 / test 1,004; 712 relevant, 292 not relevant). The deposited checkpoint reproduces the published accuracy (0.9153→0.92), F1 (0.94/0.86), and confusion matrix ([[270,22],[63,649]]) exactly. | `validation/2026_09_25_setfit_group_aware_validation.py`, `validation/setfit_group_split_manifest.csv`, `validation/rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip` |
| Topic-model sizes and stability | **Fully reproduced.** The original saved topic model reproduces Table 1 almost exactly (~0.1-0.3% differences, matching the 109 metadata-excluded documents). | `pipeline/model_pred/`, `pipeline/topic_assignment_original_model.csv`, `validation/2026_09_25_reproduce_table1_from_original_model.py`; independent re-fit sensitivity check in `validation/2026_09_25_evoc_clustering_sensitivity_analysis.py` |
| Challenge-classification pipeline: grouping step + 500-sentence audit + supervised classifier | Grouping step (`6_group_challenges.py`) recovered and deposited (deterministic label→group merge). The supervised classifier's script/weights were **not recoverable**; a documented *reconstruction* with an equivalent architecture is deposited instead, clearly marked as such. The 500-sentence audit fully reproduces. | `pipeline/6_group_challenges.py`, `validation/2026_09_25_challenge_stage2_classifier_reconstruction.py`, `validation/2026_09_25_audit_500_accuracy_F1_score_confusion_matrix.py`, `validation/amostra_500_audited.xlsx` |
| Challenge-category denominator | **Fully reproduced** from the already-public `final_corpus_data.csv`. | `validation/2026_09_25_challenge_denominator_funnel.py` |
| "Research Scope" category sample and precision/recall | **Fully reproduced**, no new data needed. | `validation/2026_09_25_research_scope_category_validation.py` |
| Experimental/computational study-type audit (100 documents) | **Fully reproduced**: 27.0% overall error rate; 26 of the errors are hybrid experimental/computational studies (23 misclassified as Experimental). | `validation/2026_09_25_audit_100_documents_study_type.py`, `validation/audit_100_documents_labels_only.csv` |
| Seed-set recall | **Reproduced**, closely matches independent recomputation (473 distinct seeds; Experimental 257/344 retrieved = 74.7%, exact match; Theoretical 71/129 retrieved = 55.0% vs. 56.6%, small discrepancy likely from DOI-matching edge cases). | `validation/2026_09_25_seed_recall_analysis.py` |
| Relevance-filter input consistency (training vs. deployment) | **Resolved.** The SI described the wrong input for global inference; corrected to describe the input actually deployed, with a validation check confirming equivalent performance (0.9153 vs. 0.9104 accuracy on the same held-out test set). | `validation/2026_09_25_abstract_vs_title_abstract_inference.py` |

## Notes on specific artifacts

**SetFit checkpoint.** `rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit.zip` is trained
on the `GroupShuffleSplit` train partition only (2,969 records; its model card
confirms 1,397 not_relevant + 1,572 relevant), so it was never exposed to the
1,004-record held-out test set used above — a fair evaluation. Retraining from
scratch was not attempted here: on Apple-Silicon (MPS) the described configuration
(9,280 steps) runs at roughly 450-550 s/step, impractical on that hardware.

**Challenge classifier.** `validation/2026_09_25_challenge_stage2_classifier_reconstruction.py`
is an explicitly-labelled reconstruction, not the original code: TF-IDF + Logistic
Regression (hyperparameters chosen by 5-fold `GridSearchCV`) trained on the deposited
label→category dictionary (`challenges_classified_v3_optimized.xlsx`). 5-fold CV
accuracy 0.922 (unweighted per unique label) / 0.969 (weighted by frequency). This is
documented explicitly as a reconstruction, not recovered code.

**Relevance-filter input consistency.** Section S-3.11 described the global-inference
input as title+abstract, whereas the deployed classifier used the abstract field
alone. The text has been corrected to describe the input actually used. To confirm
this does not affect the reported performance, the group-aware held-out test set
(n=1,004) was evaluated with abstract-only input: accuracy 0.910 (vs. 0.915 with
title+abstract), macro F1 0.894 (vs. 0.901), confusion matrix [[262,30],[60,652]].
The 0.5-percentage-point difference confirms the classifier's validated performance
is representative of the input actually used to build the corpus. Script:
`validation/2026_09_25_abstract_vs_title_abstract_inference.py`.
