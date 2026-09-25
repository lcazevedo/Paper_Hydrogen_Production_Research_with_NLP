# Hydrogen Production Research with NLP

This repository contains the code and supplementary data for the scientific article **"Mapping Electrochemical Hydrogen Production Research with NLP: Automated Relevance Filtering, Transformer-Based Topic Modeling, and Scientometric Trends"**. The study uses Natural Language Processing (NLP) techniques to analyze the global research landscape on hydrogen production.

For a component-by-component account of what is independently reproducible from this repository, see **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)**.

## ⚠️ Data Availability Notice

The original data used in this research were obtained from the **Web of Science (Clarivate)** database. Due to licensing restrictions and Clarivate's *Terms of Business*, raw title/abstract text cannot be publicly redistributed in this repository.

To ensure full **transparency and reproducibility**, we instead provide:
- the **DOIs** of every document at each stage of corpus construction, and
- for every intermediate artifact that would otherwise carry raw text, a text-free derivative (DOI/ID + numeric or categorical fields only — see `validation/`).

Researchers with institutional access to the Web of Science can use the DOI lists to retrieve the original text and run the code in this repository end to end. Files that still contain raw title/abstract text (e.g., the SetFit gold-standard labels, the preliminary-cluster map) are available from the corresponding author on request rather than deposited here.

## 📂 Repository Content

```
data/         Corpus identifiers and the final labelled corpus
pipeline/     Code that builds the corpus, topics, and challenge/direction/controversy labels
validation/   Every reproducibility check, audit sample, and validation script
analysis/     Notebooks that generate the manuscript's figures and tables
```

### `data/` — corpus definition
* `final_corpus_IDs.csv` — DOIs of the 32,114 documents in the final corpus
* `initial_corpus_IDs.csv` — DOIs of the 40,467 documents retrieved before relevance filtering
* `seed_papers_extracted_v2.csv` — the 476-document seed set used to build the search query
* `final_corpus_data.csv.zip` — per-document topic and challenge/direction/controversy group assignments

### `pipeline/` — corpus and label construction
* `classify_relevant_setfit.ipynb` — SetFit relevance filter (see `validation/` for the group-aware validation of this step)
* `clustering_EVOC.ipynb` — EVōC/BERTopic topic modeling
* `extract_controversy_from_sentences.py`, `extracts_directions_from_sentences.py` — zero-shot LLM sentence-level extraction
* `create_classes_for_research_controversies_transformers_part_1/2/3.py`, `create_classes_for_research_directions sentence_transformers_part_1/2/3.py` — embedding + clustering + LLM naming of extracted labels into macro-categories
* `6_group_challenges.py` — deterministic label→category merge for the challenge pipeline
* `create_columns_JNIF_score_log_score_sqrt.py` — Relevance Index components

### `validation/` — reproducibility checks and audit samples
* `2026_09_25_challenge_denominator_funnel.py` — reproduces the sentence-level denominator (categorized vs. "Uncategorized") behind Figure 6
* `amostra_500_audited.xlsx`, `2026_09_25_audit_500_accuracy_F1_score_confusion_matrix.py` — 500-sentence manual audit of the challenge classification (accuracy/F1/confusion matrix)
* `2026_09_25_research_scope_category_validation.py` — category-specific precision/recall for "Research Scope, Novelty, and Generalization"
* `2026_09_25_setfit_group_aware_validation.py`, `setfit_group_split_manifest.csv` — topic-held-out `GroupShuffleSplit` validation of the SetFit filter (manifest is text-free; the script needs the non-deposited gold-standard text to train)
* `2026_09_25_evoc_clustering_sensitivity_analysis.py` — topic-model stability under `noise_level` and granularity perturbations (needs the non-deposited precomputed embeddings)
* `challenges_classified_v3_optimized.xlsx`, `controversies_label_classified_sample500.csv`, `directions_label_classified_sample500.csv` — label→category mapping samples used by `pipeline/`
* `2026_09_25_challenge_stage2_classifier_reconstruction.py` — a documented **reconstruction** (not a recovery of the original code/weights — see `REPRODUCIBILITY.md`) of the supervised classifier that scales challenge classification beyond the curated label dictionary
* `audit_100_documents_labels_only.csv`, `2026_09_25_audit_100_documents_study_type.py` — 100-document experimental/computational study-type audit (labels only; the original spreadsheet carried raw WoS text and is not deposited)

### `analysis/` — figures and tables
* `charts_and_tables_for_section_4_14.ipynb`, `charts_and_tables_general.ipynb`

## 📄 License and Usage

The custom code developed for this project is intended to be released under an open-source license (LICENSE file pending — check back or contact the corresponding author). Feel free to use and adapt it, provided the original article is properly cited.

**Article Citation:**
> Azevedo, L. C., et al. (Year). *Mapping Electrochemical Hydrogen Production Research with NLP: Automated Relevance Filtering, Transformer-Based Topic Modeling, and Scientometric Trends*. Journal Name. DOI: [Link to the DOI of your paper, when published]
