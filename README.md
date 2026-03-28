# Hydrogen Production Research with NLP

This repository contains the codes and supplementary information for the scientific article **"Mapping Electrochemical Hydrogen Production Research with NLP: Automated Relevance Filtering, Transformer-Based Topic Modeling, and Scientometric Trends"**. The study uses Natural Language Processing (NLP) techniques to analyze the global research landscape on hydrogen production.

## ⚠️ Data Availability Notice

The original data used in this research were obtained from the **Web of Science (Clarivate)** database. Due to licensing restrictions and Clarivate's *Terms of Business*, the raw data (full metadata, abstracts, etc.) cannot be publicly redistributed in this repository.

However, to ensure full **transparency and reproducibility** of our study, we provide a list containing all the **DOIs (Digital Object Identifiers)** of the analyzed documents. 

Researchers with institutional access to the Web of Science can use the provided DOI list to search and download the original dataset themselves, allowing the execution of the codes provided here.

## 📂 Repository Content

* **List of DOIs of the articles included in the study:**
  * `final_corpus_IDs.csv`
  * `initial_corpus_IDs.csv`

* **Source codes for data preprocessing:**
  * `classify_relevant_setfit.ipynb`
  * `clustering_EVOC.ipynb`
  * `clustering_EVOC_translated.ipynb`
  * `create_classes_for_research_controversies_transformers_part_1.py`
  * `create_classes_for_research_controversies_transformers_part_2.py`
  * `create_classes_for_research_controversies_transformers_part_3.py`
  * `create_classes_for_research_directions sentence_transformers_part_1.py`
  * `create_classes_for_research_directions sentence_transformers_part_2.py`
  * `create_classes_for_research_directions sentence_transformers_part_3.py`
  * `create_columns_JNIF_score_log_score_sqrt.py`
  * `extract_controversy_from_sentences.py`
  * `extracts_directions_from_sentences.py`

* **Source codes for results generation:**
  * `charts_and_tables_for_section_4_14.ipynb`
  * `charts_and_tables_general.ipynb`

## 📄 License and Usage

The custom codes and scripts developed for this project are licensed under the [MIT License](LICENSE). Feel free to use and adapt them, provided the original article is properly cited.

**Article Citation:**
> Azevedo, L. C., et al. (Year). *Mapping Electrochemical Hydrogen Production Research with NLP: Automated Relevance Filtering, Transformer-Based Topic Modeling, and Scientometric Trends*. Journal Name. DOI: [Link to the DOI of your paper, when published]
