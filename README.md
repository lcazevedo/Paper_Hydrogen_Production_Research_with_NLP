# Hydrogen Production Research with NLP

This repository contains the codes and supplementary information for the scientific article **"[Insert your Article Title Here]"**. The study uses Natural Language Processing (NLP) techniques to analyze the global research landscape on hydrogen production.

## ⚠️ Data Availability Notice

The original data used in this research were obtained from the **Web of Science (Clarivate)** database. Due to licensing restrictions and Clarivate's *Terms of Business*, the raw data (full metadata, abstracts, etc.) cannot be publicly redistributed in this repository.

However, to ensure full **transparency and reproducibility** of our study, we provide a list containing all the **DOIs (Digital Object Identifiers)** of the analyzed documents. 

Researchers with institutional access to the Web of Science can use the provided DOI list to search and download the original dataset themselves, allowing the execution of the codes provided here.

## 📂 Repository Structure

* `data/`: Contains the list of DOIs of the articles included in the study (e.g., `doi_list.csv`).
* `src/`: Folder containing the source codes developed for the study.
  * `01_preprocessing.py` (or .ipynb): Codes used to clean and preprocess the raw data.
  * `02_nlp_analysis.py`: Codes for text modeling and analysis.
  * `03_generate_reports.py`: Codes to generate the charts, tables, and final reports presented in the article.
* `requirements.txt`: List of libraries and dependencies required to run the Python code.

## 🚀 How to Reproduce the Research

1. **Data Acquisition:** * Download the file with the DOI list located in the `data/` folder. 
   * Access the Web of Science and use the "Advanced Search" tool to search for the records using the identifiers (example query: `DO=(doi1 OR doi2 OR ...)`). 
   * Export the results in the format used by the scripts and save them in the `data/raw/` folder (which you must create locally).
2. **Environment Setup:** * Clone this repository to your local machine.
   * Install the dependencies by running:
     ```bash
     pip install -r requirements.txt
     ```
3. **Execution:** * Run the scripts sequentially located in the `src/` folder to replicate the preprocessing, NLP analysis, and report generation.

## 📄 License and Usage

The custom codes and scripts developed for this project are licensed under the [MIT License](LICENSE) (or insert your preferred license, such as Apache 2.0). Feel free to use and adapt them, provided the original article is properly cited.

**Article Citation:**
> Azevedo, L. C., et al. (Year). *Article Title*. Journal Name. DOI: [Link to the DOI of your paper, when published]
