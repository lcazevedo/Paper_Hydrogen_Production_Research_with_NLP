"""
Reproduces Table 1's topic sizes directly from the original saved BERTopic/EVoC
model (`pipeline/model_pred/`, recovered from the corresponding author -- the file
the deposited `clustering_EVOC.ipynb` comments describe saving via
`model.save(f'{project_path}model_pred', ...)` but that was not originally deposited).

Also closes a discrepancy raised in the audit: rerunning EVoC from scratch on
embs_pred.pkl gives a 20-topic layer closest to the paper's target of 25 (see
2026_09_25_evoc_clustering_sensitivity_analysis.py); this script instead loads the
ORIGINAL fitted model directly, removing any need to match library version or seed.

No non-deposited files are required: `topic_assignment_original_model.csv`
(pipeline/, doi + unique_id + topic_id + topic_name, no title/abstract text) already
carries the per-document topic assignment for all 32,223 documents predicted
relevant by the abstract-only SetFit inference.
"""
import json
from pathlib import Path

import pandas as pd

MODEL_DIR = Path(__file__).parent.parent / "pipeline" / "model_pred"
ASSIGNMENT_CSV = Path(__file__).parent.parent / "pipeline" / "topic_assignment_original_model.csv"

with open(MODEL_DIR / "topics.json") as f:
    model_topics = json.load(f)
print(f"Original model: {len(model_topics['topics'])} documents, "
      f"{len(model_topics['topic_sizes'])} topics (target granularity: 25)")

assignment = pd.read_csv(ASSIGNMENT_CSV)
print(f"\nTopic sizes (all 32,223 abstract-only-relevant documents):")
print(assignment["topic_name_from_model"].value_counts().to_string())

print(f"\nCompare to Table 1 / final_corpus_data.csv (32,114 documents after metadata "
      f"filtering): sizes above are consistently ~0.1-0.3% larger, matching the 109 "
      f"records excluded for missing title/abstract metadata.")
