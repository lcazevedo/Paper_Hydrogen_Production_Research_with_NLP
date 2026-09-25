import pandas as pd
from datasets import Dataset, DatasetDict
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

project_path = "/Statistical Study/"
train_file = f"{project_path}rel_nrel.csv"
excel_file = f"{project_path}papers40k_cluster1_v2.xlsx"

# Variables
label_column = "label"
title_column = "title"
abstract_column = "abstract"
base_model = "all-MiniLM-L6-v2"
saved_model = f"{project_path}rel-nrel-100perc-all-MiniLM-L6-v2-GroupSplit"

# ---------------------------------------------------------
# STEP 1: Read the Excel file and map DOI -> Cluster
# ---------------------------------------------------------
print("Reading the Excel file and mapping clusters...")
xls = pd.ExcelFile(excel_file)
cluster_data = []

for sheet_name in xls.sheet_names:
    # Read the current sheet
    df_sheet = pd.read_excel(xls, sheet_name=sheet_name)
    
    # Check whether the 'doi' column exists in the sheet
    if 'doi' in df_sheet.columns:
        temp_df = df_sheet[['doi']].copy()
        # The sheet name is the group identifier itself (e.g., '0_cluster_name')
        temp_df['cluster_id'] = sheet_name 
        cluster_data.append(temp_df)

# Combine all data extracted from the sheets into a single DataFrame
df_clusters = pd.concat(cluster_data, ignore_index=True)
# Remove duplicates if a DOI accidentally appears in more than one sheet
df_clusters.drop_duplicates(subset=['doi'], inplace=True) 

# ---------------------------------------------------------
# STEP 2: Load the CSV and cross-reference with the Clusters
# ---------------------------------------------------------
print("Loading the CSV and cross-referencing the information...")
df = pd.read_csv(train_file)
df[abstract_column] = df[abstract_column].fillna("")
df[title_column] = df[title_column].fillna("")
df['label'] = df[label_column]
df['text'] = df[title_column] + '.\n' + df[abstract_column]

# Merge using the DOI
df = df.merge(df_clusters, on='doi', how='inner')


print(f"Total documents after cross-referencing: {len(df)}")

# ---------------------------------------------------------
# STEP 3: Group-based Split (Clusters)
# ---------------------------------------------------------
# GroupShuffleSplit ensures that the same 'cluster_id' values do not leak into the test set
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# Generate the training and test indices
train_idx, test_idx = next(gss.split(df, groups=df['cluster_id']))

df_train = df.iloc[train_idx]
df_test = df.iloc[test_idx]

ds = DatasetDict()
ds['train'] = Dataset.from_pandas(df_train)
ds['eval'] = Dataset.from_pandas(df_test)

print(f"Relevant training samples: {len(df_train[df_train['label'] == 'relevant'])}")
print(f"Non-relevant training samples: {len(df_train[df_train['label'] != 'relevant'])}")
print(f"Relevant test samples: {len(df_test[df_test['label'] == 'relevant'])}")
print(f"Non-relevant test samples: {len(df_test[df_test['label'] != 'relevant'])}")

# ---------------------------------------------------------
# STEP 4: Training and Evaluation
# ---------------------------------------------------------
print("Starting SetFit training...")
model = SetFitModel.from_pretrained(base_model)

args = TrainingArguments(
    batch_size=32,
    num_iterations=5,
    num_epochs=10,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds['train'],
    eval_dataset=ds["eval"]
)

trainer.train()

# Evaluate the model without data leakage
print("Evaluating the model...")
eval_texts = ds["eval"]["text"]  #.tolist()
true_labels = ds["eval"]["label"]  #.tolist()

pred_labels = model.predict(eval_texts)

accuracy = accuracy_score(true_labels, pred_labels)
print(f"\nAccuracy (Group Split): {accuracy:.2f}")

print("\nClassification Report (Group Split):")
print(classification_report(true_labels, pred_labels))

print("\nConfusion Matrix (Group Split):")
print(confusion_matrix(true_labels, pred_labels))

# Save the new model
model.save_pretrained(saved_model)


# ---------------------------------------------------------
# STEP 5: Generate an Independent Sample for Experts
# ---------------------------------------------------------
print("\nGenerating a sample of 200 independent documents for the experts...")

# Load the complete dataset (adjust the filename for your raw 40k dataset)
# Based on your previous scripts, the file should be something like:
full_data_file = f"{project_path}bibfile_data_40k.pkl" 
full_df = pd.read_pickle(full_data_file)

# Fill missing values and create the text column
full_df['title'] = full_df['title'].fillna("")
full_df['abstract'] = full_df['abstract'].fillna("")
full_df['text'] = full_df['title'] + '.\n' + full_df['abstract']

# 1. Remove all documents that were used in training/testing (Data Leakage prevention)
used_dois = df['doi'].unique()
independent_df = full_df[~full_df['doi'].isin(used_dois)].copy()

# 2. Randomly sample 200 documents (fixed seed for reproducibility)
sample_df = independent_df.sample(n=200, random_state=42).copy()

# 3. Make predictions using the SetFit model that was just trained
# Use .tolist() to ensure compatibility with the model input
sample_df['SetFit_Prediction'] = model.predict(sample_df['text'].tolist())

# 4. Create empty columns for the experts to fill in
sample_df['Expert_1_Label'] = ""
sample_df['Expert_2_Label'] = ""

# 5. Organize and export the spreadsheet
export_cols = ['doi', 'title', 'abstract', 'SetFit_Prediction', 'Expert_1_Label', 'Expert_2_Label']
export_df = sample_df[export_cols]

export_path = f"{project_path}independent_audit_200_experts.xlsx"
export_df.to_excel(export_path, index=False)

print(f"Expert spreadsheet successfully generated at: {export_path}")