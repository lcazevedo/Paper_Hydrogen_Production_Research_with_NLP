import pandas as pd

# Load the audited spreadsheet
df = pd.read_excel("audit_100_documents.xlsx")

# Identify the columns (assuming C is the prediction and D is the evaluation)
col_pred = df.columns[2]
col_true = df.columns[3]

# Clean and standardize the text to avoid typos (spaces, capitalization)
df['pred_clean'] = df[col_pred].astype(str).str.lower().str.strip()
df['true_clean'] = df[col_true].astype(str).str.lower().str.strip()

# Map the model predictions to the terminology used by the expert
map_pred = {
    'experimental studies': 'experimental',
    'computational and simulation studies': 'computacional'
}
df['pred_mapped'] = df['pred_clean'].map(map_pred)

# Isolate the errors
erros = df[df['pred_mapped'] != df['true_clean']]
taxa_erro = (len(erros) / len(df)) * 100

print(f"Total studies evaluated: {len(df)}")
print(f"Overall error rate: {taxa_erro:.1f}%\n")

print("--- Analysis of Major Failure Modes ---")
falhas_hibridas = erros[erros['true_clean'] == 'híbrida']
print(f"1. Hybrid studies incorrectly classified: {len(falhas_hibridas)}")
print(f"   - Classified as Experimental: {len(falhas_hibridas[falhas_hibridas['pred_mapped'] == 'experimental'])}")
print(f"   - Classified as Computational: {len(falhas_hibridas[falhas_hibridas['pred_mapped'] == 'computacional'])}\n")

falhas_exp = erros[erros['true_clean'] == 'experimental']
print(f"2. Purely Experimental studies incorrectly classified as Computational: {len(falhas_exp)}")

falhas_comp = erros[erros['true_clean'] == 'computacional']
print(f"3. Purely Computational studies incorrectly classified as Experimental: {len(falhas_comp)}")