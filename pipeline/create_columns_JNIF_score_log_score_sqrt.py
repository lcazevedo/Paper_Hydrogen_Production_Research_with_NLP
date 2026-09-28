import pandas as pd
from tqdm import tqdm
import numpy as np

tqdm.pandas() 

# Attempts to import the settings from the config.py file
try:
    import config
except ImportError:
    print("ERRO: O arquivo de configuração 'config.py' não foi encontrado.")
    print("Certifique-se de que o arquivo está no mesmo diretório.")
    exit()

df_data = pd.read_parquet(config.DF_DATA_FILE_PARQUET)



# 1. Calculate the thresholds (Medians)
# We use medians because they are robust to outliers (Power Law distribution)
citation_threshold = df_data['citations_per_year'].median()  #.quantile(.75) #
if_threshold = df_data['journal_if'].median()   #.quantile(.75) # 

# 2. Define a function to categorize each row
def get_quadrant(row):
    high_cit = row['citations_per_year'] >= citation_threshold
    high_if = row['journal_if'] >= if_threshold
   
    if high_if and high_cit:
        return "Star (High IF, High Cit)"
    elif not high_if and high_cit:
        return "Hidden Gem (Low IF, High Cit)"
    elif high_if and not high_cit:
        return "Coattail (High IF, Low Cit)"
    else:
        return "Long Tail (Low IF, Low Cit)"

# 3. Apply the function
df_data['category_group'] = df_data.apply(get_quadrant, axis=1)

# Optional: View the counts to see how balanced your groups are
print("\n\n category_group")
print(df_data['category_group'].value_counts())

#######################################
# creates column with JNIF = citations of paper / expected citations (JIF)
df_data['JNIF'] = df_data.apply(
    lambda row: row['citations_tot'] / row['journal_if'] 
    if row['journal_if'] > 0 else None,
    axis=1
)


#######################################
# creates column with score_log = w1*log(citations+1) + w2*log(jif+1).
w1 = 0.7  # weight of citations
w2 = 0.3  # weight of JIF
df_data['score_log'] = (
    w1 * np.log(df_data['citations_tot'] + 1) +
    w2 * np.log(df_data['journal_if'] + 1)
)


#######################################
# creates column with score_sqrt = sqrt(citations * JIF)
df_data['score_sqrt'] = np.sqrt(df_data['citations_tot'].fillna(0) * df_data['journal_if'].fillna(0))




# === 7. (Optional) save the final result ===
df_data.to_parquet(config.DF_DATA_FILE_PARQUET, index=False)
df_data.to_csv(config.DF_DATA_FILE_PARQUET.replace(".parquet", ".csv"), index=False)

print("✅ Colunas adicionadas com sucesso!")
