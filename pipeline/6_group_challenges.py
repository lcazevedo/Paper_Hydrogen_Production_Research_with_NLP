# ======================================================================================
# --- 6_group_challenges.py: Agrupa os 'challenge_label' em uma nova categoria ---
# --- usando um arquivo CSV de mapeamento.                                      ---
# ======================================================================================
# Deterministic label -> macro-category merge for the challenge-extraction pipeline
# (referenced by charts_and_tables_for_section_4_14.ipynb, cell 5; previously missing
# from this repository -- see audit_moraes_NLPH2Mat_2026-09-25.md, Issue #12).
# Consumes "ext_<model>_challenge_label" (short LLM-extracted phrase per sentence,
# already in ChromaDB) and a curated label->group mapping CSV, and writes
# "ext_<model>_challenge_group" back to the same collection. Any label not found in
# the mapping is set to "Uncategorized".

import pandas as pd
import chromadb
from tqdm import tqdm

import config

# --- NOME DO ARQUIVO DE MAPEAMENTO ---
MAPPING_CSV_FILE = f'{config.DATA_FOLDER}/challenges_classified_v1.csv'
GROUP_COLUMN_NAME = f'ext_{config.LLM_ADVANCED_MODEL_SAFE_NAME}_challenge_group' # Nome da nova coluna a ser criada no DB

def load_challenge_map(csv_file):
    """Carrega o arquivo CSV e o transforma em um dicionário de mapeamento."""
    try:
        df_map = pd.read_csv(csv_file)
        challenge_col, cluster_col = df_map.columns[0], df_map.columns[2]
        df_map[challenge_col] = df_map[challenge_col].str.lower().str.strip()
        challenge_map = pd.Series(df_map[cluster_col].values, index=df_map[challenge_col]).to_dict()
        print(f"Mapeamento carregado com sucesso de '{csv_file}'. {len(challenge_map)} regras encontradas.")
        return challenge_map
    except FileNotFoundError:
        print(f"ERRO: O arquivo de mapeamento '{csv_file}' não foi encontrado.")
        return None
    except Exception as e:
        print(f"ERRO ao ler o arquivo de mapeamento: {e}")
        return None

def verify_mapping(collection, challenge_map):
    """
    Compara os desafios do banco de dados com as regras do arquivo de mapeamento
    para encontrar discrepâncias.
    """
    print("\n--- Verificando a cobertura do mapeamento de desafios ---")

    challenge_label_column = f"ext_{config.LLM_ADVANCED_MODEL_SAFE_NAME}_challenge_label"

    all_metadata = collection.get(include=["metadatas"])['metadatas']
    df = pd.DataFrame(all_metadata)

    if challenge_label_column not in df.columns:
        print(f"AVISO: A coluna de desafios '{challenge_label_column}' não existe no banco de dados. Verificação pulada.")
        return

    db_challenges = set(df[challenge_label_column].dropna().str.lower().str.strip())
    db_challenges.discard("no challenge")
    db_challenges.discard("nenhum desafio")

    csv_challenges = set(challenge_map.keys())

    in_db_not_in_csv = db_challenges - csv_challenges
    in_csv_not_in_db = csv_challenges - db_challenges

    if not in_db_not_in_csv and not in_csv_not_in_db:
        print("✅ Verificação completa: Todos os desafios do banco de dados estão no arquivo de mapeamento e todas as regras são utilizadas.")
        return

    if in_csv_not_in_db:
        print(f"\nINFO: {len(in_csv_not_in_db)} regras no seu arquivo CSV NUNCA foram utilizadas.")
        print("Isso geralmente indica pequenas diferenças de texto (ex: 'low stability' vs 'poor stability').")
        print("Compare estas regras não utilizadas com a lista de desafios 'faltantes' acima para encontrar as discrepâncias:")
        in_csv_not_in_db.to_csv(f"{config.DATA_FOLDER}/unused_challenge_rules.csv", index=False, header=["challenge"])
        for challenge in sorted(list(in_csv_not_in_db)):
            print(f" - '{challenge}'")
    else:
        print("\nINFO: todas as regras no seu arquivo CSV foram utilizadas.")
    print("\n")


def group_challenges(collection, challenge_map):
    """Aplica o mapeamento de desafios aos metadados da coleção."""
    print("\nIniciando o processo de agrupamento...")

    all_entries = collection.get(include=["metadatas"])
    challenge_label_column = f"ext_{config.LLM_ADVANCED_MODEL_SAFE_NAME}_challenge_label"

    to_process = []
    for i, metadata in enumerate(all_entries['metadatas']):
        to_process.append({ 'id': all_entries['ids'][i], 'metadata': metadata })

    if not to_process:
        print("Nenhuma frase nova para agrupar. O trabalho parece já estar concluído.")
        return

    print(f"Encontradas {len(to_process)} frases para agrupar.")

    BATCH_SIZE_UPDATE = 500
    ids_to_update, metadatas_to_update = [], []
    missing_challenges = set()

    for entry in tqdm(to_process, desc="Agrupando desafios"):
        original_metadata = entry['metadata']
        original_challenge = original_metadata.get(challenge_label_column)

        new_metadata = original_metadata.copy()
        if "challenge_group" in new_metadata:
            new_metadata.pop("challenge_group")

        if original_challenge and isinstance(original_challenge, str):
            normalized_challenge = original_challenge.lower().strip()
            grouped_value = challenge_map.get(normalized_challenge)

            if grouped_value:
                new_metadata[GROUP_COLUMN_NAME] = grouped_value
            else:
                new_metadata[GROUP_COLUMN_NAME] = 'Uncategorized'
                missing_challenges.add(original_challenge)
        else:
            new_metadata[GROUP_COLUMN_NAME] = None

        ids_to_update.append(entry['id'])
        metadatas_to_update.append(new_metadata)

        if len(ids_to_update) >= BATCH_SIZE_UPDATE:
            collection.update(ids=ids_to_update, metadatas=metadatas_to_update)
            ids_to_update, metadatas_to_update = [], []

    if ids_to_update:
        collection.update(ids=ids_to_update, metadatas=metadatas_to_update)

    print("\nAgrupamento de desafios concluído!")

    if missing_challenges:
        print("\nAVISO: Os seguintes desafios não foram encontrados no arquivo de mapeamento e foram marcados como 'Uncategorized':")
        for challenge in sorted(list(missing_challenges)):
            print(f" - {challenge}")


def export_collection_to_csv(collection, output_filename):
    """
    Busca todos os dados de uma coleção do ChromaDB e os salva em um arquivo CSV.
    """
    print(f"\nIniciando a exportação de todos os dados para o arquivo '{output_filename}'...")
    try:
        count = collection.count()
        if count == 0:
            print("A coleção está vazia. Nenhum dado para exportar.")
            return

        results = collection.get(limit=count, include=["metadatas"])

        if not results or not results['metadatas']:
            print("Não foi possível recuperar dados da coleção.")
            return

        df_export = pd.DataFrame(results['metadatas'])
        df_export.to_csv(output_filename, index=False, encoding='utf-8-sig')

        print(f"✅ Exportação concluída com sucesso. {len(df_export)} registros foram salvos.")

    except Exception as e:
        print(f"❌ Ocorreu um erro durante a exportação para CSV: {e}")


if __name__ == "__main__":
    challenge_map = load_challenge_map(MAPPING_CSV_FILE)

    if challenge_map:
        try:
            client_db = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
            print(f"Conectado ao ChromaDB em '{config.CHROMA_DB_PATH}'.")
            collections = client_db.list_collections()
            for collection in collections:
                print(collection.name)
            print()
            collection = client_db.get_collection(name=config.COLLECTION_NAME)

            verify_mapping(collection, challenge_map)
            group_challenges(collection, challenge_map)

            output_csv_name = f"{config.DATA_FOLDER}/{config.COLLECTION_NAME}_export_chromadb.csv"
            export_collection_to_csv(collection, output_csv_name)

        except Exception as e:
            print(f"Ocorreu um erro no processo principal: {e}")

    print("\nScript de agrupamento e exportação finalizado.")
