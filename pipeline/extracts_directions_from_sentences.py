


Resgatar oferta
Traduza os comentários do código abaixo para o inglês. IMPORTANTE: não altere nenhuma linha de código ou nome de variáveis e funções. Altere APENAS os comentários.

# ======================================================================================
# --- 5_directions.py: Extração de Rótulos de Direções de Pesquisa ---
# ======================================================================================

import pandas as pd
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import warnings
import os

import config

warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Definição do Novo Prompt (para Direções de Pesquisa) ---
RESEARCH_DIRECTION_EXTRACTION_PROMPT = """
You are a research analyst focused on identifying future research avenues and knowledge gaps in scientific literature.
Your task is to read the following sentence from a scientific paper and extract the main research direction, knowledge gap, or "next step" it suggests.

Instructions:
1.  Read the sentence carefully.
2.  If the sentence describes a research direction (e.g., using keywords like "future work", "further investigation", "is needed", "remains to be explored", "a promising avenue is..."), summarize it into a short, standardized label of 2-5 words.
3.  **Crucially, try to generalize the label and preferably start with a verb.** For instance, instead of "We need to study why the catalyst degrades", the label should be "Investigate Degradation Mechanisms". Instead of "A next step is making new materials", prefer "Develop Novel Materials".
4.  If the sentence does NOT describe a research direction, you MUST respond with the exact string "No Research Direction".
5.  Your response must ONLY be the research direction label or "No Research Direction", with no additional text.

Sentence to analyze:
"{sentence}"

Research Direction Label:
"""

async def extrair_directions_async():
    """
    Função principal assíncrona para extrair rótulos de direções de pesquisa de frases em um CSV.
    """
    print(f"Iniciando Extração de Direções de Pesquisa com o modelo '{config.LLM_ADVANCED_MODEL}'")

    # --- Definição dos Caminhos de Arquivo ---
    input_csv_path = f"{config.DATA_FOLDER}/df_data_phrases.csv"
    output_parquet_path = f"{config.DATA_FOLDER}/df_data_phrases.parquet"
    
    # --- Inicialização do Cliente vLLM ---
    try:
        client_vllm = AsyncOpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)
        print("Cliente vLLM (assíncrono) conectado.")
    except Exception as e:
        print(f"ERRO: Não foi possível conectar ao cliente vLLM: {e}")
        return

    # --- Leitura e Filtragem dos Dados (CSV) ---
    try:
        df_targets = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERRO: O arquivo de entrada '{input_csv_path}' não foi encontrado.")
        return
    except Exception as e:
        print(f"ERRO: Falha ao ler o arquivo CSV: {e}")
        return

    # Coluna de saída solicitada
    direction_key = "directions_label"

    # O DataFrame principal será processado inteiramente.
    df_to_process = df_targets.copy()
    # Garante que a coluna de destino exista
    if direction_key not in df_targets.columns:
        df_targets[direction_key] = None

    if df_to_process.empty:
        # Esta verificação agora é apenas para o caso de um CSV vazio
        print(f"O arquivo CSV '{input_csv_path}' está vazio.")
        return

    print(f"Iniciando extração de direções para {len(df_to_process)} frases (processando todas as linhas).")

    # --- Loop de Extração Assíncrono ---
    CONCURRENCY_LIMIT = 50  # Ajuste conforme a capacidade do seu servidor

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_row(index, row):
        """ Processa uma única linha do dataframe. """
        async with semaphore:
            # Usamos a coluna 'phrase', conforme solicitado
            prompt = RESEARCH_DIRECTION_EXTRACTION_PROMPT.format(sentence=row.get('phrase', ''))
            try:
                response = await client_vllm.chat.completions.create(
                    model=config.LLM_ADVANCED_MODEL,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                direction_label = response.choices[0].message.content.strip()
            except Exception as e:
                # print(f"\nAVISO: Erro na extração da direção: {e}. Pulando.")
                direction_label = "Erro na Extração"

            # Retorna o índice original do DataFrame e o rótulo extraído
            return index, direction_label

    # Cria e executa todas as tarefas
    tasks = [process_row(index, row) for index, row in df_to_process.iterrows()]
    
    # --- Alteração 2: Coleta todos os resultados antes de salvar ---
    results_list = []
    
    print("Iniciando processamento assíncrono...")
    for f in async_tqdm.as_completed(tasks, total=len(tasks), desc="Extraindo rótulos de direções"):
        # Coleta os resultados (índice e rótulo)
        results_list.append(await f)
        
    # --- Alteração 2: Atualização do DataFrame principal de uma só vez ---
    print("\nProcessamento assíncrono concluído. Atualizando DataFrame na memória...")
    for index, label in results_list:
        df_targets.loc[index, direction_key] = label

    # --- Salvamento Final ---
    print("\nExtração de rótulos de direção concluída!")
    
    # Salva o CSV final (sobrescrevendo)
    try:
        print(f"Salvando resultado final em CSV: '{input_csv_path}'")
        df_targets.to_csv(input_csv_path, index=False)
    except Exception as e:
        print(f"ERRO: Falha ao salvar arquivo CSV final: {e}")

    # Salva o Parquet final
    try:
        print(f"Salvando resultado final em Parquet: '{output_parquet_path}'")
        df_targets.to_parquet(output_parquet_path, index=False)
    except Exception as e:
        print(f"ERRO: Falha ao salvar arquivo Parquet final: {e}")
        if 'pyarrow' not in str(e):
            print("Lembre-se de instalar 'pyarrow' com: pip install pyarrow")

if __name__ == "__main__":
    asyncio.run(extrair_directions_async())
    print("Script 5 finalizado.")

# ======================================================================================
# --- 5_directions.py: Extraction of Research Direction Labels ---
# ======================================================================================

import pandas as pd
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import warnings
import os

import config

warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Definition of the New Prompt (for Research Directions) ---
RESEARCH_DIRECTION_EXTRACTION_PROMPT = """
You are a research analyst focused on identifying future research avenues and knowledge gaps in scientific literature.
Your task is to read the following sentence from a scientific paper and extract the main research direction, knowledge gap, or "next step" it suggests.

Instructions:
1.  Read the sentence carefully.
2.  If the sentence describes a research direction (e.g., using keywords like "future work", "further investigation", "is needed", "remains to be explored", "a promising avenue is..."), summarize it into a short, standardized label of 2-5 words.
3.  **Crucially, try to generalize the label and preferably start with a verb.** For instance, instead of "We need to study why the catalyst degrades", the label should be "Investigate Degradation Mechanisms". Instead of "A next step is making new materials", prefer "Develop Novel Materials".
4.  If the sentence does NOT describe a research direction, you MUST respond with the exact string "No Research Direction".
5.  Your response must ONLY be the research direction label or "No Research Direction", with no additional text.

Sentence to analyze:
"{sentence}"

Research Direction Label:
"""

async def extrair_directions_async():
    """
    Main asynchronous function to extract research direction labels from sentences in a CSV.
    """
    print(f"Starting Research Direction Extraction with model '{config.LLM_ADVANCED_MODEL}'")

    # --- File Path Definitions ---
    input_csv_path = f"{config.DATA_FOLDER}/df_data_phrases.csv"
    output_parquet_path = f"{config.DATA_FOLDER}/df_data_phrases.parquet"
    
    # --- vLLM Client Initialization ---
    try:
        client_vllm = AsyncOpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)
        print("vLLM client (asynchronous) connected.")
    except Exception as e:
        print(f"ERROR: Could not connect to vLLM client: {e}")
        return

    # --- Reading and Filtering Data (CSV) ---
    try:
        df_targets = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_csv_path}' not found.")
        return
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return

    # Requested output column
    direction_key = "directions_label"

    # The main DataFrame will be processed entirely.
    df_to_process = df_targets.copy()
    # Ensures that the target column exists
    if direction_key not in df_targets.columns:
        df_targets[direction_key] = None

    if df_to_process.empty:
        # This check now only applies to an empty CSV
        print(f"The CSV file '{input_csv_path}' is empty.")
        return

    print(f"Starting direction extraction for {len(df_to_process)} sentences (processing all rows).")

    # --- Asynchronous Extraction Loop ---
    CONCURRENCY_LIMIT = 50  # Adjust according to your server capacity

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_row(index, row):
        """ Processes a single row of the dataframe. """
        async with semaphore:
            # We use the 'phrase' column, as requested
            prompt = RESEARCH_DIRECTION_EXTRACTION_PROMPT.format(sentence=row.get('phrase', ''))
            try:
                response = await client_vllm.chat.completions.create(
                    model=config.LLM_ADVANCED_MODEL,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                direction_label = response.choices[0].message.content.strip()
            except Exception as e:
                # print(f"\nWARNING: Error extracting direction: {e}. Skipping.")
                direction_label = "Extraction Error"

            # Returns the original DataFrame index and the extracted label
            return index, direction_label

    # Creates and executes all tasks
    tasks = [process_row(index, row) for index, row in df_to_process.iterrows()]
    
    # --- Change 2: Collect all results before saving ---
    results_list = []
    
    print("Starting asynchronous processing...")
    for f in async_tqdm.as_completed(tasks, total=len(tasks), desc="Extracting direction labels"):
        # Collect results (index and label)
        results_list.append(await f)
        
    # --- Change 2: Update main DataFrame all at once ---
    print("\nAsynchronous processing completed. Updating DataFrame in memory...")
    for index, label in results_list:
        df_targets.loc[index, direction_key] = label

    # --- Final Saving ---
    print("\nDirection label extraction completed!")
    
    # Save final CSV (overwriting)
    try:
        print(f"Saving final result to CSV: '{input_csv_path}'")
        df_targets.to_csv(input_csv_path, index=False)
    except Exception as e:
        print(f"ERROR: Failed to save final CSV file: {e}")

    # Save final Parquet
    try:
        print(f"Saving final result to Parquet: '{output_parquet_path}'")
        df_targets.to_parquet(output_parquet_path, index=False)
    except Exception as e:
        print(f"ERROR: Failed to save final Parquet file: {e}")
        if 'pyarrow' not in str(e):
            print("Remember to install 'pyarrow' with: pip install pyarrow")

if __name__ == "__main__":
    asyncio.run(extrair_directions_async())
    print("Script 5 finished.")



