# ======================================================================================
# --- 6_controversies.py: Controversy Label Extraction ---
# ======================================================================================

import pandas as pd
from openai import AsyncOpenAI
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import warnings
import os

import config

warnings.simplefilter(action='ignore', category=FutureWarning)


# --- Definition of the New Prompt (for Controversies) ---
CONTROVERSY_EXTRACTION_PROMPT = """
You are a research analyst focused on identifying debates and disagreements in scientific literature.
Your task is to read the following sentence from a scientific paper and extract the main controversy, debate, or lack of consensus it describes.

Instructions:
1.  Read the sentence carefully.
2.  If the sentence describes a controversy (e.g., using keywords like "debatable", "controversial", "unclear", "disagreement", or contrasting two opposing viewpoints), summarize it into a short, standardized label of 2-5 words.
3.  **Crucially, try to generalize the label.** For instance, instead of "Researchers disagree if AEM is better than PEM", the label should be "AEM vs. PEM Efficacy". Instead of "It is unclear if this material is stable", prefer "Uncertain Material Stability".
4.  If the sentence does NOT describe a controversy, you MUST respond with the exact string "No Controversy".
5.  Your response must ONLY be the controversy label or "No Controversy", with no additional text.

Sentence to analyze:
"{sentence}"

Controversy Label:
"""

async def extrair_controversies_async():
    """
    Main asynchronous function to extract controversy labels from sentences in a CSV.
    """
    print(f"Starting Controversy Extraction with model '{config.LLM_ADVANCED_MODEL}'")

    # --- File Path Definitions ---
    input_csv_path = f"{config.DATA_FOLDER}/df_data_phrases.csv"
    output_parquet_path = f"{config.DATA_FOLDER}/df_data_phrases.parquet"
    
    # --- vLLM Client Initialization ---
    try:
        client_vllm = AsyncOpenAI(base_url=config.VLLM_BASE_URL, api_key=config.VLLM_API_KEY)
        print("vLLM client (asynchronous) connected.")
    except Exception as e:
        print(f"ERROR: Could not connect to the vLLM client: {e}")
        return

    # --- Data Reading and Filtering (CSV) ---
    try:
        df_targets = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(f"ERROR: Input file '{input_csv_path}' not found.")
        return
    except Exception as e:
        print(f"ERROR: Failed to read CSV file: {e}")
        return

    # Requested output column
    controversy_key = "controversies_label" # <-- CHANGE HERE

    # The main DataFrame will be fully processed.
    df_to_process = df_targets.copy()
    # Ensure the target column exists
    if controversy_key not in df_targets.columns:
        df_targets[controversy_key] = None

    if df_to_process.empty:
        # This check is now only for the case of an empty CSV
        print(f"The CSV file '{input_csv_path}' is empty.")
        return

    print(f"Starting controversy extraction for {len(df_to_process)} sentences (processing all rows).")

    # --- Asynchronous Extraction Loop ---
    CONCURRENCY_LIMIT = 50  # Adjust according to your server capacity

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def process_row(index, row):
        """ Processes a single row of the dataframe. """
        async with semaphore:
            # We use the 'phrase' column, as requested
            prompt = CONTROVERSY_EXTRACTION_PROMPT.format(sentence=row.get('phrase', ''))
            try:
                response = await client_vllm.chat.completions.create(
                    model=config.LLM_ADVANCED_MODEL,
                    messages=[{'role': 'user', 'content': prompt}]
                )
                controversy_label = response.choices[0].message.content.strip()
            except Exception as e:
                # print(f"\nWARNING: Error extracting controversy: {e}. Skipping.")
                controversy_label = "Extraction Error"

            # Returns the original DataFrame index and the extracted label
            return index, controversy_label

    # Create and execute all tasks
    tasks = [process_row(index, row) for index, row in df_to_process.iterrows()]
    
    # Collect all results before saving
    results_list = []
    
    print("Starting asynchronous processing...")
    for f in async_tqdm.as_completed(tasks, total=len(tasks), desc="Extracting controversy labels"):
        # Collect results (index and label)
        results_list.append(await f)

    # Update the main DataFrame all at once
    print("\nAsynchronous processing completed. Updating DataFrame in memory...")
    for index, label in results_list:
        df_targets.loc[index, controversy_key] = label

    # --- Final Saving ---
    print("\nControversy label extraction completed!")
    
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
    asyncio.run(extrair_controversies_async())
    print("Script 6 finished.")
