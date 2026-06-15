import os
import time
import pandas as pd
import logging
from deep_translator import GoogleTranslator
from tqdm import tqdm
from beverage_cleaner import ReviewCleaner, setup_logging

# Initialize logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("japanese_simulation")

DATASET_PATH = "dataset/beer_reviews_clean_100k.csv"
SIMULATED_PATH = "dataset/beer_reviews_japanese_simulated.csv"
CLEANED_PATH = "dataset/beer_reviews_multilingual_cleaned.csv"

def main():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Base dataset not found at {DATASET_PATH}")
        return

    logger.info("Step 1: Loading base reviews to simulate multilingual dataset...")
    df = pd.read_csv(DATASET_PATH, nrows=100) # Translate a small set of 100 reviews for speed
    logger.info(f"Loaded {len(df)} reviews.")

    # We will translate the first 20 reviews to Japanese to simulate mixed text
    N_TRANSLATE = 20
    logger.info(f"Translating first {N_TRANSLATE} reviews to Japanese using deep-translator...")
    
    translator = GoogleTranslator(source="en", target="ja")
    
    japanese_texts = []
    for idx, text in enumerate(df["text"].head(N_TRANSLATE)):
        try:
            # Squeeze text to first 300 chars to avoid translating excessively long reviews and save time
            truncated_text = text[:300]
            translated = translator.translate(truncated_text)
            japanese_texts.append(translated)
            logger.info(f"Translated review {idx+1}/{N_TRANSLATE} to Japanese.")
            time.sleep(0.2) # Polite delay
        except Exception as e:
            logger.warning(f"Translation failed for review {idx+1}: {e}. Using fallback.")
            japanese_texts.append("このビールは美味しいです。ホップの味が強く、泡立ちが良いです。")

    # Replace the text column for those translated rows
    df_simulated = df.copy()
    df_simulated.loc[:N_TRANSLATE-1, "text"] = japanese_texts
    df_simulated.loc[:N_TRANSLATE-1, "language"] = "ja"
    df_simulated.loc[N_TRANSLATE:, "language"] = "en"

    # Save the simulated multilingual dataset
    logger.info(f"Saving simulated multilingual dataset to {SIMULATED_PATH}...")
    df_simulated.to_csv(SIMULATED_PATH, index=False)
    
    # ----------------------------------------------------
    # Step 2: Clean the Multilingual Dataset using our Package
    # ----------------------------------------------------
    logger.info("\nStep 2: Processing simulated multilingual dataset through ReviewCleaner...")
    
    # Initialize cleaner with translation enabled!
    cleaner = ReviewCleaner(
        enable_translation=True, # WILL detect Japanese and translate to English
        enable_html_removal=True,
        enable_contractions=True,
        enable_emails_urls=True,
        enable_repeated_chars=True,
        enable_special_chars=True,
        enable_lemmatization=True,
        enable_spellcheck=False,
    )

    raw_texts = df_simulated["text"].astype(str).tolist()

    logger.info("Cleaning reviews in parallel (which batch-translates first)...")
    start_time = time.time()
    
    # Clean and translate back in parallel
    cleaned_texts = cleaner.clean_parallel(raw_texts, n_jobs=-1, batch_size=10)
    
    duration = time.time() - start_time
    logger.info(f"Cleaned and translated back {len(cleaned_texts)} reviews in {duration:.4f} seconds.")

    # Save cleaned output
    df_simulated["cleaned_text"] = cleaned_texts
    df_simulated.to_csv(CLEANED_PATH, index=False)
    logger.info(f"Saved multilingual cleaned dataset to {CLEANED_PATH}")

    # Display results validation
    print("\n=== Translation & Preprocessing Validation ===")
    for idx in range(3):
        original_eng = df["text"].iloc[idx][:120]
        japanese_sim = df_simulated["text"].iloc[idx][:120]
        cleaned_back = df_simulated["cleaned_text"].iloc[idx][:120]
        
        print(f"\n[Review {idx+1}] (Simulated Lang: {df_simulated['language'].iloc[idx]})")
        print(f"   Original English:   {original_eng}...")
        print(f"   Simulated Japanese: {japanese_sim}...")
        print(f"   Cleaned Back Eng:   {cleaned_back}...")

if __name__ == "__main__":
    main()
