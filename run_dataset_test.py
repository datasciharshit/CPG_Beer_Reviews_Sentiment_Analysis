import os
import time
import pandas as pd
import logging
from beverage_cleaner import (
    setup_logging,
    ReviewCleaner,
    register_extractor,
)

# Initialize logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("dataset_pipeline")

DATASET_PATH = "dataset/beer_reviews_clean_100k.csv"
OUTPUT_PATH = "dataset/beer_reviews_cleaned_processed.csv"

def main():
    if not os.path.exists(DATASET_PATH):
        logger.error(f"Dataset not found at {DATASET_PATH}. Please make sure the CSV is placed correctly.")
        return

    logger.info(f"Loading dataset from {DATASET_PATH}...")
    
    # We will process a subset of 25,000 reviews first to demonstrate speed & correctness.
    # To run on the full 100k reviews, simply comment out or remove 'nrows=25000'.
    N_ROWS = 25000
    df = pd.read_csv(DATASET_PATH, nrows=N_ROWS)
    logger.info(f"Successfully loaded {len(df)} rows.")

    # Initialize ReviewCleaner
    # We disable translation checks for this local run since we verified the dataset contains 0 Japanese reviews,
    # which cuts out translate check overhead entirely and speeds it up even more.
    cleaner = ReviewCleaner(
        enable_translation=False,
        enable_html_removal=True,
        enable_contractions=True,
        enable_emails_urls=True,
        enable_repeated_chars=True,
        enable_special_chars=True,
        enable_lemmatization=True,
        enable_spellcheck=False,
    )

    # Register the aspect extractor component on the spaCy pipeline
    if cleaner.nlp:
        register_extractor(cleaner.nlp)

    # Extract original text list
    logger.info("Extracting reviews text for parallel processing...")
    review_texts = df["text"].astype(str).tolist()

    # Process and clean the reviews + extract features in parallel
    logger.info(f"Processing and extracting aspects for {len(review_texts)} reviews...")
    start_time = time.time()
    
    # Run the integrated parallel method
    processed_results = cleaner.process_reviews_parallel(
        texts=review_texts,
        n_jobs=-1,        # Use all CPU cores
        batch_size=256,   # Chunk size per thread
    )
    
    duration = time.time() - start_time
    logger.info(f"Processed {len(review_texts)} reviews in {duration:.4f} seconds ({len(review_texts)/duration:.2f} reviews/sec).")

    # Convert results list of dicts to DataFrame
    df_results = pd.DataFrame(processed_results)

    # Combine back with original columns of interest
    df_output = pd.concat([
        df[["beer_name", "style", "rating_overall", "text"]],
        df_results
    ], axis=1)

    # Save output to file
    logger.info(f"Saving cleaned dataset with CPG aspects to {OUTPUT_PATH}...")
    df_output.to_csv(OUTPUT_PATH, index=False)
    logger.info("File saved successfully.")

    # ----------------------------------------------------
    # Business Jargon & Aspect Insights
    # ----------------------------------------------------
    logger.info("=== Business Aspect & Domain Jargon Insights ===")
    
    # Calculate aspect densities
    total_reviews = len(df_output)
    taste_reviews = (df_output["aspect_taste_flavor"] > 0).sum()
    mouthfeel_reviews = (df_output["aspect_mouthfeel_texture"] > 0).sum()
    appearance_reviews = (df_output["aspect_appearance"] > 0).sum()
    packaging_reviews = (df_output["aspect_packaging_price"] > 0).sum()

    logger.info(f"Total reviews analyzed: {total_reviews}")
    logger.info(f"Reviews mentioning Taste/Flavor terms: {taste_reviews} ({taste_reviews/total_reviews*100:.2f}%)")
    logger.info(f"Reviews mentioning Mouthfeel/Texture terms: {mouthfeel_reviews} ({mouthfeel_reviews/total_reviews*100:.2f}%)")
    logger.info(f"Reviews mentioning Appearance terms: {appearance_reviews} ({appearance_reviews/total_reviews*100:.2f}%)")
    logger.info(f"Reviews mentioning Packaging/Price terms: {packaging_reviews} ({packaging_reviews/total_reviews*100:.2f}%)")

    # Group by beer style and print top 5 styles by mouthfeel mentions
    logger.info("\n--- Top 5 Beer Styles by Average Mouthfeel Mentions ---")
    style_mouthfeel = df_output.groupby("style")["aspect_mouthfeel_texture"].mean().sort_values(ascending=False).head(5)
    for idx, (style, val) in enumerate(style_mouthfeel.items(), 1):
        logger.info(f"{idx}. {style}: {val:.2f} mentions/review average")

    # Print a sample review before and after processing
    logger.info("\n--- Preprocessing Validation Sample ---")
    logger.info(f"Original Text:  {df_output['text'].iloc[0][:150]}...")
    logger.info(f"Cleaned Text:   {df_output['cleaned_text'].iloc[0][:150]}...")
    logger.info(f"Aspect Counts:  Taste={df_output['aspect_taste_flavor'].iloc[0]}, "
                f"Mouthfeel={df_output['aspect_mouthfeel_texture'].iloc[0]}, "
                f"Appearance={df_output['aspect_appearance'].iloc[0]}, "
                f"Packaging={df_output['aspect_packaging_price'].iloc[0]}")

if __name__ == "__main__":
    main()
