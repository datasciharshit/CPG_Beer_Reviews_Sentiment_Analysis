import time
import logging
from typing import List
from beverage_cleaner import (
    setup_logging,
    ReviewCleaner,
    register_extractor,
    extract_beverage_features,
    parallel_process,
)

# 1. Initialize logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("demo_pipeline")

def main():
    logger.info("Starting beverage review cleaning pipeline demo...")

    # Sample reviews: containing HTML, contractions, repeated characters,
    # Japanese sentences, and beverage-specific jargon.
    reviews = [
        "<p>This IPA is sooooo good! The head retention and lacing are amazing.</p>",
        "このビールは美味しいです！ Bitter finish, but the mouthfeel is extremely smooth.",
        "Really disappointed. The packaging was damaged, and the price was too expensive.",
        "I don't like watery beers. This stout has a thick creamy body and roasted chocolate notes.",
        "非常に美味しいです。柑橘系の香りがします。 Citrus aroma and crisp carbonation.",
    ]

    # 2. Initialize the cleaner
    # For testing, we keep translation active. We leave lemmatization on and spellcheck off.
    cleaner = ReviewCleaner(
        enable_translation=True,
        enable_html_removal=True,
        enable_contractions=True,
        enable_emails_urls=True,
        enable_repeated_chars=True,
        enable_special_chars=True,
        enable_lemmatization=True,
        enable_spellcheck=False,
    )

    # Register the custom aspect extractor on spaCy pipeline
    if cleaner.nlp:
        register_extractor(cleaner.nlp)

    # ----------------------------------------------------
    # DEMO 1: Single Review Processing
    # ----------------------------------------------------
    logger.info("--- Demo 1: Processing a single review ---")
    sample_text = reviews[0]
    logger.info(f"Original: {sample_text}")
    
    cleaned_sample = cleaner.clean(sample_text)
    logger.info(f"Cleaned:  {cleaned_sample}")
    
    # Extract domain features
    if cleaner.nlp:
        features = extract_beverage_features(cleaned_sample, cleaner.nlp)
        logger.info(f"Extracted Aspects: {features}\n")

    # ----------------------------------------------------
    # DEMO 2: Batch Translation & Cleaning
    # ----------------------------------------------------
    logger.info("--- Demo 2: Processing reviews in optimized batch ---")
    start_time = time.time()
    
    # Run the full batch cleaning (translates in batch first, then cleans individual texts)
    cleaned_reviews = cleaner.clean_batch(reviews)
    
    for orig, cleaned in zip(reviews, cleaned_reviews):
        logger.info(f"Orig:    {orig}")
        logger.info(f"Cleaned: {cleaned}")
        if cleaner.nlp:
            features = extract_beverage_features(cleaned, cleaner.nlp)
            for category, details in features.items():
                if details["count"] > 0:
                    logger.info(f"   -> Aspect Match [{category}]: {[m['lemma'] for m in details['matches']]}")
        logger.info("-" * 40)
        
    duration = time.time() - start_time
    logger.info(f"Batch processing of {len(reviews)} reviews completed in {duration:.4f} seconds.\n")

    # ----------------------------------------------------
    # DEMO 3: Parallel CPU Processing
    # ----------------------------------------------------
    logger.info("--- Demo 3: Scaled Parallel Cleaning ---")
    
    # Let's mock a larger list of reviews by repeating the sample 20 times
    large_reviews_list = reviews * 20
    logger.info(f"Simulating parallel processing of {len(large_reviews_list)} reviews...")

    start_parallel = time.time()

    # Leverage clean_parallel which uses nlp.pipe
    parallel_cleaned = cleaner.clean_parallel(
        texts=large_reviews_list,
        n_jobs=-1,        # Use all CPU cores
        batch_size=10,    # Process 10 items per batch per core
    )

    duration_parallel = time.time() - start_parallel
    logger.info(f"Parallel cleaned {len(parallel_cleaned)} reviews in {duration_parallel:.4f} seconds.")

if __name__ == "__main__":
    main()
