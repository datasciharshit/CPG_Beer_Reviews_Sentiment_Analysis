import logging
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

logger = logging.getLogger("beverage_cleaner.modeling.data_loader")

# Label map for classification targets
LABEL_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def label_sentiment(rating: float) -> str:
    """Maps numerical overall rating to sentiment class label."""
    if rating >= 4.0:
        return "Positive"
    elif rating <= 2.5:
        return "Negative"
    else:
        return "Neutral"

def load_and_prepare_data(
    file_path: str = "dataset/beer_reviews_cleaned_processed.csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads processed dataset, validates schema, applies sentiment labeling,
    and returns stratified train and test DataFrames.
    """
    logger.info(f"Reading processed reviews dataset from {file_path}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        raise FileNotFoundError(f"Processed dataset not found at {file_path}. Run pipeline first.")

    # Ensure required columns exist
    required_cols = ["text", "cleaned_text", "rating_overall"]
    for col in required_cols:
        if col not in df.columns:
            if col == "cleaned_text" and "text" in df.columns:
                logger.warning("cleaned_text missing. Generating fallback clean column using raw text.")
                df["cleaned_text"] = df["text"]
            else:
                raise ValueError(f"Required column '{col}' is missing from the dataset.")

    # Apply sentiment mapping
    logger.info("Generating sentiment labels based on overall rating...")
    df["sentiment"] = df["rating_overall"].apply(label_sentiment)
    df["label"] = df["sentiment"].map(LABEL_MAP)

    logger.info(f"Target Label Distribution:\n{df['sentiment'].value_counts(normalize=True) * 100}")

    # Split dataset (80/20 stratified)
    logger.info(f"Splitting dataset (test_size={test_size}, stratify=True)...")
    
    # Stratified split to ensure train/test sets have the same ratio of Positive/Neutral/Negative
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"]
    )
    
    logger.info(f"Split complete. Train set: {len(train_df)} rows, Test set: {len(test_df)} rows.")
    return train_df, test_df
