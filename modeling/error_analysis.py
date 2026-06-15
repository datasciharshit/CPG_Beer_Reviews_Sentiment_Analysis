import logging
import torch
import pandas as pd
from typing import List
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger("beverage_cleaner.modeling.error_analysis")

def run_error_analysis(
    test_df: pd.DataFrame,
    sample_size: int = 1000,
    model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
):
    """
    Runs off-the-shelf DistilBERT inference on the holdout test set (limited to sample_size for speed),
    compares predictions against target overall rating sentiments, isolates errors,
    and analyses domain keyword gaps.
    """
    # Slice the test data down to keep execution fast during checks
    test_slice = test_df.sample(n=min(sample_size, len(test_df)), random_state=42).copy()
    logger.info(f"Running error analysis on a sample of {len(test_slice)} test reviews...")

    # Set up GPU MPS acceleration on macOS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS (Metal Performance Shaders) GPU acceleration is available. Directing model to MPS.")
    else:
        device = torch.device("cpu")
        logger.info("MPS not available. Falling back to CPU for inference.")

    logger.info(f"Loading transformer model pipeline: {model_name}...")
    try:
        # Load the sentiment analysis pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device
        )
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}. Falling back to CPU.")
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1 # CPU
        )

    # Use original clean raw text (cased) for transformer inference
    cased_texts = test_slice["text"].astype(str).tolist()

    logger.info(f"Running inference on {len(cased_texts)} reviews (batch_size=64)...")
    raw_predictions = sentiment_pipeline(cased_texts, batch_size=64, truncation=True, max_length=512)

    # Map output labels (e.g. 'positive', 'neutral', 'negative') to capitalized matching labels
    y_pred_transformer = [pred["label"].capitalize() for pred in raw_predictions]
    y_test = test_slice["sentiment"].tolist()

    # Log Transformer metrics
    logger.info("=== Off-the-Shelf DistilBERT Evaluation (CPG Review Test Slice) ===")
    print(classification_report(y_test, y_pred_transformer, target_names=["Negative", "Neutral", "Positive"], zero_division=0))

    # Construct error analysis DataFrame
    error_analysis_df = pd.DataFrame({
        "text": cased_texts,
        "true_label": y_test,
        "pred_label": y_pred_transformer
    })

    # Isolate misclassifications
    errors = error_analysis_df[error_analysis_df["true_label"] != error_analysis_df["pred_label"]]
    logger.info(f"Total Off-the-Shelf Model Errors: {len(errors)} out of {len(test_slice)} ({len(errors)/len(test_slice)*100:.2f}% error rate)")

    # Define domain vocabulary keywords
    domain_keywords = ["mouthfeel", "head retention", "aftertaste", "skunky", "lacing", "carbonation", "hoppy"]
    
    print("\n=== CPG Domain Language Gap Identification ===")
    logger.info("Scanning model errors for specific beverage terminology gaps...")

    for keyword in domain_keywords:
        keyword_errors = errors[errors["text"].str.contains(keyword, case=False, na=False)]
        if not keyword_errors.empty:
            print(f"-> 🔍 Found {len(keyword_errors)} errors containing CPG term '{keyword}'")
            # Print a representative example
            sample = keyword_errors.iloc[0]
            print(f"   Review Snippet: \"{sample['text'][:180]}...\"")
            print(f"   True Rating Label: {sample['true_label']} | Model Predicted: {sample['pred_label']}\n")
        else:
            print(f"-> 🔍 No error samples for keyword '{keyword}' in this test slice.")

    return error_analysis_df, errors
