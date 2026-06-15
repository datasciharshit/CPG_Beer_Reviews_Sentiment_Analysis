import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

from modeling.data_loader import load_and_prepare_data, REVERSE_LABEL_MAP

logger = logging.getLogger("beverage_cleaner.modeling.evaluate_fine_tuned")

def evaluate_fine_tuned_model(
    test_df: pd.DataFrame,
    sample_size: int = 1000,
    model_dir: str = "artifacts/fine_tuned_distilbert",
):
    """
    Loads local fine-tuned model, runs inference on the holdout test set,
    and prints the classification report showing domain adaptation improvements.
    """
    test_slice = test_df.head(sample_size).copy()
    logger.info(f"Evaluating fine-tuned model on {len(test_slice)} test reviews...")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Move to GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        model = model.to(device)
        logger.info("Moving model to Apple Silicon MPS GPU for evaluation.")
    else:
        device = torch.device("cpu")
        logger.info("Evaluating on CPU.")

    model.eval()

    texts = test_slice["cleaned_text"].fillna("").astype(str).tolist()
    true_labels = test_slice["label"].tolist()

    predictions = []
    
    # Process in batches to save memory
    batch_size = 64
    logger.info(f"Running evaluation inference in batches of {batch_size}...")
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            predictions.extend(preds)

    # Map predictions back to labels ('Positive', 'Neutral', 'Negative')
    y_pred_labels = [REVERSE_LABEL_MAP[p] for p in predictions]
    y_test_labels = test_slice["sentiment"].tolist()

    print("\n=== Fine-Tuned Domain Adapted Model Evaluation ===")
    print(classification_report(y_test_labels, y_pred_labels, target_names=["Negative", "Neutral", "Positive"]))

if __name__ == "__main__":
    from beverage_cleaner import setup_logging
    setup_logging(level=logging.INFO)
    _, test_df = load_and_prepare_data()
    evaluate_fine_tuned_model(test_df)
