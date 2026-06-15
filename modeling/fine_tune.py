import os
import torch
import logging
import pandas as pd
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

logger = logging.getLogger("beverage_cleaner.modeling.fine_tune")

class BeverageDataset(torch.utils.data.Dataset):
    """Custom Dataset class for formatting review encodings and labels for PyTorch training."""
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def run_fine_tuning(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_size: int = 2000,   # Set small by default for verification; user can scale to 20000
    eval_size: int = 1000,    # Evaluation subset size
    epochs: int = 1,
    model_name: str = "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    output_dir: str = "artifacts/fine_tuned_distilbert",
):
    """
    Runs domain fine-tuning on DistilBERT.
    Detects hardware capabilities (MPS for macOS, CUDA for Nvidia, CPU fallback)
    and configures mixed precision settings accordingly.
    """
    logger.info("Preparing datasets for PyTorch fine-tuning...")

    # Set up subset sizes to avoid local machine freezes
    train_subset = train_df.head(train_size)
    eval_subset = test_df.head(eval_size)
    
    logger.info(f"Fine-tuning configuration: Training size={len(train_subset)}, Eval size={len(eval_subset)}")

    train_texts = train_subset["cleaned_text"].fillna("").astype(str).tolist()
    train_labels = train_subset["label"].tolist()

    test_texts = eval_subset["cleaned_text"].fillna("").astype(str).tolist()
    test_labels = eval_subset["label"].tolist()

    logger.info(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger.info("Tokenizing texts (max_length=128)...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

    # Convert to PyTorch datasets
    train_dataset = BeverageDataset(train_encodings, train_labels)
    eval_dataset = BeverageDataset(test_encodings, test_labels)

    logger.info(f"Initializing AutoModelForSequenceClassification ({model_name})...")
    # ignore_mismatched_sizes=True allows us to swap classifier heads for CPG classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # Detect hardware capabilities
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    fp16_enabled = False
    if use_cuda:
        device_name = "cuda"
        fp16_enabled = True # Use mixed precision on Nvidia GPUs for 50% speedup
        logger.info("CUDA GPU detected. Enabling mixed precision (fp16=True).")
    elif use_mps:
        device_name = "mps"
        # MPS doesn't support full mixed precision natively for all training layers, keep False to prevent Metal crashes
        logger.info("Apple Silicon MPS GPU detected. Directing PyTorch training to Metal.")
    else:
        device_name = "cpu"
        logger.info("No GPU detected. Training will run on CPU.")

    # Configure training arguments
    logger.info("Setting up TrainingArguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        fp16=fp16_enabled,
        use_cpu=(device_name == "cpu"),
        report_to="none" # Disable integrations (e.g. W&B) to keep run clean
    )

    logger.info("Initializing Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    logger.info("🔥 Starting Domain Adaptation Fine-Tuning...")
    trainer.train()

    # Save fine-tuned model checkpoint
    logger.info(f"Saving fine-tuned model artifact to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuning completed successfully!")

    return model, tokenizer
