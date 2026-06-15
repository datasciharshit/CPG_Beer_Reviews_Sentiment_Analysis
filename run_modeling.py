import argparse
import logging
from beverage_cleaner import setup_logging
from modeling.data_loader import load_and_prepare_data
from modeling.train_baseline import train_and_evaluate_baseline
from modeling.error_analysis import run_error_analysis
from modeling.fine_tune import run_fine_tuning

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger("modeling_pipeline")

def main():
    parser = argparse.ArgumentParser(description="Beverage Review Sentiment Modeling Pipeline")
    
    # Execution flags
    parser.add_argument(
        "--data-path",
        type=str,
        default="dataset/beer_reviews_cleaned_processed.csv",
        help="Path to the cleaned reviews CSV dataset"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable baseline Logistic Regression GridSearchCV hyperparameter tuning"
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Run domain adaptation fine-tuning on DistilBERT"
    )
    parser.add_argument(
        "--no-quick-run",
        dest="quick_run",
        action="store_false",
        help="Disable quick-run mode and run on the full data slices"
    )
    
    args = parser.parse_args()

    logger.info("Starting sentiment modeling pipeline...")

    # Step 1: Load and Split Data
    try:
        train_df, test_df = load_and_prepare_data(file_path=args.data_path)
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return

    # Set parameters depending on quick-run mode
    if args.quick_run:
        logger.info("🚀 Quick-run mode active. Slicing data subsets for speed...")
        train_baseline_df = train_df.head(2000) # Use 2000 rows for baseline training
        test_baseline_df = test_df.head(500)
        error_analysis_size = 500
        fine_tune_train_size = 200
        fine_tune_eval_size = 100
        epochs = 1
    else:
        logger.info("🔥 Production run active. Processing full dataset slices...")
        train_baseline_df = train_df
        test_baseline_df = test_df
        error_analysis_size = 5000
        fine_tune_train_size = 20000
        fine_tune_eval_size = 2000
        epochs = 2

    # Step 2: Train TF-IDF + Logistic Regression Baseline
    logger.info("=== STEP 2: Training TF-IDF + Logistic Regression Baseline ===")
    train_and_evaluate_baseline(
        train_df=train_baseline_df,
        test_df=test_baseline_df,
        max_features=5000 if args.quick_run else 10000,
        run_grid_search=args.grid_search,
    )

    # Step 3: Run Zero-Shot Transformer Evaluation & Domain Gap Analysis
    logger.info("=== STEP 3: Running Off-the-Shelf DistilBERT & Domain Error Analysis ===")
    run_error_analysis(
        test_df=test_baseline_df,
        sample_size=error_analysis_size,
    )

    # Step 4: Run Fine-Tuning (Optional via flag)
    if args.fine_tune:
        logger.info("=== STEP 4: Running Transformer Domain Adaptation Fine-Tuning ===")
        run_fine_tuning(
            train_df=train_df,
            test_df=test_df,
            train_size=fine_tune_train_size,
            eval_size=fine_tune_eval_size,
            epochs=epochs,
        )
    else:
        logger.info("=== STEP 4: Skip Fine-Tuning (Pass --fine-tune to run domain adaptation) ===")

    logger.info("Sentiment Modeling Pipeline run completed successfully.")

if __name__ == "__main__":
    main()
