import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from modeling.data_loader import load_and_prepare_data

logger = logging.getLogger("beverage_cleaner.modeling.train_baseline")

# Define target directories
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def train_and_evaluate_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    max_features: int = 10000,
    run_grid_search: bool = True,
):
    """
    Fits TF-IDF vectorizer and trains a Logistic Regression baseline model.
    Saves models and reports artifacts to disk.
    """
    logger.info("Initializing TF-IDF vectorization...")
    # Squeeze NaNs
    X_train = train_df["cleaned_text"].fillna("").astype(str)
    y_train = train_df["sentiment"]
    X_test = test_df["cleaned_text"].fillna("").astype(str)
    y_test = test_df["sentiment"]

    # Fit TF-IDF Vectorizer extracting unigrams and bigrams
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    logger.info(f"Fitting TF-IDF Vectorizer (max_features={max_features})...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Save fitted vectorizer artifact
    vectorizer_path = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Saved TF-IDF Vectorizer to {vectorizer_path}")

    # Set up Logistic Regression with class weights balanced to address label skewness
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

    if run_grid_search:
        # Run GridSearchCV hyperparameter tuning
        logger.info("Setting up GridSearchCV for baseline model...")
        param_grid = {
            "C": [0.1, 1.0, 10.0],
            "solver": ["lbfgs", "liblinear"],
        }
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="f1_weighted",
            verbose=1,
            n_jobs=-1
        )
        logger.info("Starting baseline hyperparameter search...")
        grid_search.fit(X_train_tfidf, y_train)
        best_model = grid_search.best_estimator_
        logger.info(f"Best hyperparameters found: {grid_search.best_params_}")
    else:
        logger.info("Training standard Logistic Regression baseline...")
        model.fit(X_train_tfidf, y_train)
        best_model = model

    # Save model artifact
    model_path = os.path.join(ARTIFACTS_DIR, "baseline_model.pkl")
    joblib.dump(best_model, model_path)
    logger.info(f"Saved Baseline Model to {model_path}")

    # Run predictions on holdout set
    logger.info("Evaluating baseline model on holdout set...")
    y_pred = best_model.predict(X_test_tfidf)

    report = classification_report(y_test, y_pred, target_names=["Negative", "Neutral", "Positive"])
    print("\n=== Baseline English Model Evaluation ===")
    print(report)

    # Save classification report to file
    report_path = os.path.join(ARTIFACTS_DIR, "baseline_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Plot and save confusion matrix
    logger.info("Generating confusion matrix plot...")
    labels = ["Negative", "Neutral", "Positive"]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title("Baseline Confusion Matrix (Tuned)")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    
    cm_plot_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_tuned.png")
    plt.tight_layout()
    plt.savefig(cm_plot_path)
    plt.close()
    logger.info(f"Saved Confusion Matrix plot to {cm_plot_path}")

    return best_model, vectorizer
