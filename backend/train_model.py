"""
Model Training & Export Script (v2 — Entity Masking + Cross-Validation)
========================================================================
Preprocesses data with spaCy entity masking, validates the pipeline with
5-fold StratifiedKFold cross-validation, then trains on the full dataset
and exports the model to fake_news_pipeline.pkl.

Usage:
    python -m backend.train_model
"""
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

from backend.app.core.config import (
    DATASET_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    N_SAMPLES_PER_CLASS,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
)

# Import the shared text processing module (entity masking + cleaning)
from backend.app.services.text_processing import clean_text, load_spacy_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    # ──────────────────────────────────────────────────────────
    # 0. Load spaCy NER model (used inside clean_text)
    # ──────────────────────────────────────────────────────────
    logger.info("Initializing spaCy NER model for entity masking …")
    load_spacy_model()

    # ──────────────────────────────────────────────────────────
    # 1. Load dataset
    # ──────────────────────────────────────────────────────────
    logger.info("Loading dataset from %s …", DATASET_PATH)
    if not DATASET_PATH.exists():
        logger.error("Dataset not found at %s", DATASET_PATH)
        sys.exit(1)

    data = pd.read_csv(
        DATASET_PATH,
        usecols=["title", "text", "label"],
        engine="python",
        on_bad_lines="skip",
    )
    data["label"] = (
        data["label"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"fake": "0", "real": "1", "false": "0", "true": "1"})
    )
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")
    logger.info("Dataset shape: %s", data.shape)

    # ──────────────────────────────────────────────────────────
    # 2. Sample balanced classes
    # ──────────────────────────────────────────────────────────
    df_fake = data[data["label"] == 0].sample(
        N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE
    )
    df_real = data[data["label"] == 1].sample(
        N_SAMPLES_PER_CLASS, random_state=RANDOM_STATE
    )
    df = (
        pd.concat([df_fake, df_real])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )
    logger.info("Sampled %d rows (balanced)", len(df))

    # ──────────────────────────────────────────────────────────
    # 3. Drop nulls & apply entity-masked cleaning
    # ──────────────────────────────────────────────────────────
    df = df.dropna()
    logger.info("Applying entity masking + text cleaning (this may take a few minutes) …")
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    logger.info("After cleaning: %d rows", len(df))

    # Show a sample of masked text for verification
    logger.info("─── Sample masked text ───")
    for i, row in df.head(3).iterrows():
        logger.info("  [%d] %s…", i, row["content"][:120])
    logger.info("──────────────────────────")

    # ──────────────────────────────────────────────────────────
    # 4. Prepare features
    # ──────────────────────────────────────────────────────────
    X = df["content"]
    y = df["label"]

    # ──────────────────────────────────────────────────────────
    # 5. Build the pipeline
    # ──────────────────────────────────────────────────────────
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                max_features=TFIDF_MAX_FEATURES,
                ngram_range=TFIDF_NGRAM_RANGE,
                stop_words="english",
                min_df=TFIDF_MIN_DF,
            )),
            ("model", LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1500,
                random_state=RANDOM_STATE,
            )),
        ]
    )

    # ──────────────────────────────────────────────────────────
    # 6. 5-Fold Stratified Cross-Validation
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  5-FOLD STRATIFIED CROSS-VALIDATION")
    logger.info("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1, verbose=1
    )

    logger.info("─── Cross-Validation Results ───")
    for fold_idx, score in enumerate(cv_scores, 1):
        logger.info("  Fold %d: Accuracy = %.4f", fold_idx, score)
    logger.info("  ────────────────────────────")
    logger.info("  Mean Accuracy:  %.4f", cv_scores.mean())
    logger.info("  Std Deviation:  %.4f", cv_scores.std())
    logger.info("────────────────────────────────")

    # Also compute F1 macro scores
    cv_f1_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="f1_macro", n_jobs=1, verbose=0
    )
    logger.info("  Mean F1 (macro): %.4f ± %.4f", cv_f1_scores.mean(), cv_f1_scores.std())
    logger.info("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 7. Train on FULL dataset & evaluate with hold-out
    # ──────────────────────────────────────────────────────────
    logger.info("Training final model on FULL dataset …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    logger.info("Train accuracy: %.4f", train_acc)
    logger.info("Test accuracy:  %.4f", test_acc)
    logger.info("Train f1_macro: %.4f", train_f1)
    logger.info("Test f1_macro:  %.4f", test_f1)
    logger.info("Overfit gap (acc): %.4f", train_acc - test_acc)
    logger.info("Overfit gap (f1):  %.4f", train_f1 - test_f1)
    print("\n--- Test Classification Report ---")
    print(classification_report(y_test, y_test_pred, digits=4))

    # ──────────────────────────────────────────────────────────
    # 8. Export the trained pipeline
    # ──────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    logger.info(
        "✅ Pipeline saved to %s (%.1f MB)",
        MODEL_PATH,
        MODEL_PATH.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
