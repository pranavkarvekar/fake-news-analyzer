"""
ML service — loads the trained pipeline and exposes a predict function.

Uses the shared text_processing module for entity-masked cleaning,
ensuring train/inference parity.
"""
import logging
import joblib
import numpy as np
from pathlib import Path

import pandas as pd

from backend.app.core.config import (
    MODEL_PATH,
    FAKE_CLASS_THRESHOLD,
    DATASET_PATH,
    RANDOM_STATE,
    IDF_REBUILD_SAMPLES_PER_CLASS,
)

# Shared text processing (entity masking + regex cleaning)
from backend.app.services.text_processing import clean_text

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Module-level model reference (set during startup)
# ──────────────────────────────────────────────────────────────
_pipeline = None


def load_model(path: Path | None = None) -> None:
    """Load the serialized sklearn pipeline into memory."""
    global _pipeline
    model_path = path or MODEL_PATH
    logger.info("Loading ML pipeline from %s …", model_path)
    _pipeline = joblib.load(model_path)

    # Some previously saved pickles can load but fail at inference because
    # TF-IDF isn't fitted in the current sklearn version (missing `idf_`).
    tfidf = getattr(_pipeline, "named_steps", {}).get("tfidf")
    if tfidf is not None and not hasattr(tfidf, "idf_"):
        logger.warning(
            "TF-IDF in loaded pipeline is not fitted (missing idf_). Rebuilding idf_..."
        )
        _repair_tfidf_idf(_pipeline)

    # scikit-learn pickle incompatibility fix:
    # Some older pickles can load but miss the `multi_class` attribute.
    model = getattr(_pipeline, "named_steps", {}).get("model")
    if model is not None and model.__class__.__name__ == "LogisticRegression":
        if not hasattr(model, "multi_class"):
            logger.warning(
                "LogisticRegression in loaded pipeline is missing `multi_class`. "
                "Setting it to 'auto' for compatibility."
            )
            model.multi_class = "auto"

    logger.info("ML pipeline loaded successfully.")


def _repair_tfidf_idf(pipeline) -> None:
    """
    Rebuild only TF-IDF's `idf_` using the saved vocabulary and a balanced
    subset of the dataset.

    This preserves the LogisticRegression weights (coef_ aligns with the
    feature ordering derived from vocabulary).
    """
    from sklearn.model_selection import train_test_split

    tfidf = pipeline.named_steps["tfidf"]
    vocabulary = getattr(tfidf, "vocabulary_", None)
    if not vocabulary:
        raise RuntimeError("Cannot repair TF-IDF: vocabulary is missing.")

    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset not found at {DATASET_PATH}")

    # Load a balanced subset similar to train_model sampling.
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
    data = data.dropna(subset=["title", "text"])

    # WELFake dataset uses 1 for Fake and 0 for Real.
    df_fake = data[data["label"] == 1]
    df_real = data[data["label"] == 0]

    n_fake = min(IDF_REBUILD_SAMPLES_PER_CLASS, len(df_fake))
    n_real = min(IDF_REBUILD_SAMPLES_PER_CLASS, len(df_real))
    if n_fake < 10 or n_real < 10:
        raise RuntimeError(
            f"Not enough data to rebuild TF-IDF idf_ (fake={n_fake}, real={n_real})."
        )

    df_fake = df_fake.sample(n_fake, random_state=RANDOM_STATE)
    df_real = df_real.sample(n_real, random_state=RANDOM_STATE)
    df = (
        pd.concat([df_fake, df_real])
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    logger.info("Rebuilding idf_ using %d samples (approx)…", len(df))
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    X = df["content"]
    y = df["label"]

    # Mimic train_model's fit step (fit on train split).
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    params = tfidf.get_params()
    params["vocabulary"] = vocabulary

    # Recreate vectorizer with the same configuration but fixed vocabulary.
    repaired_vectorizer = tfidf.__class__(**params)
    repaired_vectorizer.fit(X_train)

    # Replace the TF-IDF step in the pipeline; keep LogisticRegression weights intact.
    pipeline.set_params(tfidf=repaired_vectorizer)

    # Persist the repaired pipeline so the server doesn't repeat the work.
    joblib.dump(pipeline, MODEL_PATH)


def is_model_loaded() -> bool:
    """Check whether the model is currently loaded."""
    return _pipeline is not None


# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────
def predict_news(title: str, text: str) -> dict:
    """
    Run the loaded pipeline on a single article.

    The text is preprocessed with the SAME entity-masking + cleaning
    pipeline used during training, ensuring consistent behavior.

    Returns:
        dict with keys:
            - prediction: "Real" or "Fake"
            - confidence: float (probability of predicted class)
            - label: int (0=Fake, 1=Real)
            - p_fake: float (raw probability of Fake class)
            - p_real: float (raw probability of Real class)
            - masked_text: str (the entity-masked input, for agent use)
    """
    if _pipeline is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")

    combined = clean_text(f"{title} {text}")
    proba = _pipeline.predict_proba([combined])[0]
    # In the trained pipeline (WELFake), class 0 is Real and class 1 is Fake
    p_real = float(proba[0])
    p_fake = float(proba[1])

    # Bias toward fewer false "Fake" alarms by requiring stronger fake evidence.
    if p_fake >= FAKE_CLASS_THRESHOLD:
        label = 0
        confidence = p_fake
    else:
        label = 1
        confidence = p_real

    return {
        "prediction": "Real" if label == 1 else "Fake",
        "confidence": round(confidence, 4),
        "label": label,
        "p_fake": round(p_fake, 4),
        "p_real": round(p_real, 4),
        "masked_text": combined,
    }
