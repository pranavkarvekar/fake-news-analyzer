"""
Shared Text Processing Module
=================================
Contains the entity-masking + regex cleaning pipeline used by BOTH
the training script and the inference service.

This ensures train/inference parity — the model always sees the same
preprocessed input format.
"""
import re
import os
import time
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Lazy singleton for the spaCy NER model
# ──────────────────────────────────────────────────────────────
_nlp = None

# Entity types we mask and their replacement tags
_ENTITY_MAP = {
    "PERSON": "[PERSON]",
    "ORG":    "[ORG]",
    "GPE":    "[GPE]",
}


def load_spacy_model() -> None:
    """
    Load the spaCy 'en_core_web_sm' model into a module-level singleton.

    Call this once at startup (lifespan) or at the top of train_model.py.
    Subsequent calls are no-ops.
    """
    global _nlp
    if _nlp is not None:
        return
    # Force a non-torch backend to avoid intermittent Windows DLL init failures
    # that can happen when thinc tries to import torch.
    os.environ.setdefault("THINC_BACKEND", "numpy")
    logger.info("Loading spaCy model 'en_core_web_sm' …")

    # Intermittent Windows torch/thinc DLL init failures can happen.
    # Retry a few times to make training reliable.
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            import spacy  # lazy import: respects env vars above
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
            logger.info("spaCy model loaded successfully.")
            return
        except OSError as e:
            last_err = e
            logger.warning(
                "spaCy load failed (attempt %d/3): %s",
                attempt,
                str(e).splitlines()[0],
            )
            time.sleep(5)

    # If we exhausted retries, re-raise the last error
    assert last_err is not None
    raise last_err


def get_spacy_model():
    """Return the loaded spaCy model (auto-loads if needed)."""
    if _nlp is None:
        load_spacy_model()
    return _nlp


# ──────────────────────────────────────────────────────────────
# Entity Masking
# ──────────────────────────────────────────────────────────────
def mask_entities(text: str) -> str:
    """
    Replace named entities (PERSON, ORG, GPE) with generic tags.

    Example:
        "Donald Trump met with Google in Washington"
        → "[PERSON] met with [ORG] in [GPE]"

    This forces TF-IDF to learn from linguistic patterns
    rather than memorizing specific proper nouns.
    """
    nlp = get_spacy_model()
    doc = nlp(text)

    # Process entities in reverse order to preserve character offsets
    masked = text
    for ent in reversed(doc.ents):
        tag = _ENTITY_MAP.get(ent.label_)
        if tag:
            masked = masked[:ent.start_char] + tag + masked[ent.end_char:]

    return masked


# ──────────────────────────────────────────────────────────────
# Full Cleaning Pipeline
# ──────────────────────────────────────────────────────────────
def clean_text(s: str) -> str:
    """
    Complete text preprocessing pipeline:
      1. Handle NaN / empty values
      2. Mask named entities (PERSON, ORG, GPE) with generic tags
      3. Lowercase
      4. Remove URLs
      5. Remove HTML tags
      6. Keep letters, numbers, basic punctuation, and entity tags []
      7. Normalize whitespace
    """
    if pd.isna(s):
        return ""
    s = str(s)

    # Step 1: Entity masking (before lowercasing, so spaCy NER works well)
    s = mask_entities(s)

    # Step 2: Lowercase
    s = s.lower()

    # Step 3: Remove URLs
    s = re.sub(r"http\S+|www\.\S+", " ", s)

    # Step 4: Remove HTML tags
    s = re.sub(r"<.*?>", " ", s)

    # Step 5: Keep letters, numbers, basic punctuation, and [] for entity tags
    s = re.sub(r"[^a-z0-9\s\.\,\!\?\-\'\[\]]", " ", s)

    # Step 6: Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s
