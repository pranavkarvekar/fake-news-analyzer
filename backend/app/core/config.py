"""
Central configuration for the Fake News Detection API.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / ".env")

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
MODEL_DIR = BASE_DIR / "models_bin"
MODEL_PATH = MODEL_DIR / "fake_news_pipeline.pkl"
DATASET_PATH = BASE_DIR.parent / "WELFake_Dataset.csv"

# ──────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"
API_TITLE = "Fake News Detection API"
API_VERSION = "2.0.0"
API_DESCRIPTION = (
    "ML-powered REST API for detecting fake news articles. "
    "Features entity-masked TF-IDF pipeline with agentic fact-checking."
)

# ──────────────────────────────────────────────────────────────
# CORS — allow frontend dev server + production origins
# ──────────────────────────────────────────────────────────────
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# ──────────────────────────────────────────────────────────────
# ML Training Defaults
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_SAMPLES_PER_CLASS = 10000
TFIDF_MAX_FEATURES = 50000

TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# Inference threshold:
# classify as Fake only if p(fake) >= threshold, else Real.
# This reduces false alarms where real news gets flagged as fake.
FAKE_CLASS_THRESHOLD = float(os.getenv("FAKE_CLASS_THRESHOLD", "0.50"))



# ──────────────────────────────────────────────────────────────
# ML model repair (when old pickle is incompatible)
# ──────────────────────────────────────────────────────────────
# If the stored pipeline can't unpickle a fitted TF-IDF (missing `idf_`),
# we rebuild only `idf_` using a balanced subset of the dataset.
IDF_REBUILD_SAMPLES_PER_CLASS = int(os.getenv("IDF_REBUILD_SAMPLES_PER_CLASS", "500"))


