"""
Central configuration for the Fake News Detection System.

Every path and tunable constant used by the training scripts and by the Flask
application lives here.  Keeping them in one module means a student can change
one value (for example ``TFIDF_MAX_FEATURES``) and have the whole project pick
it up, instead of hunting for hard-coded values scattered across files.

Values that are environment specific (secret key, debug flag, port) are read
from environment variables / a ``.env`` file so that nothing sensitive is
hard-coded in the source tree.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Optional .env support.
# python-dotenv is listed in requirements.txt, but the project must still work
# if it is missing, so the import is guarded.
# --------------------------------------------------------------------------
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# ==========================================================================
# 1. PROJECT PATHS
# ==========================================================================
# BASE_DIR points at the repository root (the folder containing app.py).
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
DATABASE_DIR: Path = BASE_DIR / "database"

# Candidate locations for the raw dataset, searched in this order.
# This lets the user drop the CSV in whichever location they find natural.
DATASET_SEARCH_PATHS: tuple[Path, ...] = (
    RAW_DATA_DIR / "fake_or_real_news.csv",
    DATA_DIR / "dataset.csv",
    RAW_DATA_DIR / "dataset.csv",
    RAW_DATA_DIR / "news.csv",
)

# Cleaned dataset produced by src/dataset.py (cached so repeated training runs
# do not have to redo the expensive NLP preprocessing).
PROCESSED_DATASET_PATH: Path = PROCESSED_DATA_DIR / "cleaned_dataset.csv"

# Saved model artefacts.
MODEL_PATH: Path = MODELS_DIR / "model.pkl"
VECTORIZER_PATH: Path = MODELS_DIR / "vectorizer.pkl"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

# SQLite database used for the prediction history.
DATABASE_PATH: Path = DATABASE_DIR / "app.db"


# ==========================================================================
# 2. DATASET SCHEMA
# ==========================================================================
# The dataset must contain a text column and a label column.  ``title`` is
# optional but is combined with the article body when present.
TEXT_COLUMN: str = "text"
TITLE_COLUMN: str = "title"
LABEL_COLUMN: str = "label"

# Canonical class names used everywhere in the project.
LABEL_FAKE: str = "FAKE"
LABEL_REAL: str = "REAL"
CLASS_NAMES: tuple[str, str] = (LABEL_FAKE, LABEL_REAL)

# Different public datasets spell the labels differently.  This map normalises
# the common variants onto our two canonical labels.
LABEL_NORMALISATION_MAP: dict[str, str] = {
    "fake": LABEL_FAKE,
    "false": LABEL_FAKE,
    "fake news": LABEL_FAKE,
    "0": LABEL_FAKE,
    "real": LABEL_REAL,
    "true": LABEL_REAL,
    "truthful": LABEL_REAL,
    "1": LABEL_REAL,
}


# ==========================================================================
# 3. MACHINE LEARNING HYPER-PARAMETERS
# ==========================================================================
# A fixed random seed makes every run reproducible, which is essential when a
# result has to be defended in a project review.
RANDOM_STATE: int = 42

# Fraction of the data held out for the final test evaluation.
TEST_SIZE: float = 0.20

# Number of folds used for cross-validation during model selection.
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Model selection policy
# ---------------------------------------------------------------------------
# By default the winning algorithm is chosen by measured cross-validated F1 on
# the training split - never by assertion. Set PRIMARY_MODEL to one of the
# names in build_candidate_models() (for example "Logistic Regression") to pin
# a specific algorithm instead; the full comparison table is still produced and
# reported either way, so the trade-off stays visible and defensible.
#
# Override from the shell without editing this file:
#     PRIMARY_MODEL="Logistic Regression" python src/train_model.py
PRIMARY_MODEL: str | None = os.environ.get("PRIMARY_MODEL") or None

# --- TF-IDF settings ---
TFIDF_MAX_FEATURES: int = 20_000   # keep the 20k most informative terms
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)  # unigrams + bigrams
TFIDF_MIN_DF: int = 3              # ignore terms appearing in < 3 documents
TFIDF_MAX_DF: float = 0.85         # ignore terms appearing in > 85% of docs
TFIDF_SUBLINEAR_TF: bool = True    # use 1 + log(tf) instead of raw tf

# Minimum number of characters a document must have after cleaning to be kept.
MIN_CLEANED_LENGTH: int = 20


# ==========================================================================
# 4. WEB APPLICATION SETTINGS
# ==========================================================================
# SECRET_KEY must come from the environment in production.  The development
# fallback is clearly marked so it is never mistaken for a real secret.
SECRET_KEY: str = os.environ.get("SECRET_KEY", "dev-only-insecure-key-change-me")

FLASK_DEBUG: bool = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
FLASK_HOST: str = os.environ.get("FLASK_HOST", "127.0.0.1")
FLASK_PORT: int = int(os.environ.get("FLASK_PORT", "5000"))

# --- Input validation limits (see src/validation.py) ---
MIN_INPUT_LENGTH: int = 20        # characters
MAX_INPUT_LENGTH: int = 20_000    # characters - protects against huge payloads
MAX_REQUEST_BYTES: int = 1 * 1024 * 1024  # 1 MB cap on any request body

# --- Prediction reliability ---
# Minimum number of words that must survive preprocessing for a prediction to
# be considered dependable. Below this the result is flagged as "low signal".
#
# Why 20: preprocessing removes stopwords, so a short sentence collapses to
# only two or three content words. With so few features the model is really
# reporting the average association of those isolated words in the training
# corpus rather than analysing an article, and it can do so with a
# misleadingly high probability. Around 20 surviving words is where a document
# carries enough distinct terms for the score to mean something. A typical
# news paragraph clears this easily; a bare headline does not.
MIN_SIGNAL_WORDS: int = 20

# Number of history rows shown per page.
HISTORY_PAGE_SIZE: int = 25

# How many influential words the explainability feature reports.
TOP_FEATURES_COUNT: int = 8

# Standard disclaimer shown anywhere a prediction is displayed.
DISCLAIMER: str = (
    "This system provides an ML-based classification and should not be "
    "treated as a substitute for professional fact-checking."
)


def ensure_directories() -> None:
    """Create every directory the project writes to, if it does not exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        REPORTS_DIR,
        DATABASE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
