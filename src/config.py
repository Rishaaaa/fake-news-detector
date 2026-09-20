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

# NOTE: there is no server-side database. Prediction history is stored in the
# visitor's own browser (see static/js/history-store.js), so the application
# holds no user data at rest and needs no persistent disk when deployed.


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

# Hosting platforms (Render, Railway, Heroku, Fly) inject the port to bind to as
# PORT, and require binding to 0.0.0.0 rather than localhost - a service bound to
# 127.0.0.1 is unreachable from outside the container and the deploy will fail
# its health check. Locally the defaults keep the server private to this machine.
FLASK_HOST: str = os.environ.get("FLASK_HOST", "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1")
FLASK_PORT: int = int(os.environ.get("PORT") or os.environ.get("FLASK_PORT", "5000"))

# --- Input validation limits (see src/validation.py) ---
MIN_INPUT_LENGTH: int = 20        # characters
MAX_INPUT_LENGTH: int = 20_000    # characters - protects against huge payloads
MAX_REQUEST_BYTES: int = 1 * 1024 * 1024  # 1 MB cap on any request body

# --- Prediction reliability ---
# Minimum number of words that must survive preprocessing before a prediction
# is presented without a reliability warning.
#
# Measured accuracy against surviving word count, on the 1,261-article held-out
# test set (see docs/PROJECT_REPORT.md section 22.7):
#
#      3 words -> 62.0%     20 words -> 77.3%     100 words -> 88.8%
#      5 words -> 66.4%     30 words -> 81.4%     200 words -> 92.8%
#     10 words -> 71.4%     50 words -> 85.1%     full      -> 94.4%
#
# Note that the model's *confidence* stays near 87-94% across this whole range,
# so it cannot detect its own degradation - the warning has to come from the
# word count, not from the probability.
MIN_SIGNAL_WORDS: int = 5

# --- Out-of-domain detection ---
# The model was trained on political news. Text from another domain still gets
# classified, usually as FAKE, because its vocabulary is unfamiliar.
#
# The detector measures what fraction of a document's terms are among the
# model's most influential features. Measured trade-off on the test set:
#
#   threshold   genuine articles wrongly warned   clearly-not-news caught
#      0.10                 4.7%                  recipes, bare claims
#      0.15                41.8%                  + business, science
#      0.18                71.1%                  + sports
#
# 0.10 is chosen deliberately: it reliably catches text that is not news at all
# at a low false-warning cost. It does NOT reliably separate sports/business/
# science *news* from political news - doing so would mean warning on most
# legitimate articles. That broader limitation is covered by a static notice in
# the UI instead, because no threshold here can fix it.
DOMAIN_MARKER_TOP_N: int = 500
DOMAIN_RATIO_THRESHOLD: float = 0.10

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
    ):
        directory.mkdir(parents=True, exist_ok=True)
