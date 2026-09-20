"""
Dataset loading, validation and cleaning.

Responsibilities
----------------
1. Locate the raw CSV on disk (several conventional locations are searched).
2. Validate that it actually looks like a fake-news dataset.
3. Clean it: missing values, duplicates, label normalisation.
4. Run the NLP preprocessing pipeline over it and cache the result so that
   re-training does not repeat several minutes of text cleaning.

Nothing in this module invents data.  If the dataset is absent the user gets
explicit instructions telling them exactly where to put the file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Allow this file to be run directly as `python src/dataset.py` as well as via
# `python -m src.dataset`. When run directly, Python puts src/ on the path
# instead of the project root, so `from src import ...` would fail. Adding the
# project root here makes both invocations work identically.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src import config
from src.exceptions import DatasetError
from src.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)

# Shown when no dataset file can be found anywhere.
DATASET_MISSING_HELP = f"""
No dataset file was found.

Place a labelled fake-news CSV at one of these locations:

{chr(10).join('    - ' + str(p) for p in config.DATASET_SEARCH_PATHS)}

The CSV must contain at least these columns:

    text   - the news article body        (required)
    label  - FAKE or REAL                 (required)
    title  - the headline                 (optional, combined with `text`)

Suggested public datasets:

    1. "Fake or Real News" (6,335 rows, already in the required format):
       https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv

       Download it directly with:
           python src/dataset.py --download

    2. Kaggle "Fake and Real News Dataset" (Fake.csv + True.csv):
       https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
       Add a `label` column (FAKE / REAL) and concatenate the two files.
""".strip()

# Direct download URL for the default dataset used by --download.
DEFAULT_DATASET_URL = (
    "https://raw.githubusercontent.com/lutzhamel/fake-news/"
    "master/data/fake_or_real_news.csv"
)


# ==========================================================================
# Locating and reading the raw file
# ==========================================================================
def find_dataset_path() -> Path:
    """
    Return the first dataset file that exists among the configured locations.

    Raises
    ------
    DatasetError
        If no candidate file exists, with instructions on how to obtain one.
    """
    for candidate in config.DATASET_SEARCH_PATHS:
        if candidate.is_file():
            logger.info("Using dataset: %s", candidate)
            return candidate
    raise DatasetError(DATASET_MISSING_HELP, user_message="Training dataset not found.")


def download_default_dataset(destination: Optional[Path] = None) -> Path:
    """
    Download the default public dataset to ``data/raw/``.

    This is a convenience for first-time setup; the project works equally well
    with any CSV the user supplies manually.
    """
    import urllib.request

    destination = destination or (config.RAW_DATA_DIR / "fake_or_real_news.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset from %s", DEFAULT_DATASET_URL)
    try:
        urllib.request.urlretrieve(DEFAULT_DATASET_URL, destination)
    except Exception as exc:  # network errors, 404, permissions...
        raise DatasetError(
            f"Download failed: {exc}\n\n{DATASET_MISSING_HELP}",
            user_message="Could not download the dataset.",
        ) from exc

    size_mb = destination.stat().st_size / (1024 * 1024)
    logger.info("Saved %.1f MB to %s", size_mb, destination)
    return destination


def load_raw_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Read the raw CSV into a DataFrame and validate its structure.

    Raises
    ------
    DatasetError
        If the file cannot be parsed, is empty, or lacks required columns.
    """
    path = path or find_dataset_path()

    try:
        frame = pd.read_csv(path, encoding="utf-8", on_bad_lines="warn")
    except UnicodeDecodeError:
        # Some public datasets ship as latin-1; retry before giving up.
        logger.warning("UTF-8 decode failed for %s - retrying as latin-1.", path)
        frame = pd.read_csv(path, encoding="latin-1", on_bad_lines="warn")
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"The dataset file '{path}' is empty.") from exc
    except Exception as exc:
        raise DatasetError(f"Could not read '{path}': {exc}") from exc

    validate_dataset(frame, path)
    logger.info("Loaded %d raw rows from %s", len(frame), path.name)
    return frame


def validate_dataset(frame: pd.DataFrame, path: Path) -> None:
    """
    Check that a loaded DataFrame has the structure the project expects.

    Verifies: non-empty, required columns present, and at least two distinct
    labels (a single-class dataset cannot train a classifier).
    """
    if frame.empty:
        raise DatasetError(f"The dataset '{path}' contains no rows.")

    # Normalise column names so "Text" / " label " also match.
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    missing = [
        col
        for col in (config.TEXT_COLUMN, config.LABEL_COLUMN)
        if col not in frame.columns
    ]
    if missing:
        raise DatasetError(
            f"The dataset '{path}' is missing required column(s): "
            f"{', '.join(missing)}.\nColumns found: {', '.join(frame.columns)}.\n\n"
            f"{DATASET_MISSING_HELP}"
        )

    distinct_labels = frame[config.LABEL_COLUMN].dropna().nunique()
    if distinct_labels < 2:
        raise DatasetError(
            f"The dataset '{path}' contains only {distinct_labels} distinct "
            "label(s). At least two (FAKE and REAL) are required to train a "
            "classifier."
        )


# ==========================================================================
# Cleaning
# ==========================================================================
def normalise_labels(series: pd.Series) -> pd.Series:
    """
    Map assorted label spellings onto the canonical FAKE / REAL strings.

    Handles "fake"/"Fake"/"FAKE", "true"/"real", and the 0/1 encoding used by
    some datasets.  Unrecognised labels become NaN and are dropped by the
    caller, rather than being silently guessed at.
    """
    normalised = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(config.LABEL_NORMALISATION_MAP)
    )
    return normalised


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply structural cleaning: missing values, duplicates, label normalisation.

    Returns a DataFrame with two columns: ``content`` (title + body) and
    ``label``.  Text preprocessing is **not** applied here - that is done by
    ``build_processed_dataset`` so the two stages can be reasoned about (and
    logged) separately.
    """
    initial_rows = len(frame)
    frame = frame.copy()

    # --- 1. Combine title and body ---------------------------------------
    # The headline is often the strongest signal in fake news (clickbait,
    # ALL CAPS, sensational wording), so it is prepended to the article body.
    if config.TITLE_COLUMN in frame.columns:
        title = frame[config.TITLE_COLUMN].fillna("").astype(str)
        body = frame[config.TEXT_COLUMN].fillna("").astype(str)
        frame["content"] = (title + ". " + body).str.strip()
        logger.info("Combined '%s' and '%s' into 'content'.",
                    config.TITLE_COLUMN, config.TEXT_COLUMN)
    else:
        frame["content"] = frame[config.TEXT_COLUMN].fillna("").astype(str)
        logger.info("No title column found - using '%s' only.", config.TEXT_COLUMN)

    # --- 2. Normalise the labels -----------------------------------------
    frame["label"] = normalise_labels(frame[config.LABEL_COLUMN])
    unrecognised = int(frame["label"].isna().sum())
    if unrecognised:
        logger.warning("Dropping %d row(s) with unrecognised labels.", unrecognised)

    # --- 3. Drop rows that are unusable ----------------------------------
    frame = frame.dropna(subset=["label"])
    frame = frame[frame["content"].str.strip().str.len() > 0]
    after_missing = len(frame)
    logger.info("Removed %d row(s) with missing text or label.",
                initial_rows - after_missing)

    # --- 4. Remove duplicates --------------------------------------------
    # Duplicate articles are a real leakage risk: the identical article could
    # otherwise land in both the training and the test split, inflating the
    # reported accuracy.
    frame = frame.drop_duplicates(subset=["content"], keep="first")
    logger.info("Removed %d duplicate article(s).", after_missing - len(frame))

    result = frame[["content", "label"]].reset_index(drop=True)
    logger.info(
        "Cleaned dataset: %d rows (%s)",
        len(result),
        ", ".join(f"{k}={v}" for k, v in result["label"].value_counts().items()),
    )
    return result


def build_processed_dataset(
    force_rebuild: bool = False,
    preprocessor: Optional[TextPreprocessor] = None,
) -> pd.DataFrame:
    """
    Produce the fully preprocessed dataset, using a cached copy when possible.

    Returns a DataFrame with columns ``content`` (raw), ``clean_text``
    (NLP-preprocessed) and ``label``.

    The cache lives at ``data/processed/cleaned_dataset.csv``.  Pass
    ``force_rebuild=True`` (or ``--rebuild`` on the command line) after
    changing the preprocessing rules.
    """
    config.ensure_directories()
    cache_path = config.PROCESSED_DATASET_PATH

    if cache_path.is_file() and not force_rebuild:
        logger.info("Loading cached processed dataset from %s", cache_path)
        cached = pd.read_csv(cache_path)
        if {"clean_text", "label"}.issubset(cached.columns) and not cached.empty:
            cached["clean_text"] = cached["clean_text"].fillna("").astype(str)
            return cached
        logger.warning("Cache at %s is invalid - rebuilding.", cache_path)

    raw = load_raw_dataset()
    cleaned = clean_dataset(raw)

    preprocessor = preprocessor or TextPreprocessor()
    logger.info("Running NLP preprocessing on %d documents "
                "(this can take a minute)...", len(cleaned))
    cleaned["clean_text"] = preprocessor.clean_many(cleaned["content"])

    # Documents that become (nearly) empty after cleaning carry no signal.
    before = len(cleaned)
    cleaned = cleaned[cleaned["clean_text"].str.len() >= config.MIN_CLEANED_LENGTH]
    cleaned = cleaned.reset_index(drop=True)
    if before != len(cleaned):
        logger.info("Dropped %d document(s) that were empty after cleaning.",
                    before - len(cleaned))

    if cleaned.empty:
        raise DatasetError(
            "Every row was discarded during preprocessing. The dataset may be "
            "in an unexpected format."
        )

    cleaned.to_csv(cache_path, index=False)
    logger.info("Cached processed dataset -> %s (%d rows)", cache_path, len(cleaned))
    return cleaned


def dataset_summary(frame: pd.DataFrame) -> dict:
    """Return a small dictionary of dataset statistics, useful for reports."""
    counts = frame["label"].value_counts().to_dict()
    return {
        "total_rows": int(len(frame)),
        "fake_rows": int(counts.get(config.LABEL_FAKE, 0)),
        "real_rows": int(counts.get(config.LABEL_REAL, 0)),
        "avg_words_per_document": round(
            float(frame["clean_text"].str.split().str.len().mean()), 1
        )
        if "clean_text" in frame.columns
        else None,
    }


def _main() -> None:
    """Command-line entry point: download, inspect or rebuild the dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Dataset utilities.")
    parser.add_argument("--download", action="store_true",
                        help="download the default public dataset")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the processed-dataset cache")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config.ensure_directories()

    if args.download:
        download_default_dataset()

    frame = build_processed_dataset(force_rebuild=args.rebuild)
    print("\nDataset summary:")
    for key, value in dataset_summary(frame).items():
        print(f"  {key:<24}: {value}")


if __name__ == "__main__":  # pragma: no cover
    _main()
