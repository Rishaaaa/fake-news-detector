"""
Training pipeline for the Fake News Detection System.

Run with:
    python src/train_model.py

Pipeline
--------
1. Load the dataset and apply NLP preprocessing (``src/dataset.py``).
2. Split into train / test using a stratified, seeded split.
3. Fit the TF-IDF vectorizer **on the training split only**.
4. Train four classifiers and score each with cross-validation.
5. Select the best model by mean cross-validated F1 (see ``select_best_model``).
6. Evaluate the winner once on the untouched test split.
7. Save model.pkl, vectorizer.pkl, metrics.json and the report charts.

Avoiding data leakage
---------------------
This is the point examiners ask about most, so it is worth being explicit:

* The TF-IDF vectorizer is fitted with ``fit_transform`` on the **training**
  data and only ``transform`` is called on the test data.  Fitting on the full
  dataset would let test-set vocabulary and document frequencies influence the
  features, inflating the score.
* Model **selection** uses cross-validation *inside the training split*.  The
  test split is touched exactly once, at the very end, to report the final
  number.  Choosing the winner by test score would make that score optimistic.
* Duplicate articles are dropped in ``src/dataset.py`` before splitting, so the
  same article cannot appear in both halves.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Allow this file to be run directly as `python src/train_model.py` as well as via
# `python -m src.train_model`. When run directly, Python puts src/ on the path
# instead of the project root, so `from src import ...` would fail. Adding the
# project root here makes both invocations work identically.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src import config, evaluate_model
from src.dataset import build_processed_dataset, dataset_summary
from src.exceptions import DatasetError

logger = logging.getLogger(__name__)


# ==========================================================================
# Model definitions
# ==========================================================================
def build_candidate_models() -> Dict[str, object]:
    """
    Return the four candidate classifiers, each freshly constructed.

    Notes on the choices
    --------------------
    Logistic Regression
        Linear, fast on high-dimensional sparse text, and - crucially for this
        project - it exposes ``predict_proba`` for a confidence score and
        signed ``coef_`` weights for the explainability feature.
    Multinomial Naive Bayes
        The classic text-classification baseline. Very fast, assumes features
        are conditionally independent (rarely true, but it works well anyway).
    Linear SVM
        Usually the strongest linear text classifier. ``LinearSVC`` has no
        ``predict_proba``, so it is wrapped in ``CalibratedClassifierCV``,
        which fits the SVM on internal folds and converts its decision-function
        output into calibrated probabilities.
    Random Forest
        A non-linear ensemble, included for contrast. Tree ensembles generally
        underperform linear models on sparse high-dimensional TF-IDF features,
        and this run is a chance to demonstrate that rather than assert it.
    """
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="liblinear",   # well suited to sparse, high-dimensional data
            random_state=config.RANDOM_STATE,
        ),
        "Naive Bayes": MultinomialNB(alpha=0.1),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=0.5, random_state=config.RANDOM_STATE),
            cv=3,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_split=5,
            n_jobs=-1,
            random_state=config.RANDOM_STATE,
        ),
    }


def build_vectorizer() -> TfidfVectorizer:
    """
    Construct the TF-IDF vectorizer from the values in ``src/config.py``.

    TF-IDF = Term Frequency x Inverse Document Frequency.  A word scores highly
    in a document when it appears often *in that document* but rarely across
    the corpus - which is exactly what makes it a useful discriminator.

    ``ngram_range=(1, 2)`` keeps single words and adjacent word pairs, so
    phrases like "hillary clinton" or "breaking news" become features in their
    own right rather than being lost to the bag-of-words assumption.
    """
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
    )


# ==========================================================================
# Training steps
# ==========================================================================
def split_dataset(
    frame: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Split into train/test with stratification on the label.

    Stratifying keeps the FAKE/REAL ratio identical in both halves, so the test
    score is not distorted by an unlucky split.
    """
    if len(frame) < 50:
        raise DatasetError(
            f"Only {len(frame)} usable rows were found - too few to train a "
            "meaningful model. Check that the dataset loaded correctly."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        frame["clean_text"],
        frame["label"],
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=frame["label"],
    )
    logger.info(
        "Split: %d training rows / %d test rows (test_size=%.0f%%)",
        len(x_train), len(x_test), config.TEST_SIZE * 100,
    )
    return x_train, x_test, y_train, y_test


def vectorize(
    x_train: pd.Series, x_test: pd.Series
) -> Tuple[TfidfVectorizer, csr_matrix, csr_matrix]:
    """
    Fit TF-IDF on the training split and transform both splits.

    ``fit_transform`` on train, plain ``transform`` on test - this asymmetry is
    what prevents test-set information leaking into the features.
    """
    vectorizer = build_vectorizer()
    logger.info("Fitting TF-IDF on %d training documents...", len(x_train))
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)  # transform only - never fit
    logger.info(
        "TF-IDF vocabulary: %d features | training matrix: %s (%.2f%% non-zero)",
        len(vectorizer.vocabulary_),
        train_matrix.shape,
        100 * train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1]),
    )
    return vectorizer, train_matrix, test_matrix


def train_and_evaluate_all(
    train_matrix: csr_matrix,
    y_train: pd.Series,
    test_matrix: csr_matrix,
    y_test: pd.Series,
) -> List[dict]:
    """
    Train every candidate model and collect its scores.

    For each model this records:
      * ``cv_f1_mean`` / ``cv_f1_std`` - cross-validated on the TRAINING split,
        used for model **selection**;
      * accuracy / precision / recall / f1 - on the TEST split, used for
        **reporting** only.
    """
    results: List[dict] = []
    folds = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    for name, model in build_candidate_models().items():
        logger.info("-" * 62)
        logger.info("Training: %s", name)

        # --- cross-validation on the training split (model selection) ------
        cv_start = time.perf_counter()
        cv_scores = cross_val_score(
            model, train_matrix, y_train,
            cv=folds,
            scoring="f1_macro",
            n_jobs=1,  # the models already parallelise internally
        )
        cv_seconds = time.perf_counter() - cv_start

        # --- fit on the full training split, score once on the test split --
        fit_start = time.perf_counter()
        model.fit(train_matrix, y_train)
        train_seconds = time.perf_counter() - fit_start

        y_pred = model.predict(test_matrix)
        metrics = evaluate_model.compute_metrics(y_test, y_pred)

        result = {
            "name": name,
            "cv_f1_mean": float(cv_scores.mean()),
            "cv_f1_std": float(cv_scores.std()),
            "cv_seconds": round(cv_seconds, 2),
            "train_seconds": round(train_seconds, 2),
            "estimator": model,
            "y_pred": y_pred,
            **metrics,
        }
        results.append(result)

        logger.info(
            "  CV F1 (macro): %.4f (+/- %.4f)  |  test accuracy: %.4f  "
            "| test F1(FAKE): %.4f  | fit %.2fs",
            result["cv_f1_mean"], result["cv_f1_std"],
            result["accuracy"], result["f1"], train_seconds,
        )

    return results


def select_best_model(results: List[dict]) -> dict:
    """
    Pick the winning model by **mean cross-validated F1** on the training split.

    Selecting on cross-validation rather than on the test score is what keeps
    the final reported number honest: the test split influences nothing except
    the number that gets printed.  Ties are broken by the faster model.

    If ``config.PRIMARY_MODEL`` is set, that algorithm is used instead.  The
    full comparison table is still generated, and the gap between the pinned
    model and the measured winner is logged, so pinning a model is an explicit,
    documented decision rather than a hidden one.
    """
    measured_best = max(results, key=lambda r: (r["cv_f1_mean"], -r["train_seconds"]))
    logger.info("=" * 62)

    pinned_name = config.PRIMARY_MODEL
    if pinned_name:
        pinned = next(
            (row for row in results if row["name"].lower() == pinned_name.lower()),
            None,
        )
        if pinned is None:
            available = ", ".join(row["name"] for row in results)
            logger.warning(
                "PRIMARY_MODEL=%r does not match any trained model (%s). "
                "Falling back to the measured best.", pinned_name, available,
            )
        else:
            if pinned is not measured_best:
                logger.warning(
                    "Using pinned model %s (CV F1 = %.4f). The measured best was "
                    "%s (CV F1 = %.4f) - a difference of %.4f.",
                    pinned["name"], pinned["cv_f1_mean"],
                    measured_best["name"], measured_best["cv_f1_mean"],
                    measured_best["cv_f1_mean"] - pinned["cv_f1_mean"],
                )
            pinned["selected_by"] = "pinned via config.PRIMARY_MODEL"
            logger.info("Selected model: %s (pinned)", pinned["name"])
            return pinned

    measured_best["selected_by"] = (
        f"highest mean F1 over {config.CV_FOLDS}-fold cross-validation"
    )
    logger.info(
        "Selected model: %s (cross-validated F1 = %.4f +/- %.4f)",
        measured_best["name"], measured_best["cv_f1_mean"], measured_best["cv_f1_std"],
    )
    return measured_best


def extract_top_features(
    model: object, vectorizer: TfidfVectorizer, top_n: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Pull the most influential terms out of a linear model.

    For a linear classifier, each vocabulary term has a signed weight: strongly
    negative weights push towards the first class (FAKE, alphabetically first)
    and strongly positive weights towards REAL.  Returns an empty dict for
    models with no accessible coefficients (e.g. Random Forest).
    """
    coefficients = _linear_coefficients(model)
    if coefficients is None:
        logger.info("%s exposes no linear coefficients - skipping top features.",
                    type(model).__name__)
        return {}

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    classes = list(getattr(model, "classes_", config.CLASS_NAMES))
    order = np.argsort(coefficients)

    return {
        # Most negative coefficients -> evidence for classes_[0].
        classes[0]: [
            (str(feature_names[i]), float(coefficients[i])) for i in order[:top_n]
        ],
        # Most positive coefficients -> evidence for classes_[1].
        classes[1]: [
            (str(feature_names[i]), float(coefficients[i]))
            for i in order[-top_n:][::-1]
        ],
    }


def _linear_coefficients(model: object) -> np.ndarray | None:
    """
    Return the 1-D coefficient vector of a binary linear model, if it has one.

    Handles ``CalibratedClassifierCV``, whose real estimator is hidden one
    level down inside its fitted calibrators.
    """
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()

    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        vectors = [
            np.asarray(cal.estimator.coef_).ravel()
            for cal in calibrated
            if hasattr(getattr(cal, "estimator", None), "coef_")
        ]
        if vectors:
            # Average the per-fold SVMs into one representative weight vector.
            return np.mean(vectors, axis=0)
    return None


def save_artifacts(
    best: dict,
    vectorizer: TfidfVectorizer,
    results: List[dict],
    y_test: pd.Series,
    data_summary: dict,
) -> None:
    """
    Persist the trained model, the vectorizer and a machine-readable report.

    The model and vectorizer are saved as **separate** files on purpose: the
    vectorizer is fitted state (vocabulary + IDF weights) that the Flask app
    needs in order to transform user input into exactly the same feature space
    the model was trained on.
    """
    config.ensure_directories()

    joblib.dump(best["estimator"], config.MODEL_PATH, compress=3)
    joblib.dump(vectorizer, config.VECTORIZER_PATH, compress=3)
    logger.info("Saved model      -> %s (%.1f KB)",
                config.MODEL_PATH, config.MODEL_PATH.stat().st_size / 1024)
    logger.info("Saved vectorizer -> %s (%.1f KB)",
                config.VECTORIZER_PATH, config.VECTORIZER_PATH.stat().st_size / 1024)

    class_metrics = evaluate_model.per_class_metrics(y_test, best["y_pred"])
    confusion = evaluate_model.confusion_matrix(
        y_test, best["y_pred"], labels=list(config.CLASS_NAMES)
    ).tolist()

    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": best["name"],
        "selection_criterion": best.get(
            "selected_by",
            f"highest mean F1 (macro) over {config.CV_FOLDS}-fold "
            "cross-validation on the training split",
        ),
        "positive_class": evaluate_model.POSITIVE_LABEL,
        "class_names": list(config.CLASS_NAMES),
        "test_metrics": {key: best[key] for key in evaluate_model.METRIC_KEYS},
        "cross_validation": {
            "folds": config.CV_FOLDS,
            "f1_mean": best["cv_f1_mean"],
            "f1_std": best["cv_f1_std"],
        },
        "per_class_metrics": class_metrics,
        "confusion_matrix": {
            "labels": list(config.CLASS_NAMES),
            "matrix": confusion,
        },
        "model_comparison": [
            {
                key: row[key]
                for key in (
                    "name", "accuracy", "precision", "recall", "f1",
                    "cv_f1_mean", "cv_f1_std", "train_seconds",
                )
            }
            for row in sorted(results, key=lambda r: r["cv_f1_mean"], reverse=True)
        ],
        "dataset": data_summary,
        "tfidf_parameters": {
            "max_features": config.TFIDF_MAX_FEATURES,
            "ngram_range": list(config.TFIDF_NGRAM_RANGE),
            "min_df": config.TFIDF_MIN_DF,
            "max_df": config.TFIDF_MAX_DF,
            "sublinear_tf": config.TFIDF_SUBLINEAR_TF,
            "vocabulary_size": len(vectorizer.vocabulary_),
        },
        "split": {
            "test_size": config.TEST_SIZE,
            "random_state": config.RANDOM_STATE,
            "test_rows": int(len(y_test)),
        },
        "top_features": extract_top_features(best["estimator"], vectorizer),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    with open(config.METRICS_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    logger.info("Saved metrics    -> %s", config.METRICS_PATH)


# ==========================================================================
# Entry point
# ==========================================================================
def train(force_rebuild: bool = False) -> dict:
    """Run the complete training pipeline and return the selected model's row."""
    config.ensure_directories()
    started = time.perf_counter()

    logger.info("=" * 62)
    logger.info("FAKE NEWS DETECTION - MODEL TRAINING")
    logger.info("=" * 62)

    # 1-2. Load + preprocess.
    frame = build_processed_dataset(force_rebuild=force_rebuild)
    data_summary = dataset_summary(frame)
    logger.info("Dataset: %d rows (FAKE=%d, REAL=%d)",
                data_summary["total_rows"],
                data_summary["fake_rows"], data_summary["real_rows"])

    # 3. Split.
    x_train, x_test, y_train, y_test = split_dataset(frame)

    # 4. Vectorize (fit on train only).
    vectorizer, train_matrix, test_matrix = vectorize(x_train, x_test)

    # 5-6. Train, compare, select.
    results = train_and_evaluate_all(train_matrix, y_train, test_matrix, y_test)
    best = select_best_model(results)

    # 7. Persist artefacts and charts.
    save_artifacts(best, vectorizer, results, y_test, data_summary)
    evaluate_model.generate_all_reports(
        y_test, best["y_pred"], best["name"], results
    )

    print("\nMODEL COMPARISON (held-out test set, positive class = FAKE)\n")
    print(evaluate_model.format_comparison_table(results))
    print(f"\nSelected model: {best['name']}")
    print(f"  Chosen by    : {best.get('selected_by', 'cross-validation')}")
    print(f"  CV F1        : {best['cv_f1_mean']:.4f} "
          f"(+/- {best['cv_f1_std']:.4f}) over {config.CV_FOLDS} folds")
    print(f"  Test accuracy: {best['accuracy']:.4f}")
    print(f"\nTotal time: {time.perf_counter() - started:.1f}s")
    print(f"Artefacts written to: {config.MODELS_DIR} and {config.REPORTS_DIR}")

    return best


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train the fake news detector.")
    parser.add_argument(
        "--rebuild", action="store_true",
        help="rebuild the preprocessed-dataset cache before training",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        train(force_rebuild=args.rebuild)
    except DatasetError as exc:
        logging.error("Training aborted.\n\n%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    _main()
