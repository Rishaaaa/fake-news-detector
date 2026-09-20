"""
Prediction service: turns raw user text into a classification result.

The module exposes a single class, :class:`FakeNewsPredictor`, which loads the
saved model and TF-IDF vectorizer once and then serves many predictions.  The
Flask app holds one shared instance, so the 20 MB of model artefacts are read
from disk once at start-up rather than on every request.

Responsible wording
-------------------
A prediction from this system is a statistical guess based on writing style and
vocabulary patterns learned from one dataset.  It is **not** a fact-check.  The
result dictionary therefore carries an ``explanation`` and a ``disclaimer``
field, and the wording used throughout is "Model prediction: FAKE" rather than
"This article is fake".
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
# Allow this file to be run directly as `python src/predict.py` as well as via
# `python -m src.predict`. When run directly, Python puts src/ on the path
# instead of the project root, so `from src import ...` would fail. Adding the
# project root here makes both invocations work identically.
# ---------------------------------------------------------------------------
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from src import config
from src.exceptions import ModelNotFoundError
from src.preprocessing import TextPreprocessor
from src.validation import validate_news_text

logger = logging.getLogger(__name__)

# Confidence bands used to describe the result in plain language.
HIGH_CONFIDENCE = 0.85
MODERATE_CONFIDENCE = 0.65

# Friendly display names for the estimator classes we might load.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "LogisticRegression": "Logistic Regression",
    "MultinomialNB": "Naive Bayes",
    "CalibratedClassifierCV": "Linear SVM",
    "RandomForestClassifier": "Random Forest",
    "LinearSVC": "Linear SVM",
}


class FakeNewsPredictor:
    """
    Loads the trained artefacts and classifies news text.

    Parameters
    ----------
    eager:
        When True the model is loaded immediately (useful in tests). The
        default is lazy loading, so importing this module never touches disk.

    Examples
    --------
    >>> predictor = FakeNewsPredictor()
    >>> result = predictor.predict("Some news article text ...")
    >>> result["prediction"], round(result["confidence"], 2)
    ('REAL', 0.93)
    """

    def __init__(self, eager: bool = False) -> None:
        self._model: Optional[object] = None
        self._vectorizer: Optional[object] = None
        self._strong_features: Optional[frozenset] = None
        self._preprocessor = TextPreprocessor()
        if eager:
            self.load()

    # -- artefact loading --------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True when both artefact files exist on disk."""
        return config.MODEL_PATH.is_file() and config.VECTORIZER_PATH.is_file()

    def load(self) -> None:
        """
        Load ``model.pkl`` and ``vectorizer.pkl`` into memory.

        Raises
        ------
        ModelNotFoundError
            If either file is missing or cannot be deserialised.
        """
        missing = [
            path
            for path in (config.MODEL_PATH, config.VECTORIZER_PATH)
            if not path.is_file()
        ]
        if missing:
            raise ModelNotFoundError(
                "Missing artefact(s): "
                + ", ".join(str(path) for path in missing)
                + ". Run: python src/train_model.py"
            )

        try:
            self._model = joblib.load(config.MODEL_PATH)
            self._vectorizer = joblib.load(config.VECTORIZER_PATH)
        except Exception as exc:
            raise ModelNotFoundError(
                f"Could not load the saved artefacts: {exc}. "
                "They may be corrupt - retrain with: python src/train_model.py"
            ) from exc

        # The marker set is derived from the model, so it must be recomputed
        # if a different model is ever loaded into this instance.
        self._strong_features = None

        logger.info(
            "Loaded %s with a %d-term vocabulary.",
            self.model_name, len(getattr(self._vectorizer, "vocabulary_", {})),
        )

    def _ensure_loaded(self) -> None:
        """Load the artefacts on first use."""
        if self._model is None or self._vectorizer is None:
            self.load()

    @property
    def model_name(self) -> str:
        """Human-readable name of the loaded estimator."""
        if self._model is None:
            return "Not loaded"
        class_name = type(self._model).__name__
        return MODEL_DISPLAY_NAMES.get(class_name, class_name)

    # -- prediction --------------------------------------------------------
    def predict(self, raw_text: object, explain: bool = True) -> dict:
        """
        Classify a news headline or article.

        Returns a dictionary with these keys:

        ``prediction``      "FAKE" or "REAL"
        ``confidence``      float in [0, 1] - the model's probability for the
                            predicted class
        ``probabilities``   per-class probabilities
        ``model``           display name of the model used
        ``explanation``     plain-language description of the result
        ``confidence_label``  "High" / "Moderate" / "Low"
        ``top_features``    influential terms (empty for non-linear models)
        ``word_count``      number of words kept after preprocessing
        ``low_signal``      True when too little text survived preprocessing
        ``out_of_domain``   True when the text does not resemble the training
                            corpus (see ``domain_ratio``)
        ``domain_ratio``    fraction of the document's terms that the model
                            considers influential
        ``disclaimer``      the standard responsible-use notice

        Raises
        ------
        ValidationError
            If the input text fails validation.
        ModelNotFoundError
            If the model artefacts are missing.
        """
        text = validate_news_text(raw_text)
        self._ensure_loaded()

        cleaned = self._preprocessor.clean(text)
        word_count = len(cleaned.split())

        # After stopword removal a short sentence can collapse to only two or
        # three content words. The model still returns a confident-looking
        # probability, but it is really reporting the average association of
        # those isolated words rather than analysing an article. The prediction
        # is still returned - flagged - so the UI can warn instead of misleading.
        low_signal = word_count < config.MIN_SIGNAL_WORDS

        features = self._vectorizer.transform([cleaned])
        predicted = str(self._model.predict(features)[0])
        probabilities = self._predict_proba(features)
        confidence = probabilities.get(predicted, 0.0)

        ratio = self.domain_ratio(features)
        out_of_domain = ratio < config.DOMAIN_RATIO_THRESHOLD

        result = {
            "prediction": predicted,
            "confidence": round(float(confidence), 4),
            "out_of_domain": out_of_domain,
            "domain_ratio": round(float(ratio), 4),
            "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()},
            "model": self.model_name,
            "confidence_label": self._confidence_label(confidence),
            "word_count": word_count,
            "low_signal": low_signal,
            "disclaimer": config.DISCLAIMER,
        }
        result["explanation"] = self._build_explanation(result)
        result["top_features"] = (
            self.explain(features, predicted) if explain else []
        )
        return result

    def _predict_proba(self, features: csr_matrix) -> Dict[str, float]:
        """
        Return per-class probabilities.

        Every model in this project supports ``predict_proba`` (the SVM is
        wrapped in ``CalibratedClassifierCV`` precisely so that it does). The
        ``decision_function`` branch is a safety net in case a model without
        probability support is ever swapped in.
        """
        classes = [str(c) for c in self._model.classes_]

        if hasattr(self._model, "predict_proba"):
            values = self._model.predict_proba(features)[0]
            return dict(zip(classes, (float(v) for v in values)))

        if hasattr(self._model, "decision_function"):
            # Squash the signed margin through a logistic function to get a
            # pseudo-probability. Clearly an approximation, not a calibrated
            # probability, but better than reporting nothing.
            margin = float(np.ravel(self._model.decision_function(features))[0])
            positive = 1.0 / (1.0 + np.exp(-margin))
            return {classes[0]: 1.0 - positive, classes[1]: positive}

        # Last resort: no confidence information available at all.
        predicted = str(self._model.predict(features)[0])
        return {label: (1.0 if label == predicted else 0.0) for label in classes}

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        """Bucket a probability into High / Moderate / Low."""
        if confidence >= HIGH_CONFIDENCE:
            return "High"
        if confidence >= MODERATE_CONFIDENCE:
            return "Moderate"
        return "Low"

    def _build_explanation(self, result: dict) -> str:
        """Compose the plain-language sentence shown under the result."""
        label = result["prediction"]
        percentage = result["confidence"] * 100
        descriptor = result["confidence_label"].lower()

        if label == config.LABEL_FAKE:
            meaning = (
                "the wording and vocabulary of this text resemble the articles "
                "labelled FAKE in the training data"
            )
        else:
            meaning = (
                "the wording and vocabulary of this text resemble the articles "
                "labelled REAL in the training data"
            )

        # A low-signal result leads with the caveat. Stating the probability
        # first and qualifying it afterwards reads as a confident answer with a
        # footnote, which is exactly the impression to avoid here.
        if result["low_signal"]:
            return (
                f"Model prediction: {label} - but this result is not reliable. "
                f"Only {result['word_count']} meaningful word(s) remained after "
                f"preprocessing, below the {config.MIN_SIGNAL_WORDS} needed for a "
                f"dependable classification. With so little text the model is "
                f"reporting the average association of a few isolated words rather "
                f"than analysing an article, so the {percentage:.1f}% figure "
                f"overstates how much it actually knows. Paste a full article "
                f"(one or more paragraphs) for a meaningful result."
            )

        # Out-of-domain text is classified anyway, usually as FAKE, because its
        # vocabulary is unfamiliar rather than because it looks deceptive.
        if result.get("out_of_domain"):
            return (
                f"Model prediction: {label} - but this text does not look like the "
                f"news articles the model was trained on. Very few of its words "
                f"carry any learned weight, so the {percentage:.1f}% figure mostly "
                f"reflects unfamiliar vocabulary rather than a judgement about the "
                f"writing. Out-of-domain text is usually classified FAKE by default. "
                f"Treat this result as uninformative."
            )

        sentence = (
            f"Model prediction: {label}. The model assigns {percentage:.1f}% "
            f"probability to this class ({descriptor} confidence), meaning "
            f"{meaning}."
        )

        if result["confidence_label"] == "Low":
            sentence += (
                " The two classes scored closely, so this result should be "
                "treated as inconclusive."
            )
        return sentence

    # -- explainability ----------------------------------------------------
    def explain(
        self, features: csr_matrix, predicted_label: str
    ) -> List[Dict[str, float]]:
        """
        Identify which terms pushed the model towards its prediction.

        How it works
        ------------
        For a linear model the decision is a weighted sum::

            score = w1*x1 + w2*x2 + ... + bias

        where ``x`` is the TF-IDF value of a term in *this* document and ``w``
        is the weight the model learned for that term. The product ``w * x`` is
        that term's contribution to the decision. Sorting the terms present in
        the document by contribution gives a direct, honest answer to "which
        words drove this result?".

        This is a description of the model's learned associations, **not**
        evidence about whether the article is factually true.

        Returns an empty list for models without accessible coefficients
        (Random Forest), which the UI handles by hiding the panel.
        """
        coefficients = self._get_coefficients()
        if coefficients is None:
            return []

        feature_names = self._vectorizer.get_feature_names_out()
        document = features.tocoo()

        # Contribution of every term that actually appears in this document.
        contributions: List[Tuple[str, float]] = [
            (str(feature_names[col]), float(coefficients[col] * value))
            for col, value in zip(document.col, document.data)
        ]
        if not contributions:
            return []

        classes = [str(c) for c in self._model.classes_]
        # Positive coefficients point at classes_[1]; negative at classes_[0].
        towards_second_class = predicted_label == classes[1]
        contributions.sort(key=lambda item: item[1], reverse=towards_second_class)

        top: List[Dict[str, float]] = []
        for term, contribution in contributions[: config.TOP_FEATURES_COUNT]:
            # Only keep terms that genuinely support the predicted class.
            if (contribution > 0) == towards_second_class and contribution != 0:
                top.append(
                    {
                        "term": term,
                        "contribution": round(abs(contribution), 5),
                        "supports": predicted_label,
                    }
                )

        # Normalise to 0-1 so the UI can draw comparable bars.
        if top:
            largest = max(item["contribution"] for item in top)
            for item in top:
                item["weight"] = round(item["contribution"] / largest, 4) if largest else 0.0
        return top

    # -- out-of-domain detection ------------------------------------------
    def _strong_feature_indices(self) -> Optional[frozenset]:
        """
        Return the indices of the model's most influential features.

        These are derived from the loaded model's own coefficients, so no extra
        artefact has to be saved and the set automatically matches whichever
        model was selected during training. Cached after the first call.
        """
        if self._strong_features is not None:
            return self._strong_features

        coefficients = self._get_coefficients()
        if coefficients is None:
            # Non-linear model (e.g. Random Forest): fall back to impurity
            # importances if available, otherwise disable the check.
            importances = getattr(self._model, "feature_importances_", None)
            if importances is None:
                self._strong_features = frozenset()
                return self._strong_features
            coefficients = np.asarray(importances)

        ranked = np.argsort(np.abs(coefficients))[-config.DOMAIN_MARKER_TOP_N:]
        self._strong_features = frozenset(int(i) for i in ranked)
        return self._strong_features

    def domain_ratio(self, features: csr_matrix) -> float:
        """
        Fraction of this document's terms that are influential for the model.

        A political news article typically scores around 0.16-0.24. Text from
        outside the training domain scores lower because almost none of its
        vocabulary carries learned weight. Returns 1.0 (i.e. "no concern") when
        the check cannot be performed, so a missing signal never produces a
        spurious warning.
        """
        strong = self._strong_feature_indices()
        if not strong:
            return 1.0

        present = features.tocsr().indices
        if len(present) == 0:
            return 0.0
        return sum(1 for index in present if int(index) in strong) / len(present)

    def _get_coefficients(self) -> Optional[np.ndarray]:
        """Return the flat coefficient vector of the loaded linear model."""
        if hasattr(self._model, "coef_"):
            return np.asarray(self._model.coef_).ravel()

        # CalibratedClassifierCV hides the real estimator one level down.
        calibrated = getattr(self._model, "calibrated_classifiers_", None)
        if calibrated:
            vectors = [
                np.asarray(cal.estimator.coef_).ravel()
                for cal in calibrated
                if hasattr(getattr(cal, "estimator", None), "coef_")
            ]
            if vectors:
                return np.mean(vectors, axis=0)
        return None


# --------------------------------------------------------------------------
# Shared instance used by the Flask application.
# --------------------------------------------------------------------------
_shared_predictor: Optional[FakeNewsPredictor] = None


def get_predictor() -> FakeNewsPredictor:
    """Return the process-wide predictor, creating it on first call."""
    global _shared_predictor
    if _shared_predictor is None:
        _shared_predictor = FakeNewsPredictor()
    return _shared_predictor


def predict_news(text: object) -> dict:
    """Convenience wrapper: classify ``text`` with the shared predictor."""
    return get_predictor().predict(text)


def _main() -> None:  # pragma: no cover - manual CLI
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Classify news text.")
    parser.add_argument("text", nargs="*", help="the news text to classify")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    text = " ".join(args.text) or input("Enter news text: ")
    print(json.dumps(predict_news(text), indent=2))


if __name__ == "__main__":  # pragma: no cover
    _main()
