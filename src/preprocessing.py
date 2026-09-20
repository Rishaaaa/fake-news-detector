"""
Reusable NLP preprocessing pipeline.

This is the single most important module for correctness of the project:
the **exact same** cleaning steps are applied when the model is trained and
when a user submits text through the web app.  If the two ever diverged, the
model would see differently-shaped input at prediction time and accuracy would
silently collapse.  Both ``src/train_model.py`` and ``src/predict.py``
therefore import ``TextPreprocessor`` from here rather than re-implementing
any cleaning logic.

Pipeline steps (in order):
    1. Lowercasing
    2. URL removal
    3. HTML tag / HTML entity removal
    4. Special-character removal (keeps letters and spaces)
    5. Number handling (numeric tokens are dropped)
    6. Tokenization
    7. Stopword removal
    8. Lemmatization (default) or stemming
    9. Whitespace normalisation

Note on data leakage
--------------------
Every step here operates on **one document at a time** and uses no information
from any other document.  That matters: a cleaning step that used corpus-wide
statistics (for example "drop words that are rare across the whole dataset")
would leak information from the test set into the training set.  Corpus-level
statistics belong in the TF-IDF vectorizer, which is fitted on the training
split only.
"""

from __future__ import annotations

import html
import logging
import re
from functools import lru_cache
from typing import Iterable, List, Literal

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Compiled regular expressions.
# Compiling once at import time is much faster than recompiling per document,
# which matters when cleaning thousands of news articles.
# --------------------------------------------------------------------------
URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
# Keep only ASCII letters and whitespace.  This removes punctuation, digits,
# emoji and other symbols in one pass.
NON_ALPHA_PATTERN = re.compile(r"[^a-z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")
# Simple, dependency-free tokenizer used as a fallback when NLTK's `punkt`
# tokenizer data has not been downloaded.
SIMPLE_TOKEN_PATTERN = re.compile(r"[a-z]+")

# Minimum token length to keep.  One- and two-letter leftovers ("s", "th")
# carry almost no signal but inflate the vocabulary.
MIN_TOKEN_LENGTH = 3

# Fallback stopword list, used only if the NLTK `stopwords` corpus is missing.
# Keeping this here means the project still runs on a machine with no internet
# access instead of crashing at import time.
FALLBACK_STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are aren't as at be because
    been before being below between both but by can't cannot could couldn't did
    didn't do does doesn't doing don't down during each few for from further had
    hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
    herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
    it's its itself let's me more most mustn't my myself no nor not of off on once
    only or other ought our ours ourselves out over own same shan't she she'd
    she'll she's should shouldn't so some such than that that's the their theirs
    them themselves then there there's these they they'd they'll they're they've
    this those through to too under until up very was wasn't we we'd we'll we're
    we've were weren't what what's when when's where where's which while who who's
    whom why why's with won't would wouldn't you you'd you'll you're you've your
    yours yourself yourselves
    """.split()
)


@lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    """
    Load English stopwords from NLTK, falling back to a built-in list.

    Cached with ``lru_cache`` so the corpus is read from disk only once per
    process, no matter how many documents are cleaned.
    """
    try:
        from nltk.corpus import stopwords

        words = frozenset(stopwords.words("english"))
        logger.debug("Loaded %d NLTK stopwords.", len(words))
        return words
    except Exception:  # LookupError if not downloaded, ImportError if no NLTK
        logger.warning(
            "NLTK stopwords unavailable - using the built-in fallback list. "
            "Run: python -c \"import nltk; nltk.download('stopwords')\""
        )
        return FALLBACK_STOPWORDS


class TextPreprocessor:
    """
    Applies the full cleaning pipeline to raw news text.

    Parameters
    ----------
    mode:
        ``"lemmatize"`` (default) reduces a word to its dictionary form
        ("studies" -> "study").  ``"stem"`` chops suffixes off more crudely
        ("studies" -> "studi").  Lemmatization produces real words, which makes
        the explainability feature much easier to read, so it is the default.
    remove_stopwords:
        Whether to drop common English words that carry little signal.
    min_token_length:
        Tokens shorter than this are discarded.

    Examples
    --------
    >>> TextPreprocessor().clean("BREAKING!! Visit http://x.com <b>NOW</b> 2024")
    'breaking visit'
    """

    def __init__(
        self,
        mode: Literal["lemmatize", "stem"] = "lemmatize",
        remove_stopwords: bool = True,
        min_token_length: int = MIN_TOKEN_LENGTH,
    ) -> None:
        if mode not in ("lemmatize", "stem"):
            raise ValueError("mode must be either 'lemmatize' or 'stem'")

        self.mode = mode
        self.remove_stopwords = remove_stopwords
        self.min_token_length = min_token_length
        self.stopwords = _load_stopwords() if remove_stopwords else frozenset()
        self._normaliser = self._build_normaliser()

    # -- construction helpers ---------------------------------------------
    def _build_normaliser(self):
        """Return the lemmatizer/stemmer callable, or identity if unavailable."""
        if self.mode == "stem":
            try:
                from nltk.stem import PorterStemmer

                return PorterStemmer().stem
            except ImportError:
                logger.warning("NLTK PorterStemmer unavailable - skipping stemming.")
                return lambda token: token

        try:
            from nltk.stem import WordNetLemmatizer

            lemmatizer = WordNetLemmatizer()
            # Force the WordNet corpus to load now so that a missing download
            # is reported once here rather than on every single token.
            lemmatizer.lemmatize("tests")
            return lemmatizer.lemmatize
        except Exception:
            logger.warning(
                "NLTK WordNet unavailable - skipping lemmatization. "
                "Run: python -c \"import nltk; nltk.download('wordnet')\""
            )
            return lambda token: token

    # -- individual pipeline steps ----------------------------------------
    @staticmethod
    def remove_urls(text: str) -> str:
        """Step 2: strip http(s):// and www. links."""
        return URL_PATTERN.sub(" ", text)

    @staticmethod
    def remove_html(text: str) -> str:
        """Step 3: unescape entities (&amp;) then strip <tags>."""
        return HTML_TAG_PATTERN.sub(" ", html.unescape(text))

    @staticmethod
    def remove_special_characters(text: str) -> str:
        """Steps 4 and 5: drop punctuation, symbols and digits."""
        return NON_ALPHA_PATTERN.sub(" ", text)

    def tokenize(self, text: str) -> List[str]:
        """
        Step 6: split cleaned text into word tokens.

        After ``remove_special_characters`` the text contains only lowercase
        letters and spaces, so a regex tokenizer is both sufficient and far
        faster than NLTK's ``word_tokenize``.
        """
        return SIMPLE_TOKEN_PATTERN.findall(text)

    def filter_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Step 7: drop stopwords and very short tokens."""
        return [
            token
            for token in tokens
            if len(token) >= self.min_token_length and token not in self.stopwords
        ]

    def normalise_tokens(self, tokens: Iterable[str]) -> List[str]:
        """Step 8: lemmatize or stem each token."""
        return [self._normaliser(token) for token in tokens]

    # -- public API --------------------------------------------------------
    def clean(self, text: object) -> str:
        """
        Run the complete pipeline and return a space-joined token string.

        Accepts any object so that NaN / None values coming out of a pandas
        column do not crash the pipeline; they simply produce an empty string.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            # Covers float('nan') from pandas and any other unexpected type.
            text = str(text)
            if text.lower() in {"nan", "none", "nat"}:
                return ""

        text = text.lower()                       # 1. lowercase
        text = self.remove_urls(text)             # 2. URLs
        text = self.remove_html(text)             # 3. HTML
        text = self.remove_special_characters(text)  # 4 + 5. symbols, digits

        tokens = self.tokenize(text)              # 6. tokenize
        tokens = self.filter_tokens(tokens)       # 7. stopwords
        tokens = self.normalise_tokens(tokens)    # 8. lemmatize / stem

        # 9. whitespace normalisation: joining on a single space guarantees
        # there are no double or trailing spaces left.
        return WHITESPACE_PATTERN.sub(" ", " ".join(tokens)).strip()

    def clean_many(self, texts: Iterable[object]) -> List[str]:
        """Clean an iterable of documents (convenience wrapper over ``clean``)."""
        return [self.clean(text) for text in texts]

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"TextPreprocessor(mode={self.mode!r}, "
            f"remove_stopwords={self.remove_stopwords}, "
            f"min_token_length={self.min_token_length})"
        )


# --------------------------------------------------------------------------
# Module-level convenience wrapper.
# A single shared instance avoids re-loading the stopword corpus and the
# lemmatizer on every call, which is important inside the Flask request path.
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_default_preprocessor() -> TextPreprocessor:
    """Return a shared, lazily-created ``TextPreprocessor`` instance."""
    return TextPreprocessor()


def preprocess_text(text: object) -> str:
    """Clean a single document using the shared default preprocessor."""
    return get_default_preprocessor().clean(text)


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    logging.basicConfig(level=logging.INFO)
    SAMPLE = (
        "BREAKING!!! Scientists <b>SHOCKED</b> by this 2024 discovery... "
        "Read more at https://example.com/story?id=99 &amp; share it!"
    )
    print("RAW    :", SAMPLE)
    print("CLEANED:", preprocess_text(SAMPLE))
