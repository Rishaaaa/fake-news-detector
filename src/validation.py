"""
Input validation and sanitisation for user-supplied news text.

Validation lives in its own module so the web form and the JSON API enforce
exactly the same rules, and so the limits can be unit-tested without starting
a server.

Note on escaping: this module does **not** HTML-escape the text.  Escaping is
an output concern, and Jinja2 autoescaping already handles it when the text is
rendered in a template.  Escaping on input would corrupt the stored text and
produce double-escaped output ("&amp;amp;").
"""

from __future__ import annotations

import re
import unicodedata

from src import config
from src.exceptions import ValidationError

# Matches C0/C1 control characters except tab, newline and carriage return,
# which are legitimate in pasted articles.
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def sanitise_text(raw: object) -> str:
    """
    Normalise raw input into a clean unicode string.

    Steps: coerce to ``str``, apply NFKC unicode normalisation (so visually
    identical characters compare equal), strip control characters that could
    corrupt log files or terminal output, and collapse runs of blank lines.
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)

    text = unicodedata.normalize("NFKC", raw)
    text = CONTROL_CHAR_PATTERN.sub("", text)
    text = EXCESS_NEWLINES_PATTERN.sub("\n\n", text)
    return text.strip()


def validate_news_text(raw: object) -> str:
    """
    Sanitise and validate text submitted for classification.

    Returns
    -------
    str
        The cleaned text, safe to preprocess and store.

    Raises
    ------
    ValidationError
        If the text is empty, too short to classify, too long to accept, or
        contains no alphabetic content at all.
    """
    text = sanitise_text(raw)

    if not text:
        raise ValidationError(
            "Empty input submitted.",
            user_message="Please enter a news headline or article to analyse.",
        )

    if len(text) > config.MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input of {len(text)} characters exceeds the limit.",
            user_message=(
                f"That text is too long ({len(text):,} characters). "
                f"Please submit at most {config.MAX_INPUT_LENGTH:,} characters."
            ),
        )

    if len(text) < config.MIN_INPUT_LENGTH:
        raise ValidationError(
            f"Input of {len(text)} characters is below the minimum.",
            user_message=(
                f"That text is too short ({len(text)} characters). "
                f"Please enter at least {config.MIN_INPUT_LENGTH} characters "
                "so the model has enough context."
            ),
        )

    # Reject input such as "!!!!!!!!!!!!!!!!!!!!" that survives the length
    # check but contains nothing the model can actually use.
    if not any(char.isalpha() for char in text):
        raise ValidationError(
            "Input contains no alphabetic characters.",
            user_message="Please enter readable text containing words.",
        )

    return text
