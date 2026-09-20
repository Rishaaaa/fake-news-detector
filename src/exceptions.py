"""
Custom exception types for the Fake News Detection System.

Defining project-specific exceptions (instead of raising bare ``Exception``)
lets the Flask layer catch exactly the failure it knows how to explain, and
show the user a friendly message while the full technical detail goes to the
server log only.
"""

from __future__ import annotations


class FakeNewsError(Exception):
    """Base class for every error raised by this project."""

    #: Short, safe message that may be shown directly to an end user.
    user_message: str = "An unexpected error occurred. Please try again."

    def __init__(self, message: str, user_message: str | None = None) -> None:
        super().__init__(message)
        if user_message is not None:
            self.user_message = user_message


class DatasetError(FakeNewsError):
    """Raised when the dataset is missing, empty, or has the wrong columns."""

    user_message = "The training dataset could not be loaded."


class ModelNotFoundError(FakeNewsError):
    """Raised when model.pkl or vectorizer.pkl is missing or unreadable."""

    user_message = (
        "The prediction model has not been trained yet. "
        "Please run: python src/train_model.py"
    )


class ValidationError(FakeNewsError):
    """Raised when user-supplied input fails validation."""

    user_message = "The submitted text is not valid."
