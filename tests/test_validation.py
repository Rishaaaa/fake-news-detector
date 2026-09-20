"""Unit tests for user input validation."""

import unittest

from src import config
from src.exceptions import ValidationError
from src.validation import sanitise_text, validate_news_text

VALID_TEXT = "The Senate approved the federal budget agreement on Tuesday evening."


class TestSanitiseText(unittest.TestCase):
    def test_strips_surrounding_whitespace(self):
        self.assertEqual(sanitise_text("   hello   "), "hello")

    def test_removes_control_characters(self):
        self.assertEqual(sanitise_text("he\x00l\x07lo"), "hello")

    def test_keeps_newlines_and_tabs(self):
        self.assertIn("\n", sanitise_text("line one\nline two"))

    def test_collapses_excess_blank_lines(self):
        self.assertNotIn("\n\n\n", sanitise_text("a\n\n\n\n\nb"))

    def test_handles_none(self):
        self.assertEqual(sanitise_text(None), "")


class TestValidateNewsText(unittest.TestCase):
    def test_accepts_valid_text(self):
        self.assertEqual(validate_news_text(VALID_TEXT), VALID_TEXT)

    def test_rejects_empty_string(self):
        with self.assertRaises(ValidationError):
            validate_news_text("")

    def test_rejects_whitespace_only(self):
        with self.assertRaises(ValidationError):
            validate_news_text("      \n\n   ")

    def test_rejects_none(self):
        with self.assertRaises(ValidationError):
            validate_news_text(None)

    def test_rejects_text_below_minimum_length(self):
        with self.assertRaises(ValidationError):
            validate_news_text("too short")

    def test_rejects_text_above_maximum_length(self):
        with self.assertRaises(ValidationError):
            validate_news_text("word " * (config.MAX_INPUT_LENGTH // 2))

    def test_rejects_text_with_no_letters(self):
        with self.assertRaises(ValidationError):
            validate_news_text("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def test_error_carries_a_safe_user_message(self):
        try:
            validate_news_text("")
        except ValidationError as exc:
            self.assertTrue(exc.user_message)
            # The user-facing message must not leak internals.
            self.assertNotIn("Traceback", exc.user_message)


if __name__ == "__main__":
    unittest.main()
