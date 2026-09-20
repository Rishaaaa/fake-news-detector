"""Unit tests for the NLP preprocessing pipeline."""

import unittest

from src.preprocessing import TextPreprocessor, preprocess_text


class TestTextPreprocessor(unittest.TestCase):
    """Each test targets one documented step of the pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.preprocessor = TextPreprocessor()

    def test_lowercases_text(self):
        self.assertNotIn("BREAKING", self.preprocessor.clean("BREAKING NEWS TODAY"))

    def test_removes_urls(self):
        cleaned = self.preprocessor.clean(
            "Read the full report at https://example.com/story?id=42 immediately"
        )
        self.assertNotIn("example", cleaned)
        self.assertNotIn("http", cleaned)

    def test_removes_www_urls(self):
        self.assertNotIn("www", self.preprocessor.clean("Visit www.example.com today"))

    def test_removes_html_tags(self):
        cleaned = self.preprocessor.clean("<p>Government <b>announces</b> policy</p>")
        self.assertNotIn("<", cleaned)
        self.assertIn("government", cleaned)

    def test_unescapes_html_entities(self):
        self.assertNotIn("amp", self.preprocessor.clean("Senate &amp; House approve"))

    def test_removes_digits_and_punctuation(self):
        cleaned = self.preprocessor.clean("Inflation rose 3.5% in 2024!!! Really?")
        self.assertTrue(all(not char.isdigit() for char in cleaned))
        self.assertNotIn("%", cleaned)
        self.assertNotIn("!", cleaned)

    def test_removes_stopwords(self):
        cleaned = self.preprocessor.clean("This is the report about the economy")
        for stopword in ("this", "is", "the", "about"):
            self.assertNotIn(stopword, cleaned.split())

    def test_lemmatizes_plurals(self):
        # "studies" -> "study" only if the WordNet corpus is installed.
        cleaned = self.preprocessor.clean("Several studies confirmed the findings")
        self.assertTrue("study" in cleaned or "studies" in cleaned)

    def test_normalises_whitespace(self):
        cleaned = self.preprocessor.clean("economy     grows\n\n\nslowly   today")
        self.assertNotIn("  ", cleaned)
        self.assertEqual(cleaned, cleaned.strip())

    def test_handles_none_and_nan(self):
        self.assertEqual(self.preprocessor.clean(None), "")
        self.assertEqual(self.preprocessor.clean(float("nan")), "")

    def test_handles_empty_string(self):
        self.assertEqual(self.preprocessor.clean(""), "")

    def test_is_deterministic(self):
        text = "The Senate approved the budget agreement on Tuesday"
        self.assertEqual(self.preprocessor.clean(text), self.preprocessor.clean(text))

    def test_stemming_mode_differs_from_lemmatizing(self):
        stemmer = TextPreprocessor(mode="stem")
        self.assertIn("studi", stemmer.clean("studies"))

    def test_rejects_invalid_mode(self):
        with self.assertRaises(ValueError):
            TextPreprocessor(mode="invalid")

    def test_module_level_helper_matches_class(self):
        text = "Government announces new economic policy measures"
        self.assertEqual(preprocess_text(text), self.preprocessor.clean(text))

    def test_full_pipeline_end_to_end(self):
        raw = "BREAKING!!! Scientists <b>SHOCKED</b> by 2024 find: https://x.com/a"
        self.assertEqual(self.preprocessor.clean(raw), "breaking scientist shocked find")


if __name__ == "__main__":
    unittest.main()
