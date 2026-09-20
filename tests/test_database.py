"""
Unit tests for the SQLite history layer.

Each test runs against a temporary database file so the developer's real
history in database/app.db is never touched.
"""

import tempfile
import unittest
from pathlib import Path

from src import config, database


class TestDatabase(unittest.TestCase):
    def setUp(self):
        # Point the module at a throwaway database for the duration of the test.
        self._temp_dir = tempfile.TemporaryDirectory()
        self._original_path = config.DATABASE_PATH
        config.DATABASE_PATH = Path(self._temp_dir.name) / "test.db"
        database.init_database()

    def tearDown(self):
        config.DATABASE_PATH = self._original_path
        self._temp_dir.cleanup()

    def _insert(self, prediction="FAKE", confidence=0.9, text="sample news text"):
        return database.save_prediction(
            input_text=text,
            prediction=prediction,
            confidence=confidence,
            model_name="Linear SVM",
            word_count=12,
        )

    def test_save_returns_row_id(self):
        self.assertGreater(self._insert(), 0)

    def test_saved_row_is_retrievable(self):
        self._insert(text="the senate approved the budget")
        rows = database.fetch_predictions()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "FAKE")
        self.assertEqual(rows[0]["model_name"], "Linear SVM")

    def test_long_text_is_truncated(self):
        self._insert(text="x" * 5000)
        stored = database.fetch_predictions()[0]["input_text"]
        self.assertLessEqual(len(stored), database.STORED_TEXT_LIMIT + 3)

    def test_count_matches_inserts(self):
        for _ in range(5):
            self._insert()
        self.assertEqual(database.count_predictions(), 5)

    def test_filter_by_label(self):
        self._insert(prediction="FAKE")
        self._insert(prediction="REAL")
        self._insert(prediction="REAL")
        self.assertEqual(database.count_predictions(label_filter="REAL"), 2)
        self.assertEqual(database.count_predictions(label_filter="FAKE"), 1)

    def test_search_is_case_insensitive(self):
        self._insert(text="The Senate Approved The Budget")
        self.assertEqual(database.count_predictions(search="senate"), 1)
        self.assertEqual(database.count_predictions(search="SENATE"), 1)

    def test_search_with_sql_metacharacters_is_safe(self):
        """A classic injection string must be treated as literal text."""
        self._insert(text="normal article")
        malicious = "'; DROP TABLE predictions; --"
        self.assertEqual(database.count_predictions(search=malicious), 0)
        # The table must still exist and still hold the original row.
        self.assertEqual(database.count_predictions(), 1)

    def test_invalid_sort_column_falls_back_safely(self):
        self._insert()
        rows = database.fetch_predictions(sort_by="id; DROP TABLE predictions")
        self.assertEqual(len(rows), 1)
        self.assertEqual(database.count_predictions(), 1)

    def test_pagination(self):
        for index in range(10):
            self._insert(text=f"article number {index}")
        self.assertEqual(len(database.fetch_predictions(limit=4, offset=0)), 4)
        self.assertEqual(len(database.fetch_predictions(limit=4, offset=8)), 2)

    def test_delete_single_row(self):
        row_id = self._insert()
        self.assertTrue(database.delete_prediction(row_id))
        self.assertEqual(database.count_predictions(), 0)

    def test_delete_missing_row_returns_false(self):
        self.assertFalse(database.delete_prediction(9999))

    def test_clear_history_removes_everything(self):
        for _ in range(3):
            self._insert()
        self.assertEqual(database.clear_history(), 3)
        self.assertEqual(database.count_predictions(), 0)

    def test_statistics_are_consistent(self):
        self._insert(prediction="FAKE", confidence=0.8)
        self._insert(prediction="REAL", confidence=0.6)
        stats = database.get_statistics()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["fake_count"], 1)
        self.assertEqual(stats["real_count"], 1)
        self.assertAlmostEqual(stats["avg_confidence"], 0.7, places=3)
        self.assertEqual(stats["fake_percentage"], 50.0)

    def test_statistics_on_empty_database(self):
        stats = database.get_statistics()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["fake_percentage"], 0.0)


if __name__ == "__main__":
    unittest.main()
