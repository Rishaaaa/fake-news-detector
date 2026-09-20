"""
Integration tests for the Flask routes and JSON API.

These use Flask's built-in test client, so no server needs to be running.
Tests that require a trained model are skipped automatically when
models/model.pkl is absent, so the suite still passes on a clean checkout.
"""

import json
import tempfile
import unittest
from pathlib import Path

from src import config, database

VALID_TEXT = (
    "Washington (Reuters) - The Senate approved a bipartisan spending agreement "
    "on Tuesday, with lawmakers backing the measure after weeks of negotiation."
)


def model_is_trained() -> bool:
    return config.MODEL_PATH.is_file() and config.VECTORIZER_PATH.is_file()


class FlaskTestCase(unittest.TestCase):
    def setUp(self):
        import app as app_module

        self._temp_dir = tempfile.TemporaryDirectory()
        self._original_path = config.DATABASE_PATH
        config.DATABASE_PATH = Path(self._temp_dir.name) / "test.db"
        database.init_database()

        app_module.app.config["TESTING"] = True
        self.client = app_module.app.test_client()

    def tearDown(self):
        config.DATABASE_PATH = self._original_path
        self._temp_dir.cleanup()


class TestPages(FlaskTestCase):
    def test_index_returns_200(self):
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_dashboard_returns_200(self):
        self.assertEqual(self.client.get("/dashboard").status_code, 200)

    def test_history_returns_200(self):
        self.assertEqual(self.client.get("/history").status_code, 200)

    def test_about_returns_200(self):
        self.assertEqual(self.client.get("/about").status_code, 200)

    def test_unknown_page_returns_404(self):
        self.assertEqual(self.client.get("/no-such-page").status_code, 404)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("model_ready", response.get_json())

    def test_report_path_traversal_is_blocked(self):
        response = self.client.get("/reports/../models/model.pkl")
        self.assertIn(response.status_code, (400, 404))


class TestApiValidation(FlaskTestCase):
    """These run without a trained model - validation happens first."""

    def test_malformed_json_returns_400(self):
        response = self.client.post(
            "/api/predict", data="{not json",
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_missing_text_field_returns_400(self):
        response = self.client.post("/api/predict", json={"foo": "bar"})
        self.assertEqual(response.status_code, 400)

    def test_empty_text_returns_400(self):
        response = self.client.post("/api/predict", json={"text": ""})
        self.assertEqual(response.status_code, 400)

    def test_short_text_returns_400(self):
        response = self.client.post("/api/predict", json={"text": "hi"})
        self.assertEqual(response.status_code, 400)

    def test_overlong_text_returns_400(self):
        long_text = "word " * (config.MAX_INPUT_LENGTH // 2)
        response = self.client.post("/api/predict", json={"text": long_text})
        self.assertEqual(response.status_code, 400)

    def test_error_response_never_leaks_a_traceback(self):
        response = self.client.post("/api/predict", json={"text": ""})
        self.assertNotIn("Traceback", response.get_data(as_text=True))


@unittest.skipUnless(model_is_trained(), "requires a trained model")
class TestApiPrediction(FlaskTestCase):
    def test_valid_prediction_returns_expected_shape(self):
        response = self.client.post("/api/predict", json={"text": VALID_TEXT})
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()
        for key in ("prediction", "confidence", "model", "disclaimer"):
            self.assertIn(key, payload)

        self.assertIn(payload["prediction"], config.CLASS_NAMES)
        self.assertGreaterEqual(payload["confidence"], 0.0)
        self.assertLessEqual(payload["confidence"], 1.0)

    def test_probabilities_sum_to_one(self):
        payload = self.client.post(
            "/api/predict", json={"text": VALID_TEXT}
        ).get_json()
        self.assertAlmostEqual(sum(payload["probabilities"].values()), 1.0, places=3)

    def test_prediction_is_saved_to_history(self):
        self.client.post("/api/predict", json={"text": VALID_TEXT})
        self.assertEqual(database.count_predictions(), 1)

    def test_short_input_is_flagged_as_low_signal(self):
        """A one-line claim must be flagged, not reported confidently."""
        payload = self.client.post(
            "/api/predict", json={"text": "Apple is good for your health today"}
        ).get_json()
        self.assertTrue(payload["word_count"] < config.MIN_SIGNAL_WORDS)
        self.assertIn("not reliable", payload["explanation"])

    def test_full_article_is_not_flagged_as_low_signal(self):
        """A real article comfortably clears the threshold."""
        article = VALID_TEXT + (
            " Senator Patty Murray, who helped negotiate the agreement, said it "
            "would provide certainty for federal agencies through the year. A "
            "senior administration official said the president intended to sign "
            "the legislation once it reaches his desk later this week."
        )
        payload = self.client.post(
            "/api/predict", json={"text": article}
        ).get_json()
        self.assertGreaterEqual(payload["word_count"], config.MIN_SIGNAL_WORDS)
        self.assertNotIn("not reliable", payload["explanation"])

    def test_low_signal_warning_is_shown_on_the_result_page(self):
        body = self.client.post(
            "/predict", data={"news_text": "Apple is good for your health today"}
        ).get_data(as_text=True)
        self.assertIn("This result is not reliable", body)
        self.assertIn("is-unreliable", body)

    def test_out_of_domain_text_is_flagged(self):
        """A recipe is not news - the domain check must catch it."""
        recipe = (
            "Preheat the oven to 180 degrees and grease a round cake tin. Cream "
            "the butter and sugar together until pale, then beat in the eggs one "
            "at a time. Fold in the flour and bake for 35 minutes."
        )
        payload = self.client.post("/api/predict", json={"text": recipe}).get_json()
        self.assertTrue(payload["out_of_domain"])
        self.assertLess(payload["domain_ratio"], config.DOMAIN_RATIO_THRESHOLD)
        self.assertIn("does not look like", payload["explanation"])

    def test_political_news_is_not_flagged_as_out_of_domain(self):
        """In-domain news must not trigger a spurious domain warning."""
        payload = self.client.post(
            "/api/predict", json={"text": VALID_TEXT}
        ).get_json()
        self.assertFalse(payload["out_of_domain"])
        self.assertGreaterEqual(
            payload["domain_ratio"], config.DOMAIN_RATIO_THRESHOLD
        )

    def test_out_of_domain_warning_is_shown_on_the_result_page(self):
        recipe = (
            "Preheat the oven to 180 degrees and grease a round cake tin. Cream "
            "the butter and sugar together until pale, then beat in the eggs one "
            "at a time. Fold in the flour and bake for 35 minutes."
        )
        body = self.client.post(
            "/predict", data={"news_text": recipe}
        ).get_data(as_text=True)
        self.assertIn("look like news the model knows", body)
        self.assertIn("is-unreliable", body)

    def test_form_submission_renders_result(self):
        response = self.client.post("/predict", data={"news_text": VALID_TEXT})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Model prediction", response.get_data(as_text=True))

    def test_user_text_is_escaped_in_the_response(self):
        payload = f'<script>alert("xss")</script> {VALID_TEXT}'
        body = self.client.post(
            "/predict", data={"news_text": payload}
        ).get_data(as_text=True)
        self.assertNotIn('<script>alert("xss")</script>', body)
        self.assertIn("&lt;script&gt;", body)


if __name__ == "__main__":
    unittest.main()
