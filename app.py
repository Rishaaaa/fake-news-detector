"""
Flask web application for the Fake News Detection System.

Run with:
    python app.py

Routes
------
GET  /                    prediction form (main dashboard)
POST /predict             classify text submitted from the form
GET  /dashboard           analytics dashboard with charts
GET  /history             prediction history (rendered from browser storage)
GET  /about               project information
POST /api/predict         JSON prediction API
GET  /api/metrics         JSON model evaluation metrics
GET  /reports/<filename>  serve a generated report chart

Security posture
----------------
* ``SECRET_KEY`` is read from the environment; the fallback is clearly marked
  as development-only.
* ``MAX_CONTENT_LENGTH`` rejects oversized request bodies before they are read
  into memory.
* All user text passes through ``src.validation`` before use.
* Prediction history is stored in the visitor's browser, never server-side,
  so the application keeps no user data at rest.
* Jinja2 autoescaping (on by default for .html) escapes user text on output.
* Internal exception details are logged server-side and never sent to the
  client; users get a generic message plus an error id.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src import config
from src.evaluate_model import load_saved_metrics
from src.exceptions import ModelNotFoundError, ValidationError
from src.predict import get_predictor

# --------------------------------------------------------------------------
# Application setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fake_news_app")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=config.SECRET_KEY,
    MAX_CONTENT_LENGTH=config.MAX_REQUEST_BYTES,
    JSON_SORT_KEYS=False,
    TEMPLATES_AUTO_RELOAD=config.FLASK_DEBUG,
)

# Only these files may be served from reports/ - stops any path traversal.
ALLOWED_REPORT_FILES = frozenset(
    {"confusion_matrix.png", "model_comparison.png", "performance_metrics.png"}
)


@app.context_processor
def inject_globals() -> dict:
    """Make shared values available to every template without passing them."""
    return {
        "disclaimer": config.DISCLAIMER,
        "max_input_length": config.MAX_INPUT_LENGTH,
        "min_input_length": config.MIN_INPUT_LENGTH,
        "min_signal_words": config.MIN_SIGNAL_WORDS,
        "app_title": "Fake News Detection System",
    }


def _model_is_available() -> bool:
    """True when the trained artefacts exist on disk."""
    return get_predictor().is_ready


# ==========================================================================
# Page routes
# ==========================================================================
@app.route("/")
def index():
    """Render the main prediction form."""
    return render_template("index.html", model_ready=_model_is_available())


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle a form submission from the main page.

    Validation failures are shown on the form itself (with the text preserved)
    rather than on an error page, so the user can simply correct and resubmit.
    """
    raw_text = request.form.get("news_text", "")

    try:
        result = get_predictor().predict(raw_text)
    except ValidationError as exc:
        flash(exc.user_message, "warning")
        return (
            render_template(
                "index.html",
                model_ready=_model_is_available(),
                submitted_text=raw_text,
            ),
            400,
        )
    except ModelNotFoundError as exc:
        logger.error("Prediction failed - model unavailable: %s", exc)
        flash(exc.user_message, "danger")
        return (
            render_template(
                "index.html", model_ready=False, submitted_text=raw_text,
            ),
            503,
        )

    # History is written to the visitor's own browser by result.html, not to
    # the server - see static/js/history-store.js.
    return render_template("result.html", result=result, submitted_text=raw_text)


@app.route("/dashboard")
def dashboard():
    """Analytics dashboard: usage statistics plus model evaluation metrics."""
    return render_template(
        "dashboard.html",
        metrics=load_saved_metrics(),
        model_ready=_model_is_available(),
        report_images=[
            name for name in sorted(ALLOWED_REPORT_FILES)
            if (config.REPORTS_DIR / name).is_file()
        ],
    )


@app.route("/history")
def history():
    """
    Render the history page shell.

    The page has no server-side data: entries are read from the visitor's
    browser storage by static/js/history-store.js and rendered client-side.
    """
    return render_template("history.html")


@app.route("/about")
def about():
    """Static information page about the project and its limitations."""
    return render_template(
        "about.html",
        metrics=load_saved_metrics(),
        model_ready=_model_is_available(),
    )


@app.route("/reports/<path:filename>")
def report_image(filename: str):
    """
    Serve a generated chart from ``reports/``.

    Only the three known report filenames are allowed. Because the name is
    checked against a fixed set before it reaches the filesystem, "../"
    traversal attempts cannot escape the directory.
    """
    if filename not in ALLOWED_REPORT_FILES:
        abort(404)
    if not (config.REPORTS_DIR / filename).is_file():
        abort(404)
    return send_from_directory(config.REPORTS_DIR, filename)


# ==========================================================================
# JSON API
# ==========================================================================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Classify news text supplied as JSON.

    Request
    -------
    POST /api/predict
    Content-Type: application/json
    {"text": "news article text"}

    Response (200)
    --------------
    {"prediction": "FAKE", "confidence": 0.91, "model": "Linear SVM", ...}

    Errors
    ------
    400  invalid JSON, missing "text", or text failing validation
    413  request body larger than the configured limit
    503  the model has not been trained yet
    500  unexpected server error
    """
    # ``silent=True`` makes Flask return None on malformed JSON instead of
    # raising, so a bad body becomes a clean 400 rather than a 500.
    payload = request.get_json(silent=True)
    if payload is None:
        return _api_error(
            "Request body must be valid JSON with Content-Type: application/json.",
            400,
        )
    if not isinstance(payload, dict) or "text" not in payload:
        return _api_error("Request JSON must contain a 'text' field.", 400)

    try:
        result = get_predictor().predict(payload["text"])
    except ValidationError as exc:
        return _api_error(exc.user_message, 400)
    except ModelNotFoundError as exc:
        logger.error("API prediction failed - model unavailable: %s", exc)
        return _api_error(exc.user_message, 503)

    return jsonify(
        {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "model": result["model"],
            "probabilities": result["probabilities"],
            "confidence_label": result["confidence_label"],
            "low_signal": result["low_signal"],
            "out_of_domain": result["out_of_domain"],
            "domain_ratio": result["domain_ratio"],
            "explanation": result["explanation"],
            "top_features": result["top_features"],
            "word_count": result["word_count"],
            "disclaimer": result["disclaimer"],
        }
    )


@app.route("/api/metrics")
def api_metrics():
    """Return the saved model evaluation metrics as JSON."""
    metrics = load_saved_metrics()
    if metrics is None:
        return _api_error(
            "No evaluation metrics found. Run: python src/train_model.py", 503
        )
    return jsonify(metrics)


@app.route("/api/health")
def api_health():
    """Lightweight readiness probe."""
    return jsonify(
        {
            "status": "ok" if _model_is_available() else "model_not_trained",
            "model_ready": _model_is_available(),
        }
    )


# ==========================================================================
# Helpers and error handling
# ==========================================================================
def _api_error(message: str, status: int, error_id: str | None = None):
    """Build a consistent JSON error response."""
    body = {"error": message, "status": status}
    if error_id:
        body["error_id"] = error_id
    return jsonify(body), status


def _wants_json() -> bool:
    """True when the client is calling the API rather than browsing."""
    return request.path.startswith("/api/") or request.is_json


@app.errorhandler(400)
def handle_bad_request(error):
    if _wants_json():
        return _api_error("Bad request.", 400)
    return render_template("error.html", code=400,
                           message="The request could not be understood."), 400


@app.errorhandler(404)
def handle_not_found(error):
    if _wants_json():
        return _api_error("Endpoint not found.", 404)
    return render_template("error.html", code=404,
                           message="That page does not exist."), 404


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(error):
    message = (
        f"The submitted content is too large. The limit is "
        f"{config.MAX_REQUEST_BYTES // 1024} KB."
    )
    if _wants_json():
        return _api_error(message, 413)
    return render_template("error.html", code=413, message=message), 413


@app.errorhandler(Exception)
def handle_unexpected(error):
    """
    Catch-all handler for unhandled exceptions.

    The full traceback goes to the server log together with a short random
    error id. The user sees only that id, so a support request can be traced
    without ever exposing stack traces, file paths or SQL to the client.
    """
    # Let Flask handle its own HTTP exceptions (404, 405, ...) normally.
    if isinstance(error, HTTPException):
        return error

    error_id = uuid.uuid4().hex[:8]
    logger.exception("Unhandled error [%s] on %s", error_id, request.path)

    if _wants_json():
        return _api_error(
            "An internal server error occurred.", 500, error_id=error_id
        )
    return (
        render_template(
            "error.html",
            code=500,
            message="Something went wrong on our side.",
            error_id=error_id,
        ),
        500,
    )


def create_app() -> Flask:
    """Initialise dependencies and return the configured Flask app."""
    config.ensure_directories()

    if _model_is_available():
        try:
            get_predictor().load()
        except ModelNotFoundError:
            logger.exception("Model artefacts present but unreadable.")
    else:
        logger.warning(
            "No trained model found at %s. The app will start, but predictions "
            "are disabled until you run: python src/train_model.py",
            config.MODEL_PATH,
        )
    return app


if __name__ == "__main__":
    create_app()
    logger.info(
        "Starting server on http://%s:%d (debug=%s)",
        config.FLASK_HOST, config.FLASK_PORT, config.FLASK_DEBUG,
    )
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
