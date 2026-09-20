#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build script for deployment (Render, Railway, and similar platforms).
#
# Runs once per deploy, before the server starts. It installs dependencies and
# downloads the NLTK corpora into a project-local folder, which
# src/preprocessing.py registers on NLTK's search path at import time.
#
# The trained model is committed to the repository, so NO training happens
# here - a deploy takes seconds rather than re-running the full pipeline.
# ---------------------------------------------------------------------------
set -o errexit   # abort the deploy if any step fails

echo "--> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "--> Downloading NLTK corpora into ./nltk_data"
python -m nltk.downloader -d ./nltk_data stopwords wordnet omw-1.4

echo "--> Verifying model artefacts are present"
python - <<'PYCHECK'
import sys
from pathlib import Path

required = ["models/model.pkl", "models/vectorizer.pkl", "models/metrics.json"]
missing = [name for name in required if not Path(name).is_file()]
if missing:
    sys.exit(
        "ERROR: missing model artefact(s): " + ", ".join(missing) +
        "\nCommit them, or run 'python src/train_model.py' before deploying."
    )
print("All model artefacts present.")
PYCHECK

echo "--> Build complete"
