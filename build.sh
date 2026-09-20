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
set -o errexit   # abort the deploy if a required step fails

echo "--> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

# NLTK refuses to write into a directory that is group- or world-writable, and
# on failure its command-line downloader prompts for input - which aborts a
# non-interactive build with EOFError. Creating the directory with 755 and
# driving the Python API directly avoids both problems.
echo "--> Downloading NLTK corpora into ./nltk_data"
mkdir -p ./nltk_data
chmod 755 ./nltk_data
python - <<'PYNLTK'
import sys

TARGET = "./nltk_data"
CORPORA = ["stopwords", "wordnet", "omw-1.4"]

try:
    import nltk
except ImportError:
    sys.exit("ERROR: nltk is not installed - check requirements.txt")

failed = []
for name in CORPORA:
    try:
        nltk.download(name, download_dir=TARGET, quiet=True, raise_on_error=True)
        print(f"    ok: {name}")
    except Exception as exc:
        failed.append(name)
        print(f"    WARNING: could not download {name}: {exc}")

if failed:
    # Not fatal. src/preprocessing.py falls back to a built-in stopword list
    # and skips lemmatization, which costs roughly 0.6 percentage points of
    # accuracy - degraded, but a working deploy beats a failed one.
    print(f"    {len(failed)} corpus/corpora unavailable: {', '.join(failed)}")
    print("    The app will run with reduced preprocessing accuracy.")
else:
    print("    All corpora installed.")
PYNLTK

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
print("    All model artefacts present.")
PYCHECK

echo "--> Build complete"
