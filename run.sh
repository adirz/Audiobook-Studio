#!/usr/bin/env bash
# ─── Audiobook Studio launcher ───────────────────────────────
# Usage: ./run.sh [--port PORT]

set -e
cd "$(dirname "$0")"

PORT="${1:-8899}"
if [ "$1" = "--port" ]; then PORT="$2"; fi

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: Python 3 is required."
    exit 1
fi

# Install dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Checking dependencies..."
pip install -q -r requirements.txt 2>/dev/null

# Download nltk data if needed
python3 -c "
import nltk
for pkg in ['words', 'wordnet', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'corpora/{pkg}' if pkg != 'averaged_perceptron_tagger_eng' else f'taggers/{pkg}')
    except LookupError:
        print(f'Downloading NLTK {pkg}...')
        nltk.download(pkg, quiet=True)
" 2>/dev/null

echo ""
echo "═══════════════════════════════════════════════"
echo "  Audiobook Studio"
echo "  Open http://localhost:${PORT} in your browser"
echo "═══════════════════════════════════════════════"
echo ""

python3 -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --reload
