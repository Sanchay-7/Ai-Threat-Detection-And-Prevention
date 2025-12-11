#!/usr/bin/env bash
set -euo pipefail

# Ensure the script is run with sudo
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or with sudo"
  exit 1
fi

echo "Setting up Python virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installing/updating dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Starting AI DDoS Shield on http://0.0.0.0:8000"
# The --reload flag is useful for development but can be removed for production
exec uvicorn app:app --host 0.0.0.0 --port 8000 --log-level info --reload
