#!/usr/bin/env bash
# Reset the demo to a clean state before a run:
#   1. Clear outbox/ (the fake sent-messages log from send_followup)
#   2. Re-run ingest.py, which itself does cognee.forget(everything=True),
#      re-remembers everything in data/, and regenerates brain.html.
#
# Usage: ./reset.sh && uv run python demo.py
set -euo pipefail
cd "$(dirname "$0")"

echo "Clearing outbox/..."
rm -rf outbox
mkdir -p outbox

echo "Rebuilding the cognee graph (forget + re-ingest data/ + regenerate brain.html)..."
uv run python ingest.py
