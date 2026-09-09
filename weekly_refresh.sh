#!/usr/bin/env bash
# Weekly refresh: pull fresh data, refit weights, rebuild the tuner page's
# data payload. Run from the nfl_edge project root:
#   bash weekly_refresh.sh
# After this succeeds, republish build/tuner_published.html to the
# existing Artifact URL (the Artifact tool, not this script, does that).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
python -m nfl_edge.cli refresh-data
python -m nfl_edge.export_client_data
python build_tuner.py
echo "Done. build/tuner_published.html is up to date -- republish it to the live Artifact URL."
