#!/usr/bin/env bash
set -euo pipefail

# Model is normally baked at build time; retrain on first boot only if missing.
if [ ! -f "${DS_MODEL_ROOT:-/app/models}/serving_model.json" ]; then
  echo "No model found — training..."
  python -m driver_state.cli train
fi

PORT="${PORT:-8000}"
echo "Starting API + dashboard on :${PORT}"
exec uvicorn driver_state.api.main:app --host 0.0.0.0 --port "${PORT}"
