#!/usr/bin/env bash
# End-to-end pipeline: generate poisoned datasets -> extract features ->
# detect -> mitigate -> plots.
#
# Usage: bash scripts/07_run_full_pipeline.sh [config_path]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/default.yaml}"

echo "================================================================"
echo " 1. Generate poisoned datasets"
echo "================================================================"
bash scripts/02_reproduce_flip.sh "$CONFIG"

echo "================================================================"
echo " 2. Extract SSL features"
echo "================================================================"
python scripts/03_extract_features.py --config "$CONFIG"

echo "================================================================"
echo " 3. Run detection on each poisoned dataset"
echo "================================================================"
shopt -s nullglob
for poisoned in data/poisoned/*.pt; do
  echo "--> Detection on $poisoned"
  python scripts/04_run_detection.py --config "$CONFIG" --poisoned "$poisoned" --skip_validation
done

echo "================================================================"
echo " 4. Mitigation on the canonical 2% FLIP dataset"
echo "================================================================"
SCORES_FILE="results/detection/flip_$(python - <<'PY'
import yaml; print(yaml.safe_load(open("configs/default.yaml"))["dataset"]["name"])
PY
)_$(python - <<'PY'
import yaml; print(yaml.safe_load(open("configs/default.yaml"))["attack"]["num_flips"])
PY
)_scores.npz"

POISONED_FILE="data/poisoned/flip_$(python - <<'PY'
import yaml; print(yaml.safe_load(open("configs/default.yaml"))["dataset"]["name"])
PY
)_$(python - <<'PY'
import yaml; print(yaml.safe_load(open("configs/default.yaml"))["attack"]["num_flips"])
PY
).pt"

if [ -f "$SCORES_FILE" ] && [ -f "$POISONED_FILE" ]; then
  python scripts/05_run_mitigation.py --config "$CONFIG" \
      --poisoned "$POISONED_FILE" --scores "$SCORES_FILE" --mode remove
  python scripts/05_run_mitigation.py --config "$CONFIG" \
      --poisoned "$POISONED_FILE" --scores "$SCORES_FILE" --mode none
else
  echo "Skipping mitigation: missing $SCORES_FILE or $POISONED_FILE"
fi

echo "================================================================"
echo " 5. Aggregate plots"
echo "================================================================"
python scripts/06_generate_plots.py --config "$CONFIG"

echo "Pipeline complete. See results/ for outputs."
