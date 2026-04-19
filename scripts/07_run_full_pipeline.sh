#!/usr/bin/env bash
# End-to-end pipeline: generate FLIP-poisoned datasets -> extract features ->
# detect -> mitigate -> plots.
#
# Usage:
#   bash scripts/07_run_full_pipeline.sh [config_path]
#   bash scripts/07_run_full_pipeline.sh [config_path] --include-random-baseline
#
# The default workflow is FLIP-only because the thesis defense targets the
# FLIP attack from "Label Poisoning is All You Need". Random label flips are
# optional and must be requested explicitly.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="configs/default.yaml"
INCLUDE_RANDOM=0

for arg in "$@"; do
  if [ "$arg" = "--include-random-baseline" ]; then
    INCLUDE_RANDOM=1
  else
    CONFIG="$arg"
  fi
done

REPRO_ARGS=("$CONFIG")
if [ "$INCLUDE_RANDOM" -eq 1 ]; then
  REPRO_ARGS+=("--include-random-baseline")
fi

echo "================================================================"
echo " 1. Generate poisoned datasets"
echo "================================================================"
bash scripts/02_reproduce_flip.sh "${REPRO_ARGS[@]}"

echo "================================================================"
echo " 2. Extract SSL features"
echo "================================================================"
python scripts/03_extract_features.py --config "$CONFIG"

echo "================================================================"
echo " 3. Run detection on FLIP-poisoned datasets"
echo "================================================================"
shopt -s nullglob
for poisoned in data/poisoned/flip_*.pt; do
  echo "--> Detection on $poisoned"
  python scripts/04_run_detection.py --config "$CONFIG" --poisoned "$poisoned" --skip_validation
done

if [ "$INCLUDE_RANDOM" -eq 1 ]; then
  for poisoned in data/poisoned/random_*.pt; do
    echo "--> Detection on optional baseline $poisoned"
    python scripts/04_run_detection.py --config "$CONFIG" --poisoned "$poisoned" --skip_validation
  done
fi

echo "================================================================"
echo " 4. Mitigation on the canonical 2% FLIP dataset"
echo "================================================================"
SCORES_FILE="results/detection/flip_$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["dataset"]["name"])
PY
)_$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["attack"]["num_flips"])
PY
)_scores.npz"

POISONED_FILE="data/poisoned/flip_$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["dataset"]["name"])
PY
)_$(python - "$CONFIG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["attack"]["num_flips"])
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
