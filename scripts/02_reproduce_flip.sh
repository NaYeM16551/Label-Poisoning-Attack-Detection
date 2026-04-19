#!/usr/bin/env bash
# Generate FLIP-poisoned CIFAR datasets at every requested poisoning rate.
#
# Usage:
#   bash scripts/02_reproduce_flip.sh [config_path]
#   bash scripts/02_reproduce_flip.sh [config_path] --include-random-baseline
#
# The default thesis workflow is FLIP-only. The random attack is kept only as
# an optional comparator and is not generated unless explicitly requested.
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

# Parameters: poisoning rate -> num_flips (50K * rate for CIFAR-10).
RATES=(250 500 1000 2500)

for n in "${RATES[@]}"; do
  echo "==> Generating FLIP poisoned dataset with num_flips=$n"
  python -m src.attacks.generate_poisoned --config "$CONFIG" --attack flip --num_flips "$n"
done

if [ "$INCLUDE_RANDOM" -eq 1 ]; then
  echo "Generating optional random-flip baseline (num_flips=1000)"
  python -m src.attacks.generate_poisoned --config "$CONFIG" --attack random --num_flips 1000
else
  echo "Skipping random-flip baseline; default workflow is FLIP-only."
fi

echo "Done. Poisoned datasets under data/poisoned/"
