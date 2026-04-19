#!/usr/bin/env bash
# Generate poisoned CIFAR datasets at every requested poisoning rate.
#
# Usage: bash scripts/02_reproduce_flip.sh [config_path]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-configs/default.yaml}"

# Parameters: poisoning rate -> num_flips (50K * rate for CIFAR-10).
RATES=(250 500 1000 2500)

for n in "${RATES[@]}"; do
  echo "==> Generating FLIP poisoned dataset with num_flips=$n"
  python -m src.attacks.generate_poisoned --config "$CONFIG" --attack flip --num_flips "$n"
done

echo "Generating random-flip baseline (num_flips=1000)"
python -m src.attacks.generate_poisoned --config "$CONFIG" --attack random --num_flips 1000

echo "Done. Poisoned datasets under data/poisoned/"
