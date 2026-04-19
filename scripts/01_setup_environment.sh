#!/usr/bin/env bash
# Bootstrap the project environment.
#
# Usage: bash scripts/01_setup_environment.sh
#
# Idempotent: re-running is safe; everything that already exists is skipped.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/4] Installing Python requirements"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

echo "[2/4] Cloning the FLIP repository (if missing)"
if [ ! -d "external/FLIP" ]; then
  git clone https://github.com/SewoongLab/FLIP external/FLIP
else
  echo "  external/FLIP already present, skipping clone"
fi

echo "[3/4] Downloading CIFAR-10"
python - <<'PY'
import torchvision
torchvision.datasets.CIFAR10(root="./data/raw", train=True, download=True)
torchvision.datasets.CIFAR10(root="./data/raw", train=False, download=True)
print("CIFAR-10 ready under ./data/raw")
PY

echo "[4/4] Creating result directories"
mkdir -p results/detection results/mitigation results/figures results/tables results/validation
mkdir -p data/poisoned data/features

echo "Environment setup complete."
echo "Next: download FLIP precomputed soft labels into external/FLIP/precomputed/ "
echo "      (see external/FLIP/README.md), then run scripts/02_reproduce_flip.sh"
