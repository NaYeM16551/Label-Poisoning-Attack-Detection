# Detecting FLIP Label Poisoning via Self-Supervised Feature Auditing

A modular Python implementation of the thesis project: detecting FLIP-style
label-only backdoor attacks using k-NN neighbourhood consistency in a
self-supervised feature space.

The codebase follows the layout described in
`Implementation_Blueprint_English.md` (one directory above this folder).

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install -e .

# 2. (One-time) clone FLIP and download CIFAR-10
bash scripts/01_setup_environment.sh

# 3. Run the full pipeline (FLIP -> features -> detection -> mitigation -> plots)
bash scripts/07_run_full_pipeline.sh
```

Individual stages can be run with the numbered scripts under `scripts/`. All
behaviour is driven by YAML configs under `configs/`.

The default workflow is FLIP-only and targets the attack from
`Label_Poisoning_is_all_you_need.pdf`. A random label-flip attack remains in
the repository only as an optional baseline for side comparisons.

## Layout

```
Detection/
|-- configs/        YAML experiment configurations
|-- src/            Library code (importable as `from src...`)
|   |-- attacks/    FLIP wrapper + optional random baseline
|   |-- features/   SSL & supervised feature extractors + validation
|   |-- detectors/  k-NN, loss, random detectors + scoring helpers
|   |-- mitigation/ Remove/downweight + retrain
|   |-- evaluation/ Detection + attack-success metrics
|   |-- visualization/  t-SNE, ROC/PR, histograms, ablations
|   |-- utils/      data, config, seeding, logging
|-- scripts/        Numbered entry-point scripts (01..07)
|-- external/       Third-party code (FLIP repo lives here)
|-- data/           Datasets + cached features (gitignored)
|-- results/        Detection / mitigation outputs + figures
|-- notebooks/      Exploratory analysis notebooks
|-- paper/          Thesis sources + publication figures
```

## Method overview

1. **Attack:** load FLIP-poisoned labels from the FLIP codebase and wrap
   CIFAR-10 with a `PoisonedCIFAR10` dataset that tracks ground truth.
2. **Features:** extract embeddings with a frozen SSL encoder
   (`dinov2_vits14` by default). Crucially, these features never see the
   poisoned labels.
3. **Detect:** for each sample, compare its label against the labels of its
   k nearest neighbours in feature space; high disagreement -> suspicious.
4. **Mitigate:** remove (or downweight) the suspicious samples and retrain a
   classifier; report Clean Test Accuracy (CTA) and Poison Test Accuracy
   (PTA).

Optional only: `bash scripts/02_reproduce_flip.sh configs/default.yaml --include-random-baseline`
will also generate a random label-flip comparator. That comparator is not part
of the main thesis defense path.

See `Implementation_Blueprint_English.md` for the full design rationale,
expected results at each checkpoint, and the week-by-week execution plan.
