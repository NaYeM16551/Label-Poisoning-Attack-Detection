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

---

## Running in Kaggle

This project is designed to run in Kaggle notebooks. Follow these steps:

### Kaggle Notebook Settings

First, enable the required resources in your Kaggle notebook settings:

- Turn **Internet** ON
- Turn **GPU** ON (if available)

### Setup Cells

#### Cell 1: Clone Repository

```python
!git clone https://github.com/NaYeM16551/Label-Poisoning-Attack-Detection.git
```

#### Cell 2: Navigate to Detection Folder

```python
%cd /kaggle/working/Label-Poisoning-Attack-Detection/Detection
```

#### Cell 3: Install Dependencies

```python
!pip install -q -r requirements.txt
!pip install -q -e .
```

### Important: FLIP Soft-Label File

This project requires the FLIP soft-label file to generate poisoned labels. The file path is configured in [configs/default.yaml](configs/default.yaml):

```
external/FLIP/precomputed/cifar10_sinusoidal_soft_labels.pt
```

**Where do I get this file?**

**Option A: Generate it locally (then upload to Kaggle)**

- Run `bash scripts/01_setup_environment.sh` (takes ~10 mins)
- File will be at: `external/FLIP/precomputed/cifar10_sinusoidal_soft_labels.pt`
- Then upload it as a Kaggle Dataset (see steps below)

**Option B: Generate it directly in Kaggle (no local setup needed)**

Add this cell to your Kaggle notebook (after navigating to Detection folder):

```python
!bash scripts/01_setup_environment.sh
```

- This generates the file directly in Kaggle (~10 mins)
- The file will be created at: `external/FLIP/precomputed/cifar10_sinusoidal_soft_labels.pt`
- Full Kaggle path: `/kaggle/working/Label-Poisoning-Attack-Detection/Detection/external/FLIP/precomputed/cifar10_sinusoidal_soft_labels.pt`
- No need to upload as a dataset since it's already there

**How to use it in Kaggle:**

**If you chose Option B above (generated in Kaggle):**

- Skip this entire section! The file is already in the correct location.
- Proceed directly to "Run Detection Only" below.

**If you chose Option A (uploading from local):**

1. **Upload to Kaggle as a Dataset:**
   - Go to https://www.kaggle.com/datasets/create/new
   - Upload your local `cifar10_sinusoidal_soft_labels.pt` file
   - Name it (e.g., `flip-soft-labels`)
   - Click "Create"

2. **Link dataset in your Kaggle notebook:**
   - In the notebook: click **+ Add input** (right sidebar)
   - Search for and select your dataset

3. **Copy it to the correct folder** (add this as a cell in your notebook):

```python
!mkdir -p external/FLIP/precomputed
!find /kaggle/input -name cifar10_sinusoidal_soft_labels.pt
!cp "$(find /kaggle/input -name cifar10_sinusoidal_soft_labels.pt | head -n 1)" external/FLIP/precomputed/
!ls -lh external/FLIP/precomputed/cifar10_sinusoidal_soft_labels.pt
```

> **Why this works:** Kaggle input folder names depend on your dataset slug, so auto-discovery avoids path mismatch errors.

> **Critical:** Without this file, FLIP generation will fail in [src/attacks/flip_wrapper.py](src/attacks/flip_wrapper.py).

### Run Detection Only (Recommended Start)

If you want to run only the FLIP defense pipeline without mitigation training:

```python
# 1) Generate FLIP-poisoned label metadata
!python -m src.attacks.generate_poisoned --config configs/default.yaml --attack flip --num_flips 1000

# 2) Extract SSL features
!python scripts/03_extract_features.py --config configs/default.yaml

# 3) Run detection
!python scripts/04_run_detection.py --config configs/default.yaml --poisoned data/poisoned/flip_cifar10_1000.pt
```

This is the main FLIP-defense path.

### Run Full Pipeline (Detection + Mitigation)

If you also want mitigation strategies and full evaluation:

```python
!python -m src.attacks.generate_poisoned --config configs/default.yaml --attack flip --num_flips 1000
!python scripts/03_extract_features.py --config configs/default.yaml
!python scripts/04_run_detection.py --config configs/default.yaml --poisoned data/poisoned/flip_cifar10_1000.pt
!python scripts/05_run_mitigation.py --config configs/default.yaml --poisoned data/poisoned/flip_cifar10_1000.pt --scores results/detection/flip_cifar10_1000_scores.npz --mode remove
!python scripts/05_run_mitigation.py --config configs/default.yaml --poisoned data/poisoned/flip_cifar10_1000.pt --scores results/detection/flip_cifar10_1000_scores.npz --mode none
```

### GPU and CPU Considerations

**Detection-only:**

- GPU is not mandatory but strongly recommended for faster processing
- CPU runs are feasible but slower

**Mitigation/Training:**

- GPU is strongly recommended

**If running on CPU**, edit [configs/default.yaml](configs/default.yaml) to:

- Set `device: "cpu"` (default is `"cuda"`)
- Reduce batch sizes from `256` to `32` or `64`

Quick CPU setup:

```python
!sed -i 's/device: "cuda"/device: "cpu"/' configs/default.yaml
!sed -i 's/batch_size: 256/batch_size: 64/' configs/default.yaml
```

### Training Requirements

- **generate_poisoned, extract_features, run_detection**: No training from scratch
- **05_run_mitigation.py**: Requires training a classifier (GPU recommended)

### First Run Notes

- The first run of `03_extract_features.py` will download pretrained SSL models
- Ensure **Internet** is enabled in Kaggle for initial model weight downloads
- Subsequent runs will use cached weights

### Example Kaggle Notebook Order

1. Install dependencies
2. Copy FLIP soft-label file
3. Run detection-only pipeline OR full pipeline
4. Inspect results in `results/` folder
5. Optionally generate visualizations from `notebooks/04_analysis.ipynb`
