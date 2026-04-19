"""Package setup for the FLIP SSL defence project."""
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
LONG_DESCRIPTION = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""

setup(
    name="flip-ssl-defense",
    version="0.1.0",
    description="Detecting FLIP label poisoning via self-supervised feature auditing.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Thesis Project",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "faiss-cpu>=1.7.4",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "umap-learn>=0.5.0",
        "timm>=0.9.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
        "gpu": ["faiss-gpu>=1.7.4"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
