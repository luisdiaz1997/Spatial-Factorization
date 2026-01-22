# Spatial Factorization Development Notes

---

## URGENT: Active Implementation Plan

**READ THIS FIRST:** [SLIDESEQ_PNMF_PLANNING.md](./SLIDESEQ_PNMF_PLANNING.md)

This document contains the detailed implementation plan for:
- CLI commands: `preprocess`, `train`, `analyze`, `figures`
- Standardized data format (X, Y, C as .npy files)
- Code references to borrow from GPzoo/PNMF
- Four-stage pipeline architecture

---

## Environment Setup

**IMPORTANT:** Use a dedicated conda environment to avoid dependency conflicts.

```bash
# Create new conda environment named 'factorization'
conda create -n factorization python=3.10 -y
conda activate factorization

# Install core dependencies
pip install torch>=2.0.0 numpy>=1.21.0 scipy>=1.7.0 pandas>=1.3.0
pip install scikit-learn>=1.0.0 tqdm>=4.62.0 pyyaml>=6.0 click>=8.0

# Install bioinformatics packages
pip install scanpy>=1.9.0 anndata>=0.8.0 squidpy>=1.2.0

# Install testing dependencies
pip install pytest>=7.0.0

# Install this package
pip install -e .

# Install sibling packages (PNMF, GPzoo) - adjust paths as needed
pip install -e ../Probabilistic-NMF
pip install -e ../GPzoo
```

**Current environment:** `factorization` (Python 3.10)

**Note:** The previous environment had squidpy/numba segfault issues. Using specific versions (scanpy==1.10.0, squidpy==1.5.0) with Python 3.10 should resolve this.

---

## Project Overview

This repository provides dataset loaders and configuration utilities for spatial transcriptomics analysis. It is designed to work with the PNMF package for model fitting.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Spatial-Factorization                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Configs   │  │  Datasets   │  │     Notebooks       │ │
│  │   (YAML)    │  │  (loaders)  │  │    (analysis)       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│        PNMF         │         │       GPzoo         │
│  - sklearn API      │         │  - SVGP, VNNGP      │
│  - spatial=True     │◄────────│  - LCGP, kernels    │
│  - ELBO modes       │         │  - modules          │
└─────────────────────┘         └─────────────────────┘
```

## Package Structure

```
Spatial-Factorization/
├── spatial_factorization/
│   ├── __init__.py             # Package exports
│   ├── config.py               # Config dataclasses
│   ├── cli.py                  # Click CLI entry point
│   ├── analysis.py             # plot_factors, etc.
│   ├── commands/               # CLI commands
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Standardize data format
│   │   ├── train.py            # Train PNMF model
│   │   ├── analyze.py          # Compute metrics
│   │   └── figures.py          # Generate plots
│   └── datasets/               # Dataset loaders (dataset-specific)
│       ├── __init__.py
│       ├── base.py             # SpatialData container
│       ├── preprocessed.py     # Load preprocessed .npy files
│       ├── slideseq.py         # SlideseqV2 loader
│       └── tenxvisium.py       # 10x Visium loader
├── scripts/
│   └── install_deps.sh         # Install PNMF/GPzoo from sibling dirs
├── configs/                    # YAML experiment configs
│   └── slideseq/
│       └── pnmf.yaml
├── outputs/                    # Generated (git-ignored)
├── setup.py                    # Minimal wrapper (PEP 517 compat)
├── pyproject.toml
└── README.md
```

## Key Design Decisions

### 1. Models and Training Live in PNMF

This package does NOT implement models or training loops. Those belong in PNMF:

```python
# Correct: use PNMF for models
from PNMF import PNMF
model = PNMF(n_components=10, spatial=True, gp_class='LCGP')
model.fit(Y, X=coordinates)

# Wrong: don't create model factories here
# from spatial_factorization.models import build_model  # NO
```

### 2. Config Stores Settings, PNMF Builds Models

Configs are for reproducibility and experiment tracking:

```python
from spatial_factorization import Config, load_dataset
from PNMF import PNMF

config = Config.from_yaml("configs/slideseq/lcgp.yaml")
data = load_dataset(config.dataset)

# Config provides kwargs, PNMF builds the model
model = PNMF(**config.model.to_pnmf_kwargs(), **config.training.to_pnmf_kwargs())
model.fit(data.Y.T.numpy(), X=data.X.numpy())
```

### 3. SpatialData Container

The `SpatialData` dataclass holds:
- `X`: Spatial coordinates (N, 2) - used in analysis/plotting, ignored in non-spatial training
- `Y`: Count matrix (D, N) - genes x spots (GPzoo convention) - used in training
- `C`: Group codes (N,) - used in analysis/plotting, ignored in non-spatial training
- `gene_names`, `spot_names`, `group_names`: Metadata

Note: PNMF's sklearn API expects (N, D), so use `data.Y.T`.

## Dependencies

- **pnmf**: Model fitting (sklearn API with spatial=True)
- **gpzoo**: GP backends (used internally by PNMF)
- **scanpy, squidpy**: Data loading
- **pyyaml**: Config parsing
- **click**: CLI framework

### Installing PNMF and GPzoo

PNMF and GPzoo are sibling repositories that must be installed locally for development. The `scripts/install_deps.sh` helper script handles this:

```bash
# Install PNMF and GPzoo from sibling directories
./scripts/install_deps.sh
```

The script expects:
- `../Probabilistic-NMF/` - PNMF repository
- `../GPzoo/` - GPzoo repository

Adjust paths if your directory structure differs.

### Installation

```bash
# 1. Install dependencies (PNMF, GPzoo)
./scripts/install_deps.sh

# 2. Install spatial-factorization
pip install -e .

# 3. Verify CLI works
spatial_factorization --help
```

**Note:** The CLI entry point is configured in `pyproject.toml` as:
```
[project.scripts]
spatial_factorization = "spatial_factorization.cli:cli"
```

A minimal `setup.py` wrapper is provided for PEP 517 backwards compatibility.

## Relationship to Other Repos

| Repo | Purpose | What Lives Here |
|------|---------|-----------------|
| **GPzoo** | GP backends | SVGP, VNNGP, LCGP, kernels, modules |
| **PNMF** | sklearn API | `PNMF` class, ELBO computation, `spatial=True` support |
| **Spatial-Factorization** | Analysis | Dataset loaders, configs, notebooks |

## Implementation Status

See [SLIDESEQ_PNMF_PLANNING.md](./SLIDESEQ_PNMF_PLANNING.md) for the detailed 4-stage implementation plan.

| Stage | Status | Description |
|-------|--------|-------------|
| 0 | **DONE** | Setup & Installation (CLI entry point, package structure) |
| 1 | TODO | Preprocess command (standardize data format) |
| 2 | TODO | Train command (PNMF model fitting) |
| 3 | TODO | Analyze command (Moran's I, reconstruction metrics) |
| 4 | TODO | Figures command (publication plots) |

## Future Work

- [ ] Implement Stages 1-4 (see SLIDESEQ_PNMF_PLANNING.md)
- [ ] Add more dataset loaders (Stereo-seq, MERFISH, etc.)
- [ ] Create analysis notebooks for each dataset
- [ ] Add visualization utilities
- [ ] Benchmark scripts

## CLI Usage

```bash
# Install
pip install -e .

# Four-stage pipeline
spatial_factorization preprocess -c configs/slideseq/pnmf.yaml  # Run once
spatial_factorization train      -c configs/slideseq/pnmf.yaml
spatial_factorization analyze    -c configs/slideseq/pnmf.yaml
spatial_factorization figures    -c configs/slideseq/pnmf.yaml
```

## Python API Example

```python
import torch
from spatial_factorization import Config, load_dataset
from PNMF import PNMF

# Load config
config = Config.from_yaml("configs/slideseq/pnmf.yaml")
torch.manual_seed(config.seed)

# Load data
data = load_dataset(config.dataset)
print(f"Loaded {data.n_spots} spots, {data.n_genes} genes")

# Fit model
model = PNMF(
    n_components=config.model.n_components,
    mode=config.model.mode,
    max_iter=config.training.max_iter,
    learning_rate=config.training.learning_rate,
    verbose=config.training.verbose,
)

# PNMF expects (n_samples, n_features), SpatialData has (D, N)
Y_sklearn = data.Y.T.numpy()  # (N, D)
model.fit(Y_sklearn)
print(f"ELBO: {model.elbo_}")
```
