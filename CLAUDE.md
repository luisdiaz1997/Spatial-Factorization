# Spatial Factorization Development Notes

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
├── src/
│   └── spatial_factorization/
│       ├── __init__.py         # Package exports
│       ├── config.py           # Config dataclasses
│       └── datasets/           # Dataset loaders
│           ├── __init__.py
│           ├── base.py         # SpatialData container
│           ├── slideseq.py     # SlideseqV2 loader
│           └── tenxvisium.py   # 10x Visium loader
├── configs/                    # YAML experiment configs
│   ├── slideseq/
│   └── tenxvisium/
├── notebooks/                  # Analysis notebooks
├── scripts/                    # CLI scripts
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
- `X`: Spatial coordinates (N, 2)
- `Y`: Count matrix (D, N) - genes x spots (GPzoo convention)
- `V`: Size factors (N,)
- `groups`: Optional group labels for MGGP
- `gene_names`, `spot_names`: Metadata

Note: PNMF's sklearn API expects (N, D), so use `data.Y.T`.

## Dependencies

- **pnmf**: Model fitting (sklearn API with spatial=True)
- **gpzoo**: GP backends (used internally by PNMF)
- **scanpy, squidpy**: Data loading
- **pyyaml**: Config parsing

## Relationship to Other Repos

| Repo | Purpose | What Lives Here |
|------|---------|-----------------|
| **GPzoo** | GP backends | SVGP, VNNGP, LCGP, kernels, modules |
| **PNMF** | sklearn API | `PNMF` class, ELBO computation, `spatial=True` support |
| **Spatial-Factorization** | Analysis | Dataset loaders, configs, notebooks |

## Future Work

- [ ] Add more dataset loaders (Stereo-seq, MERFISH, etc.)
- [ ] Create analysis notebooks for each dataset
- [ ] Add visualization utilities
- [ ] Benchmark scripts

## Usage Example

```python
import torch
from spatial_factorization import Config, load_dataset
from PNMF import PNMF

# Load config
config = Config.from_yaml("configs/slideseq/lcgp.yaml")
torch.manual_seed(config.seed)

# Load data
data = load_dataset(config.dataset)
print(f"Loaded {data.n_spots} spots, {data.n_genes} genes")

# Fit model (when PNMF spatial=True is implemented)
model = PNMF(
    n_components=config.model.n_components,
    spatial=config.model.spatial,
    gp_class=config.model.gp_class,
    mode=config.model.mode,
    max_iter=config.training.max_iter,
    learning_rate=config.training.learning_rate,
    verbose=config.training.verbose,
)

# PNMF expects (n_samples, n_features), SpatialData has (D, N)
Y_sklearn = data.Y.T.numpy()  # (N, D)
X_coords = data.X.numpy()      # (N, 2)

model.fit(Y_sklearn, X=X_coords)
print(f"ELBO: {model.elbo_}")
```
