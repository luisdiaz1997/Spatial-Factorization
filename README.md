# Spatial Factorization

Config-driven spatial transcriptomics analysis using probabilistic non-negative matrix factorization with Gaussian process priors.

## Overview

This repository provides scripts and notebooks for analyzing spatial transcriptomics datasets using:

- **[PNMF](https://github.com/luisdiaz1997/Probabilistic-NMF)**: Probabilistic Non-negative Matrix Factorization with sklearn-compatible API
- **[GPzoo](https://github.com/luisdiaz1997/GPzoo)**: Scalable Gaussian Process backends (SVGP, VNNGP, LCGP)

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Spatial-Factorization                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Configs   │  │  Datasets   │  │  Training Scripts   │ │
│  │   (YAML)    │  │  (loaders)  │  │    & Notebooks      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│        PNMF         │         │       GPzoo         │
│  - sklearn API      │         │  - SVGP, VNNGP      │
│  - ELBO modes       │         │  - LCGP, MGGP       │
│  - Non-spatial      │         │  - Kernels          │
└─────────────────────┘         └─────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/luisdiaz1997/Spatial-Factorization.git
cd Spatial-Factorization

# Install in development mode
pip install -e ".[all]"

# Install backends (from GitHub for now)
pip install git+https://github.com/luisdiaz1997/GPzoo.git
pip install git+https://github.com/luisdiaz1997/Probabilistic-NMF.git
```

## Quick Start

### Using Config Files

```bash
# Train a model using a config file
spatial-train --config configs/slideseq/lcgp.yaml

# Override specific parameters
spatial-train --config configs/slideseq/lcgp.yaml --steps 5000 --device cuda:1
```

### Programmatic API

```python
from spatial_factorization import Config, load_dataset, build_model, train

# Load config
config = Config.from_yaml("configs/slideseq/lcgp.yaml")

# Load data
data = load_dataset(config.dataset)

# Build and train model
model = build_model(config.model, data)
losses = train(model, data, config.training)
```

## Project Structure

```
Spatial-Factorization/
├── configs/                    # YAML experiment configs
│   ├── slideseq/
│   │   ├── svgp.yaml
│   │   ├── lcgp.yaml
│   │   └── vnngp.yaml
│   └── tenxvisium/
│       └── ...
├── src/
│   └── spatial_factorization/
│       ├── config.py           # Config loading/validation
│       ├── datasets/           # Dataset loaders
│       │   ├── slideseq.py
│       │   └── tenxvisium.py
│       ├── models/             # Model factory
│       └── training/           # Training utilities
├── notebooks/                  # Analysis notebooks
├── scripts/                    # CLI entry points
└── outputs/                    # Checkpoints and logs (git-ignored)
```

## Supported Datasets

| Dataset | Description | Loader |
|---------|-------------|--------|
| SlideseqV2 | Spatial transcriptomics (beads) | `squidpy.datasets.slideseqv2()` |
| 10x Visium | Spatial transcriptomics (spots) | `squidpy.datasets.visium_*()` |

## Supported Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `svgp_nsf` | SVGP + Poisson factorization | Standard spatial NMF |
| `vnngp_nsf` | VNNGP + Poisson factorization | Large datasets, local structure |
| `lcgp_nsf` | LCGP + Poisson factorization | Low-rank approximation |
| `*_mggp_nsf` | Multi-group variants | Datasets with known cell types |

## Config Reference

```yaml
name: experiment_name
seed: 42

dataset:
  name: slideseq              # Dataset name
  spatial_scale: 50.0         # Coordinate scaling
  filter_mt: true             # Filter mitochondrial genes

model:
  type: lcgp_nsf              # Model type
  n_components: 10            # Latent factors
  lengthscale: 4.0            # Kernel lengthscale
  K: 50                       # Nearest neighbors (VNNGP/LCGP)
  loadings_mode: projected    # W constraint mode

training:
  steps: 10000
  optimizer: adam
  learning_rates:
    default: 0.01
    loading: 0.001

output:
  dir: outputs/slideseq/lcgp
```

## Related Projects

- **[PNMF](https://github.com/luisdiaz1997/Probabilistic-NMF)**: Core probabilistic NMF library
- **[GPzoo](https://github.com/luisdiaz1997/GPzoo)**: Gaussian process backends
- **[NSF](https://github.com/willtownes/nsf-paper)**: Original non-negative spatial factorization paper

## License

GPL-2.0

## Citation

If you use this software, please cite:

```bibtex
@software{spatial_factorization,
  author = {Chumpitaz Diaz, Luis},
  title = {Spatial Factorization},
  url = {https://github.com/luisdiaz1997/Spatial-Factorization},
  year = {2025}
}
```
