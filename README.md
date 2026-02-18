# Spatial Factorization

Config-driven spatial transcriptomics analysis using probabilistic non-negative matrix factorization with Gaussian process priors.

## Overview

Four-stage CLI pipeline for analyzing spatial transcriptomics datasets:

```
preprocess → train → analyze → figures
```

Built on:
- **[PNMF](https://github.com/luisdiaz1997/Probabilistic-NMF)**: Probabilistic Non-negative Matrix Factorization with sklearn-compatible API
- **[GPzoo](https://github.com/luisdiaz1997/GPzoo)**: Scalable Gaussian Process backends (SVGP, LCGP, MGGP variants)

## Installation

```bash
git clone https://github.com/luisdiaz1997/Spatial-Factorization.git
cd Spatial-Factorization

# Install dependencies
pip install -e ../Probabilistic-NMF
pip install -e ../GPzoo
pip install -e .
```

## Usage

### Single model

```bash
# Run all stages for one model
spatial_factorization run all -c configs/slideseq/svgp.yaml

# Run specific stages
spatial_factorization run train analyze figures -c configs/slideseq/svgp.yaml

# Run stages individually
spatial_factorization preprocess -c configs/slideseq/svgp.yaml
spatial_factorization train      -c configs/slideseq/svgp.yaml
spatial_factorization analyze    -c configs/slideseq/svgp.yaml
spatial_factorization figures    -c configs/slideseq/svgp.yaml

# Resume training from a checkpoint
spatial_factorization train --resume -c configs/slideseq/svgp.yaml
```

### Multiple models in parallel

```bash
# Generate per-model configs from a general config
spatial_factorization generate -c configs/slideseq/general.yaml

# Run all 5 models in parallel (pnmf, svgp, mggp_svgp, lcgp, mggp_lcgp)
spatial_factorization run all -c configs/slideseq/general.yaml

# Dry run to see the execution plan
spatial_factorization run all -c configs/slideseq/general.yaml --dry-run

# Force re-preprocessing
spatial_factorization run all -c configs/slideseq/general.yaml --force

# Run all datasets at once
spatial_factorization run all -c configs/
```

## Models

| Model | Config key | Groups | Description |
|-------|-----------|--------|-------------|
| PNMF | `spatial: false` | — | Non-spatial baseline |
| SVGP | `spatial: true, local: false, groups: false` | No | Sparse variational GP (M<<N inducing pts) |
| MGGP-SVGP | `spatial: true, local: false, groups: true` | Yes | Multi-group SVGP |
| LCGP | `spatial: true, local: true, groups: false` | No | Locally conditioned GP (M=N, VNNGP-style) |
| MGGP-LCGP | `spatial: true, local: true, groups: true` | Yes | Multi-group LCGP |

## Config Reference

```yaml
seed: 42
output_dir: outputs/slideseq

dataset:
  name: slideseq
  path: data/slideseq/Puck_200115_08.h5ad

model:
  n_components: 10
  spatial: true
  local: true           # true = LCGP, false = SVGP
  groups: false         # true = multi-group variant
  K: 50                 # KNN neighbors (LCGP only)
  num_inducing: 3000    # inducing points (SVGP only)
  kernel: Matern32
  lengthscale: 8.0
  train_lengthscale: false
  mode: expanded
  loadings_mode: projected

training:
  max_iter: 20000
  batch_size: 7000
  y_batch_size: null
  learning_rate: 0.01
  device: cuda
```

## Project Structure

```
Spatial-Factorization/
├── configs/slideseq/
│   ├── general.yaml        # Superset config for multiplex training
│   ├── general_test.yaml   # Quick test (200 epochs)
│   ├── pnmf.yaml
│   ├── svgp.yaml
│   ├── mggp_svgp.yaml
│   ├── lcgp.yaml
│   └── mggp_lcgp.yaml
├── spatial_factorization/
│   ├── cli.py              # Click CLI entry point
│   ├── config.py           # Config dataclass
│   ├── runner.py           # Parallel job runner
│   ├── commands/
│   │   ├── preprocess.py
│   │   ├── train.py
│   │   ├── analyze.py
│   │   └── figures.py
│   └── datasets/
│       ├── slideseq.py
│       └── tenxvisium.py
└── outputs/                # Generated (git-ignored)
```

## Related Projects

- **[PNMF](https://github.com/luisdiaz1997/Probabilistic-NMF)**: Core probabilistic NMF library
- **[GPzoo](https://github.com/luisdiaz1997/GPzoo)**: Gaussian process backends

## License

GPL-2.0
