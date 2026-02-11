# Spatial Factorization Development Notes

---

## Environment Setup

**CRITICAL: Always use the `factorization` conda environment!**

```bash
conda activate factorization
# Or for non-interactive contexts:
conda run -n factorization <command>
```

**Python:** 3.14 | **Location:** `/gladstone/engelhardt/home/lchumpitaz/miniconda3/envs/factorization`

---

## Project Overview

Four-stage CLI pipeline for spatial transcriptomics analysis using PNMF + GP priors.

```
preprocess → train → analyze → figures
```

All stages are **implemented and working**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Spatial-Factorization                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   Configs   │  │  Datasets   │  │   Commands (CLI)    │ │
│  │   (YAML)    │  │  (loaders)  │  │ train/analyze/figs  │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│        PNMF         │         │       GPzoo         │
│  - sklearn API      │         │  - SVGP, MGGP_SVGP  │
│  - spatial=True     │◄────────│  - batched kernels   │
│  - multigroup       │         │  - CholeskyParameter  │
└─────────────────────┘         └─────────────────────┘
```

## Package Structure

```
Spatial-Factorization/
├── spatial_factorization/
│   ├── __init__.py
│   ├── config.py               # Config dataclass with model_name, groups, spatial properties
│   ├── cli.py                  # Click CLI entry point
│   ├── commands/
│   │   ├── preprocess.py       # Stage 1: Standardize data format
│   │   ├── train.py            # Stage 2: Train PNMF model
│   │   ├── analyze.py          # Stage 3: Compute metrics, factors, loadings
│   │   └── figures.py          # Stage 4: Generate plots
│   └── datasets/
│       ├── base.py             # SpatialData container + load_preprocessed()
│       ├── slideseq.py         # SlideseqV2 loader
│       └── tenxvisium.py       # 10x Visium loader
├── configs/slideseq/
│   ├── pnmf.yaml               # Non-spatial baseline
│   ├── svgp.yaml                # MGGP_SVGP (full training)
│   ├── svgp_test.yaml           # MGGP_SVGP (10 epochs, testing)
│   └── svgp_no_groups_test.yaml # SVGP no groups (10 epochs, testing)
├── tests/
│   └── test_svgp_model.py       # SVGP model inspection tests
├── outputs/                     # Generated (git-ignored)
├── setup.py
├── pyproject.toml
└── CLAUDE.md
```

---

## Output Directory Naming Convention

Model output directories are determined by `config.model_name`:

| Config | `model_name` | Directory |
|--------|-------------|-----------|
| `spatial: false` | `pnmf` | `outputs/slideseq/pnmf/` |
| `spatial: true, groups: false` | `SVGP` | `outputs/slideseq/SVGP/` |
| `spatial: true, groups: true` | `MGGP_SVGP` | `outputs/slideseq/MGGP_SVGP/` |

The naming uses the prior name directly (uppercase). When `groups: true`, the `MGGP_` prefix is added.

### Output Directory Structure

```
outputs/slideseq/
├── preprocessed/           # Shared by all models (Stage 1)
│   ├── X.npy               # (N, 2) spatial coordinates
│   ├── Y.npz               # (D, N) count matrix (sparse)
│   ├── C.npy               # (N,) group codes (integers 0..G-1)
│   └── metadata.json       # gene_names, group_names, etc.
│
├── MGGP_SVGP/              # groups=true output
│   ├── model.pth            # PyTorch state dict
│   ├── training.json        # Metadata
│   ├── elbo_history.csv     # ELBO convergence
│   ├── config.yaml          # Config snapshot
│   ├── factors.npy          # (N, L) factor values
│   ├── scales.npy           # (N, L) factor uncertainty
│   ├── loadings.npy         # (D, L) global loadings
│   ├── loadings_group_*.npy # (D, L) per-group loadings
│   ├── Z.npy                # (M, 2) inducing point locations
│   ├── groupsZ.npy          # (M,) inducing point group assignments
│   ├── Lu.pt                # (L, M, M) Cholesky variational covariance
│   ├── moran_i.csv          # Moran's I per factor
│   ├── gene_enrichment.json # LFC per factor per group
│   ├── metrics.json         # All computed metrics
│   └── figures/
│       ├── factors_spatial.png
│       ├── scales_spatial.png
│       ├── lu_scales_inducing.png
│       ├── groups.png
│       ├── elbo_curve.png
│       ├── top_genes.png
│       ├── factors_with_genes.png
│       ├── gene_enrichment.png
│       ├── enrichment_factor_*.png
│       └── enrichment_by_group/
│
└── SVGP/                   # groups=false output
    ├── model.pth            # (pickle also works for non-MGGP)
    ├── model.pkl
    ├── factors.npy, scales.npy, loadings.npy
    ├── Z.npy, Lu.pt         # Inducing data (NO groupsZ)
    ├── moran_i.csv          # (NO gene_enrichment, NO group loadings)
    ├── metrics.json
    └── figures/              # (NO enrichment or group plots)
```

---

## Key Config Properties (`config.py`)

```python
config = Config.from_yaml("configs/slideseq/svgp_test.yaml")

config.spatial      # bool: model.spatial (default False)
config.groups       # bool: model.groups (default False)
config.prior        # str: model.prior (e.g., "SVGP")
config.model_name   # str: "pnmf" | "SVGP" | "MGGP_SVGP" (used for output dir)
config.to_pnmf_kwargs()  # dict: merged model+training kwargs for PNMF constructor
```

The `groups` config field maps to PNMF's `multigroup` parameter:
- `groups: true` → `multigroup=True` → uses `MGGP_SVGP` + `batched_MGGP_Matern32`
- `groups: false` → `multigroup=False` → uses `SVGP` + `batched_Matern32` + K-means inducing points

---

## Key Design Decisions

### 1. Models Live in PNMF, Analysis Lives Here

This package does NOT implement models or training loops. It provides:
- Dataset loaders and preprocessing
- Config management
- Analysis pipeline (metrics, factors, loadings, enrichment)
- Figure generation

### 2. Spatial Model Variants

| Variant | PNMF params | GPzoo classes | Groups required |
|---------|-------------|---------------|-----------------|
| **MGGP_SVGP** | `spatial=True, multigroup=True` | `MGGP_SVGP`, `batched_MGGP_Matern32` | Yes |
| **SVGP** | `spatial=True, multigroup=False` | `SVGP`, `batched_Matern32` | No |
| **pnmf** | `spatial=False` | None (GaussianPrior) | No |

### 3. Model Loading in analyze.py

`_load_model()` detects MGGP vs SVGP by checking if `groupsZ` exists in the prior state dict:
- `groupsZ` present → reconstruct `MGGP_SVGP` + `batched_MGGP_Matern32`
- `groupsZ` absent → reconstruct `SVGP` + `batched_Matern32`

### 4. Groups vs No-Groups Pipeline Behavior

When `groups: false`:
- **train**: Only passes `coordinates` to `model.fit()` (no `groups` arg)
- **analyze**: Skips group-specific loadings, gene enrichment, groupsZ saving
- **figures**: Skips enrichment plots, group plots still show data groups (from C.npy) but no inducing groups

---

## CLI Usage

```bash
# Install
pip install -e .

# Single stage
spatial_factorization preprocess -c configs/slideseq/pnmf.yaml      # Run once per dataset
spatial_factorization train      -c configs/slideseq/svgp_test.yaml  # Model-specific
spatial_factorization analyze    -c configs/slideseq/svgp_test.yaml
spatial_factorization figures    -c configs/slideseq/svgp_test.yaml

# Chain multiple stages with `run`
spatial_factorization run train analyze figures -c configs/slideseq/svgp_test.yaml
spatial_factorization run preprocess train analyze figures -c configs/slideseq/pnmf.yaml
```

Stages passed to `run` are automatically sorted into pipeline order (`preprocess → train → analyze → figures`).

## Available Configs

| Config | Model | Groups | Epochs | Use |
|--------|-------|--------|--------|-----|
| `pnmf.yaml` | Non-spatial PNMF | N/A | 10000 | Baseline |
| `svgp.yaml` | MGGP_SVGP | true | 10000 | Full training |
| `svgp_test.yaml` | MGGP_SVGP | true | 10 | Quick testing |
| `svgp_no_groups_test.yaml` | SVGP | false | 10 | Quick testing |

---

## Implementation Status

| Stage | Status | Description |
|-------|--------|-------------|
| 0 | **DONE** | Setup & Installation |
| 1 | **DONE** | Preprocess command |
| 2 | **DONE** | Train command (PNMF, SVGP, MGGP_SVGP) |
| 3 | **DONE** | Analyze command (Moran's I, reconstruction, group loadings, enrichment) |
| 4 | **DONE** | Figures command (spatial plots, enrichment, gene plots) |

## Relationship to Other Repos

| Repo | Purpose | Key Classes |
|------|---------|-------------|
| **GPzoo** | GP backends | `SVGP`, `MGGP_SVGP`, `batched_Matern32`, `batched_MGGP_Matern32`, `CholeskyParameter` |
| **PNMF** | sklearn API | `PNMF` class, `multigroup` param, `spatial=True`, transforms (`get_factors`, `_get_spatial_qF`, `transform_W`) |
| **Spatial-Factorization** | Pipeline | Dataset loaders, configs, CLI commands, analysis, figures |

### PNMF Recent Changes (no-groups branch)

- `multigroup` default changed from `True` to `False`
- Non-multigroup path uses K-means inducing points (not random)
- Non-multigroup uses `batched_Matern32` kernel + `SVGP` class

---

## Dependencies

```bash
# Core
pip install -e .                          # This package
pip install -e ../Probabilistic-NMF       # PNMF
pip install -e ../GPzoo                   # GP backends

# Or via helper script
./scripts/install_deps.sh
```

## Running Tests

```bash
conda run -n factorization python -m pytest tests/ -v -s
```

Tests in `test_svgp_model.py` require a trained MGGP_SVGP model at `outputs/slideseq/MGGP_SVGP/`.
