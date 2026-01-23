# SlideSeq PNMF Planning

## Overview

This plan outlines running **non-spatial PNMF** on SlideSeq data as a first step. We will use:
- **Spatial-Factorization**: Data loading via `SlideseqLoader`
- **PNMF**: Model fitting via sklearn-like API

This is a simpler baseline before adding spatial GP priors.

---

## Development Strategy: 5-Stage Implementation Plan

**Status Legend:** `⬜` NOT DONE | `🟩` DONE

We will implement this in stages, verifying each stage works before moving to the next. Each stage builds on the previous ones.

### Stage 0: Setup & Installation `🟩 DONE`

**Goal:** Ensure the package is installable and CLI entry points work.

```bash
# Create setup.py (see ../Probabilistic-NMF/setup.py for reference)
# All metadata is in pyproject.toml, setup.py just calls setup()

# Verify installation
pip install -e .

# Test CLI entry point
spatial_factorization --help
# Should show: preprocess, train, analyze, figures commands
```

**Deliverables:**
- [x] `setup.py` - Minimal wrapper (see Probabilistic-NMF)
- [x] Update `pyproject.toml` entry point: `spatial_factorization = "spatial_factorization.cli:cli"`
- [x] Add `click>=8.0` to dependencies
- [x] Verify `spatial_factorization --help` works

---

### Stage 1: Preprocess Command `🟩 DONE`

**Goal:** Standardize data format (run once per dataset).

**IMPORTANT:** Preprocessing is dataset-specific, NOT model-specific. The output directory should be named after the dataset (e.g., `outputs/slideseq`), not the model. Model-specific outputs are saved in subdirectories during later stages (e.g., `outputs/slideseq/pnmf/`).

```bash
spatial_factorization preprocess -c configs/slideseq/pnmf.yaml
```

**Expected output:**
```
Preprocessing dataset: slideseq
  Spots (N): 41783, Genes (D): 17702
Preprocessed data saved to: outputs/slideseq/preprocessed/
  X: (41783, 2)
  Y: (17702, 41783)
  C: 14 groups
```

**Files created:**
- `outputs/slideseq/preprocessed/X.npy`
- `outputs/slideseq/preprocessed/Y.npy`
- `outputs/slideseq/preprocessed/C.npy`
- `outputs/slideseq/preprocessed/metadata.json`

**Directory structure:**
```
outputs/
└── slideseq/                    # Dataset-specific (shared by all models)
    ├── preprocessed/            # Standardized data format (Stage 1)
    │   ├── X.npy
    │   ├── Y.npy
    │   ├── C.npy
    │   └── metadata.json
    ├── pnmf/                    # Model-specific outputs (Stage 2+)
    │   ├── model.pkl
    │   └── training.json
    └── lcgp/                    # Model-specific outputs (Stage 2+)
        ├── model.pkl
        └── training.json
```

**Deliverables:**
- [x] `spatial_factorization/commands/preprocess.py`
- [x] `spatial_factorization/datasets/base.py` - added `group_names` and `load_preprocessed()`
- [x] `spatial_factorization/datasets/slideseq.py` - extract real cluster names
- [x] `spatial_factorization/datasets/tenxvisium.py` - extract real cluster names
- [x] `configs/slideseq/pnmf.yaml`
- [x] `tests/test_preprocessed.py` - integration test
- [x] `scripts/setup_env.sh` - environment setup script

---

### Stage 2: Train Command `⬜ NOT DONE`

**Goal:** Train PNMF model and save results.

```bash
spatial_factorization train --config configs/slideseq/pnmf.yaml
```

**Expected output:**
```
Loading preprocessed data: 34000 spots, 1000 genes
Training PNMF with 10 components...
[Iteration 100] ELBO: -1234567.89
[Iteration 200] ELBO: -1234000.00
...
Training complete!
  ELBO: -1230000.00
  Time: 123.4s
  Saved to: outputs/slideseq_pnmf/
```

**Files created:**
- `outputs/slideseq_pnmf/model.pth`
- `outputs/slideseq_pnmf/elbo_history.csv`
- `outputs/slideseq_pnmf/train_metadata.json`

**Deliverables:**
- [ ] `spatial_factorization/commands/train.py`

---

### Stage 3: Analyze Command `⬜ NOT DONE`

**Goal:** Compute metrics (Moran's I, reconstruction error).

```bash
spatial_factorization analyze --config configs/slideseq/pnmf.yaml
```

**Expected output:**
```
Loading preprocessed data...
Loading trained model...
Computing Moran's I for spatial autocorrelation...
Computing reconstruction error...
Analysis complete!
  Moran's I: [0.12, 0.08, ..., 0.15]
  Reconstruction error: 0.234
  Saved to: outputs/slideseq_pnmf/
```

**Files created:**
- `outputs/slideseq_pnmf/metrics.json`
- `outputs/slideseq_pnmf/factors.npy`
- `outputs/slideseq_pnmf/moran_i.csv`

**Deliverables:**
- [ ] `spatial_factorization/commands/analyze.py`
- [ ] `spatial_factorization/analysis.py` - `plot_factors()`, `dims_autocorr()`, etc.

---

### Stage 4: Figures Command `⬜ NOT DONE`

**Goal:** Generate publication figures.

```bash
spatial_factorization figures --config configs/slideseq/pnmf.yaml
```

**Expected output:**
```
Loading preprocessed data...
Loading analysis results...
Generating figures...
  figures/factors_spatial.png
  figures/elbo_curve.png
  figures/top_genes.png
Figures saved to: outputs/slideseq_pnmf/figures/
```

**Files created:**
- `outputs/slideseq_pnmf/figures/factors_spatial.png`
- `outputs/slideseq_pnmf/figures/elbo_curve.png`
- `outputs/slideseq_pnmf/figures/top_genes.png`

**Deliverables:**
- [ ] `spatial_factorization/commands/figures.py`

---

### Summary: Complete Pipeline

```bash
# Stage 0: Install
pip install -e .

# Stage 1: Preprocess (run once per dataset)
spatial_factorization preprocess -c configs/slideseq/pnmf.yaml

# Stage 2: Train
spatial_factorization train --config configs/slideseq/pnmf.yaml

# Stage 3: Analyze
spatial_factorization analyze --config configs/slideseq/pnmf.yaml

# Stage 4: Generate figures
spatial_factorization figures --config configs/slideseq/pnmf.yaml
```

**Note:** Preprocess uses `-c` (short), train/analyze/figures use `--config` (long) - both work for all commands.

---

## Code References (Borrow From Here)

### Data Loading - REUSE DIRECTLY

| What | Path | Notes |
|------|------|-------|
| SlideSeq loader | `Spatial-Factorization/src/spatial_factorization/datasets/slideseq.py` | Use `SlideseqLoader.load()` directly |
| SpatialData container | `Spatial-Factorization/src/spatial_factorization/datasets/base.py` | Has `n_spots`, `n_genes` properties |
| Config dataclasses | `Spatial-Factorization/src/spatial_factorization/config.py` | `DatasetConfig`, `ModelConfig`, `TrainingConfig` |

### PNMF API - REUSE DIRECTLY

| What | Path | Notes |
|------|------|-------|
| PNMF sklearn class | `Probabilistic-NMF/PNMF/models.py:108-536` | Main API: `fit()`, `transform()`, `fit_transform()` |
| ELBO computation | `Probabilistic-NMF/PNMF/elbo.py` | Modes: `simple`, `expanded`, `lower-bound` |
| Usage example | `Probabilistic-NMF/CLAUDE.md:268-286` | Quick verification code |
| Benchmark script | `Probabilistic-NMF/benchmarks/simple_vs_expanded.py` | **USE THIS** - tested patterns for training |

**Key functions from `simple_vs_expanded.py`:**
- `run_benchmark()` (line 56-108): Shows proper PNMF initialization and fit with history
- `generate_synthetic_data()` (line 21-53): Synthetic data generation pattern
- `print_summary()` (line 167-223): Results reporting pattern

### GPzoo Training Scripts - REFERENCE FOR PATTERNS

| What | Path | Notes |
|------|------|-------|
| SVGP training script | `GPzoo/gpzoo/datasets/slideseq/svgp_nsf.py` | Reference for training loop structure |
| Data loading (original) | `GPzoo/gpzoo/datasets/slideseq/common.py:27-64` | `load_slideseq_with_groups()` - already ported to Spatial-Factorization |
| Config values | `GPzoo/gpzoo/datasets/slideseq/config.py` | Hyperparameters: `SEED=67`, `L_FACTORS=10`, `LR=0.01` |
| Run all script | `GPzoo/gpzoo/datasets/slideseq/run_all.sh` | Shows how multiple models are launched |
| Training utilities | `GPzoo/gpzoo/training_utilities.py` | `run_training()` pattern for saving results |
| **Analysis utilities** | `GPzoo/gpzoo/utilities.py` | `dims_autocorr()` for Moran's I, `rescale_spatial_coords()` |

### GPzoo Notebooks - ANALYSIS/FIGURE PATTERNS

| What | Path | Notes |
|------|------|-------|
| Liver MGGP notebook | `GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb` | `plot_factors()`, `get_groupwise_factors()`, `plot_groupwise_factors()` |
| SlideSeq MGGP notebook | `GPzoo/notebooks/Slideseqv2_MGGP_november.ipynb` | `plot_factors()`, `plot_top_genes()`, `dims_autocorr()` usage |
| Model saving pattern | Both notebooks | `torch.save(model.state_dict(), 'path.pth')` |

### Key Code Snippets to Copy

**From GPzoo config (hyperparameters):**
```python
# GPzoo/gpzoo/datasets/slideseq/config.py:21-30
SEED = 67
STEPS = 10000
X_BATCH = 34000
Y_BATCH = 1000
L_FACTORS = 10
LR = 1e-2
SPATIAL_SCALE = 50.0
LENGTHSCALE = 4.0
```

**From GPzoo common.py (results saving pattern):**
```python
# GPzoo/gpzoo/datasets/slideseq/common.py:86-96
def _save_losses(losses, weights_path: Path) -> Dict[str, str]:
    csv_path = base.with_name(base.stem + "_losses.csv")
    npy_path = base.with_name(base.stem + "_losses.npy")
    df = pd.DataFrame({"step": np.arange(len(losses)), "loss": np.asarray(losses)})
    df.to_csv(csv_path, index=False)
    np.save(npy_path, np.asarray(losses))
```

**From PNMF benchmark (run_benchmark pattern):**
```python
# Probabilistic-NMF/benchmarks/simple_vs_expanded.py:56-108
def run_benchmark(mode='expanded', n_components=5, max_iter=100, random_state=42,
                  verbose=False, optimizer='Adam', learning_rate=0.01):
    # Initialize and fit model
    model = PNMF(
        n_components=n_components,
        mode=mode,
        loadings_mode='projected',
        E=10,
        max_iter=max_iter,
        tol=1e-4,
        learning_rate=learning_rate,
        optimizer=optimizer,
        random_state=random_state,
        verbose=verbose
    )

    elbo_history, model = model.fit(X, return_history=True)

    # Compute reconstruction error using exp(qF.mean) @ W.T
    exp_F_mean = np.exp(model._prior.mean.detach().cpu().numpy().T)  # (N, L)
    W = model.components_.T  # (D, L)
    X_reconstructed = exp_F_mean @ W.T  # (N, D)
    reconstruction_error = np.linalg.norm(X - X_reconstructed, 'fro') / np.linalg.norm(X, 'fro')

    return {
        'mode': mode,
        'n_iterations': model.n_iter_,
        'final_elbo': model.elbo_,
        'elbo_history': elbo_history,
        'reconstruction_error': reconstruction_error,
    }
```

---

## Current State Analysis

### What Exists

**Spatial-Factorization repo:**
- `SlideseqLoader` in `src/spatial_factorization/datasets/slideseq.py` - loads data from squidpy
- `SpatialData` container with `X` (coordinates), `Y` (counts)
- Config system in `src/spatial_factorization/config.py`

**PNMF repo:**
- `PNMF` class with sklearn API: `fit()`, `transform()`, `fit_transform()`
- Supports ELBO modes: `'simple'`, `'expanded'`, `'lower-bound'`
- Supports training modes: `'standard'`, `'natural'`
- Does NOT yet support `spatial=True` (planned for future)

**GPzoo SlideSeq training (reference):**
- Uses spatial GP models (SVGP, VNNGP, LCGP) from `gpzoo.models`
- Complex training with freezing/unfreezing schedules
- Tensorboard logging, checkpointing

### Data Flow Summary

```
SlideseqV2 (squidpy)
        │
        ▼
┌─────────────────┐
│ SlideseqLoader  │  (Spatial-Factorization)
│ - QC filtering  │
│ - MT gene filter│
│ - Coord scaling │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   SpatialData   │
│ - X: (N, 2)     │  coordinates
│ - Y: (D, N)     │  counts (genes x spots)
│ - C: (N,)       │  group codes
└────────┬────────┘

Note: For non-spatial PNMF, X and C are IGNORED during training
      but USED in analysis (Moran's I) and plotting (spatial figures)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│      PNMF       │  (Probabilistic-NMF)
│ - sklearn API   │
│ - non-spatial   │
│ X ≈ exp(F) @ W.T│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Results      │
│ - components_   │  (L, D) loadings
│ - transformed   │  (N, L) factors
│ - elbo_         │  final ELBO
└─────────────────┘
```

---

## Implementation Plan

### Step 1: Create CLI and Commands Module

**`spatial_factorization/cli.py`:**
```python
"""CLI entry point for spatial_factorization using Click."""
import click


@click.group()
@click.version_option()
def cli():
    """Spatial transcriptomics factorization toolkit."""
    pass


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def preprocess(config):
    """Preprocess dataset into standardized format (run once)."""
    from .commands import preprocess as preprocess_cmd
    preprocess_cmd.run(config)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def train(config):
    """Train a PNMF model."""
    from .commands import train as train_cmd
    train_cmd.run(config)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def analyze(config):
    """Analyze a trained model."""
    from .commands import analyze as analyze_cmd
    analyze_cmd.run(config)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def figures(config):
    """Generate publication figures."""
    from .commands import figures as figures_cmd
    figures_cmd.run(config)


if __name__ == '__main__':
    cli()
```

**`spatial_factorization/commands/train.py`:**
```python
"""Generic training command - works with any dataset."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PNMF import PNMF
from ..config import Config
from ..datasets import load_dataset


def run(config_path: str):
    """Train a PNMF model from config."""
    config = Config.from_yaml(config_path)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load data (dataset-agnostic!)
    print(f"Loading dataset: {config.dataset.name}")
    data = load_dataset(config.dataset)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Prepare for sklearn API
    Y_sklearn = data.Y.T.numpy()  # (N, D)

    # Build model from config
    model = PNMF(
        n_components=config.model.n_components,
        mode=config.model.mode,
        loadings_mode=config.model.loadings_mode,
        max_iter=config.training.max_iter,
        learning_rate=config.training.learning_rate,
        tol=config.training.tol,
        random_state=config.seed,
        verbose=config.training.verbose,
    )

    # Train
    print(f"Training PNMF with {config.model.n_components} components...")
    t0 = time.perf_counter()
    history, model = model.fit(Y_sklearn, return_history=True)
    train_time = time.perf_counter() - t0

    # Save results
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model state
    torch.save(model._model.state_dict(), output_dir / "model.pth")

    # Save history
    df = pd.DataFrame({"iteration": range(len(history)), "elbo": history})
    df.to_csv(output_dir / "elbo_history.csv", index=False)

    # Save metadata
    metadata = {
        "config": config_path,
        "n_components": model.n_components_,
        "n_spots": data.n_spots,
        "n_genes": data.n_genes,
        "final_elbo": float(model.elbo_),
        "n_iterations": model.n_iter_,
        "train_time_sec": train_time,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_dir / "train_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete!")
    print(f"  ELBO: {model.elbo_:.2f}")
    print(f"  Time: {train_time:.1f}s")
    print(f"  Saved to: {output_dir}")
```

### Step 2: Create Config File

```yaml
# configs/slideseq/pnmf.yaml
name: slideseq_pnmf
seed: 67

dataset:
  name: slideseq
  spatial_scale: 50.0
  filter_mt: true
  min_counts: 100
  min_cells: 10

model:
  n_components: 10
  spatial: false           # Non-spatial PNMF
  mode: expanded           # 'simple', 'expanded', or 'lower-bound'
  loadings_mode: projected # 'softplus', 'exp', or 'projected'

training:
  max_iter: 500
  learning_rate: 0.01
  tol: 1e-4
  verbose: true

output:
  dir: outputs/slideseq_pnmf
```

### Step 3: Add Entry Point and Click Dependency to pyproject.toml

```toml
[project.scripts]
spatial_factorization = "spatial_factorization.cli:cli"

[project.dependencies]
# ... existing deps ...
click = ">=8.0"
```

---

## Key Differences from GPzoo Training

| Aspect | GPzoo (SVGP_NSF) | PNMF (non-spatial) |
|--------|------------------|-------------------|
| Prior on F | Spatial GP (SVGP) | Standard Gaussian |
| Coordinates | Used in GP kernel | Ignored (for now) |
| Training | Complex freezing schedules | Simple Adam |
| Output | Spatial factor maps | Factor values (no spatial smoothing) |
| Complexity | ~200 lines config | ~50 lines |

---

## Expected Results

For SlideSeq data (~34k spots, ~1000 genes after filtering):

1. **Data dimensions**:
   - N ≈ 34,000 spots
   - D ≈ 1,000 genes
   - L = 10 components

2. **Training time**:
   - ~5-10 min on MPS/CUDA for 500 iterations
   - Faster with `mode='lower-bound'`

3. **ELBO**:
   - Should converge to stable value
   - `mode='expanded'` typically gives best final ELBO
   - `mode='lower-bound'` converges fastest

4. **Reconstruction**:
   - MSE should decrease during training
   - Non-spatial model won't capture spatial patterns

---

## Library CLI Interface

The package provides CLI commands via entry points. All commands are **dataset-agnostic** - only the config and dataset loaders are dataset-specific.

```bash
# Install the library
pip install -e .

# Four-stage pipeline via CLI
spatial_factorization preprocess -c configs/slideseq/pnmf.yaml  # Run once
spatial_factorization train      -c configs/slideseq/pnmf.yaml
spatial_factorization analyze    -c configs/slideseq/pnmf.yaml
spatial_factorization figures    -c configs/slideseq/pnmf.yaml
```

---

## Preprocessing: Standardized Data Format

Each dataset has different naming conventions for groups (cluster, region, cell_type, Cell_Type, etc.). The `preprocess` command standardizes everything into a common format that train/analyze/figures can load quickly.

### Standardized Output Format

```python
# Saved to: outputs/{dataset}/preprocessed/
# ├── X.npy          # (N, 2) spatial coordinates
# ├── Y.npy          # (D, N) count matrix (genes x spots)
# ├── C.npy          # (N,) group codes (integers 0..G-1)
# ├── metadata.json  # gene_names, spot_names, group_names, etc.
```

### Metadata JSON Structure

```json
{
  "n_spots": 34000,
  "n_genes": 1000,
  "n_groups": 14,
  "gene_names": ["Gene1", "Gene2", ...],
  "spot_names": ["AAACCT...", ...],
  "group_names": ["Astrocytes", "CA1_CA2_CA3", ...],
  "dataset": "slideseq",
  "preprocessing": {
    "spatial_scale": 50.0,
    "filter_mt": true,
    "min_counts": 100,
    "min_cells": 10
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### Dataset-Specific Group Column Mapping

| Dataset | Raw Column | Example Values |
|---------|------------|----------------|
| SlideSeq | `cluster` | "Astrocytes", "CA1_CA2_CA3_Subiculum" |
| 10x Visium | `region` | "cortex", "hippocampus" |
| Liver MERFISH | `Cell_Type` | "Hepatocyte 1", "LSEC", "Mac 1" |

Each dataset loader handles the mapping to standardized `C` (integer codes) and `group_names` (category labels).

### Preprocess Command Implementation

```python
# spatial_factorization/commands/preprocess.py
"""Preprocess dataset into standardized format."""
import json
import time
from pathlib import Path

import numpy as np

from ..config import Config
from ..datasets import load_dataset


def run(config_path: str):
    """Preprocess dataset and save standardized files."""
    config = Config.from_yaml(config_path)

    print(f"Preprocessing dataset: {config.dataset.name}")
    data = load_dataset(config.dataset)

    # Output directory
    output_dir = config.output_dir / "preprocessed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save arrays
    np.save(output_dir / "X.npy", data.X.numpy())      # (N, 2)
    np.save(output_dir / "Y.npy", data.Y.numpy())      # (D, N)

    # Groups: convert to integer codes
    if data.groups is not None:
        np.save(output_dir / "C.npy", data.groups.numpy())  # (N,)
    else:
        # No groups - create dummy single group
        np.save(output_dir / "C.npy", np.zeros(data.n_spots, dtype=np.int64))

    # Metadata
    metadata = {
        "n_spots": data.n_spots,
        "n_genes": data.n_genes,
        "n_groups": data.n_groups if data.n_groups > 0 else 1,
        "gene_names": data.gene_names,
        "spot_names": data.spot_names,
        "group_names": data.group_names if hasattr(data, 'group_names') else None,
        "dataset": config.dataset.name,
        "preprocessing": {
            "spatial_scale": config.dataset.spatial_scale,
            "filter_mt": config.dataset.filter_mt,
            "min_counts": config.dataset.min_counts,
            "min_cells": config.dataset.min_cells,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Preprocessed data saved to: {output_dir}")
    print(f"  X: {data.X.shape}")
    print(f"  Y: {data.Y.shape}")
    print(f"  C: {data.n_groups} groups")
```

### Loading Preprocessed Data

```python
# spatial_factorization/datasets/preprocessed.py
"""Load preprocessed data (fast path for train/analyze/figures)."""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


@dataclass
class PreprocessedData:
    """Container for preprocessed spatial data."""
    X: torch.Tensor           # (N, 2) spatial coordinates
    Y: torch.Tensor           # (D, N) count matrix
    C: torch.Tensor           # (N,) group codes
    n_groups: int
    gene_names: List[str]
    spot_names: List[str]
    group_names: Optional[List[str]]

    @property
    def n_spots(self) -> int:
        return self.X.shape[0]

    @property
    def n_genes(self) -> int:
        return self.Y.shape[0]


def load_preprocessed(output_dir: Path) -> PreprocessedData:
    """Load preprocessed data from directory."""
    prep_dir = output_dir / "preprocessed"

    X = torch.from_numpy(np.load(prep_dir / "X.npy"))
    Y = torch.from_numpy(np.load(prep_dir / "Y.npy"))
    C = torch.from_numpy(np.load(prep_dir / "C.npy"))

    with open(prep_dir / "metadata.json") as f:
        meta = json.load(f)

    return PreprocessedData(
        X=X, Y=Y, C=C,
        n_groups=meta["n_groups"],
        gene_names=meta["gene_names"],
        spot_names=meta["spot_names"],
        group_names=meta.get("group_names"),
    )
```

### Updated Train Command (Uses Preprocessed)

```python
# spatial_factorization/commands/train.py
def run(config_path: str):
    config = Config.from_yaml(config_path)

    # Load preprocessed data (fast!)
    from ..datasets.preprocessed import load_preprocessed
    data = load_preprocessed(config.output_dir)

    print(f"Loaded preprocessed data: {data.n_spots} spots, {data.n_genes} genes")

    # ... rest of training
```

### Pipeline Flow

```
┌─────────────────┐
│  0. PREPROCESS  │  (run once per dataset)
│                 │
│ - Load raw data │
│ - QC filtering  │
│ - Standardize   │
│   X, Y, V, C    │
└────────┬────────┘
         │
         ▼
   preprocessed/
   ├── X.npy
   ├── Y.npy
   ├── C.npy
   └── metadata.json
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   1. TRAIN      │     │   2. ANALYZE    │     │   3. FIGURES    │
│                 │     │                 │     │                 │
│ - Load preproc  │     │ - Load preproc  │     │ - Load preproc  │
│ - Fit PNMF      │────▶│ - Load model    │────▶│ - Load analysis │
│ - Save .pth     │     │ - Moran's I     │     │ - plot_factors  │
│ - Save history  │     │ - Reconstruction│     │ - plot_top_genes│
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
  model.pth              metrics.json            figures/*.svg
  elbo_history.csv       factors.npy
```

### CLI Entry Points (pyproject.toml)

```toml
[project.scripts]
spatial_factorization = "spatial_factorization.cli:cli"

[project.dependencies]
click = ">=8.0"
```

### CLI Implementation (using Click)

```python
# spatial_factorization/cli.py
import click
from .commands import train, analyze, figures


@click.group()
@click.version_option()
def cli():
    """Spatial transcriptomics factorization toolkit."""
    pass


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def train(config):
    """Train a PNMF model."""
    from .commands import train as train_cmd
    train_cmd.run(config)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def analyze(config):
    """Analyze a trained model (Moran's I, reconstruction, etc.)."""
    from .commands import analyze as analyze_cmd
    analyze_cmd.run(config)


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to config YAML')
def figures(config):
    """Generate publication figures."""
    from .commands import figures as figures_cmd
    figures_cmd.run(config)


if __name__ == '__main__':
    cli()
```

**Click advantages:**
- Cleaner decorator-based syntax
- Automatic `--help` generation
- Built-in path validation (`type=click.Path(exists=True)`)
- Short options (`-c` for `--config`)

---

## Model Saving and Loading

### Saving (from GPzoo notebooks)

```python
# GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb (cell 39)
# GPzoo/notebooks/Slideseqv2_MGGP_november.ipynb (cell 50)
torch.save(model.state_dict(), 'outputs/slideseq_pnmf/model.pth')
```

### Loading (pattern from notebooks)

```python
# To load a saved model:
# 1. Recreate the model architecture
# 2. Load state dict
model = PNMF(n_components=10, ...)
model._prior = GaussianPrior(...)  # Recreate prior
model._model = PoissonFactorization(...)  # Recreate model
model._model.load_state_dict(torch.load('outputs/slideseq_pnmf/model.pth'))
```

### What Gets Saved

For PNMF, the state dict contains:
- `prior.mean`: Variational mean (L, N)
- `prior._scale_raw`: Variational scale (L, N)
- `W._raw`: Loadings matrix (D, L)

---

## Analysis Code References

### From GPzoo Notebooks - REUSE THESE

| Function | Path | Notes |
|----------|------|-------|
| `plot_factors()` | `GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb` (cell 2) | Spatial factor visualization |
| `dims_autocorr()` | `GPzoo/gpzoo/utilities.py` | Computes Moran's I for each factor |
| `plot_top_genes()` | `GPzoo/notebooks/Slideseqv2_MGGP_november.ipynb` (cell 70) | Factors + top genes by loading |
| `get_groupwise_factors()` | `GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb` (cell 41) | **TODO: Future spatial models** |
| `plot_groupwise_factors()` | `GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb` (cell 42) | **TODO: Future spatial models** |

### Key Code Snippets

**plot_factors() - from liver notebook:**
```python
# GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb (cell 2)
def plot_factors(factors, X, moran_idx=None, ax=None, size=7, alpha=0.8, s=0.1,
                 names=None, vmin=None, vmax=None, cmap='turbo'):
    if vmin is None:
        vmin = np.percentile(factors, 1)
    if vmax is None:
        vmax = np.percentile(factors, 99)

    if moran_idx is not None:
        factors = factors[moran_idx]

    L = len(factors)
    if ax is None:
        fig, ax = plt.subplots(2, 6, figsize=(size*6, size*2), tight_layout=True)

    for i in range(L):
        curr_ax = ax[i // 6, i % 6]
        curr_ax.scatter(X[:, 0], X[:, 1], c=factors[i], vmin=vmin, vmax=vmax,
                        alpha=alpha, cmap=cmap, s=s)
        curr_ax.invert_yaxis()
        curr_ax.set_xticks([])
        curr_ax.set_yticks([])
        curr_ax.set_facecolor('xkcd:gray')
```

**dims_autocorr() - Moran's I computation:**
```python
# GPzoo/gpzoo/utilities.py
from gpzoo.utilities import dims_autocorr

# Usage:
moran_idx, moranI = dims_autocorr(factors, X)
# moran_idx: indices sorted by Moran's I (descending)
# moranI: Moran's I values for each factor
```

---

## Groupwise Functions (Spatial Only)

These functions will be in the codebase but only run when `spatial=True`:

```python
# src/spatial_factorization/analysis.py

def get_groupwise_factors(model, X, groups, L=12):
    """Condition GP on different group identities.

    Only works for spatial models (MGGP). For non-spatial PNMF,
    this function should not be called.
    """
    if not hasattr(model, 'spatial') or not model.spatial:
        raise ValueError("get_groupwise_factors requires spatial=True")
    # ... implementation from liver notebook cell 41

def plot_groupwise_factors(results, X, categories, mean, L=12, ...):
    """Visualize how factors change per cell type.

    Only meaningful for spatial models with group conditioning.
    """
    # ... implementation from liver notebook cell 42
```

**Usage pattern:**
```python
# In analyze_pnmf.py
if model.spatial:
    results = get_groupwise_factors(model, data.X, data.groups)
    # ... save groupwise results
else:
    print("Skipping groupwise analysis (non-spatial model)")
```

### Code to Copy

**get_groupwise_factors() - from liver notebook cell 41:**
```python
# GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb
def get_groupwise_factors(model, X_test, codes, L=12):
    model.cpu()
    model.eval()

    with torch.no_grad():
        group_ids = codes.unique().tolist()
        results = []

        knn_idx = model.prior.calculate_knn(X_test)
        model.prior.knn_idx = knn_idx[:, :-1]

        for group_index in tqdm(group_ids):
            group_mask = (codes == group_index)
            groupsX_test = torch.full_like(codes, fill_value=group_index)

            qF_test, _, _ = model.prior(X_test, groupsX=groupsX_test)
            factors_test = torch.exp(qF_test.mean).cpu().numpy()

            results.append({
                'group_index': group_index,
                'group_mask': group_mask.cpu().numpy(),
                'factors': factors_test
            })

        return results
```

**plot_groupwise_factors() - from liver notebook cell 42:**
```python
# GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb
def plot_groupwise_factors(results, X_test, categories, mean, L=12, size=2,
                           s=0.2, alpha=0.9, vmin=0, vmax=2.0):
    cols = [f"Factor {i}" for i in range(1, L + 1)]
    nrows = len(results)

    fig, axes = plt.subplots(nrows=nrows+1, ncols=L + 1,
                             figsize=(size * (L + 1), size * (nrows+1)),
                             squeeze=False)

    # Row 0: mean factors
    axes[0, 0].axis("off")
    for j in range(L):
        axes[0, j + 1].scatter(X_test[:, 0], X_test[:, 1], c=mean[j],
                               vmin=vmin, vmax=vmax, alpha=alpha, s=s, cmap='turbo')
        axes[0, j + 1].invert_yaxis()
        axes[0, j + 1].set(xticks=[], yticks=[])
        axes[0, j + 1].set_title(cols[j], fontsize=10)

    # Rows 1+: per-group factors
    for i, result in enumerate(results):
        row_idx = i + 1
        group_index = result['group_index']
        group_mask = result['group_mask']
        factors_test = result['factors']
        display_name = ' '.join(categories[group_index].split('_')[:2])

        # Column 0: group mask
        cell_fake_factor = np.zeros_like(factors_test[0])
        cell_fake_factor[group_mask] = 1.0
        axes[row_idx, 0].scatter(X_test[:, 0], X_test[:, 1], c=cell_fake_factor,
                                 vmin=0, vmax=1, alpha=alpha, s=s, cmap='copper')
        axes[row_idx, 0].invert_yaxis()
        axes[row_idx, 0].set(xticks=[], yticks=[])
        axes[row_idx, 0].set_ylabel(display_name, rotation=0, fontsize=10, labelpad=50)

        # Columns 1+: factors conditioned on this group
        for j in range(L):
            axes[row_idx, j + 1].scatter(X_test[:, 0], X_test[:, 1], c=factors_test[j],
                                         vmin=vmin, vmax=vmax, alpha=alpha, s=s, cmap='turbo')
            axes[row_idx, j + 1].invert_yaxis()
            axes[row_idx, j + 1].set(xticks=[], yticks=[])

    fig.tight_layout()
    return fig
```

---

## Next Steps (After Baseline)

1. **Add spatial support to PNMF**:
   - Implement `spatial=True` parameter
   - Replace `GaussianPrior` with GP-based prior from GPzoo
   - Use spatial coordinates `X` in GP kernel

2. **Compare spatial vs non-spatial**:
   - Same L, same data
   - Compare ELBO, reconstruction, factor maps

3. **Integrate with GPzoo GP backends**:
   - Import `SVGP`, `VNNGP`, `LCGP` from `gpzoo.gp`
   - Create `SpatialPrior` class wrapping GP

---

## File Checklist

### Library Structure (To Create)

**CLI & Commands**
- [ ] `spatial_factorization/cli.py` - Main CLI entry point (Click)
- [ ] `spatial_factorization/commands/__init__.py`
- [ ] `spatial_factorization/commands/preprocess.py` - Standardize data format (run once)
- [ ] `spatial_factorization/commands/train.py` - Generic training command
- [ ] `spatial_factorization/commands/analyze.py` - Generic analysis command
- [ ] `spatial_factorization/commands/figures.py` - Generic figure generation command

**Preprocessed Data Loader**
- [ ] `spatial_factorization/datasets/preprocessed.py` - Load preprocessed .npy files (fast path)

**Analysis Utilities**
- [ ] `spatial_factorization/analysis.py` - Analysis functions:
  - `plot_factors()` - Spatial factor visualization
  - `plot_top_genes()` - Factors + top genes by loading
  - `get_groupwise_factors()` - Condition on groups (spatial only)
  - `plot_groupwise_factors()` - Per-group factor maps (spatial only)

**Package Setup**
- [ ] `pyproject.toml` - Add CLI entry point: `spatial_factorization = "spatial_factorization.cli:main"`

**Configs (per dataset)**
- [ ] `configs/slideseq/pnmf.yaml` - SlideSeq PNMF config

### Already Exists ✓
- [x] `spatial_factorization/datasets/slideseq.py` - SlideSeq loader
- [x] `spatial_factorization/datasets/tenxvisium.py` - 10x Visium loader
- [x] `spatial_factorization/datasets/base.py` - SpatialData container
- [x] `spatial_factorization/config.py` - Config dataclasses

### Dependencies
- [ ] Add `click>=8.0` to pyproject.toml
- [ ] Ensure PNMF is installed: `pip install -e ../Probabilistic-NMF`
- [ ] Ensure squidpy/scanpy available for data loading
- [ ] Ensure GPzoo is installed (for `dims_autocorr`): `pip install -e ../GPzoo`

---

## Quick Start Commands

```bash
# Install the library and dependencies
cd Spatial-Factorization
pip install -e .
pip install -e ../Probabilistic-NMF
pip install -e ../GPzoo  # for dims_autocorr

# Run the three-stage pipeline
spatial_factorization train   --config configs/slideseq/pnmf.yaml
spatial_factorization analyze --config configs/slideseq/pnmf.yaml
spatial_factorization figures --config configs/slideseq/pnmf.yaml

# View results
ls outputs/slideseq_pnmf/
```

---

## Summary

This plan provides a minimal first step:
1. **Load SlideSeq data** using existing `SlideseqLoader`
2. **Run non-spatial PNMF** using the sklearn-like API
3. **Save and analyze results**

This establishes a baseline before adding spatial GP priors. The non-spatial PNMF will factor the count matrix but won't leverage spatial coordinates - that's the next phase.

---

## Exact Code Structure (Copy-Paste Ready)

### Imports

```python
# Standard
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# From Spatial-Factorization (this repo)
from spatial_factorization.datasets.slideseq import SlideseqLoader
from spatial_factorization.config import DatasetConfig

# From Probabilistic-NMF
from PNMF import PNMF
```

### Data Loading (from slideseq.py)

```python
# Reuse SlideseqLoader directly - no changes needed
config = DatasetConfig(
    name="slideseq",
    spatial_scale=50.0,  # From GPzoo config.py:33
    filter_mt=True,
    min_counts=100,
    min_cells=10,
)
loader = SlideseqLoader()
data = loader.load(config)

# Data shapes:
# data.X: (N, 2) - spatial coordinates (ignored in training, used in analysis/plotting)
# data.Y: (D, N) - genes x spots (used in training)
# data.C: (N,)   - group codes (ignored in training, used in analysis/plotting)
```

### PNMF Fitting (from PNMF/models.py)

```python
# Transpose for sklearn API: (N, D)
Y_sklearn = data.Y.T.numpy()

# Create model with same hyperparams as GPzoo
model = PNMF(
    n_components=10,           # GPzoo: L_FACTORS = 10
    mode='expanded',           # Best ELBO
    training_mode='standard',  # Or 'natural' for better convergence
    max_iter=500,              # Shorter than GPzoo's 10000 (no spatial)
    learning_rate=0.01,        # GPzoo: LR = 1e-2
    loadings_mode='projected', # GPzoo: LOADINGS_MODE = 'projected'
    random_state=67,           # GPzoo: SEED = 67
    verbose=False,
)

# Fit with history (from PNMF/models.py:278-283)
history, model = model.fit(Y_sklearn, return_history=True)

# Results
print(f"ELBO: {model.elbo_}")
print(f"Components: {model.components_.shape}")  # (L, D)
```

### Results Saving (adapted from GPzoo common.py:86-96)

```python
def save_results(model, history, output_dir: Path, label: str):
    """Save model results - pattern from GPzoo/gpzoo/datasets/slideseq/common.py."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save components (loadings)
    np.save(output_dir / f"{label}_components.npy", model.components_)

    # Save ELBO history (adapted from _save_losses)
    df = pd.DataFrame({
        "iteration": np.arange(len(history)),
        "elbo": np.asarray(history),
    })
    df.to_csv(output_dir / f"{label}_elbo_history.csv", index=False)
    np.save(output_dir / f"{label}_elbo_history.npy", np.asarray(history))

    # Save metadata
    metadata = {
        "n_components": model.n_components_,
        "n_features": model.n_features_in_,
        "final_elbo": float(model.elbo_),
        "n_iterations": model.n_iter_,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(output_dir / f"{label}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata
```

---

## File Mapping (What Goes Where)

```
Probabilistic-NMF/
├── PNMF/
│   ├── models.py      ← PNMF class (use as-is)
│   └── elbo.py        ← ELBO computation (use as-is)
├── benchmarks/
│   └── simple_vs_expanded.py  ← Training patterns (copy from here)

GPzoo/
├── gpzoo/
│   ├── utilities.py   ← dims_autocorr(), rescale_spatial_coords() (import)
│   └── datasets/slideseq/
│       ├── config.py      ← Hyperparameters (copy values)
│       ├── common.py      ← save_losses pattern (adapt)
│       └── svgp_nsf.py    ← Training script pattern (reference)
├── notebooks/
│   ├── liver_mggp_healthy_matern32_umap_init.ipynb  ← plot_factors(), groupwise (copy)
│   └── Slideseqv2_MGGP_november.ipynb               ← plot_top_genes() (copy)

Spatial-Factorization/                    ← INSTALLABLE LIBRARY
├── pyproject.toml                        ← Package config + CLI entry point
├── spatial_factorization/
│   ├── __init__.py
│   ├── config.py                         ← Config dataclasses (exists)
│   ├── cli.py                            ← TO CREATE: CLI entry point
│   ├── analysis.py                       ← TO CREATE: plot_factors, etc.
│   ├── commands/                         ← TO CREATE: CLI commands
│   │   ├── __init__.py
│   │   ├── train.py                      ← Generic training (any dataset)
│   │   ├── analyze.py                    ← Generic analysis (any dataset)
│   │   └── figures.py                    ← Generic figures (any dataset)
│   └── datasets/                         ← DATASET-SPECIFIC (only this!)
│       ├── __init__.py
│       ├── base.py                       ← SpatialData container (exists)
│       ├── slideseq.py                   ← SlideSeq loader (exists)
│       └── tenxvisium.py                 ← 10x Visium loader (exists)
├── configs/                              ← YAML configs per dataset
│   ├── slideseq/
│   │   └── pnmf.yaml
│   └── tenxvisium/
│       └── pnmf.yaml
└── outputs/                              ← Generated by CLI (git-ignored)
    └── {dataset}_{model}/
        ├── model.pth
        ├── elbo_history.csv
        ├── metrics.json
        ├── factors.npy
        └── figures/
```

### Key Insight: Only `datasets/` is Dataset-Specific

Everything else (CLI, commands, analysis) is generic and works with any dataset:
- `train.py` reads config → calls `load_dataset(config.dataset)` → fits PNMF → saves
- `analyze.py` reads config → loads model → computes metrics → saves
- `figures.py` reads config → loads analysis → generates plots → saves

Adding a new dataset only requires:
1. New loader in `spatial_factorization/datasets/newdataset.py`
2. New config in `configs/newdataset/pnmf.yaml`
