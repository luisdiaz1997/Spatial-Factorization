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
│  - spatial=True     │◄────────│  - LCGP, MGGP_LCGP  │
│  - multigroup       │         │  - batched kernels   │
│  - local (LCGP)     │         │  - CholeskyParameter │
└─────────────────────┘         └─────────────────────┘
```

## Package Structure

```
Spatial-Factorization/
├── spatial_factorization/
│   ├── __init__.py
│   ├── __main__.py            # Enables `python -m spatial_factorization`
│   ├── config.py              # Config dataclass with model_name, groups, spatial properties
│   ├── cli.py                 # Click CLI entry point
│   ├── generate.py            # Generate per-model configs from general.yaml
│   ├── runner.py              # Parallel job runner with GPU/CPU resource management
│   ├── status.py              # Live status display for parallel training (rich)
│   ├── commands/
│   │   ├── preprocess.py      # Stage 1: Standardize data format
│   │   ├── train.py           # Stage 2: Train PNMF model
│   │   ├── analyze.py         # Stage 3: Compute metrics, factors, loadings
│   │   └── figures.py         # Stage 4: Generate plots
│   └── datasets/
│       ├── base.py            # SpatialData container + load_preprocessed()
│       ├── slideseq.py        # SlideseqV2 loader
│       └── tenxvisium.py      # 10x Visium loader
├── configs/slideseq/
│   ├── general.yaml           # Superset config for multiplex training (generates 5 models)
│   ├── pnmf.yaml              # Non-spatial baseline
│   ├── svgp.yaml              # SVGP (no groups, full training)
│   ├── mggp_svgp.yaml         # MGGP_SVGP (groups, full training)
│   ├── lcgp.yaml              # LCGP (no groups, generated from general.yaml)
│   ├── mggp_lcgp.yaml         # MGGP_LCGP (groups, generated from general.yaml)
│   ├── svgp_test.yaml         # MGGP_SVGP (10 epochs, testing)
│   ├── svgp_no_groups_test.yaml # SVGP no groups (10 epochs, testing)
│   ├── lcgp_test.yaml         # LCGP (10 epochs, testing)
│   └── mggp_lcgp_test.yaml    # MGGP_LCGP (10 epochs, testing)
├── tests/
│   └── test_svgp_model.py     # SVGP model inspection tests
├── outputs/                   # Generated (git-ignored)
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
| `spatial: true, groups: false, local: false` | `svgp` | `outputs/slideseq/svgp/` |
| `spatial: true, groups: true, local: false` | `mggp_svgp` | `outputs/slideseq/mggp_svgp/` |
| `spatial: true, groups: false, local: true` | `lcgp` | `outputs/slideseq/lcgp/` |
| `spatial: true, groups: true, local: true` | `mggp_lcgp` | `outputs/slideseq/mggp_lcgp/` |

All names are lowercase. The prior name is lowercased, and when `groups: true`, the `mggp_` prefix is added.

### Output Directory Structure

```
outputs/slideseq/
├── preprocessed/           # Shared by all models (Stage 1)
│   ├── X.npy               # (N, 2) spatial coordinates
│   ├── Y.npz               # (D, N) count matrix (sparse)
│   ├── C.npy               # (N,) group codes (integers 0..G-1)
│   └── metadata.json       # gene_names, group_names, etc.
│
├── logs/                   # Multiplex training logs
│   ├── pnmf.log            # stdout/stderr from pnmf training
│   ├── svgp.log            # stdout/stderr from SVGP training
│   ├── mggp_svgp.log       # stdout/stderr from MGGP_SVGP training
│   ├── lcgp.log            # stdout/stderr from LCGP training
│   └── mggp_lcgp.log       # stdout/stderr from MGGP_LCGP training
│
├── run_status.json         # Multiplex run summary (JSON)
│
├── mggp_svgp/              # groups=true output
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
│       ├── points.png
│       ├── elbo_curve.png
│       ├── top_genes.png
│       ├── factors_with_genes.png
│       ├── gene_enrichment.png
│       ├── enrichment_factor_*.png
│       └── enrichment_by_group/
│
└── svgp/                   # groups=false output
    ├── model.pth            # (pickle also works for non-MGGP)
    ├── model.pkl
    ├── factors.npy, scales.npy, loadings.npy
    ├── loadings_group_*.npy # (D, L) per-group loadings (computed post-hoc)
    ├── Z.npy, Lu.pt         # Inducing data (NO groupsZ)
    ├── moran_i.csv
    ├── gene_enrichment.json # LFC per factor per group (computed post-hoc)
    ├── metrics.json
    └── figures/
        ├── points.png       # Data groups + inducing points (no group assignment)
        ├── gene_enrichment.png
        ├── enrichment_factor_*.png
        └── enrichment_by_group/

├── lcgp/                    # LCGP (local=True, groups=false)
│   ├── model.pth            # PyTorch state dict
│   ├── factors.npy, scales.npy, loadings.npy
│   ├── loadings_group_*.npy # Per-group loadings (computed post-hoc)
│   ├── Z.npy                # (N, 2) ALL data points as inducing (M=N)
│   ├── Lu.npy               # (L, N, K) VNNGP-style Lu parameter (S = Lu @ Lu.T)
│   ├── moran_i.csv, gene_enrichment.json, metrics.json
│   └── figures/
│       ├── (same as svgp, but NO lu_scales_inducing.png since M=N)
│       └── points.png       # Data groups only (no inducing overlay)

└── mggp_lcgp/               # MGGP_LCGP (local=True, groups=true)
    ├── model.pth
    ├── factors.npy, scales.npy, loadings.npy
    ├── loadings_group_*.npy
    ├── Z.npy, groupsZ.npy   # All data points + group assignments
    ├── Lu.npy               # (L, N, K) VNNGP-style Lu parameter (S = Lu @ Lu.T)
    ├── moran_i.csv, gene_enrichment.json, metrics.json
    └── figures/
        ├── (same as mggp_svgp, but NO lu_scales_inducing.png)
        └── points.png       # Data groups (no separate inducing overlay)
```

---

## Key Config Properties (`config.py`)

```python
config = Config.from_yaml("configs/slideseq/svgp_test.yaml")

config.spatial      # bool: model.spatial (default False)
config.groups       # bool: model.groups (default False)
config.local        # bool: model.local (default False) - LCGP mode
config.prior        # str: model.prior (used for output dir naming, not passed to PNMF)
config.model_name   # str: "pnmf" | "svgp" | "mggp_svgp" | "lcgp" | "mggp_lcgp"
config.to_pnmf_kwargs()  # dict: merged model+training kwargs for PNMF constructor (no prior)
```

The `groups` config field maps to PNMF's `multigroup` parameter:
- `groups: true` → `multigroup=True` → PNMF uses MGGP variants
- `groups: false` → `multigroup=False` → PNMF uses non-group variants

The `local` config field selects LCGP vs SVGP:
- `local: false` → SVGP variants (sparse variational GP with M inducing points)
- `local: true` → LCGP variants (locally conditioned GP, M=N inducing points)

---

## Key Design Decisions

### 1. Models Live in PNMF, Analysis Lives Here

This package does NOT implement models or training loops. It provides:
- Dataset loaders and preprocessing
- Config management
- Analysis pipeline (metrics, factors, loadings, enrichment)
- Figure generation

### 2. Spatial Model Variants

| Variant | PNMF params | GPzoo classes | Groups | Complexity |
|---------|-------------|---------------|--------|------------|
| **MGGP_SVGP** | `spatial=True, multigroup=True, local=False` | `MGGP_SVGP`, `batched_MGGP_Matern32` | Yes | O(M²) |
| **SVGP** | `spatial=True, multigroup=False, local=False` | `SVGP`, `batched_Matern32` | No | O(M²) |
| **MGGP_LCGP** | `spatial=True, multigroup=True, local=True` | `MGGP_LCGP`, `batched_MGGP_Matern32` | Yes | O(NR) |
| **LCGP** | `spatial=True, multigroup=False, local=True` | `LCGP`, `batched_Matern32` | No | O(NR) |
| **pnmf** | `spatial=False` | None (GaussianPrior) | No | O(NL) |

- **SVGP**: Sparse variational GP with M<<N inducing points, O(M²) covariance
- **LCGP**: Locally conditioned GP with ALL N points as inducing, VNNGP-style S=Lu@Lu.T covariance (K neighbors)

### 3. LCGP vs SVGP Trade-offs

| Aspect | SVGP | LCGP |
|--------|------|------|
| Inducing points | M<<N (e.g., 3000) | M=N (all data) |
| Covariance complexity | O(M²) full Cholesky | O(NK²) VNNGP-style S=Lu@Lu.T |
| GPU memory | ~7.6 GB peak | ~7.6 GB peak |
| Saved Lu | `Lu.pt` (L,M,M) ~344MB | `Lu.npy` (L,N,K) |
| Best for | Large datasets | When full spatial coverage needed |

### 4. Prior Selection and Model Loading

**Training:** The `prior` field in YAML is NOT passed to PNMF. PNMF auto-selects the prior based on `spatial`, `multigroup`, and `local` flags:
- `spatial=False` → `GaussianPrior`
- `spatial=True, local=False, multigroup=False` → `SVGP`
- `spatial=True, local=False, multigroup=True` → `MGGP_SVGP`
- `spatial=True, local=True, multigroup=False` → `LCGP`
- `spatial=True, local=True, multigroup=True` → `MGGP_LCGP`

**Loading:** `_load_model()` detects model type from saved state dict:
1. **LCGP vs SVGP**: Check for raw `Lu` key without `Lu._raw` (LCGP has raw `Lu`, SVGP has `Lu._raw` from CholeskyParameter)
2. **Multigroup vs not**: Check for `groupsZ` key in state dict
3. Reconstruct appropriate GP class with correct kernel and covariance parameter type

The `prior` field in YAML/config is only used for:
- Naming output directories (`model_name` property)
- Print statements during training
- Saved to `model.pth` hyperparameters for documentation

### 5. Factor Ordering by Moran's I

The analyze stage reorders all factor-related outputs by descending Moran's I before saving. Factor 0 always has the highest spatial autocorrelation. This applies to: `factors.npy`, `scales.npy`, `loadings.npy`, `Lu.pt`, group loadings, and gene enrichment. The `moran_i.csv` reflects the new order (already sorted descending).

### 6. Groups vs No-Groups Pipeline Behavior

When `groups: false`:
- **train**: Only passes `coordinates` to `model.fit()` (no `groups` arg)
- **analyze**: Computes group-specific loadings and gene enrichment post-hoc (using data groups from C.npy); skips groupsZ saving
- **figures**: Generates all enrichment plots; `points.png` shows data groups + inducing points (no group coloring)

When `groups: true`:
- **train**: Passes both `coordinates` and `groups` to `model.fit()`
- **analyze**: Computes group-specific loadings and gene enrichment; saves groupsZ.npy
- **figures**: Generates all enrichment plots; `points.png` shows data groups + inducing points colored by group

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

---

## Multiplex Pipeline (Parallel Training)

Run multiple models in parallel with resource-aware GPU/CPU scheduling.

### Usage

```bash
# Generate per-model configs from general.yaml
spatial_factorization generate -c configs/slideseq/general.yaml

# Run all models in parallel (pnmf, SVGP, MGGP_SVGP, LCGP, MGGP_LCGP)
spatial_factorization run all -c configs/slideseq/general.yaml

# Force re-preprocessing
spatial_factorization run all -c configs/slideseq/general.yaml --force

# Dry run (show plan without executing)
spatial_factorization run all -c configs/slideseq/general.yaml --dry-run
```

### Live Status Display

During parallel training, a live-updating table shows progress with separate rows for train and analyze tasks:

```
                                          Training Progress
┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Job         ┃ Task      ┃ Device  ┃ Status    ┃ Epoch       ┃ ELBO             ┃ Time             ┃
┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ lcgp        │ train     │ cuda:0  │ completed │ 200/200     │ -335521443.5     │ 0:00:15/-        │
│ lcgp        │ analyze   │ cuda:0  │ analyzing │ 600/1000    │ -                │ 0:00:00/00:01    │
│ mggp_lcgp   │ train     │ cuda:1  │ training  │ 50/200      │ -49833232.0      │ 0:00:02/0:01:30  │
│ mggp_lcgp   │ analyze   │ pending │ pending   │ -           │ -                │ 0:00:00/-        │
│ mggp_svgp   │ train     │ cpu     │ completed │ 200/200     │ -45042568.0      │ 0:01:43/-        │
│ mggp_svgp   │ analyze   │ cpu     │ analyzing │ 114/1000    │ -                │ 0:00:00/01:42    │
│ pnmf        │ train     │ cuda:0  │ completed │ 200/200     │ -50454488.0      │ 0:00:08/-        │
│ pnmf        │ analyze   │ cuda:0  │ completed │ 82/1000     │ -                │ 0:00:00/00:03    │
│ svgp        │ train     │ cuda:1  │ completed │ 200/200     │ -47521234.5      │ 0:01:29/-        │
│ svgp        │ analyze   │ cuda:1  │ analyzing │ 80/1000     │ -                │ 0:00:00/00:02    │
└─────────────┴───────────┴─────────┴───────────┴─────────────┴──────────────────┴──────────────────┘
```

### Status Module (`status.py`)

| Component | Purpose |
|-----------|---------|
| `JobStatus` | Dataclass tracking name, model, task, device, status, epoch, elbo, remaining_time, elapsed |
| `StatusManager` | Context manager with live `rich` table display |
| `stream_output()` | Non-blocking subprocess stdout capture with tqdm parsing |

Parses both PNMF output formats:
- `verbose=True`: `Iteration 500: ELBO = -12345.67`
- `verbose=False` (tqdm): `5000/10000 [...ELBO=-5.475e+05, ...<02:45...]`

### Resource Management (`runner.py`)

- **Training**: Uses available GPUs (1 job per GPU exclusive) + 1 CPU fallback
- **Analyze**: Same pattern - parallel with GPU + CPU fallback
- **CPU fallback**: When all GPUs are busy, at least one job runs on CPU
- **Training priority**: Training jobs get priority over analyze jobs for GPU/CPU resources. Analyze only starts when no pending training jobs are waiting for resources.
- **Logs**: Each job writes to `outputs/{dataset}/logs/{model}.log`

## Available Configs

| Config | Model | Groups | Local | Epochs | Use |
|--------|-------|--------|-------|--------|-----|
| `general.yaml` | All 5 models | N/A | N/A | 20000 | Multiplex training |
| `general_test.yaml` | All 5 models | N/A | N/A | 200 | **Quick multiplex test** |
| `pnmf.yaml` | Non-spatial PNMF | N/A | N/A | 20000 | Baseline |
| `svgp.yaml` | SVGP | false | false | 20000 | Full training |
| `mggp_svgp.yaml` | MGGP_SVGP | true | false | 20000 | Full training |
| `lcgp.yaml` | LCGP | false | true | 20000 | Full training (generated) |
| `mggp_lcgp.yaml` | MGGP_LCGP | true | true | 20000 | Full training (generated) |
| `svgp_no_groups_test.yaml` | SVGP | false | false | 10 | Quick testing |
| `svgp_test.yaml` | MGGP_SVGP | true | false | 10 | Quick testing |
| `lcgp_test.yaml` | LCGP | false | true | 10 | Quick testing |
| `mggp_lcgp_test.yaml` | MGGP_LCGP | true | true | 10 | Quick testing |

### Testing

**IMPORTANT: Always clean up test outputs before running a new test!**

```bash
# Remove all model outputs (keep preprocessed data)
rm -rf outputs/slideseq_test/pnmf outputs/slideseq_test/svgp outputs/slideseq_test/mggp_svgp \
       outputs/slideseq_test/lcgp outputs/slideseq_test/mggp_lcgp \
       outputs/slideseq_test/logs outputs/slideseq_test/run_status.json

# Or use the helper script
./scripts/clean_test_outputs.sh
```

**IMPORTANT: Use `general_test.yaml` for quick tests, NOT `general.yaml`!**

```bash
# Quick test of full multiplex pipeline (10 epochs, outputs to slideseq_test/)
spatial_factorization run all -c configs/slideseq/general_test.yaml

# Single model quick test
spatial_factorization run train analyze figures -c configs/slideseq/svgp_test.yaml
```

**DO NOT run `general.yaml` for testing** - it trains 20000 epochs and takes hours.

#### Running Tests in Tmux (for live output visibility)

To run tests with visible live output in a split tmux pane:

```bash
# Split current window horizontally and run test on right pane
tmux split-window -t 15:0 -h
tmux send-keys -t 15:0.1 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate factorization && python -m spatial_factorization run all -c configs/slideseq/general_test.yaml' Enter
```

This allows:
- Viewing the live ELBO training progress table
- Monitoring analyze/figures stages in real-time
- Both Claude and user can see output simultaneously

---

## Implementation Status

| Stage | Status | Description |
|-------|--------|-------------|
| 0 | **DONE** | Setup & Installation |
| 1 | **DONE** | Preprocess command |
| 2 | **DONE** | Train command (PNMF, SVGP, MGGP_SVGP, LCGP, MGGP_LCGP) |
| 3 | **DONE** | Analyze command (Moran's I, reconstruction, group loadings, enrichment) |
| 4 | **DONE** | Figures command (spatial plots, enrichment, gene plots) |
| 5 | **DONE** | Multiplex pipeline (parallel training, live status, GPU/CPU scheduling) |
| 6 | **DONE** | LCGP integration (local=True support, VNNGP-style covariance) |

## Relationship to Other Repos

| Repo | Purpose | Key Classes |
|------|---------|-------------|
| **GPzoo** | GP backends | `SVGP`, `MGGP_SVGP`, `LCGP`, `MGGP_LCGP`, `batched_Matern32`, `batched_MGGP_Matern32`, `CholeskyParameter` |
| **PNMF** | sklearn API | `PNMF` class, `multigroup` param, `local` param, `spatial=True`, transforms (`get_factors`, `_get_spatial_qF`, `transform_W`) |
| **Spatial-Factorization** | Pipeline | Dataset loaders, configs, CLI commands, analysis, figures |

### PNMF API (LCGP branch)

- `multigroup` default: `False`
- `local` default: `False` (set `True` for LCGP)
- `local=True` params: `K=50`, `precompute_knn=True`
- Non-multigroup SVGP uses K-means inducing points + `batched_Matern32` kernel
- LCGP uses all data points as inducing (M=N) with VNNGP-style `S = Lu @ Lu.T` covariance

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

Tests in `test_svgp_model.py` require a trained mggp_svgp model at `outputs/slideseq/mggp_svgp/`.
