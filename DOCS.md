# Spatial-Factorization ReadTheDocs Plan

This document is the detailed plan for the Spatial-Factorization ReadTheDocs site. It follows the same Sphinx/RST structure as PNMF's docs. Each section below maps to an `.rst` file (or major section within one) and describes what content to include.

---

## Site Structure

```
docs/
├── conf.py
├── requirements.txt
├── index.rst              ← Landing page + toctree
├── installation.rst       ← Install from source; deps
├── quickstart.rst         ← End-to-end 5-minute example
├── math.rst               ← Model math (PNMF + GP priors)
├── models.rst             ← The 5 model variants + comparison
├── configuration.rst      ← YAML config reference (all fields)
├── cli.rst                ← Full CLI reference (all commands)
├── datasets.rst           ← All 7 dataset loaders
├── pipeline.rst           ← Stage-by-stage pipeline walkthrough
├── multiplex.rst          ← Parallel runner, GPU scheduling, status
├── advanced.rst           ← Resume, probabilistic KNN, video, multianalyze
├── outputs.rst            ← File-by-file output reference
└── examples/
    ├── slideseq.rst       ← Slide-seq reference walkthrough
    ├── merfish.rst        ← MERFISH large-scale example
    ├── sdmbench.rst       ← SDMBench 12-slide multi-dataset
    └── colon.rst          ← Colon (1.2M cells) scale example
```

---

## `index.rst` — Landing Page

**Purpose:** First impression. What is this package, why use it.

**Content:**
- One-paragraph description: "Spatial-Factorization is a four-stage CLI pipeline for spatial transcriptomics analysis. It applies probabilistic non-negative matrix factorization (PNMF) with Gaussian process (GP) priors to decompose spatial gene expression into spatially coherent latent factors."
- Bullet list of key features:
  - Five model variants: non-spatial PNMF, SVGP, MGGP_SVGP, LCGP, MGGP_LCGP
  - Seven dataset loaders (Slide-seq, MERFISH, Visium, liver, colon, osmFISH, SDMBench)
  - Four-stage pipeline: `preprocess → train → analyze → figures`
  - Parallel multi-model/multi-dataset runner with GPU/CPU scheduling
  - Checkpoint resume, probabilistic KNN, factor comparison (`multianalyze`)
- Dependency diagram showing the three-package ecosystem:
  ```
  Spatial-Factorization (pipeline/CLI)
       ├── PNMF (sklearn-API model)
       └── GPzoo (GP backends)
  ```
- toctree linking all pages

---

## `installation.rst` — Installation

**Content:**

### Prerequisites
- Python 3.14
- CUDA-capable GPU recommended (training SVGP/LCGP on large datasets)
- Conda recommended: `factorization` environment

### Install from Source
```bash
# Clone all three repos
git clone https://github.com/luisdiaz1997/Spatial-Factorization.git
git clone https://github.com/luisdiaz1997/Probabilistic-NMF.git
git clone https://github.com/luisdiaz1997/GPzoo.git

# Create conda env
conda create -n factorization python=3.14
conda activate factorization

# Install dependencies in order
pip install -e Probabilistic-NMF/
pip install -e GPzoo/
pip install -e Spatial-Factorization/
```

### Verify Installation
```bash
spatial_factorization --version
spatial_factorization --help
```

### Optional Dependencies
- `faiss-gpu` (instead of `faiss-cpu`) for faster KNN on GPU
- `rich` for live status display during parallel training
- `matplotlib`, `seaborn` for figure generation

---

## `quickstart.rst` — Quick Start

**Purpose:** Someone should be able to run the full pipeline in ~15 minutes on the Slide-seq dataset.

**Content:**

### 5-Minute Pipeline

1. Generate per-model configs:
```bash
spatial_factorization generate -c configs/slideseq/general.yaml
```

2. Run the full pipeline (all 5 models in parallel):
```bash
spatial_factorization run all -c configs/slideseq/general_test.yaml
```

3. Check outputs:
```
outputs/slideseq/
├── svgp/figures/factors_spatial.png    ← Spatial factor maps
├── svgp/figures/top_genes.png          ← Top loading genes per factor
├── svgp/moran_i.csv                    ← Spatial autocorrelation
└── svgp/metrics.json                   ← Reconstruction error, deviance
```

### Single Model Example
```bash
# Preprocess (once per dataset)
spatial_factorization preprocess -c configs/slideseq/svgp_test.yaml

# Train, analyze, plot
spatial_factorization run train analyze figures -c configs/slideseq/svgp_test.yaml
```

### Reading Results
```python
import numpy as np

factors = np.load("outputs/slideseq/svgp/factors.npy")   # (N, L) spatial factors
loadings = np.load("outputs/slideseq/svgp/loadings.npy") # (D, L) gene loadings
```

---

## `math.rst` — Mathematical Background

**Purpose:** Describe the model precisely for readers who want to understand what is being computed.

### PNMF Base Model

Given spatial transcriptomics data Y ∈ ℝ₊^{N×D} (N spots, D genes):

```math
y_{ij} ∼ Poisson(λ_{ij})

λ_{ij} = Σ_ℓ W_{jℓ} exp(F_{iℓ})
```

Where:
- **W** ∈ ℝ₊^{D×L}: gene loadings (non-negative)
- **F** ∈ ℝ^{N×L}: latent factors (real-valued, log-space)
- **L**: number of components

### Variational Inference

Approximate posterior q(F) is Gaussian with learnable mean μ and variance σ²:

```math
F_{iℓ} ∼ q(F_{iℓ}) = N(μ_{iℓ}, σ²_{iℓ})
```

ELBO maximized over μ, σ², W:

```math
L = E_{q(F)}[log p(Y|F)] - KL[q(F) || p(F)]
```

Expected log-likelihood uses moment-generating function for the analytic term:

```math
E[exp(F_{iℓ})] = exp(μ_{iℓ} + σ²_{iℓ}/2)
```

### GP Priors

All spatial models replace the isotropic Gaussian prior p(F) with a GP prior over spatial coordinates X ∈ ℝ^{N×2}:

```math
f_ℓ(·) ∼ GP(0, k(·, ·))
```

The kernel k is **Matérn-3/2** by default:

```math
k(x, x') = σ²(1 + √3 r / ℓ) exp(-√3 r / ℓ),   r = ||x - x'||
```

Parameters: σ (signal variance, fixed), ℓ (lengthscale, trainable optionally).

### SVGP (Sparse Variational GP)

Induces on M << N inducing points Z ∈ ℝ^{M×2}. Variational distribution q(u) = N(m, S):

```math
q(f | u) = N(K_{NM} K_{MM}^{-1} m,  K_{NN} - K_{NM} K_{MM}^{-1}(K_{MM} - S) K_{MM}^{-1} K_{MN})
```

Covariance stored as Cholesky Lu (L, M, M). Memory/compute: O(NM + M²) per component.

### MGGP_SVGP (Multi-Group SVGP)

Extends SVGP with per-group kernel. Given group labels C ∈ {0,...,G-1}^N and group-specific inducing points:

```math
k_MGGP((x,c), (x',c')) = k_base(x,x') · ρ(c,c')
```

where ρ(c,c) = 1 (within group), ρ(c,c') = exp(-group_diff_param) (across groups).

Inducing points Z have associated group labels groupsZ. Stored: Lu (L, M, M), groupsZ (M,).

### LCGP (Locally Conditioned GP)

Uses all N training points as inducing (M=N). Covariance is VNNGP-style:

```math
S = Lu @ Lu^T,   Lu ∈ ℝ^{L×N×K}
```

K-nearest neighbors define the sparse structure. Memory: O(NK²) per component.
KNN can be computed via:
- **FAISS L2** (`neighbors='knn'`): deterministic, fast
- **Kernel-weighted probabilistic** (`neighbors='probabilistic'`): samples neighborhoods proportional to k(x_i, z_j), Gumbel-max trick

### MGGP_LCGP

Combines local conditioning with multi-group kernel. groupsZ = group labels of all N training points.

### Factor Ordering

All factors are sorted by descending **Moran's I** after training. Factor 0 has the highest spatial autocorrelation. This reordering is applied to factors, scales, loadings, Lu, groupsZ, and gene enrichment.

---

## `models.rst` — Model Variants

**Content:**

### Comparison Table

| Model | PNMF flags | Prior | Groups | M | Complexity |
|-------|-----------|-------|--------|---|------------|
| `pnmf` | `spatial=False` | GaussianPrior | No | — | O(NL) |
| `svgp` | `spatial=True, local=False, multigroup=False` | SVGP | No | M<<N (e.g. 3000) | O(NM + M²) |
| `mggp_svgp` | `spatial=True, local=False, multigroup=True` | MGGP_SVGP | Yes | M<<N | O(NM + M²) |
| `lcgp` | `spatial=True, local=True, multigroup=False` | LCGP | No | M=N | O(NK²) |
| `mggp_lcgp` | `spatial=True, local=True, multigroup=True` | MGGP_LCGP | Yes | M=N | O(NK²) |

### When to Use Each Model

- **pnmf**: Baseline. Use when spatial structure is not expected or as a sanity check.
- **svgp**: Standard sparse GP. Efficient for large N (N~40K–300K). Good default spatial model.
- **mggp_svgp**: Use when cell types/tissue regions are annotated and you want group-aware covariance.
- **lcgp**: Use when you need full-data coverage and can afford O(NK²) memory.
- **mggp_lcgp**: LCGP + group-aware. Most expressive spatial model.

### Key Hyperparameters

For SVGP/MGGP_SVGP:
- `num_inducing` (M): Number of inducing points. Default 3000. Clamped to N if N < 3000.
- `lengthscale`: GP kernel lengthscale in spatial units. Default 8.0 (tuned for Slide-seq).
- `kernel`: `Matern32` (default) or `RBF`

For LCGP/MGGP_LCGP:
- `K`: Number of local neighbors per point. Default 50.
- `neighbors`: `knn` (FAISS L2, deterministic) or `probabilistic` (kernel-weighted sampling)

For MGGP variants:
- `group_diff_param`: Cross-group kernel penalty. Higher = more independent groups. Default 1.0.

---

## `configuration.rst` — Configuration Reference

**Purpose:** Complete reference for every YAML field.

### Config File Structure

```yaml
name: slideseq               # Experiment name (used in logs)
seed: 67                     # Random seed for reproducibility
dataset: slideseq            # Dataset key (see Datasets page)
output_dir: outputs/slideseq # Where to write all outputs

preprocessing:
  spatial_scale: 50.0        # Coordinate normalization factor
  filter_mt: true            # Drop mitochondrial genes (slideseq)
  min_counts: 100            # Min total counts per cell
  min_cells: 10              # Min cells expressing a gene
  min_group_fraction: 0.01   # Drop groups < 1% of cells (set or min_group_size)
  min_group_size: 10         # Alternative: drop groups < N cells
  subsample: null            # Subsample cells to this count (null = no subsampling)

model:
  # Universal
  n_components: 10           # L: number of latent factors
  mode: expanded             # ELBO mode: 'expanded' | 'simple' | 'lower-bound'
  loadings_mode: multiplicative  # W constraint: 'projected' | 'softplus' | 'exp' | 'multiplicative'
  training_mode: standard    # 'standard' or 'natural_gradient'
  E: 3                       # Monte Carlo samples for ELBO

  # Spatial (all spatial models)
  spatial: true              # Enable GP prior (false for pnmf)
  groups: false              # Enable MGGP (multigroup) mode
  local: false               # Enable LCGP (local conditioning) mode
  kernel: Matern32           # 'Matern32' or 'RBF'
  lengthscale: 8.0           # GP kernel lengthscale
  sigma: 1.0                 # GP signal variance (fixed during training)
  train_lengthscale: false   # Whether to optimize lengthscale
  group_diff_param: 1.0      # MGGP cross-group penalty (MGGP only)

  # SVGP-specific
  num_inducing: 3000         # M: number of inducing points
  cholesky_mode: exp         # Cholesky parameterization: 'exp' | 'softplus'
  diagonal_only: false       # Use diagonal-only covariance (faster, less expressive)
  inducing_allocation: derived  # How to distribute inducing points across groups

  # LCGP-specific
  K: 50                      # Number of local neighbors
  precompute_knn: true       # Precompute KNN at training start
  neighbors: knn             # Neighbor strategy: 'knn' | 'probabilistic'

  # ELBO scaling
  scale_ll_D: true           # Scale log-likelihood by 1/D
  scale_kl_NM: true          # Scale KL by N/M

  # Naming override (optional)
  model_name_override: null  # Override output directory name

training:
  max_iter: 20000            # Maximum training iterations
  learning_rate: 2e-3        # Adam learning rate
  optimizer: Adam            # Optimizer (only Adam supported)
  tol: 1e-4                  # Convergence tolerance
  verbose: false             # Print per-iteration ELBO
  device: gpu                # 'cpu', 'gpu', or 'cuda:N'
  batch_size: 7000           # Cells per mini-batch (null = full batch)
  y_batch_size: 2000         # Genes per mini-batch (null = full batch)
  shuffle: true              # Shuffle mini-batches each epoch
  analyze_batch_size: 10000  # Cells per batch in analyze GP forward pass
  video_interval: 20         # Iterations between video frame captures
```

### General vs Per-Model Configs

A **general config** (e.g. `general.yaml`) has no `model.spatial` key. It contains all hyperparameters for all model variants. Running `spatial_factorization generate` expands it into 5 per-model YAML files (pnmf, svgp, mggp_svgp, lcgp, mggp_lcgp), each with the correct `spatial`/`groups`/`local` flags injected.

A **per-model config** (e.g. `svgp.yaml`) has `model.spatial: true|false` set explicitly and runs exactly one model.

### `model_name` Resolution

The output directory name is determined by:

| Config flags | `model_name` | Output dir |
|-------------|-------------|-----------|
| `spatial: false` | `pnmf` | `outputs/<dataset>/pnmf/` |
| `spatial: true, groups: false, local: false` | `svgp` | `outputs/<dataset>/svgp/` |
| `spatial: true, groups: true, local: false` | `mggp_svgp` | `outputs/<dataset>/mggp_svgp/` |
| `spatial: true, groups: false, local: true` | `lcgp` | `outputs/<dataset>/lcgp/` |
| `spatial: true, groups: true, local: true` | `mggp_lcgp` | `outputs/<dataset>/mggp_lcgp/` |
| `model_name_override: foo` | `foo` | `outputs/<dataset>/foo/` |

---

## `cli.rst` — CLI Reference

**Purpose:** Full reference for every command and flag, with examples.

### `preprocess`

```bash
spatial_factorization preprocess -c CONFIG
```

Standardize raw dataset into `{output_dir}/preprocessed/`:
- `X.npy` — (N, 2) spatial coordinates
- `Y.npz` — (D, N) count matrix (sparse)
- `C.npy` — (N,) group codes (integers 0..G-1)
- `metadata.json` — gene names, group names, filter stats

Run once per dataset. Subsequent model runs reuse this data.

### `train`

```bash
spatial_factorization train -c CONFIG [--resume] [--video] [--probabilistic]
```

Flags:
- `--resume`: Warm-start from `model.pth` checkpoint. Appends to ELBO history. Falls back to scratch if no checkpoint exists.
- `--video`: Capture factor snapshots every `video_interval` iterations and save as `video_frames.npy`.
- `--probabilistic`: Override KNN strategy to probabilistic for LCGP/MGGP_LCGP training. No-op for other models.

Outputs: `model.pth`, `model.pkl` (when picklable), `training.json`, `elbo_history.csv`.

### `analyze`

```bash
spatial_factorization analyze -c CONFIG [--probabilistic]
```

Flags:
- `--probabilistic`: Reload model with probabilistic KNN (LCGP/MGGP_LCGP only). Overwrites analyze outputs in place.

Outputs: `factors.npy`, `scales.npy`, `loadings.npy`, `loadings_group_*.npy`, `Z.npy`, `Lu.pt` (SVGP) or `Lu.npy` (LCGP), `moran_i.csv`, `gene_enrichment.json`, `metrics.json`.

### `figures`

```bash
spatial_factorization figures -c CONFIG [--no-heatmap]
```

Flags:
- `--no-heatmap`: Skip `celltype_gene_loadings.png` and `factor_gene_loadings.png` (slow for large D).

Outputs: `figures/factors_spatial.png`, `figures/scales_spatial.png`, `figures/top_genes.png`, `figures/elbo_curve.png`, `figures/gene_enrichment.png`, `figures/points.png`, and per-factor `enrichment_factor_*.png`.

### `generate`

```bash
spatial_factorization generate -c general.yaml
```

Reads a general config and writes 5 per-model YAML files into the same directory. Prints generated file paths.

### `run`

```bash
spatial_factorization run STAGES... -c CONFIG [OPTIONS]
```

**STAGES**: `preprocess`, `train`, `analyze`, `figures`, or `all`. Multiple stages run in pipeline order.

**CONFIG behavior:**
- Per-model YAML: runs that single model
- General YAML: generates per-model configs, runs all in parallel
- Directory: recursively finds files matching `--config-name`. General files are expanded; per-model files are used as-is. Falls back to every `*.yaml` if no match.

**Key flags:**
- `--config-name NAME`: Filename to search when config is a directory (default: `general.yaml`)
- `--skip-general`: Ignore general configs; treat all non-general `*.yaml` as per-model
- `--resume`: Warm-start from checkpoint when available; train from scratch otherwise
- `--force`: Re-run preprocessing even if preprocessed data exists
- `--probabilistic`: Override LCGP KNN to probabilistic for train and analyze stages
- `--video`: Capture training video frames
- `--gpu-only`: Never fall back to CPU
- `--failed`: Re-run only jobs that failed in the previous `run_status.json`
- `--dry-run`: Show plan without executing
- `--no-heatmap`: Skip heatmap figures

**Examples:**
```bash
# All models for one dataset
spatial_factorization run all -c configs/slideseq/general.yaml

# Quick test (10 epochs)
spatial_factorization run all -c configs/slideseq/general_test.yaml

# All datasets at once
spatial_factorization run all -c configs/ --config-name general_test.yaml

# Resume + probabilistic KNN for all LCGP models across datasets
spatial_factorization run all -c configs/ --config-name mggp_lcgp.yaml --resume --probabilistic

# Single model, specific stages
spatial_factorization run train analyze figures -c configs/slideseq/svgp.yaml

# Re-run failed jobs
spatial_factorization run all -c configs/ --failed

# Dry run to inspect plan
spatial_factorization run all -c configs/slideseq/general.yaml --dry-run
```

### `multianalyze`

```bash
spatial_factorization multianalyze -c CONFIG MODEL1 MODEL2 [MODEL3...] [OPTIONS]
```

Compare matched factors across trained models.

**2-model mode:** Greedy pairwise matching. Shows top `--n-pairs` matched pairs as [2D | 3D] side-by-side.

**3+-model mode:** Finds the single best reference factor between MODEL1 and MODEL2 (or `--match-against`), then matches it against all remaining models. Layout: top row = 2D spatial, bottom row = 3D surface; one column per model.

Flags:
- `--n-pairs N`: Number of matched pairs to show (2-model mode only). Default 2.
- `--match-against MODEL`: Which model to use as reference in 3+ model mode. Default: MODEL2.
- `--output PATH`: Output file path. Default: `{output_dir}/figures/comparison_*.png`.

**Examples:**
```bash
# Compare SVGP and MGGP_SVGP — 2 pairs
spatial_factorization multianalyze -c configs/slideseq/general.yaml svgp mggp_svgp

# All 5 models side-by-side
spatial_factorization multianalyze -c configs/slideseq/general.yaml \
    svgp mggp_svgp pnmf lcgp mggp_lcgp

# Custom reference model
spatial_factorization multianalyze -c configs/slideseq/general.yaml \
    svgp pnmf mggp_svgp lcgp mggp_lcgp --match-against mggp_svgp
```

---

## `datasets.rst` — Datasets

**Content:**

### Supported Datasets

| Key | Source | N | D | Groups column |
|-----|--------|---|---|---------------|
| `slideseq` | `sq.datasets.slideseq_v2()` | ~41K | ~4K | `obs["cluster"]` |
| `tenxvisium` | `sq.datasets.visium_hne_adata()` | ~3K | ~15K | `obs["cluster"]` |
| `sdmbench` | h5ad path | ~4.2K | ~33K | `obs["Region"]` |
| `merfish` | `sq.datasets.merfish()` | 73K | 161 | `obs["Cell_class"]` |
| `liver` | h5ad path | 90K | 317 | `obs["Cell_Type"]` |
| `liver_diseased` | h5ad path | 310K | 317 | `obs["Cell_Type_final"]` |
| `osmfish` | h5ad path | 4.8K | 33 | `obs["ClusterName"]` |
| `colon` | h5ad + CSV labels | 1.2M | 492 | CSV column `cl46v1SubShort_ds` |

### Config Fields per Dataset

Each dataset is configured via the `dataset` key and `preprocessing` block. Dataset-specific fields (paths, column names) go in `preprocessing`:

```yaml
# SDMBench example
dataset: sdmbench
preprocessing:
  path: /path/to/151507.h5ad

# Liver example
dataset: liver
preprocessing:
  path: /path/to/adata_healthy_merfish.h5ad
  cell_type_column: Cell_Type       # or Cell_Type_final for diseased

# Colon example
dataset: colon
preprocessing:
  path: /path/to/colon.h5ad
  labels_path: /path/to/labels.csv
  subsample: null   # or integer to subsample
```

### Preprocessing Filters

**NaN filter**: drops cells where coordinates, expression, or group code is NaN.

**Small-group filter**: drops groups smaller than `min_group_fraction` (fraction of total cells) or `min_group_size` (absolute). Surviving group codes are re-encoded contiguously 0..G'-1.

**Coordinate normalization**: coordinates are divided by `spatial_scale` (default 50.0) so that lengthscale=8.0 corresponds to ~400 spatial units.

### SDMBench (12 slides)

SDMBench includes 12 DLPFC slides (151507–151676). Each has its own config directory:
```
configs/sdmbench/
├── 151507/general.yaml
├── 151508/general.yaml
...
└── 151676/general.yaml
```

Run all 12 slides in parallel:
```bash
spatial_factorization run all -c configs/sdmbench/ --config-name general_test.yaml
```

---

## `pipeline.rst` — Pipeline Stages

**Purpose:** Detailed walkthrough of what each stage does, what it reads, what it writes.

### Stage 1: Preprocess

**Input:** Raw dataset (loaded via dataset-specific loader)
**Output:** `{output_dir}/preprocessed/`

Steps:
1. Load dataset with appropriate loader (squidpy, h5ad path, etc.)
2. Extract spatial coordinates from `obsm["spatial"]` (or `obsm["X_spatial"]` for liver)
3. Extract group codes from the configured `obs` column
4. Apply NaN filter (drop cells with NaN in coords, expression, or group)
5. Apply small-group filter (drop rare groups, re-encode codes)
6. Normalize coordinates by `spatial_scale`
7. Optionally subsample cells
8. Save `X.npy`, `Y.npz`, `C.npy`, `metadata.json`

### Stage 2: Train

**Input:** `{output_dir}/preprocessed/`, config YAML
**Output:** `{output_dir}/{model_name}/`

Steps:
1. Load preprocessed data
2. Clamp `num_inducing`, `batch_size`, `y_batch_size` to data dimensions
3. Construct PNMF model (`PNMF(**config.to_pnmf_kwargs())`)
4. For spatial models: pass `coordinates` (and optionally `groups`) to `model.fit()`
5. Save `model.pth`, `model.pkl` (when picklable), `training.json`, `elbo_history.csv`, `config.yaml`

Note: MGGP models (mggp_svgp, mggp_lcgp) cannot be pickled due to a local class wrapper in GPzoo. They always fall back to `.pth`.

### Stage 3: Analyze

**Input:** `{model_dir}/model.pth`, `{output_dir}/preprocessed/`
**Output:** `{model_dir}/`

Steps:
1. Load model from `model.pth` (reconstructs GP class from state dict)
2. Compute factors (N, L) and scales (N, L) via batched GP forward pass
3. Compute Moran's I per factor
4. Sort all outputs by descending Moran's I
5. Compute global loadings (D, L) via `transform_W`
6. Compute per-group loadings for each group
7. Compute gene enrichment (log fold change per factor per group)
8. Compute reconstruction error and Poisson deviance
9. Save all outputs

### Stage 4: Figures

**Input:** All numpy files from analyze
**Output:** `{model_dir}/figures/`

Figures generated:
- `factors_spatial.png`: L panels of 2D spatial factor maps
- `scales_spatial.png`: L panels of factor uncertainty
- `elbo_curve.png`: ELBO convergence over training iterations
- `points.png`: Data group scatter + inducing points (SVGP) or just groups (LCGP)
- `top_genes.png`: Top 10 genes per factor (bar chart of loadings)
- `factors_with_genes.png`: Factor maps annotated with top gene names
- `gene_enrichment.png`: Heatmap of LFC across groups and factors
- `enrichment_factor_*.png`: Per-factor enrichment bar charts
- `enrichment_by_group/*.png`: Per-group enrichment bar charts
- `lu_scales_inducing.png`: Variational covariance scale at inducing points (SVGP only)
- `celltype_gene_loadings.png`: Heatmap of group-specific loadings (skipped with `--no-heatmap`)
- `factor_gene_loadings.png`: Factor × gene loadings heatmap (skipped with `--no-heatmap`)

---

## `multiplex.rst` — Multiplex Pipeline

**Purpose:** Explain how parallel training works: job scheduling, live status, logging.

### How Parallelism Works

The `run all` command with a general config or directory spawns a `JobRunner` which:
1. Collects all per-model configs (expanding general configs via `generate`)
2. Runs preprocessing sequentially once per unique `output_dir`
3. Dispatches training jobs to available GPUs (1 job per GPU) + CPU fallback
4. After each training job completes, dispatches its analyze job
5. After analyze completes, dispatches figures
6. Maintains a live `rich` status table throughout

### Resource Scheduling

- **GPU assignment**: each GPU runs at most one training job at a time (exclusive)
- **CPU fallback**: at least one slot is always available on CPU when all GPUs are occupied
- **Training priority**: training jobs always get resources before analyze/figures jobs (analyze only starts when no pending training jobs are waiting)
- **Multi-dataset**: when `-c configs/` is used, each dataset's preprocessing runs first in that dataset's `output_dir`, then models train in parallel across all datasets

### Live Status Table

```
                                Training Progress
┏━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Job       ┃ Task      ┃ Device  ┃ Status    ┃ Epoch     ┃ ELBO             ┃ Time        ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ lcgp      │ train     │ cuda:0  │ training  │ 50/200    │ -335521443.5     │ 0:00:15/-   │
│ mggp_lcgp │ train     │ cuda:1  │ training  │ 30/200    │ -49833232.0      │ 0:00:02/-   │
│ pnmf      │ train     │ cpu     │ completed │ 200/200   │ -50454488.0      │ 0:00:08/-   │
│ pnmf      │ analyze   │ cpu     │ analyzing │ 82/1000   │ -                │ 0:00:03/-   │
└───────────┴───────────┴─────────┴───────────┴───────────┴──────────────────┴─────────────┘
```

### Logs

Each job writes to `{output_dir}/logs/{model_name}.log`. For multi-dataset runs, each dataset gets its own `logs/` directory.

### Run Status

After completion, `run_status.json` records:
- Which jobs succeeded/failed
- Training time per job
- Final ELBO per job

Use `--failed` flag in subsequent runs to re-run only failed jobs.

---

## `advanced.rst` — Advanced Usage

### Resume Training

```bash
spatial_factorization train --resume -c configs/slideseq/svgp.yaml
```

`--resume` warm-starts from `model.pth`:
- Loads saved prior and W
- Continues ELBO history (appends to existing CSV)
- Falls back to training from scratch if no checkpoint exists (safe for batch runs)

**Internals:** `_create_warm_start_pnmf` subclasses PNMF to inject the loaded prior instead of random initialization, then runs `fit()` normally. After training, `model.__class__` is reset to `PNMF` so the model can be pickled identically to a normal run.

### Probabilistic KNN (LCGP/MGGP_LCGP)

By default LCGP uses FAISS L2 to find K nearest neighbors. The `--probabilistic` flag switches to kernel-weighted probabilistic neighbor sampling (Gumbel-max trick):

```bash
# Re-analyze with probabilistic KNN (outputs overwrite in place)
spatial_factorization analyze --probabilistic -c configs/slideseq/lcgp.yaml
spatial_factorization figures -c configs/slideseq/lcgp.yaml

# Resume training using probabilistic KNN
spatial_factorization train --resume --probabilistic -c configs/slideseq/lcgp.yaml

# All LCGP models across all datasets
spatial_factorization run all -c configs/ --config-name mggp_lcgp.yaml --resume --probabilistic
```

Once `--probabilistic` is used during training, the saved checkpoint records `neighbors: probabilistic`. Future plain `analyze` calls will read this and use probabilistic KNN automatically.

Note: `--probabilistic` is a no-op for non-LCGP models (pnmf, svgp, mggp_svgp).

### Training Video

```bash
spatial_factorization train --video -c configs/slideseq/svgp.yaml
```

Captures factor snapshots every `video_interval` iterations (default 20). Saves `video_frames.npy` (n_frames, N, L) and `video_frame_iters.npy` to the model directory. The `figures` stage renders these into an animation.

### Multi-Model Factor Comparison

```bash
# Compare SVGP vs MGGP_SVGP — best 3 matched pairs
spatial_factorization multianalyze -c configs/slideseq/general.yaml \
    svgp mggp_svgp --n-pairs 3

# All 5 models — reference factor from svgp matched against all others
spatial_factorization multianalyze -c configs/slideseq/general.yaml \
    svgp mggp_svgp pnmf lcgp mggp_lcgp
```

Factor matching uses greedy normalized L2 distance between `factors.npy` arrays. Output figure is written to `{output_dir}/figures/comparison_*.png`.

### Auto-Clamping

`num_inducing`, `batch_size`, and `y_batch_size` are automatically clamped to the dataset dimensions at training time. This means configs written for large datasets (e.g. `num_inducing=3000`) work on small datasets (e.g. osmFISH N=4.8K) without manual adjustment.

### ELBO Scaling Flags

Two flags control ELBO normalization (default behavior is both `True`):
- `scale_ll_D: true` — scale log-likelihood by 1/D (number of genes)
- `scale_kl_NM: true` — scale KL divergence by N/M

Set these to `false` for video demos or direct comparison with older checkpoints.

---

## `outputs.rst` — Output Files Reference

**Purpose:** Describe every file produced by the pipeline.

### Preprocessed (shared by all models)

| File | Shape | Description |
|------|-------|-------------|
| `X.npy` | (N, 2) | Spatial coordinates (normalized) |
| `Y.npz` | (D, N) | Count matrix (sparse CSR) |
| `C.npy` | (N,) | Group codes (0..G-1) |
| `metadata.json` | — | gene_names, group_names, N, D, G, filter stats |

### Training Outputs

| File | Description |
|------|-------------|
| `model.pth` | PyTorch state dict (always saved) |
| `model.pkl` | Full model pickle (pnmf/svgp/lcgp only; MGGP models not picklable) |
| `training.json` | n_components, ELBO, training_time, converged, n_iterations, timestamp |
| `elbo_history.csv` | iteration, elbo — one row per training iteration |
| `config.yaml` | Snapshot of config used for this run |

### Analyze Outputs

| File | Shape | Description |
|------|-------|-------------|
| `factors.npy` | (N, L) | exp(F) factor values, sorted by Moran's I |
| `scales.npy` | (N, L) | Factor uncertainty (std of q(F)) |
| `loadings.npy` | (D, L) | Global gene loadings W |
| `loadings_group_*.npy` | (D, L) | Per-group gene loadings (one file per group) |
| `Z.npy` | (M, 2) | Inducing point coordinates (SVGP: M<<N; LCGP: M=N=all data) |
| `groupsZ.npy` | (M,) | Inducing point group assignments (MGGP only) |
| `Lu.pt` | (L, M, M) | Cholesky variational covariance (SVGP) |
| `Lu.npy` | (L, N, K) | VNNGP-style covariance (LCGP) |
| `moran_i.csv` | — | factor_idx, moran_i — sorted descending |
| `gene_enrichment.json` | — | LFC per factor per group |
| `metrics.json` | — | reconstruction_error, poisson_deviance, moran_i_mean |

### Runner Outputs

| File | Description |
|------|-------------|
| `logs/{model}.log` | stdout/stderr from training subprocess |
| `run_status.json` | Per-job success/failure, ELBO, training_time |

---

## `examples/slideseq.rst` — Slide-seq Example

**Purpose:** Full end-to-end walkthrough with explanation of every step.

Content:
1. Dataset description (Slide-seq V2 mouse cerebellum, ~41K spots, ~4K genes)
2. Config setup: `general.yaml` → `generate` → per-model configs
3. Running the full pipeline (`run all`)
4. Interpreting the live status table
5. Reading factor outputs
6. Comparing models with `multianalyze`
7. Annotating factors with gene names

---

## `examples/merfish.rst` — MERFISH Example

**Purpose:** Large-scale dataset with many cell types and 161 genes.

Content:
1. Dataset description (73K cells, 161 genes, 12 cell type groups)
2. Why MGGP_SVGP is natural here (explicit cell type annotations)
3. Batch size tuning (`batch_size: 15000, y_batch_size: 161`)
4. Running with `--resume` to extend training
5. Gene enrichment interpretation

---

## `examples/sdmbench.rst` — SDMBench Multi-Slide Example

**Purpose:** 12-slide parallel training.

Content:
1. Dataset description (12 DLPFC Visium slides, ~4K spots, ~33K genes)
2. Running all 12 slides at once: `run all -c configs/sdmbench/ --config-name general_test.yaml`
3. Per-slide `run_status.json` inspection
4. Cross-slide factor comparison (can compare factors across slides with `multianalyze`)

---

## `examples/colon.rst` — Large-Scale Colon Example

**Purpose:** 1.2M cell dataset — extreme scale challenges.

Content:
1. Dataset description (Colon Cancer Vizgen MERFISH, 1.2M cells, 492 genes)
2. Why SVGP (not LCGP) is preferred at this scale (LCGP M=N=1.2M → too large)
3. Batch size config for 1.2M cells
4. Memory considerations: GPU vs CPU
5. Using `analyze_batch_size` to avoid OOM in analyze stage

---

## Sphinx Configuration (`conf.py`)

Key settings to include:
- `extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax', 'myst_parser']`
- `html_theme = 'furo'` (same clean look as PNMF)
- `project = 'Spatial-Factorization'`
- `author = 'Luis Diaz'`
- Math: use `mathjax` for rendered equations

---

## Implementation Order

1. Create `docs/conf.py` and `docs/requirements.txt`
2. Write `index.rst` + `installation.rst` + `quickstart.rst` (no math needed, easy wins)
3. Write `math.rst` (core content, reference for other pages)
4. Write `models.rst` and `configuration.rst` (tables/reference material)
5. Write `cli.rst` (can largely be drawn from `--help` output)
6. Write `pipeline.rst` and `outputs.rst`
7. Write `multiplex.rst` and `advanced.rst`
8. Write `datasets.rst`
9. Write examples pages
10. Set up ReadTheDocs integration (`.readthedocs.yaml`, link to GitHub)

---

## Notes

- **No auto-generated API docs** for this package — the public interface is the CLI, not a Python API. Use hand-written RST with explicit code blocks, not `autoclass`/`autofunction`.
- **Cross-link liberally**: CLI page links to configuration page; pipeline page links to output files page.
- **Math blocks**: Use `.. math::` RST directive for all equations.
- **Admonitions**: Use `.. note::`, `.. warning::`, `.. tip::` for important callouts (e.g. "MGGP models are not picklable", "Use `general_test.yaml` for testing, not `general.yaml`").
- **Version pinning**: Note minimum versions for PNMF and GPzoo in requirements.
