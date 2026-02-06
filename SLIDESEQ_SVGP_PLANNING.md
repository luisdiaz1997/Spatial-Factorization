# SlideSeq SVGP Planning

## Overview

This plan outlines running **spatial PNMF with SVGP prior** on SlideSeq data. Unlike the non-spatial PNMF baseline (see [SLIDESEQ_PNMF_PLANNING.md](./SLIDESEQ_PNMF_PLANNING.md)), here the latent factors F are modeled by a **Sparse Variational Gaussian Process** over spatial coordinates, enabling spatially smooth factor maps with group-aware correlations (MGGP).

We will use:
- **Spatial-Factorization**: Data loading, CLI pipeline, analysis, figures
- **PNMF** (SVGP branch): Model fitting with `spatial=True` and SVGP prior from GPzoo
- **GPzoo**: GP backends (MGGP_SVGP, batched_MGGP_Matern32, CholeskyParameter)

This builds on the completed PNMF baseline. The preprocessing stage is **shared** (dataset-specific, not model-specific), so `outputs/slideseq/preprocessed/` is reused.

---

## Development Strategy: 5-Stage Implementation Plan

**Status Legend:** `⬜` NOT DONE | `🟩` DONE

### Stage 0: Setup & Prerequisites `⬜ NOT DONE`

**Goal:** Ensure PNMF's SVGP branch is merged/installed and GPzoo is available.

**Prerequisites:**
- [x] PNMF baseline working (non-spatial, `SLIDESEQ_PNMF_PLANNING.md` Stage 0-4 done)
- [x] Preprocessing complete (`outputs/slideseq/preprocessed/` exists)
- [ ] PNMF SVGP branch merged or installable
- [ ] GPzoo installed (`pip install -e ../GPzoo`)
- [ ] `configs/slideseq/svgp.yaml` finalized

**Verification:**

```bash
# Activate environment
conda activate factorization

# Verify PNMF has spatial support
python -c "from PNMF import PNMF; m = PNMF(spatial=True); print('spatial OK')"

# Verify GPzoo is importable
python -c "from gpzoo.gp import MGGP_SVGP; print('GPzoo OK')"

# Verify preprocessed data exists
ls outputs/slideseq/preprocessed/
# Should show: X.npy, Y.npz, C.npy, metadata.json
```

**Deliverables:**
- [ ] PNMF SVGP branch finalized (see [PNMF PLAN.md](../Probabilistic-NMF/PLAN.md))
- [ ] GPzoo installed in factorization environment
- [ ] `configs/slideseq/svgp.yaml` created
- [ ] Quick smoke test passes

---

### Stage 1: Preprocess Command `🟩 DONE (shared with PNMF)`

Preprocessing is **dataset-specific**, not model-specific. The same preprocessed data is used by both PNMF and SVGP configs.

```bash
# Already done for PNMF - reuse the same preprocessed data
spatial_factorization preprocess -c configs/slideseq/pnmf.yaml
# OR equivalently:
spatial_factorization preprocess -c configs/slideseq/svgp.yaml
```

Both configs point to `output_dir: outputs/slideseq` and use the same preprocessing parameters.

**Directory structure (shared):**
```
outputs/
└── slideseq/
    ├── preprocessed/           # Shared by ALL models
    │   ├── X.npy               # (N, 2) spatial coordinates
    │   ├── Y.npz               # (N, D) count matrix (sparse CSR)
    │   ├── C.npy               # (N,) group codes
    │   └── metadata.json       # Gene names, group names, etc.
    ├── pnmf/                   # Non-spatial baseline (done)
    │   ├── model.pkl
    │   ├── model.pth
    │   ├── training.json
    │   ├── elbo_history.csv
    │   ├── factors.npy
    │   ├── loadings.npy
    │   ├── ...
    │   └── figures/
    └── svgp/                   # Spatial SVGP model (this plan)
        ├── model.pkl
        ├── model.pth
        ├── training.json
        ├── elbo_history.csv
        ├── factors.npy
        ├── loadings.npy
        ├── ...
        └── figures/
```

---

### Stage 2: Train Command `⬜ NOT DONE`

**Goal:** Train spatial PNMF with SVGP prior and save results.

```bash
spatial_factorization train -c configs/slideseq/svgp.yaml
```

**Expected output:**
```
Loading preprocessed data from: outputs/slideseq/preprocessed/
  Spots (N): 41783, Genes (D): 17702
Training PNMF with 10 components (mode=lower-bound, device=auto)...
  spatial=True, prior=SVGP, groups=True
  Inducing points (M): 3000
  Kernel: Matern32 (lengthscale=4.0)
[Iteration 100/2000] ELBO: -1234567.89
[Iteration 200/2000] ELBO: -1234000.00
...
Training complete!
  Final ELBO:     -1230000.00
  Training time:  456.7s
  Converged:      True
Model saved to: outputs/slideseq/svgp/
```

**Files created:**
- `outputs/slideseq/svgp/model.pkl` - Trained model (pickle)
- `outputs/slideseq/svgp/model.pth` - PyTorch state dict (includes GP params)
- `outputs/slideseq/svgp/training.json` - Training metadata
- `outputs/slideseq/svgp/elbo_history.csv` - ELBO history
- `outputs/slideseq/svgp/config.yaml` - Copy of config

**What changes in train.py:**

The current `train.py` already determines model directory via:
```python
spatial = config.model.get("spatial", False)
prior = config.model.get("prior", "GaussianPrior")
model_name = prior.lower() if spatial else "pnmf"  # -> "svgp"
```

The key change is passing coordinates and groups to `model.fit()`:

```python
# Current (non-spatial):
model = PNMF(random_state=config.seed, **config.to_pnmf_kwargs())
elbo_history, model = model.fit(data.Y.numpy(), return_history=True)

# New (spatial):
model = PNMF(random_state=config.seed, **config.to_pnmf_kwargs())
if config.spatial:
    elbo_history, model = model.fit(
        data.Y.numpy(),
        coordinates=data.X.numpy(),
        groups=data.groups.numpy(),
        return_history=True,
    )
else:
    elbo_history, model = model.fit(data.Y.numpy(), return_history=True)
```

**Config-to-PNMF mapping (to_pnmf_kwargs needs updating):**

New SVGP-specific kwargs that `to_pnmf_kwargs()` must pass through:
```python
# Model params (new for spatial)
kwargs["spatial"] = self.model.get("spatial", False)
kwargs["prior"] = self.model.get("prior", "GaussianPrior")
kwargs["kernel"] = self.model.get("kernel", "Matern32")
kwargs["multigroup"] = self.model.get("groups", True)  # config uses "groups"
kwargs["num_inducing"] = self.model.get("num_inducing", 3000)
kwargs["lengthscale"] = float(self.model.get("lengthscale", 1.0))
kwargs["sigma"] = float(self.model.get("sigma", 1.0))
kwargs["group_diff_param"] = float(self.model.get("group_diff_param", 10.0))
kwargs["train_lengthscale"] = self.model.get("train_lengthscale", False)
kwargs["cholesky_mode"] = self.model.get("cholesky_mode", "exp")
kwargs["diagonal_only"] = self.model.get("diagonal_only", False)
kwargs["inducing_allocation"] = self.model.get("inducing_allocation", "proportional")
```

**Model saving (model.pth) for spatial:**

The `_save_model()` function saves `model._prior.state_dict()`. For SVGP, this includes:
- `mu` (L, M) - variational means
- `Lu` (L, M, M) - variational Cholesky factors
- `Z` (M, 2) - inducing point locations
- `groupsZ` (M,) - inducing point group assignments
- Kernel parameters (lengthscale, sigma)

```python
# Updated _save_model for spatial:
state = {
    "model_state_dict": model._model.state_dict(),
    "prior_state_dict": model._prior.state_dict(),
    "components": model.components_,
    "hyperparameters": {
        "n_components": model.n_components,
        "spatial": model.spatial,
        "mode": config.model.get("mode", "lower-bound"),
        "loadings_mode": config.model.get("loadings_mode", "multiplicative"),
        "random_state": config.seed,
    },
}
if model.spatial:
    state["spatial_info"] = {
        "num_inducing": model.num_inducing,
        "lengthscale": float(model._prior.kernel.lengthscale.item()),
        "n_groups": int(model._groups.max().item() + 1),
    }
```

**Deliverables:**
- [ ] Update `config.py`: `to_pnmf_kwargs()` passes spatial params
- [ ] Update `train.py`: Pass coordinates/groups when `spatial=True`
- [ ] Update `train.py`: `_save_model()` handles GP state dict
- [ ] Verify training runs end-to-end with SVGP config

---

### Stage 3: Analyze Command `⬜ NOT DONE`

**Goal:** Compute metrics for the SVGP model (Moran's I, reconstruction, deviance, group loadings, enrichment).

```bash
spatial_factorization analyze -c configs/slideseq/svgp.yaml
```

**Expected output:**
```
Loading preprocessed data...
Loading trained model from: outputs/slideseq/svgp/
Extracting factors...
  Factors shape: (41783, 10)
Computing Moran's I for spatial autocorrelation...
  Top 3 Moran's I: [0.45, 0.38, 0.31]    # Higher than PNMF baseline!
Computing reconstruction error...
  Reconstruction error: 0.1800
Computing Poisson deviance...
  Poisson deviance (mean): 1.2345
Computing group-specific loadings for 14 groups...
  Group reconstruction errors:
    Astrocytes: 0.1900
    CA1_CA2_CA3: 0.1750
    ...
Computing relative gene enrichment...
Analysis complete!
  Saved to: outputs/slideseq/svgp/
```

**What changes:**

The analyze command is **already model-agnostic**. It:
1. Loads model from pickle (works for any PNMF model, spatial or not)
2. Calls `get_factors(model)` - already works for spatial models (PNMF transforms.py handles it)
3. Calls `factor_uncertainty(model)` - already works for spatial models
4. Computes Moran's I using coordinates (same for all models)
5. Computes reconstruction error using factors and loadings (same API)
6. Computes group-specific loadings via `transform_W()` (same API)

**Potential changes needed:**

1. **Factor extraction for spatial**: `get_factors()` and `factor_uncertainty()` in `PNMF/transforms.py` need coordinates when `model.spatial=True`. The PNMF SVGP branch already handles this by storing `model._coordinates` and `model._groups` during `fit()`, so `get_factors(model)` can use them internally.

2. **Group-conditioned factors (SVGP-specific)**: For spatial models, we can also extract factors *conditioned on specific groups*:
   ```python
   # New analysis for spatial models only
   if model.spatial:
       groupwise_factors = get_groupwise_factors(model, data.X, data.groups)
   ```
   This produces per-group factor maps showing how each factor looks when the GP is conditioned on a specific cell type. This is one of the key advantages of the MGGP model.

3. **GP-specific metrics**:
   - Learned lengthscale (if `train_lengthscale=True`)
   - Inducing point coverage
   - Variational posterior quality

**Files created (same format as PNMF):**
- `outputs/slideseq/svgp/factors.npy` (N, L)
- `outputs/slideseq/svgp/scales.npy` (N, L) - GP posterior uncertainty
- `outputs/slideseq/svgp/loadings.npy` (D, L)
- `outputs/slideseq/svgp/loadings_group_{g}.npy` (D, L)
- `outputs/slideseq/svgp/moran_i.csv`
- `outputs/slideseq/svgp/metrics.json`
- `outputs/slideseq/svgp/gene_enrichment.json`
- **NEW:** `outputs/slideseq/svgp/groupwise_factors.npz` (optional, SVGP-specific)

**Deliverables:**
- [ ] Verify existing analyze.py works unchanged with spatial model pickle
- [ ] Add groupwise factor extraction for spatial models (optional)
- [ ] Add GP-specific metrics to metrics.json (lengthscale, inducing info)

---

### Stage 4: Figures Command `⬜ NOT DONE`

**Goal:** Generate publication figures for the SVGP model.

```bash
spatial_factorization figures -c configs/slideseq/svgp.yaml
```

**Expected output:**
```
Loading preprocessed data...
Loading analysis results from: outputs/slideseq/svgp/
Generating spatial factor plot...
  Saved: outputs/slideseq/svgp/figures/factors_spatial.png
Generating factor uncertainty (scales) spatial plot...
  Saved: outputs/slideseq/svgp/figures/scales_spatial.png
Generating ELBO curve...
  Saved: outputs/slideseq/svgp/figures/elbo_curve.png
Generating top genes plot...
  Saved: outputs/slideseq/svgp/figures/top_genes.png
Generating factors + top genes spatial plot...
  Saved: outputs/slideseq/svgp/figures/factors_with_genes.png
Generating gene enrichment heatmap...
  Saved: outputs/slideseq/svgp/figures/gene_enrichment.png
Generating groupwise factor plots...                          # NEW (SVGP-specific)
  Saved: outputs/slideseq/svgp/figures/groupwise_factors.png
Figures generation complete!
```

**What changes:**

The figures command is **already model-agnostic** for all existing plots. The same plots work for both PNMF and SVGP:
- `factors_spatial.png` - Spatial factors (sorted by Moran's I)
- `scales_spatial.png` - Factor uncertainty
- `elbo_curve.png` - Training convergence
- `top_genes.png` - Top genes per factor
- `factors_with_genes.png` - Factors + gene expression
- `gene_enrichment.png` - LFC heatmap
- `enrichment_by_group/` - Per-group enrichment bars

**New figures (SVGP-specific):**

1. **Groupwise factors plot** (`groupwise_factors.png`):
   - Grid: rows = cell types, columns = factors
   - Row 0: mean factors (all groups)
   - Rows 1+: factors conditioned on each group
   - Shows how spatial patterns change per cell type
   - Pattern from: `GPzoo/notebooks/liver_mggp_healthy_matern32_umap_init.ipynb` (cell 42)

2. **PNMF vs SVGP comparison** (optional):
   - Side-by-side Moran's I comparison
   - Factor maps from both models

**Deliverables:**
- [ ] Verify existing figures.py works unchanged with SVGP analysis outputs
- [ ] Add groupwise factor plot for spatial models
- [ ] Add comparison plot (optional, could be a separate notebook)

---

### Summary: Complete Pipeline

```bash
# Stage 0: Prerequisites
pip install -e .
pip install -e ../Probabilistic-NMF  # SVGP branch
pip install -e ../GPzoo

# Stage 1: Preprocess (already done for PNMF, shared)
spatial_factorization preprocess -c configs/slideseq/svgp.yaml

# Stage 2: Train (SVGP)
spatial_factorization train -c configs/slideseq/svgp.yaml

# Stage 3: Analyze
spatial_factorization analyze -c configs/slideseq/svgp.yaml

# Stage 4: Figures
spatial_factorization figures -c configs/slideseq/svgp.yaml
```

---

## Config File

```yaml
# configs/slideseq/svgp.yaml
# SlideseqV2 with spatial PNMF (SVGP prior, multi-group)

name: slideseq_svgp
seed: 67
dataset: slideseq
output_dir: outputs/slideseq

preprocessing:
  spatial_scale: 50.0
  filter_mt: true
  min_counts: 100
  min_cells: 10

model:
  n_components: 10
  spatial: true
  prior: SVGP
  groups: true                  # Use MGGP (multi-group GP)
  mode: lower-bound
  loadings_mode: multiplicative
  # GP-specific parameters
  kernel: Matern32
  num_inducing: 3000
  lengthscale: 4.0              # From GPzoo config.py: LENGTHSCALE = 4.0
  sigma: 1.0
  group_diff_param: 10.0
  train_lengthscale: false      # Keep kernel fixed
  cholesky_mode: exp
  diagonal_only: false
  inducing_allocation: proportional

training:
  max_iter: 2000
  learning_rate: 1e-1
  optimizer: Adam
  tol: 1e-4
  verbose: false
  device: gpu
  batch_size: 7000
  y_batch_size: 2000
  shuffle: true
```

---

## Key Differences: PNMF vs SVGP

| Aspect | PNMF (baseline) | SVGP (spatial) |
|--------|-----------------|----------------|
| Prior on F | Independent Gaussian N(mu, sigma) | Spatial GP via MGGP_SVGP |
| Prior params | mu (L,N), sigma (L,N) | mu (L,M), Lu (L,M,M), Z (M,2), kernel |
| Coordinates | Ignored during training | Used in GP kernel |
| Groups | Ignored during training | Used in MGGP group correlations |
| KL divergence | Gaussian KL over (L,N) data points | Whitened KL over (L,M) inducing points |
| KL scaling | Scales with N/batch_size | No scaling (fixed M inducing points) |
| Factor smoothness | No spatial structure | Smooth via Matern32 kernel |
| Config `spatial` | false | true |
| Config `prior` | GaussianPrior | SVGP |
| Output dir | `outputs/slideseq/pnmf/` | `outputs/slideseq/svgp/` |
| Training time | ~2-5 min (500 iter) | ~10-30 min (2000 iter) |
| Memory | Low (no kernel matrices) | Higher (M=3000 inducing) |
| Dependencies | PyTorch only | PyTorch + GPzoo |
| ELBO modes | All three work | All three work (same qF type) |

---

## Data Flow

```
SlideseqV2 (squidpy)
        |
        v
┌─────────────────┐
│ SlideseqLoader  │  (Spatial-Factorization)
│ - QC filtering  │
│ - MT gene filter│
│ - Coord scaling │
└────────┬────────┘
         |
         v
┌─────────────────┐
│   SpatialData   │
│ - X: (N, 2)     │  coordinates ─────────────┐
│ - Y: (N, D)     │  counts (spots x genes) ──┤
│ - C: (N,)       │  group codes ─────────────┤
└────────┬────────┘                            │
         |                                     │
         v                                     v
   preprocessed/                    ┌─────────────────────────┐
   ├── X.npy                        │     PNMF (spatial=True)  │
   ├── Y.npz                        │                           │
   ├── C.npy                         │  MGGP_SVGP prior:        │
   └── metadata.json                │    Z (M,2) inducing pts  │
         |                           │    mu (L,M) var. mean    │
         |                           │    Lu (L,M,M) var. cov   │
         |                           │    kernel: Matern32      │
         |                           │                           │
         |                           │  Likelihood:              │
         |                           │    W (D,L) loadings      │
         |                           │    rate = W @ exp(F)     │
         |                           │                           │
         |                           │  ELBO = E[log p(Y|F)]   │
         |                           │       - KL(q(U)||p(U))  │
         |                           └───────────┬─────────────┘
         |                                       │
         v                                       v
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   2. TRAIN      │     │   3. ANALYZE    │     │   4. FIGURES    │
│                 │     │                 │     │                 │
│ - Load preproc  │     │ - Load preproc  │     │ - Load preproc  │
│ - Fit SVGP-PNMF│────>│ - Load model    │────>│ - Load analysis │
│ - Save .pkl/.pth│     │ - get_factors() │     │ - plot_factors  │
│ - Save history  │     │ - Moran's I     │     │ - plot_top_genes│
│                 │     │ - Reconstruction│     │ - groupwise_fac │
│                 │     │ - Group loadings│     │                 │
│                 │     │ - Enrichment    │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        |                       |                       |
        v                       v                       v
  model.pkl              metrics.json            figures/*.png
  model.pth              factors.npy             factors_spatial.png
  training.json          scales.npy              scales_spatial.png
  elbo_history.csv       loadings.npy            elbo_curve.png
                         moran_i.csv             top_genes.png
                         gene_enrichment.json    gene_enrichment.png
                                                 groupwise_factors.png
```

---

## PNMF SVGP Branch: Current State

The PNMF repo is on the `SVGP` branch with uncommitted changes that add full spatial GP support. Here is a summary of what exists:

### Modified Files (uncommitted on SVGP branch)

**`PNMF/models.py` (+464/-88 lines):**
- `PNMF.__init__()`: Added ~15 spatial GP parameters (`spatial`, `prior`, `kernel`, `multigroup`, `num_inducing`, `lengthscale`, `sigma`, `group_diff_param`, `jitter`, `train_lengthscale`, `cholesky_mode`, `diagonal_only`, `inducing_allocation`)
- `PNMF._validate_params()`: Spatial-specific validation
- `PNMF._create_optimizer()`: Refactored into helper method
- `PNMF._create_spatial_prior()`: ~100-line method creating MGGP_SVGP from GPzoo
- `PNMF.fit()`: Extended with `coordinates` and `groups` parameters; spatial training loop with GP forward, exp_ll via `expected_log_likelihood`, KL via `self._prior.kl_divergence(qU, pU)` (no N-scaling)
- `PNMF.transform()`: Extended for spatial (runs GP predictive at new coordinates)
- `PNMF.fit_transform()`: Passes through coordinates and groups
- `PoissonFactorization.forward()`: Extended with `coordinates`, `groups`, `spatial` parameters

**`PNMF/transforms.py` (+106 lines):**
- `_get_spatial_qF()`: Helper to run GP forward pass
- `log_factors()`, `get_factors()`, `factor_uncertainty()`, `factor_samples()`: All extended with optional `coordinates` and `groups` parameters for spatial models

**`pyproject.toml` (+4 lines):**
- Added `[project.optional-dependencies] spatial` section with `gpzoo` and `faiss-cpu`

**New files (untracked):**
- `PLAN.md`: Detailed implementation plan for SVGP integration
- `tests/test_spatial.py`: Spatial mode tests

### Status

The PNMF SVGP branch changes are **complete but uncommitted**. Before Stage 2 can run:
1. Commit and merge the SVGP branch in PNMF
2. Reinstall PNMF: `pip install -e ../Probabilistic-NMF`

---

## Spatial-Factorization Changes Required

### 1. `config.py` - Update `to_pnmf_kwargs()`

Current `to_pnmf_kwargs()` only passes non-spatial parameters. Needs to include:

```python
def to_pnmf_kwargs(self) -> Dict[str, Any]:
    kwargs = {}

    # ... existing model params ...

    # Spatial params (pass through if present)
    if self.model.get("spatial", False):
        kwargs["spatial"] = True
        kwargs["prior"] = self.model.get("prior", "SVGP")
        kwargs["kernel"] = self.model.get("kernel", "Matern32")
        kwargs["multigroup"] = self.model.get("groups", True)
        kwargs["num_inducing"] = self.model.get("num_inducing", 3000)
        kwargs["lengthscale"] = float(self.model.get("lengthscale", 1.0))
        kwargs["sigma"] = float(self.model.get("sigma", 1.0))
        kwargs["group_diff_param"] = float(self.model.get("group_diff_param", 10.0))
        kwargs["train_lengthscale"] = self.model.get("train_lengthscale", False)
        kwargs["cholesky_mode"] = self.model.get("cholesky_mode", "exp")
        kwargs["diagonal_only"] = self.model.get("diagonal_only", False)
        kwargs["inducing_allocation"] = self.model.get("inducing_allocation", "proportional")

    # ... existing training params ...

    return kwargs
```

### 2. `commands/train.py` - Pass coordinates and groups

```python
def run(config_path: str):
    # ... existing setup ...

    model = PNMF(random_state=config.seed, **config.to_pnmf_kwargs())

    # Branch on spatial
    t0 = time.perf_counter()
    if config.spatial:
        print(f"  spatial=True, prior={config.prior}, groups={config.model.get('groups', True)}")
        print(f"  Inducing points (M): {config.model.get('num_inducing', 3000)}")
        print(f"  Kernel: {config.model.get('kernel', 'Matern32')} (lengthscale={config.model.get('lengthscale', 1.0)})")
        elbo_history, model = model.fit(
            data.Y.numpy(),
            coordinates=data.X.numpy(),
            groups=data.groups.numpy(),
            return_history=True,
        )
    else:
        elbo_history, model = model.fit(data.Y.numpy(), return_history=True)
    train_time = time.perf_counter() - t0

    # ... rest is the same (model_name is already "svgp") ...
```

### 3. `commands/analyze.py` - Minimal changes

The analyze command should work mostly unchanged because:
- `get_factors(model)` handles spatial models internally (PNMF transforms.py)
- `factor_uncertainty(model)` handles spatial models internally
- `model.components_` returns loadings (same API)
- `transform_W()` uses exp-space factors (same API)

**Optional additions for spatial models:**
```python
# In run():
if model.spatial:
    # Add GP-specific metrics
    metrics["gp_info"] = {
        "num_inducing": model.num_inducing,
        "lengthscale": float(model._prior.kernel.lengthscale.item()),
        "sigma": float(model._prior.kernel.sigma.item()),
    }
```

### 4. `commands/figures.py` - Minimal changes

The figures command should work unchanged for all existing plots. Optional addition:

```python
# In run():
# New: Groupwise factor plot (SVGP-specific)
if config.spatial and results.get("groupwise_factors") is not None:
    print("Generating groupwise factor plot...")
    fig = plot_groupwise_factors(
        results["groupwise_factors"], coords, group_names,
        moran_idx=moran_idx
    )
    fig.savefig(figures_dir / "groupwise_factors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
```

---

## Code References

### PNMF (SVGP Branch) - What Exists

| What | Path | Notes |
|------|------|-------|
| PNMF spatial constructor | `Probabilistic-NMF/PNMF/models.py` | `spatial=True` params added |
| `_create_spatial_prior()` | `Probabilistic-NMF/PNMF/models.py` | Creates MGGP_SVGP from GPzoo |
| Spatial fit() | `Probabilistic-NMF/PNMF/models.py` | Handles coords, groups, GP KL |
| Spatial transforms | `Probabilistic-NMF/PNMF/transforms.py` | `_get_spatial_qF()`, extended factor extraction |
| Spatial tests | `Probabilistic-NMF/tests/test_spatial.py` | Test suite for spatial mode |
| SVGP plan | `Probabilistic-NMF/PLAN.md` | Detailed implementation plan |

### GPzoo - What Gets Imported

| What | Path | Import |
|------|------|--------|
| MGGP_SVGP | `gpzoo/gp.py` | `from gpzoo.gp import MGGP_SVGP` |
| SVGP | `gpzoo/gp.py` | `from gpzoo.gp import SVGP` |
| batched_MGGP_Matern32 | `gpzoo/kernels.py` | `from gpzoo.kernels import batched_MGGP_Matern32` |
| batched_Matern32 | `gpzoo/kernels.py` | `from gpzoo.kernels import batched_Matern32` |
| CholeskyParameter | `gpzoo/modules.py` | `from gpzoo.modules import CholeskyParameter` |
| mggp_kmeans_inducing_points | `gpzoo/model_utilities.py` | `from gpzoo.model_utilities import mggp_kmeans_inducing_points` |
| dims_autocorr | `gpzoo/utilities.py` | `from gpzoo.utilities import dims_autocorr` |

### GPzoo - Training Patterns (Reference)

| What | Path | Notes |
|------|------|-------|
| SVGP training script | `GPzoo/gpzoo/datasets/slideseq/svgp_nsf.py` | Reference for training loop, param groups, freezing |
| SlideSeq config | `GPzoo/gpzoo/datasets/slideseq/config.py` | Hyperparameters: L=10, M=3000, LR=0.01, etc. |
| Training utilities | `GPzoo/gpzoo/training_utilities.py` | `train_svgp_batched_with_tracking()` |
| SVGP_NSF model | `GPzoo/gpzoo/models/nsf.py` | Model architecture reference |

### Spatial-Factorization - What Exists

| What | Path | Status |
|------|------|--------|
| CLI entry point | `spatial_factorization/cli.py` | Done, works for all configs |
| Config system | `spatial_factorization/config.py` | Needs `to_pnmf_kwargs()` update |
| Train command | `spatial_factorization/commands/train.py` | Needs spatial branching |
| Analyze command | `spatial_factorization/commands/analyze.py` | Should work unchanged |
| Figures command | `spatial_factorization/commands/figures.py` | Should work unchanged |
| Data loading | `spatial_factorization/datasets/base.py` | Done, model-agnostic |

---

## GP Forward Pass (How SVGP Works)

For reference, here's what happens when PNMF calls the SVGP prior during training:

```
Input: X_batch (B, 2), groups_batch (B,)

1. Kernel computations (batched via vmap over L factors):
   Kxx_diag = kernel(X_batch, X_batch, groups_batch, groups_batch, diag=True)  # (L, B)
   Kzx = kernel(Z, X_batch, groupsZ, groups_batch)                             # (L, M, B)
   Kzz = kernel(Z, Z, groupsZ, groupsZ)                                        # (L, M, M)

2. Cholesky:
   Lzz = cholesky(Kzz + jitter * I)                                            # (L, M, M)

3. GP predictive equations:
   alpha = Lzz^{-1} @ Kzx                                                      # (L, M, B)
   mean = alpha.T @ mu                                                          # (L, B)
   var  = Kxx_diag - sum(alpha^2, dim=M) + sum((alpha.T @ Lu)^2, dim=M)       # (L, B)
   qF = Normal(mean, sqrt(var))                                                 # (L, B)

4. ELBO terms:
   F_samples ~ qF.rsample((E,))                                                # (E, L, B)
   rate = W @ exp(F_samples)                                                    # (E, D, B)
   exp_ll = E[log Poisson(Y | rate)]                                           # scalar

   KL = 0.5 * sum(-2*log(diag(Lu)) + trace(Lu@Lu.T) + mu.T@mu - M)           # scalar (whitened)
   loss = KL - exp_ll                                                          # minimize
```

---

## Mini-Batch Scaling

**Critical difference from non-spatial:**

For non-spatial PNMF:
```python
# Both exp_ll and KL scale with batch size
if batch_size: exp_ll *= N / batch_size
if batch_size: kl *= N / batch_size
```

For spatial SVGP:
```python
# Only exp_ll scales - KL is over fixed M inducing points
if batch_size: exp_ll *= N / batch_size
if y_batch_size: exp_ll *= D / y_batch_size
# KL does NOT scale (already covers all inducing points)
```

This is handled inside PNMF's fit() method on the SVGP branch.

---

## Expected Results

For SlideSeq data (~41K spots, ~17K genes after filtering):

1. **Data dimensions**:
   - N ~ 41,000 spots
   - D ~ 17,000 genes
   - L = 10 components
   - M = 3,000 inducing points
   - G = 14 cell type groups

2. **Moran's I** (spatial autocorrelation):
   - SVGP factors should have **higher Moran's I** than PNMF baseline
   - The GP prior explicitly encourages spatial smoothness
   - Expected: mean Moran's I ~0.3-0.5 (vs ~0.1-0.2 for PNMF)

3. **ELBO**:
   - Not directly comparable to PNMF ELBO (different KL terms)
   - Should converge within 2000 iterations with `mode=lower-bound`

4. **Reconstruction**:
   - Similar or slightly worse than PNMF (spatial smoothing trades off reconstruction)
   - The GP regularization prevents overfitting to noise

5. **Group-specific patterns**:
   - With MGGP, factors should show cell-type-specific spatial patterns
   - Groupwise factors will reveal how each cell type modulates spatial patterns

---

## Hyperparameter Choices (from GPzoo config)

| Parameter | Value | Source | Rationale |
|-----------|-------|--------|-----------|
| `n_components` | 10 | GPzoo config: `L_FACTORS=10` | Standard for SlideSeq |
| `num_inducing` | 3000 | GPzoo config: `SVGP_INDUCING=3000` | ~7% of N, good coverage |
| `lengthscale` | 4.0 | GPzoo config: `LENGTHSCALE=4.0` | After spatial_scale=50 rescaling |
| `learning_rate` | 0.1 | PNMF default for lower-bound | Higher LR works with multiplicative W |
| `batch_size` | 7000 | ~17% of N | Memory-efficient, good gradient estimate |
| `y_batch_size` | 2000 | ~11% of D | Subsample genes per iteration |
| `max_iter` | 2000 | Empirical | More iterations needed for GP convergence |
| `mode` | lower-bound | Best for multiplicative W | Fully analytic, fast per iteration |
| `loadings_mode` | multiplicative | No gradient for W | Multiplicative updates (NNLS-style) |
| `seed` | 67 | GPzoo config: `SEED=67` | Consistency with GPzoo experiments |
| `group_diff_param` | 10.0 | GPzoo default | Controls inter-group correlation |
| `sigma` | 1.0 | GPzoo default | Kernel output scale |

---

## File Checklist

### Spatial-Factorization (To Modify)

- [ ] `spatial_factorization/config.py` - Update `to_pnmf_kwargs()` for spatial params
- [ ] `spatial_factorization/commands/train.py` - Pass coordinates/groups when spatial
- [ ] `spatial_factorization/commands/analyze.py` - Add GP-specific metrics (optional)
- [ ] `spatial_factorization/commands/figures.py` - Add groupwise factor plot (optional)

### Already Done / No Changes Needed

- [x] `spatial_factorization/cli.py` - Works as-is (model_name = "svgp" when spatial+SVGP)
- [x] `spatial_factorization/datasets/base.py` - Works as-is (returns X, Y, groups)
- [x] `spatial_factorization/datasets/preprocessed.py` - Works as-is
- [x] `spatial_factorization/commands/preprocess.py` - Works as-is (shared preprocessing)
- [x] `configs/slideseq/svgp.yaml` - Created

### External Dependencies

- [ ] PNMF SVGP branch: Commit and merge (or install from branch)
- [ ] GPzoo: Ensure installed (`pip install -e ../GPzoo`)

---

## Implementation Order

1. **PNMF SVGP branch** - Commit, test, merge (blocking dependency)
2. **Config update** - `to_pnmf_kwargs()` passes spatial params
3. **Train update** - Pass coordinates/groups
4. **Smoke test** - Run full pipeline end-to-end
5. **Analyze verify** - Confirm existing analyze works with spatial pickle
6. **Figures verify** - Confirm existing figures works with spatial outputs
7. **Optional: Groupwise factors** - Add SVGP-specific analysis and figures

---

## Quick Start

```bash
# Prerequisites
conda activate factorization
pip install -e .
pip install -e ../Probabilistic-NMF    # SVGP branch
pip install -e ../GPzoo

# Full pipeline
spatial_factorization preprocess -c configs/slideseq/svgp.yaml   # if not already done
spatial_factorization train      -c configs/slideseq/svgp.yaml
spatial_factorization analyze    -c configs/slideseq/svgp.yaml
spatial_factorization figures    -c configs/slideseq/svgp.yaml

# View results
ls outputs/slideseq/svgp/
ls outputs/slideseq/svgp/figures/
```
