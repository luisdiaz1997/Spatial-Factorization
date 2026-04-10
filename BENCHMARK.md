# Benchmark Command — Design Plan

A new `benchmark` top-level CLI group with two sub-stages: `analyze` and `figures`.

---

## Attribution

Metric implementations (CHAOS, PAS, ASW, ARI, NMI, HOM, COM) are imported from
the **SDMBench** package by Zhao et al.:

> Zhao F, et al. *Benchmarking Spatial Domain Identification Methods for Spatial Transcriptomics*.
> GitHub: https://github.com/zhaofangyuan98/SDMBench  
> Website: http://sdmbench.drai.cn/  
> Source file: [`SDMBench/SDMBench/SDMBench.py`][sdm]

Each borrowed function is annotated with its source line range in that file.

[sdm]: /gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/SDMBench/SDMBench.py

---

## Motivation

The GPzoo repo already has a two-step benchmarking pipeline for 10X Visium DLPFC slides
([`generate_annotations.py`][ga] + [`benchmark.py`][bm] / [`benchmark_parallel.py`][bp]).
That pipeline is tied to hardcoded paths, two conda environments, and flat-file `.txt`
annotations. This plan integrates equivalent functionality directly into the
`spatial_factorization` CLI with:

- Config-driven scoping (one slide, one dataset, or all datasets)
- The same `outputs/<dataset>/` layout already used by train/analyze/figures
- **Primary metric**: mean Moran's I across factors (already in `metrics.json` — free to read)
- **Secondary metrics**: ARI, NMI, HOM, COM, CHAOS, PAS, ASW (adapted from SDMBench, Zhao et al.)
- PCA and NMF baselines alongside our 5 models
- One figure per scope (single-dataset bar chart, or multi-dataset mean ± std aggregate)

[ga]: /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo/gpzoo/datasets/tenxvisium/generate_annotations.py
[bm]: /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo/gpzoo/datasets/tenxvisium/benchmark.py
[bp]: /gladstone/engelhardt/home/lchumpitaz/gitclones/GPzoo/gpzoo/datasets/tenxvisium/benchmark_parallel.py

---

## CLI Design

```bash
# --- analyze stage ---

# One SDMBench slide
spatial_factorization benchmark analyze -c configs/sdmbench/151507/general.yaml

# All 12 SDMBench slides (directory scope)
spatial_factorization benchmark analyze -c configs/sdmbench/

# All datasets at once
spatial_factorization benchmark analyze -c configs/

# Only specific models (skip others)
spatial_factorization benchmark analyze -c configs/sdmbench/ --models svgp mggp_svgp pnmf

# Skip PCA/NMF baselines
spatial_factorization benchmark analyze -c configs/sdmbench/ --no-baselines

# --- figures stage ---

# One slide
spatial_factorization benchmark figures -c configs/sdmbench/151507/general.yaml

# All slides → aggregate figure
spatial_factorization benchmark figures -c configs/sdmbench/

# --- both stages in one call ---
spatial_factorization benchmark run analyze figures -c configs/sdmbench/

# --- per-model config also works (figures only for that model) ---
spatial_factorization benchmark analyze -c configs/sdmbench/151507/svgp.yaml
```

### Click structure

```python
# cli.py additions

@cli.group("benchmark")
def benchmark():
    """Evaluate trained models with SDMBench metrics."""
    pass

@benchmark.command("analyze")
@click.option("--config", "-c", required=True, ...)
@click.option("--models", multiple=True, default=None,
              help="Only benchmark these model names (default: all available)")
@click.option("--no-baselines", is_flag=True,
              help="Skip PCA and NMF baseline computation")
@click.option("--config-name", default="general.yaml", ...)
def benchmark_analyze(config, models, no_baselines, config_name):
    ...

@benchmark.command("figures")
@click.option("--config", "-c", required=True, ...)
@click.option("--config-name", default="general.yaml", ...)
def benchmark_figures(config, config_name):
    ...

@benchmark.command("run")
@click.argument("stages", nargs=-1, required=True)
@click.option("--config", "-c", required=True, ...)
@click.option("--models", multiple=True, default=None, ...)
@click.option("--no-baselines", is_flag=True, ...)
@click.option("--config-name", default="general.yaml", ...)
def benchmark_run(stages, config, models, no_baselines, config_name):
    ...
```

---

## Scoping Logic (config resolution)

Mirrors the existing `run` command logic in `runner.py`:

| `--config` argument | Behavior |
|---------------------|----------|
| Per-model YAML (e.g., `svgp.yaml`) | Benchmark only that model for that dataset |
| General YAML (e.g., `general.yaml`) | Benchmark all 5 models for that one dataset |
| Directory (e.g., `configs/sdmbench/`) | Recurse, find all `general.yaml` files, benchmark each |
| Root directory (`configs/`) | Benchmark ALL datasets |

The resolve logic is a thin wrapper that collects (output_dir, model_names) pairs:

```python
def resolve_benchmark_targets(config: str, config_name: str = "general.yaml",
                              model_filter: tuple = ()) -> list[BenchmarkTarget]:
    """
    Returns a list of BenchmarkTarget(output_dir, slide_id, config_path, model_names).
    Mirrors JobRunner._collect_configs() in runner.py.
    """
    ...
```

---

## Output Directory Layout

```
outputs/<dataset>/
├── preprocessed/            # existing (Stage 1)
├── svgp/                    # existing (Stage 2-4)
│   ├── factors.npy          # (N, L)  ← primary input to benchmark analyze
│   ├── moran_i.csv          # already sorted descending
│   └── metrics.json         # moran_i.mean already here — FREE metric
├── benchmark/               # NEW — created by benchmark analyze
│   ├── annotations/
│   │   ├── pnmf.npy           # (N,) K-means labels
│   │   ├── svgp.npy
│   │   ├── mggp_svgp.npy
│   │   ├── lcgp.npy
│   │   ├── mggp_lcgp.npy
│   │   ├── pca_baseline.npy   # (N,) K-means on PCA factors
│   │   └── nmf_baseline.npy   # (N,) K-means on NMF factors
│   ├── pnmf_results.json      # per-model metrics dict
│   ├── svgp_results.json
│   ├── mggp_svgp_results.json
│   ├── lcgp_results.json
│   ├── mggp_lcgp_results.json
│   ├── pca_baseline_results.json
│   ├── nmf_baseline_results.json
│   └── benchmark_results.csv  # all models × all metrics (indexed by model name)
└── figures/
    └── benchmark.png          # NEW — created by benchmark figures
```

For multi-dataset runs, each dataset gets its own `benchmark/` dir plus a top-level aggregate:

```
outputs/benchmark_aggregate/       # when scope is a directory
├── sdmbench_151507_results.csv
├── sdmbench_151508_results.csv
├── ...
├── summary.csv                    # mean ± std across all datasets in scope
└── benchmark_figure.png           # aggregate figure (mean ± std bars per model)
```

---

## `benchmark analyze` — Implementation Details

### File: `spatial_factorization/commands/benchmark_analyze.py`

```python
def run(config_path: str, model_filter: tuple = (), include_baselines: bool = True):
    """
    Main entry point for benchmark analyze (single dataset scope).

    Args:
        config_path: Path to a per-model or general YAML (resolved by caller for multi-dataset).
        model_filter: If non-empty, only benchmark these model names.
        include_baselines: Whether to compute PCA/NMF baselines.
    """
```

#### Step 1 — Collect available models

```python
output_dir = Path(config.output_dir)  # e.g. outputs/sdmbench/151507
model_dirs = {
    name: output_dir / name
    for name in ["pnmf", "svgp", "mggp_svgp", "lcgp", "mggp_lcgp"]
    if (output_dir / name / "factors.npy").exists()
}
if model_filter:
    model_dirs = {k: v for k, v in model_dirs.items() if k in model_filter}
```

#### Step 2 — Load preprocessed data (ground truth groups)

```python
from ..datasets.base import load_preprocessed

data = load_preprocessed(output_dir / "preprocessed")
X = data["X"]        # (N, 2) spatial coords — already in preprocessed/X.npy
C = data["C"]        # (N,) group codes (integers 0..G-1)
metadata = data["metadata"]  # has group_names list

# Ground truth labels for SDMBench accuracy metrics
gt_labels = C  # integer codes, directly comparable to K-means output
n_clusters = len(metadata["group_names"])
group_names = metadata["group_names"]
```

**Why C.npy is the right ground truth:**
- `C.npy` is the preprocessed group code array (0..G-1) already filtered by `_filter_small_groups`
- It corresponds 1:1 with the N rows in `factors.npy`
- The group names come from `metadata["group_names"]` in the same preprocessed dir
- This means we don't need to re-load the original h5ad file at benchmark time

#### Step 3 — Load factors and compute K-means annotations

```python
from sklearn.cluster import KMeans

def _kmeans_from_factors(factors: np.ndarray, n_clusters: int, random_state: int = 0) -> np.ndarray:
    """
    Apply K-means on factors (N, L) → cluster labels (N,).
    Uses log-space factors for clustering (consistent with generate_annotations.py:line 343).
    factors is exp(F) as stored in factors.npy; take log before clustering.
    """
    log_factors = np.log(factors + 1e-8)  # (N, L) in log space
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(log_factors)
```

**Why log-space**: `generate_annotations.py` (line 343: `factors_for_kmeans = mean_log.T`) uses log-space factors for K-means because exp-space factors are highly skewed.  
**Why `factors.npy` stores exp(F)**: per CLAUDE.md "factors.npy stores exp(F) (non-negative, minimum > 0)".  
So we must take `log(factors + 1e-8)` before clustering.

```python
annotations = {}
for model_name, model_dir in model_dirs.items():
    factors = np.load(model_dir / "factors.npy")  # (N, L)
    labels = _kmeans_from_factors(factors, n_clusters)
    annotations[model_name] = labels
    np.save(benchmark_dir / "annotations" / f"{model_name}.npy", labels)
```

#### Step 4 — PCA and NMF baselines (optional)

```python
def _pca_baseline(Y_dense: np.ndarray, n_components: int, n_clusters: int) -> np.ndarray:
    """PCA on (N, D) expression → K-means labels."""
    # mirrors generate_annotations.py:generate_pca_baseline() lines 259-273
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=0)
    factors = pca.fit_transform(Y_dense)  # (N, n_components)
    return KMeans(n_clusters=n_clusters, random_state=0).fit_predict(factors)

def _nmf_baseline(Y_dense: np.ndarray, n_components: int, n_clusters: int) -> np.ndarray:
    """Size-normalized NMF → K-means labels (log-space for clustering)."""
    # mirrors generate_annotations.py:generate_nmf_baseline() lines 229-256
    from sklearn.decomposition import NMF
    from gpzoo.utilities import scanpy_sizefactors
    V = scanpy_sizefactors(Y_dense)
    nmf = NMF(n_components=n_components, max_iter=300, init='random', random_state=0, alpha_W=1e-2)
    exp_factors = nmf.fit_transform(Y_dense / V) / 5   # (N, L)
    log_factors = np.log(exp_factors + 1e-2)           # consistent with analyze log-space
    return KMeans(n_clusters=n_clusters, random_state=0).fit_predict(log_factors)
```

Y is loaded from `preprocessed/Y.npz` (sparse, D×N). Convert to dense (N×D) for PCA/NMF:

```python
from scipy.sparse import load_npz
Y_sparse = load_npz(output_dir / "preprocessed" / "Y.npz")   # (D, N)
Y_dense = Y_sparse.T.toarray()                                # (N, D) for sklearn
n_components = config.model.get("n_components", 10)
```

#### Step 5 — Moran's I mean (FREE — read from metrics.json)

```python
def _load_moran_i_mean(model_dir: Path) -> float | None:
    """Read moran_i.mean from already-computed metrics.json."""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics.get("moran_i", {}).get("mean")
```

This is the PRIMARY metric from the user's spec: **"we will use the moran index mean of factors in a given model"**.
No re-computation needed — analyze stage already computed and saved it.

#### Step 6 — SDMBench metrics (optional, graceful if package missing)

All SDMBench metrics are imported directly from the `SDMBench` package (Zhao et al.),
which has no unmet dependencies in the `factorization` env and installs with one command:

```bash
python -m pip install -e /gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/SDMBench/
```

```python
from SDMBench import sdmbench  # Zhao et al., https://github.com/zhaofangyuan98/SDMBench

ari   = sdmbench.compute_ARI(adata_with_pred, gt_key, "pred")
nmi   = sdmbench.compute_NMI(adata_with_pred, gt_key, "pred")
hom   = sdmbench.compute_HOM(adata_with_pred, gt_key, "pred")
com   = sdmbench.compute_COM(adata_with_pred, gt_key, "pred")
chaos = sdmbench.compute_CHAOS(adata_with_pred, "pred")
pas   = sdmbench.compute_PAS(adata_with_pred, "pred", spatial_key="spatial")
asw   = sdmbench.compute_ASW(adata_with_pred, "pred", spatial_key="spatial")
```

`benchmark_metrics.py` is therefore a thin wrapper that loads the AnnData view needed
by `sdmbench` and calls the package directly — no reimplementation needed.

The `benchmark.py` script in GPzoo uses the same pattern (see [`benchmark.py:55-72`][bm]).

#### Step 7 — Save results

```python
# Per-model JSON
results = {
    "model": model_name,
    "moran_i_mean": moran_i_mean,         # PRIMARY metric (from metrics.json)
    "n_clusters": n_clusters,
    "ARI": ari,
    "NMI": nmi,
    "HOM": hom,
    "COM": com,
    "CHAOS": chaos,
    "PAS": pas,
    "ASW": asw,
    "timestamp": datetime.now().isoformat(),
}
with open(benchmark_dir / f"{model_name}_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Combined CSV
results_df = pd.DataFrame(all_results).set_index("model")
results_df.to_csv(benchmark_dir / "benchmark_results.csv")
```

---

## `benchmark figures` — Implementation Details

### File: `spatial_factorization/commands/benchmark_figures.py`

#### Single-dataset figure (`outputs/<dataset>/figures/benchmark.png`)

```python
def plot_benchmark_single(benchmark_results_csv: Path, output_path: Path):
    """
    Bar chart comparing all models on all metrics for one dataset.

    Layout: one subplot per metric group, models on x-axis.
    Primary metric (Moran's I mean) gets its own panel, highlighted.
    """
    df = pd.read_csv(benchmark_results_csv, index_col="model")

    metrics_groups = {
        "Spatial Quality\n(Moran's I mean)": ["moran_i_mean"],
        "Accuracy": ["ARI", "NMI", "HOM", "COM"],
        "Continuity": ["CHAOS", "PAS", "ASW"],
    }
    # For CHAOS and PAS: lower is better — invert or annotate
    # Model color order: pca_baseline, nmf_baseline, pnmf, svgp, mggp_svgp, lcgp, mggp_lcgp
    ...
```

Figure dimensions: ~14×5 inches, 3 subplots side by side.

#### Multi-dataset aggregate figure

When scope is a directory (e.g., `configs/sdmbench/`), `benchmark figures` reads per-dataset
CSVs and produces one aggregate figure:

```python
def plot_benchmark_aggregate(dataset_results: dict[str, pd.DataFrame], output_path: Path):
    """
    Bar chart with mean ± std across all datasets in scope.
    Same layout as single-dataset but bars show error bars.
    """
    ...
```

The aggregate figure goes to: `{scope_output_dir}/figures/benchmark_aggregate.png`
For `configs/sdmbench/` scope → `outputs/sdmbench/figures/benchmark_aggregate.png`
For `configs/` scope → `outputs/benchmark_aggregate.png`

---

## Metric Reference

| Metric | Higher=Better | Requires GT | Source |
|--------|---------------|-------------|--------|
| Moran's I mean | ✓ | No | `metrics.json["moran_i"]["mean"]` (already computed!) |
| ARI | ✓ | Yes | `sdmbench.compute_ARI` (SDMBench.py line 140) |
| NMI | ✓ | Yes | `sdmbench.compute_NMI` (SDMBench.py line 143) |
| HOM | ✓ | Yes | `sdmbench.compute_HOM` (SDMBench.py line 146) |
| COM | ✓ | Yes | `sdmbench.compute_COM` (SDMBench.py line 149) |
| CHAOS | ✗ | No | `sdmbench.compute_CHAOS` (SDMBench.py lines 64–83) |
| PAS | ✗ | No | `sdmbench.compute_PAS` (SDMBench.py lines 113–119) |
| ASW | ✓ | No | `sdmbench.compute_ASW` (SDMBench.py lines 152–154) |
All SDMBench metrics: Zhao et al., https://github.com/zhaofangyuan98/SDMBench

### Why ground truth is always available for sdmbench

`C.npy` in `preprocessed/` stores the group codes. For sdmbench:
- Groups = cortical layers (Region column): Layer1..Layer6, WM
- These ARE the ground truth used by the original SDMBench benchmarking

For other datasets (liver, merfish, etc.):
- Groups = cell types (not region/layer ground truth)
- ARI/NMI/etc. would measure cell-type recovery, not spatial domain assignment
- Still valid as a secondary metric, but Moran's I is the primary spatial quality metric

---

## Config Changes

No new YAML fields strictly required. We infer everything from existing configs:

```python
# n_clusters = len(metadata["group_names"]) from preprocessed/metadata.json
# ground truth = C.npy (group codes 0..G-1)
# n_components = config.model.get("n_components", 10)
```

Optionally add a `benchmark:` section to general.yaml for overrides:
```yaml
# Optional — all values can be inferred if omitted
benchmark:
  n_clusters: ~          # null = infer from preprocessed groups (default)
  include_baselines: true
```

---

## Datasets with Ground Truth vs. Cell-Type Groups

| Dataset | Group column | Nature | GT for SDMBench accuracy? |
|---------|-------------|--------|---------------------------|
| sdmbench | `obs["Region"]` | Cortical layers | **Yes** (primary use case) |
| slideseq | `obs["cluster"]` | Leiden clusters | Partial (no anatomical GT) |
| tenxvisium | `obs["cluster"]` | Leiden clusters | Partial |
| merfish | `obs["Cell_class"]` | Cell types | Cell-type recovery metric |
| liver | `obs["Cell_Type"]` | Cell types | Cell-type recovery metric |
| osmfish | `obs["ClusterName"]` | Clusters | Partial |
| colon | CSV labels | Cell subtypes | Cell-type recovery metric |

For all datasets, Moran's I mean is the universal spatial quality metric.
For sdmbench, ARI/NMI/CHAOS/PAS/ASW are the full suite (same as original SDMBench benchmark).

---

## Implementation Plan

### Files to create

```
spatial_factorization/commands/
├── benchmark_analyze.py     # Stage: load factors, K-means, compute metrics, save CSV
├── benchmark_figures.py     # Stage: read CSV, plot bar charts
└── benchmark_metrics.py     # SDMBench wrappers + K-means helper
```

### Files to modify

```
spatial_factorization/cli.py     # Add @cli.group("benchmark") with analyze/figures/run subcommands
```

### Step-by-step

**Step 1: `benchmark_metrics.py`**

Thin wrapper around `sdmbench` (imported directly) plus the K-means helper:

```python
"""
Benchmark metric helpers.
Metrics are computed via the SDMBench package (Zhao et al.):
    https://github.com/zhaofangyuan98/SDMBench
Install once: python -m pip install -e <repo>/SDMBench/SDMBench/
"""
from SDMBench import sdmbench
from sklearn.cluster import KMeans
import numpy as np

def kmeans_on_factors(factors: np.ndarray, n_clusters: int, random_state: int = 0) -> np.ndarray:
    """K-means on (N, L) exp-space factors after log-transform."""
    # factors.npy stores exp(F); log before clustering (see generate_annotations.py:343)
    log_factors = np.log(factors + 1e-8)
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit_predict(log_factors)

def compute_all_metrics(adata_pred, gt_key: str, include_gt: bool = True) -> dict:
    """Run full SDMBench metric suite. adata_pred must have obs['pred'] set."""
    results = {
        "CHAOS": sdmbench.compute_CHAOS(adata_pred, "pred"),
        "PAS":   sdmbench.compute_PAS(adata_pred, "pred", spatial_key="spatial"),
        "ASW":   sdmbench.compute_ASW(adata_pred, "pred", spatial_key="spatial"),
    }
    if include_gt:
        results["ARI"] = sdmbench.compute_ARI(adata_pred, gt_key, "pred")
        results["NMI"] = sdmbench.compute_NMI(adata_pred, gt_key, "pred")
        results["HOM"] = sdmbench.compute_HOM(adata_pred, gt_key, "pred")
        results["COM"] = sdmbench.compute_COM(adata_pred, gt_key, "pred")
    return results
```

**Step 2: `benchmark_analyze.py`**

```python
def run(config_path: str, model_filter=(), include_baselines=True):
    config = Config.from_yaml(config_path)
    output_dir = Path(config.output_dir)
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(exist_ok=True)
    (benchmark_dir / "annotations").mkdir(exist_ok=True)

    # Load preprocessed data
    data = load_preprocessed(output_dir / "preprocessed")
    X = data["X"]          # (N, 2)
    C = data["C"]          # (N,) ground truth codes
    metadata = data["metadata"]
    n_clusters = len(metadata["group_names"])
    n_components = config.model.get("n_components", 10)

    all_results = []

    # --- Our 5 models ---
    for model_name in ["pnmf", "svgp", "mggp_svgp", "lcgp", "mggp_lcgp"]:
        if model_filter and model_name not in model_filter:
            continue
        model_dir = output_dir / model_name
        if not (model_dir / "factors.npy").exists():
            continue
        factors = np.load(model_dir / "factors.npy")   # (N, L) exp-space
        pred = kmeans_on_factors(factors, n_clusters)
        np.save(benchmark_dir / "annotations" / f"{model_name}.npy", pred)
        moran_i_mean = _load_moran_i_mean(model_dir)   # FREE from metrics.json
        metrics = _compute_all_metrics(C, pred, X, moran_i_mean)
        all_results.append({"model": model_name, **metrics})
        _save_model_results(benchmark_dir, model_name, metrics)

    # --- Baselines ---
    if include_baselines:
        from scipy.sparse import load_npz
        Y_dense = load_npz(output_dir / "preprocessed" / "Y.npz").T.toarray()
        for name, labels in [
            ("pca_baseline", _pca_baseline(Y_dense, n_components, n_clusters)),
            ("nmf_baseline", _nmf_baseline(Y_dense, n_components, n_clusters)),
        ]:
            if model_filter and name not in model_filter:
                continue
            np.save(benchmark_dir / "annotations" / f"{name}.npy", labels)
            metrics = _compute_all_metrics(C, labels, X, moran_i_mean=None)
            all_results.append({"model": name, **metrics})
            _save_model_results(benchmark_dir, name, metrics)

    # --- Save combined CSV ---
    df = pd.DataFrame(all_results).set_index("model")
    df.to_csv(benchmark_dir / "benchmark_results.csv")
    print(f"Saved: {benchmark_dir / 'benchmark_results.csv'}")
```

**Step 3: `benchmark_figures.py`**

```python
MODEL_ORDER = ["pca_baseline", "nmf_baseline", "pnmf", "svgp", "mggp_svgp", "lcgp", "mggp_lcgp"]
MODEL_COLORS = { ... }  # consistent with existing figures.py color scheme

def run_single(benchmark_dir: Path, figures_dir: Path):
    """One-dataset figure."""
    df = pd.read_csv(benchmark_dir / "benchmark_results.csv", index_col="model")
    df = df.reindex([m for m in MODEL_ORDER if m in df.index])
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    # Panel 0: Moran's I mean (PRIMARY, highlighted background)
    # Panel 1: Accuracy (ARI, NMI, HOM, COM) — grouped bars
    # Panel 2: Continuity (CHAOS lower=better, PAS lower=better, ASW)
    fig.savefig(figures_dir / "benchmark.png", dpi=150, bbox_inches="tight")

def run_aggregate(all_dfs: dict[str, pd.DataFrame], figures_dir: Path):
    """Multi-dataset aggregate figure (mean ± std bars)."""
    ...
```

**Step 4: `cli.py` additions**

```python
@cli.group("benchmark")
def benchmark_group():
    """Evaluate trained models with SDMBench metrics (Moran's I, ARI, NMI, CHAOS, ...)."""
    pass

@benchmark_group.command("analyze")
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--models", multiple=True, default=(),
              help="Only benchmark these model names (e.g. --models svgp mggp_svgp)")
@click.option("--no-baselines", is_flag=True, default=False)
@click.option("--config-name", default="general.yaml", show_default=True)
def benchmark_analyze(config, models, no_baselines, config_name):
    """Compute benchmark metrics for all trained models."""
    from .commands.benchmark_analyze import run_from_cli
    run_from_cli(config, model_filter=models, include_baselines=not no_baselines,
                 config_name=config_name)

@benchmark_group.command("figures")
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--config-name", default="general.yaml", show_default=True)
def benchmark_figures(config, config_name):
    """Generate benchmark comparison figure."""
    from .commands.benchmark_figures import run_from_cli
    run_from_cli(config, config_name=config_name)

@benchmark_group.command("run")
@click.argument("stages", nargs=-1, required=True)
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--models", multiple=True, default=())
@click.option("--no-baselines", is_flag=True, default=False)
@click.option("--config-name", default="general.yaml", show_default=True)
def benchmark_run(stages, config, models, no_baselines, config_name):
    """Run benchmark stages sequentially (analyze and/or figures)."""
    ...
```

---

## Multi-Dataset Scoping in `run_from_cli`

```python
def run_from_cli(config: str, model_filter=(), include_baselines=True,
                 config_name="general.yaml", **kwargs):
    """
    Entry point from CLI. Resolves config to a list of (dataset_name, per-model config paths)
    and runs benchmark_analyze.run() for each.
    """
    from pathlib import Path as P
    from ..config import Config

    config_path = P(config)

    if config_path.is_dir():
        # Recurse for all config_name files in directory tree
        targets = _resolve_directory(config_path, config_name)
    elif Config.is_general_config(config_path):
        targets = _resolve_general(config_path)
    else:
        # Per-model config — single target
        targets = [config_path]

    all_dfs = {}
    for target_config in targets:
        cfg = Config.from_yaml(target_config)
        output_dir = P(cfg.output_dir)
        run(str(target_config), model_filter=model_filter,
            include_baselines=include_baselines, **kwargs)
        bm_csv = output_dir / "benchmark" / "benchmark_results.csv"
        if bm_csv.exists():
            all_dfs[cfg.name] = pd.read_csv(bm_csv, index_col="model")

    # Save aggregate CSV alongside the scope config
    if len(all_dfs) > 1:
        _save_aggregate(all_dfs, config_path)
```

---

## Relationship to Existing Pipeline

```
preprocess → train → analyze → figures
                         ↓
                    benchmark analyze → benchmark figures
```

`benchmark analyze` only reads the OUTPUT of `analyze` (`factors.npy`, `metrics.json`).
It NEVER re-trains or re-runs the GP forward pass.

The `analyze` stage MUST have been run first. If `factors.npy` is missing for a model,
that model is silently skipped (same pattern as figures stage).

---

## Notes on Environments

Unlike the GPzoo pipeline which requires a separate `sdmbench` conda env, this implementation
imports directly from the `SDMBench` package (Zhao et al.,
https://github.com/zhaofangyuan98/SDMBench). All of its dependencies (scanpy, squidpy,
sklearn, scipy) are already present in the `factorization` env — only the `SDMBench` package
itself needs to be installed once:

```bash
conda run -n factorization python -m pip install -e \
    /gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/SDMBench/
```

After that, `from SDMBench import sdmbench` works and the entire benchmark runs in the
`factorization` environment with no subprocess or environment switching.

This means **the entire benchmark runs in the `factorization` environment** with no subprocess
or environment switching.

---

## Example Expected Output

```bash
$ spatial_factorization benchmark analyze -c configs/sdmbench/151507/general.yaml
Loading preprocessed data: outputs/sdmbench/151507/preprocessed/
  N=4221 spots, G=7 groups (Layer1, Layer2, Layer3, Layer4, Layer5, Layer6, WM)
  n_clusters=7 (inferred from groups)

Computing benchmark metrics...
  svgp        moran_i_mean=0.580  ARI=0.412  NMI=0.511  CHAOS=0.021  PAS=0.063  ASW=0.441
  mggp_svgp   moran_i_mean=0.572  ARI=0.428  NMI=0.519  CHAOS=0.020  PAS=0.061  ASW=0.453
  lcgp        moran_i_mean=0.561  ARI=0.391  NMI=0.487  CHAOS=0.023  PAS=0.068  ASW=0.428
  mggp_lcgp   moran_i_mean=0.567  ARI=0.415  NMI=0.503  CHAOS=0.021  PAS=0.064  ASW=0.437
  pnmf        moran_i_mean=0.311  ARI=0.253  NMI=0.347  CHAOS=0.031  PAS=0.089  ASW=0.312
Computing PCA baseline...
  pca_baseline            ARI=0.234  NMI=0.321  CHAOS=0.034  PAS=0.095  ASW=0.289
Computing NMF baseline...
  nmf_baseline            ARI=0.248  NMI=0.339  CHAOS=0.029  PAS=0.082  ASW=0.307

Saved: outputs/sdmbench/151507/benchmark/benchmark_results.csv

$ spatial_factorization benchmark figures -c configs/sdmbench/151507/general.yaml
Saved: outputs/sdmbench/151507/figures/benchmark.png

$ spatial_factorization benchmark analyze -c configs/sdmbench/
[151507] ... [151508] ... ... [151676] ...
Saved aggregate: outputs/sdmbench/benchmark_aggregate/summary.csv

$ spatial_factorization benchmark figures -c configs/sdmbench/
Saved aggregate figure: outputs/sdmbench/figures/benchmark_aggregate.png
```
