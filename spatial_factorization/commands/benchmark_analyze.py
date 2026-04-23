"""Benchmark trained models using SDMBench metrics (Zhao et al.).

Computes K-means cluster assignments from learned factors, then evaluates
with ARI, NMI, HOM, COM, CHAOS, PAS, ASW (via SDMBench) plus Moran's I
mean (read from existing metrics.json).

SDMBench: Zhao F, et al. https://github.com/zhaofangyuan98/SDMBench
"""

from __future__ import annotations

import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from ..config import Config
from ..datasets.base import load_preprocessed
from ..status import JobStatus, StatusManager

# SDMBench path — imported by path since the package's editable install is broken
_SDMBENCH_PATH = Path(__file__).resolve().parents[3] / "SDMBench" / "SDMBench" / "SDMBench.py"

MODEL_NAMES = ["pnmf", "svgp", "mggp_svgp", "lcgp", "mggp_lcgp"]


def _load_sdmbench():
    """Import sdmbench class from SDMBench.py by file path."""
    # Try normal import first (in case it gets fixed)
    try:
        from SDMBench import sdmbench
        return sdmbench
    except ImportError:
        pass

    # Fall back to path-based import
    sdmbench_path = _SDMBENCH_PATH
    if not sdmbench_path.exists():
        # Try relative to gitclones
        alt = Path.home() / "gitclones" / "SDMBench" / "SDMBench" / "SDMBench.py"
        if alt.exists():
            sdmbench_path = alt
        else:
            raise ImportError(
                f"SDMBench not found at {_SDMBENCH_PATH} or {alt}. "
                "Install: pip install -e /path/to/SDMBench/SDMBench/"
            )
    spec = importlib.util.spec_from_file_location("SDMBench", str(sdmbench_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.sdmbench


def _kmeans_on_factors(factors: np.ndarray, n_clusters: int) -> np.ndarray:
    """K-means on (N, L) exp-space factors after log-transform.

    factors.npy stores exp(F); we log-transform before clustering
    (consistent with GPzoo generate_annotations.py:343).
    """
    log_factors = np.log(factors + 1e-8)
    return KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(log_factors)



def _load_moran_i_mean(model_dir: Path) -> Optional[float]:
    """Read moran_i.mean from already-computed metrics.json."""
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        metrics = json.load(f)
    return metrics.get("moran_i", {}).get("mean")


def _compute_sdmbench_metrics(
    sdmbench_cls, X: np.ndarray, gt_labels: np.ndarray, pred_labels: np.ndarray
) -> dict:
    """Compute SDMBench metrics for one set of predictions.

    Constructs a minimal AnnData with obs['gt'], obs['pred'], obsm['spatial'].
    """
    adata = anndata.AnnData(
        obs=pd.DataFrame({
            "gt": pd.Categorical(gt_labels),
            "pred": pd.Categorical(pred_labels),
        })
    )
    adata.obsm["spatial"] = X

    results = {}
    # Ground-truth-dependent metrics
    results["ARI"] = sdmbench_cls.compute_ARI(adata, "gt", "pred")
    results["NMI"] = sdmbench_cls.compute_NMI(adata, "gt", "pred")
    results["HOM"] = sdmbench_cls.compute_HOM(adata, "gt", "pred")
    results["COM"] = sdmbench_cls.compute_COM(adata, "gt", "pred")
    # Spatial continuity metrics
    results["CHAOS"] = sdmbench_cls.compute_CHAOS(adata, "pred")
    results["PAS"] = sdmbench_cls.compute_PAS(adata, "pred", spatial_key="spatial")
    results["ASW"] = sdmbench_cls.compute_ASW(adata, "pred", spatial_key="spatial")
    return results


def _benchmark_one_model(
    model_name: str,
    model_dir: Path,
    benchmark_dir: Path,
    sdmbench_cls,
    X: np.ndarray,
    gt_labels: np.ndarray,
    n_clusters: int,
    status_manager: StatusManager,
    job_prefix: str,
) -> Optional[dict]:
    """Benchmark a single model. Returns results dict or None on failure."""
    job_name = f"{job_prefix}_{model_name}"
    status_manager.update_job(job_name, status="running", start_time=time.time())

    try:
        factors = np.load(model_dir / "factors.npy")
        pred = _kmeans_on_factors(factors, n_clusters)
        np.save(benchmark_dir / "annotations" / f"{model_name}.npy", pred)

        moran_i_mean = _load_moran_i_mean(model_dir)
        metrics = _compute_sdmbench_metrics(sdmbench_cls, X, gt_labels, pred)
        result = {"model": model_name, "moran_i_mean": moran_i_mean, **metrics}

        # Save per-model JSON
        with open(benchmark_dir / f"{model_name}_results.json", "w") as f:
            json.dump(result, f, indent=2)

        status_manager.update_job(job_name, status="completed", end_time=time.time())
        return result
    except Exception as e:
        status_manager.update_job(job_name, status="failed", end_time=time.time())
        print(f"  [{model_name}] Error: {e}")
        return None


def _compute_moran_i_mean(factors: np.ndarray, coords: np.ndarray) -> float:
    """Compute mean Moran's I across factors. Reuses analyze module."""
    from .analyze import _compute_moran_i
    _, moran_values = _compute_moran_i(factors, coords)
    return float(np.mean(moran_values))


def _compute_groupwise_moran_i(
    model_dir: Path, output_dir: Path, group_names: list
) -> Optional[np.ndarray]:
    """Compute per-group per-factor Moran's I from saved groupwise_factors/.

    groupwise_factors/group_g.npy: (N, L) exp-space, columns in global Moran's I order.
    Saves groupwise_moran_i.csv with columns: group_idx, group_name, factor_idx, moran_i.
    Returns flat moran_i array (G*L values) for bulk use in box plots.
    """
    gf_dir = model_dir / "groupwise_factors"
    if not gf_dir.exists():
        return None

    coords = np.load(output_dir / "preprocessed" / "X.npy")
    from .analyze import _compute_moran_i

    records = []
    group_files = sorted(
        gf_dir.glob("group_*.npy"), key=lambda p: int(p.stem.split("_")[1])
    )
    for gf_path in group_files:
        g = int(gf_path.stem.split("_")[1])
        group_name = group_names[g] if g < len(group_names) else str(g)
        factors_g = np.load(gf_path)  # (N, L) exp-space

        # _compute_moran_i returns sorted (idx, values); invert to get per-column values
        moran_idx, moran_sorted = _compute_moran_i(factors_g, coords)
        moran_per_factor = np.empty(len(moran_idx))
        moran_per_factor[moran_idx] = moran_sorted

        for factor_idx, mi in enumerate(moran_per_factor):
            records.append({
                "group_idx": g,
                "group_name": group_name,
                "factor_idx": factor_idx,
                "moran_i": float(mi),
            })

    df = pd.DataFrame(records)
    out_path = model_dir / "groupwise_moran_i.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return df["moran_i"].values


def _compute_factor_specificity(
    model_dir: Path, output_dir: Path, group_names: list
) -> None:
    """Compute L1 specificity: ||cond||_1 / ||marginal||_1 per group per factor.

    Clips both marginal and conditional at the per-factor 99th percentile of the
    marginal before computing L1, to prevent GP-extrapolation outliers in the
    top 1% from dominating the sum.
    """
    gf_dir = model_dir / "groupwise_factors"
    if not gf_dir.exists():
        return

    marginal = np.load(model_dir / "factors.npy")
    p99 = np.percentile(marginal, 99, axis=0)
    marginal_clipped = np.minimum(marginal, p99[None, :])
    m_l1 = marginal_clipped.sum(axis=0)

    records = []
    for gf_path in sorted(gf_dir.glob("group_*.npy"), key=lambda p: int(p.stem.split("_")[1])):
        g = int(gf_path.stem.split("_")[1])
        group_name = group_names[g] if g < len(group_names) else str(g)
        cond = np.load(gf_path)
        cond_clipped = np.minimum(cond, p99[None, :])
        c_l1 = cond_clipped.sum(axis=0)
        ratios = c_l1 / (m_l1 + 1e-10)
        for fi, r in enumerate(ratios):
            records.append({
                "group_idx": g,
                "group_name": group_name,
                "factor_idx": fi,
                "l1_ratio": float(r),
            })

    df = pd.DataFrame(records)
    out_path = model_dir / "factor_specificity.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")

    # Shannon entropy per factor from normalized L1 ratios
    n_groups = df["group_idx"].nunique()
    n_factors = df["factor_idx"].nunique()
    ent_records = []
    for fi in range(n_factors):
        r = df[df["factor_idx"] == fi]["l1_ratio"].values
        lr = np.log2(r)
        lr_shifted = np.clip(lr + 1.0, 0, None)
        p = lr_shifted / (lr_shifted.sum() + 1e-10)
        p = p[p > 0]
        h = -np.sum(p * np.log2(p)) / np.log2(n_groups)
        ent_records.append({"factor_idx": fi, "shannon_entropy": float(h)})
    ent_df = pd.DataFrame(ent_records)

    # Classify each factor: factor_specific | celltype_enriched | universal
    # - factor_specific: low entropy (H<0.9), most CTs depleted, 1-2 enriched
    # - celltype_enriched: multiple CTs enriched or strong enrichment (>1.5x),
    #   shows new patterns not seen in marginal
    # - universal: high entropy, all CTs ~preserved (ratio≈1)
    class_records = []
    for fi in range(n_factors):
        h = ent_df[ent_df["factor_idx"] == fi]["shannon_entropy"].values[0]
        ratios = df[df["factor_idx"] == fi]["l1_ratio"].values
        n_enriched = int((ratios > 1.0).sum())
        n_enriched_strong = int((ratios > 1.5).sum())
        max_r = float(ratios.max())

        mean_r = float(ratios.mean())
        if max_r > 1.5:
            cls = "celltype_enriched"
        elif h < 0.9 and (n_enriched >= 1 or mean_r < 0.9):
            cls = "factor_specific"
        else:
            cls = "universal"
        class_records.append({"factor_idx": fi, "class": cls})

    ent_df = ent_df.merge(pd.DataFrame(class_records), on="factor_idx")
    ent_path = model_dir / "factor_entropy.csv"
    ent_df.to_csv(ent_path, index=False)
    print(f"  Saved: {ent_path}")

    # Per-gene factor entropy (B2 direction): each gene's loading distribution across factors
    loadings_path = model_dir / "loadings.npy"
    meta_path = output_dir / "preprocessed" / "metadata.json"
    if loadings_path.exists() and meta_path.exists():
        import json
        loadings = np.load(loadings_path)  # (D, L)
        gene_names = json.load(open(meta_path))["gene_names"]
        D, L = loadings.shape

        eps = 1e-10
        W_pos = np.maximum(loadings, eps)
        p = W_pos / W_pos.sum(axis=1, keepdims=True)
        log_p = np.log2(np.clip(p, 1e-30, None))
        H = -(p * log_p).sum(axis=1) / np.log2(L)

        gene_ent_df = pd.DataFrame({
            "gene_idx": range(D),
            "gene_name": gene_names,
            "shannon_entropy": H,
        })
        gene_ent_path = model_dir / "gene_factor_entropy.csv"
        gene_ent_df.to_csv(gene_ent_path, index=False)
        print(f"  Saved: {gene_ent_path}")


def _benchmark_pca_baseline(
    Y: np.ndarray,
    n_components: int,
    benchmark_dir: Path,
    sdmbench_cls,
    X: np.ndarray,
    gt_labels: np.ndarray,
    n_clusters: int,
    status_manager: StatusManager,
    job_prefix: str,
    model_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Compute PCA baseline and benchmark it."""
    job_name = f"{job_prefix}_pca_baseline"
    status_manager.update_job(job_name, status="running", start_time=time.time())

    try:
        pca = PCA(n_components=n_components, random_state=0)
        pca_factors = pca.fit_transform(Y)
        if model_dir is not None:
            model_dir.mkdir(parents=True, exist_ok=True)
            np.save(model_dir / "factors.npy", pca_factors)
        pred = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit_predict(pca_factors)
        np.save(benchmark_dir / "annotations" / "pca_baseline.npy", pred)

        # Compute per-component Moran's I and save CSV
        from .analyze import _compute_moran_i
        moran_idx, moran_sorted = _compute_moran_i(pca_factors, X)
        # Reconstruct per-PC values in original PC order
        moran_per_pc = np.empty(len(moran_idx))
        moran_per_pc[moran_idx] = moran_sorted
        moran_i_mean = float(np.mean(moran_per_pc))
        if model_dir is not None:
            pd.DataFrame({"factor_idx": np.arange(len(moran_per_pc)), "moran_i": moran_per_pc}).to_csv(
                model_dir / "moran_i.csv", index=False
            )

        metrics = _compute_sdmbench_metrics(sdmbench_cls, X, gt_labels, pred)
        result = {"model": "pca_baseline", "moran_i_mean": moran_i_mean, **metrics}

        with open(benchmark_dir / "pca_baseline_results.json", "w") as f:
            json.dump(result, f, indent=2)

        status_manager.update_job(job_name, status="completed", end_time=time.time())
        return result
    except Exception as e:
        status_manager.update_job(job_name, status="failed", end_time=time.time())
        print(f"  [pca_baseline] Error: {e}")
        return None


def run(config_path: str, model_filter: tuple = (), include_baselines: bool = True):
    """Benchmark all trained models for a single dataset.

    Args:
        config_path: Path to a per-model or general YAML.
        model_filter: If non-empty, only benchmark these model names.
        include_baselines: Whether to compute PCA baseline.
    """
    config = Config.from_yaml(config_path)
    output_dir = Path(config.output_dir)
    benchmark_dir = output_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "annotations").mkdir(exist_ok=True)

    # Load preprocessed data
    data = load_preprocessed(output_dir)
    X = data.X.numpy()
    gt_labels = data.groups.numpy()
    n_clusters = data.n_groups
    n_components = config.model.get("n_components", 10)
    group_names = data.group_names or []

    print(f"  N={X.shape[0]}, G={n_clusters} groups ({', '.join(group_names[:5])}{'...' if len(group_names) > 5 else ''})")
    print(f"  n_clusters={n_clusters} (inferred from groups)")

    # Collect available models
    model_dirs = {}
    for name in MODEL_NAMES:
        if model_filter and name not in model_filter:
            continue
        d = output_dir / name
        if (d / "factors.npy").exists():
            model_dirs[name] = d

    if not model_dirs and not include_baselines:
        print("  No models found to benchmark.")
        return

    # Load SDMBench
    sdmbench_cls = _load_sdmbench()

    # Derive job prefix from output_dir
    out_parts = output_dir.parts
    job_prefix = "_".join(out_parts[1:]) if len(out_parts) > 1 else out_parts[0] if out_parts else "benchmark"

    # Setup status manager with all jobs
    status_manager = StatusManager()
    for name in model_dirs:
        status_manager.add_job(JobStatus(
            name=f"{job_prefix}_{name}", model=name, task="benchmark",
            config_path=Path(config_path), device="cpu", status="pending",
        ))
    if include_baselines:
        status_manager.add_job(JobStatus(
            name=f"{job_prefix}_pca_baseline", model="pca_baseline", task="benchmark",
            config_path=Path(config_path), device="cpu", status="pending",
        ))

    all_results = []

    # Run sequentially to avoid OpenBLAS segfaults from concurrent KNN on large N
    with status_manager:
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {}

            for model_name, model_dir in model_dirs.items():
                fut = executor.submit(
                    _benchmark_one_model,
                    model_name, model_dir, benchmark_dir, sdmbench_cls,
                    X, gt_labels, n_clusters, status_manager, job_prefix,
                )
                futures[fut] = model_name

            if include_baselines:
                Y = data.Y.numpy()
                fut = executor.submit(
                    _benchmark_pca_baseline,
                    Y, n_components, benchmark_dir, sdmbench_cls,
                    X, gt_labels, n_clusters, status_manager, job_prefix,
                    output_dir / "pca",
                )
                futures[fut] = "pca_baseline"

            for fut in as_completed(futures):
                result = fut.result()
                if result:
                    all_results.append(result)

    # Compute groupwise Moran's I for MGGP models (uses saved groupwise_factors/)
    for mggp_name in ["mggp_svgp", "mggp_lcgp"]:
        if mggp_name in model_dirs:
            print(f"  Computing groupwise Moran's I for {mggp_name}...")
            _compute_groupwise_moran_i(model_dirs[mggp_name], output_dir, group_names)
            _compute_factor_specificity(model_dirs[mggp_name], output_dir, group_names)

    # Save combined CSV
    if all_results:
        df = pd.DataFrame(all_results).set_index("model")
        # Reorder to canonical order
        order = [m for m in MODEL_NAMES + ["pca_baseline"] if m in df.index]
        df = df.loc[order]
        df.to_csv(benchmark_dir / "benchmark_results.csv")
        print(f"\nSaved: {benchmark_dir / 'benchmark_results.csv'}")

        # Print summary table
        print(f"\n{'Model':<16} {'Moran I':>8} {'ARI':>6} {'NMI':>6} {'CHAOS':>7} {'PAS':>6} {'ASW':>6}")
        print("-" * 62)
        for _, row in df.iterrows():
            mi = f"{row['moran_i_mean']:.3f}" if pd.notna(row.get('moran_i_mean')) else "  -"
            print(f"{row.name:<16} {mi:>8} {row['ARI']:>6.3f} {row['NMI']:>6.3f} "
                  f"{row['CHAOS']:>7.4f} {row['PAS']:>6.3f} {row['ASW']:>6.3f}")


def _resolve_configs(config_path: Path, config_name: str = "general.yaml") -> List[Path]:
    """Resolve config path to list of general/per-model configs.

    Mirrors JobRunner._resolve_configs() logic.
    """
    if config_path.is_dir():
        matched = sorted(config_path.rglob(config_name))
        if matched:
            config_paths = []
            for p in matched:
                if Config.is_general_config(p):
                    from ..generate import generate_configs
                    generated = generate_configs(p)
                    # We only need one config per dataset to get output_dir
                    config_paths.append(next(iter(generated.values())))
                else:
                    config_paths.append(p)
            return config_paths
        # Fallback: all yamls
        return sorted(config_path.rglob("*.yaml"))

    if Config.is_general_config(config_path):
        from ..generate import generate_configs
        generated = generate_configs(config_path)
        return [next(iter(generated.values()))]

    return [config_path]


def _unique_output_dirs(config_paths: List[Path]) -> List[Tuple[Path, Path]]:
    """Deduplicate configs by output_dir, returning (output_dir, config_path) pairs."""
    seen = set()
    result = []
    for p in config_paths:
        cfg = Config.from_yaml(p)
        od = cfg.output_dir
        if od not in seen:
            seen.add(od)
            result.append((Path(od), p))
    return result


def run_from_cli(config: str, model_filter: tuple = (), include_baselines: bool = True,
                 config_name: str = "general.yaml"):
    """Entry point from CLI. Handles directory scoping."""
    config_path = Path(config)
    config_paths = _resolve_configs(config_path, config_name)

    if not config_paths:
        print("No configs found.")
        return

    targets = _unique_output_dirs(config_paths)
    print(f"\nBenchmarking {len(targets)} dataset(s)...")

    all_dfs = {}
    benchmarked_output_dirs = []
    for output_dir, cfg_path in targets:
        dataset_label = "/".join(output_dir.parts[1:]) if len(output_dir.parts) > 1 else str(output_dir)
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset_label}")
        print(f"{'='*60}")

        if not (output_dir / "preprocessed").exists():
            print(f"  Skipping (no preprocessed data)")
            continue

        run(str(cfg_path), model_filter=model_filter, include_baselines=include_baselines)

        bm_csv = output_dir / "benchmark" / "benchmark_results.csv"
        if bm_csv.exists():
            all_dfs[dataset_label] = pd.read_csv(bm_csv, index_col="model")
            benchmarked_output_dirs.append(output_dir)

    # Save aggregate if multiple datasets
    if len(all_dfs) > 1:
        _save_aggregate(all_dfs, benchmarked_output_dirs)


def _common_output_parent(output_dirs: list) -> Path:
    """Find the common parent of all output directories."""
    parts_list = [Path(d).parts for d in output_dirs]
    common = []
    for level in zip(*parts_list):
        if len(set(level)) == 1:
            common.append(level[0])
        else:
            break
    return Path(*common) if common else Path("outputs")


def _save_aggregate(all_dfs: dict, output_dirs: list):
    """Save aggregate summary CSV in the common parent of all output dirs."""
    agg_dir = _common_output_parent(output_dirs) / "benchmark_aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)

    # Save per-dataset CSVs
    for name, df in all_dfs.items():
        safe_name = name.replace("/", "_")
        df.to_csv(agg_dir / f"{safe_name}_results.csv")

    # Compute mean ± std across datasets
    combined = pd.concat(all_dfs.values(), keys=all_dfs.keys())
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    summary_mean = combined.groupby("model")[numeric_cols].mean()
    summary_std = combined.groupby("model")[numeric_cols].std()

    summary = summary_mean.round(4).astype(str) + " ± " + summary_std.round(4).astype(str)
    summary.to_csv(agg_dir / "summary.csv")
    print(f"\nAggregate saved: {agg_dir / 'summary.csv'}")

    return agg_dir
