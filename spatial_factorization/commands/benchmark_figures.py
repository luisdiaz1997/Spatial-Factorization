"""Generate benchmark comparison figures.

Single-dataset: bar chart comparing all models on all metrics.
Multi-dataset: aggregate figure with mean +/- std bars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import Config
from .benchmark_analyze import _common_output_parent, _resolve_configs, _unique_output_dirs

MODEL_ORDER = ["pca_baseline", "pnmf", "svgp", "mggp_svgp", "lcgp", "mggp_lcgp"]

# Maps benchmark model key → output subdirectory name
MODEL_DIRNAME = {
    "pca_baseline": "pca",
    "pnmf": "pnmf",
    "svgp": "svgp",
    "mggp_svgp": "mggp_svgp",
    "lcgp": "lcgp",
    "mggp_lcgp": "mggp_lcgp",
}

MODEL_LABELS = {
    "pca_baseline": "PCA",
    "pnmf": "PNMF",
    "svgp": "SVGP",
    "mggp_svgp": "MGGP-SVGP",
    "lcgp": "LCGP",
    "mggp_lcgp": "MGGP-LCGP",
}

MODEL_COLORS = {
    "pca_baseline": "#999999",
    "pnmf": "#e377c2",
    "svgp": "#1f77b4",
    "mggp_svgp": "#ff7f0e",
    "lcgp": "#2ca02c",
    "mggp_lcgp": "#d62728",
}

# Metric groupings for subplots
METRIC_GROUPS = {
    "Spatial Quality": ["moran_i_mean"],
    "Accuracy": ["ARI", "NMI", "HOM", "COM"],
    "Continuity": ["CHAOS", "PAS", "ASW"],
}

# Lower is better for these
LOWER_IS_BETTER = {"CHAOS", "PAS"}

# SDMBench baseline methods (canonical order: alphabetical)
SDMBENCH_METHODS = [
    "BASS", "BayesSpace", "CCST", "GraphST", "Leiden", "Louvain",
    "SCAN-IT", "SEDR", "STAGATE", "SpaGCN", "SpaGCN(HE)", "SpaceFlow", "conST", "stLearn",
]
SDMBENCH_COLOR = "#bbbbbb"


def _order_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder rows to canonical model order."""
    order = [m for m in MODEL_ORDER if m in df.index]
    return df.loc[order]


_GROUPWISE_MODELS = ["mggp_svgp", "mggp_lcgp"]


def _load_moran_per_factor(output_dir: Path, models: list) -> dict:
    """Load per-factor Moran's I arrays from each model's moran_i.csv (marginal)."""
    result = {}
    for model in models:
        dirname = MODEL_DIRNAME.get(model, model)
        csv_path = output_dir / dirname / "moran_i.csv"
        if csv_path.exists():
            try:
                result[model] = pd.read_csv(csv_path)["moran_i"].values.astype(float)
            except Exception:
                result[model] = None
        else:
            result[model] = None
    return result


def _load_moran_conditional(output_dir: Path) -> dict:
    """Load per-group per-factor Moran's I for MGGP models from groupwise_moran_i.csv.

    Returns {mggp_svgp: array, mggp_lcgp: array} with values only for models
    that have a groupwise CSV; missing models are omitted.
    """
    result = {}
    for model in _GROUPWISE_MODELS:
        dirname = MODEL_DIRNAME.get(model, model)
        gw_path = output_dir / dirname / "groupwise_moran_i.csv"
        if gw_path.exists():
            try:
                result[model] = pd.read_csv(gw_path)["moran_i"].values.astype(float)
            except Exception:
                pass
    return result


def plot_single(benchmark_dir: Path, figures_dir: Path, dataset_name: str = ""):
    """Bar chart comparing all models for one dataset.

    Spatial Quality panel uses per-factor Moran's I box plots when available;
    Accuracy and Continuity panels remain grouped bar charts.
    """
    csv_path = benchmark_dir / "benchmark_results.csv"
    if not csv_path.exists():
        print(f"  No benchmark_results.csv found in {benchmark_dir}")
        return

    df = _order_df(pd.read_csv(csv_path, index_col="model"))
    models = df.index.tolist()
    labels = [MODEL_LABELS.get(m, m) for m in models]
    colors = [MODEL_COLORS.get(m, "#666666") for m in models]

    output_dir = benchmark_dir.parent
    moran_per_factor = _load_moran_per_factor(output_dir, models)
    moran_conditional = _load_moran_conditional(output_dir)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    title = f"Benchmark: {dataset_name}" if dataset_name else "Benchmark Results"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, (group_name, metrics) in zip(axes, METRIC_GROUPS.items()):
        available = [m for m in metrics if m in df.columns]
        if not available:
            ax.set_visible(False)
            continue

        # --- Spatial Quality: box plot of per-factor Moran's I ---
        if group_name == "Spatial Quality":
            # Marginal boxes (one per model, 10 values each)
            box_data = [moran_per_factor.get(m) for m in models]
            box_colors = list(colors)
            box_labels = list(labels)

            # Append MGGP conditional boxes (one per MGGP model, G*L values each)
            for mggp_name in _GROUPWISE_MODELS:
                if mggp_name in moran_conditional:
                    box_data.append(moran_conditional[mggp_name])
                    box_colors.append(MODEL_COLORS.get(mggp_name, "#666666"))
                    box_labels.append(MODEL_LABELS.get(mggp_name, mggp_name) + "\nconditional")

            has_boxes = any(v is not None for v in box_data)
            if has_boxes:
                positions = np.arange(len(box_data))
                valid_idx = [i for i, v in enumerate(box_data) if v is not None]
                n_marginal = len(models)

                bp = ax.boxplot(
                    [box_data[i] for i in valid_idx],
                    positions=positions[valid_idx],
                    widths=0.5,
                    patch_artist=True,
                    showfliers=False,
                    showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="black", markersize=5),
                    medianprops=dict(color="black", linewidth=1.5),
                )
                for patch, i in zip(bp["boxes"], valid_idx):
                    patch.set_facecolor(box_colors[i])
                    patch.set_alpha(0.75)
                    # Hatch the conditional boxes to visually distinguish them
                    if i >= n_marginal:
                        patch.set_hatch("///")
                        patch.set_edgecolor("black")

                # Overlay individual points
                rng = np.random.default_rng(0)
                for i in valid_idx:
                    vals = box_data[i]
                    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
                    ax.scatter(positions[i] + jitter, vals,
                               color=box_colors[i], alpha=0.5,
                               s=12 if i < n_marginal else 6,
                               zorder=3, edgecolors="gray", linewidths=0.4)

                # Vertical separator between marginal and conditional boxes
                if len(box_data) > n_marginal:
                    ax.axvline(n_marginal - 0.5, color="black",
                               linewidth=0.8, linestyle="--", alpha=0.4)

                ax.set_xticks(positions)
                ax.set_xticklabels(box_labels, rotation=45, ha="right", fontsize=8)
                ax.set_ylabel("Moran's I")
                ax.set_title(group_name, fontsize=11)
                ax.set_facecolor("#f0f7ff")
                continue  # skip the bar-chart block below

        # --- Accuracy / Continuity: grouped bar chart ---
        x = np.arange(len(models))
        width = 0.8 / max(len(available), 1)

        for i, metric in enumerate(available):
            vals = df[metric].values.astype(float)
            offset = (i - (len(available) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width * 0.9, color=colors, alpha=0.85)

            # Label each bar — rotated 90° so text never clips into adjacent bars
            if len(available) > 1:
                for bar, val in zip(bars, vals):
                    if not np.isnan(val):
                        y_pos = bar.get_height()
                        va = "bottom" if val >= 0 else "top"
                        ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                                f"{val:.2f}", ha="center", va=va,
                                fontsize=6, rotation=90)

        # Expand y-axis so rotated labels don't get clipped at the top/bottom
        ylo, yhi = ax.get_ylim()
        pad = (yhi - ylo) * 0.18
        ax.set_ylim(ylo - pad if ylo < 0 else ylo, yhi + pad)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_title(group_name, fontsize=11)

        if len(available) > 1:
            ax.legend(available, fontsize=8, loc="upper right",
                      bbox_to_anchor=(1, 1), framealpha=0.9)
        elif len(available) == 1:
            ax.set_ylabel(available[0])

        # Annotate lower-is-better
        lower_in_group = [m for m in available if m in LOWER_IS_BETTER]
        if lower_in_group:
            ax.text(0.02, 0.98, f"{', '.join(lower_in_group)}: lower = better",
                    transform=ax.transAxes, fontsize=7, va="top", color="gray")

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "benchmark.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _load_moran_conditional_per_dataset_mean(
    output_dirs: Dict[str, Path],
) -> Dict[str, Dict[str, float]]:
    """For each MGGP model, load groupwise_moran_i.csv per dataset and return
    {model: {dataset_label: mean_moran_i}} — one scalar per dataset.
    """
    result: Dict[str, Dict[str, float]] = {m: {} for m in _GROUPWISE_MODELS}
    for dataset_label, output_dir in output_dirs.items():
        for model in _GROUPWISE_MODELS:
            dirname = MODEL_DIRNAME.get(model, model)
            gw_path = Path(output_dir) / dirname / "groupwise_moran_i.csv"
            if gw_path.exists():
                try:
                    vals = pd.read_csv(gw_path)["moran_i"].values.astype(float)
                    result[model][dataset_label] = float(np.mean(vals))
                except Exception:
                    pass
    return result


def plot_aggregate(
    all_dfs: Dict[str, pd.DataFrame],
    figures_dir: Path,
    output_dirs: Dict[str, Path] = None,
):
    """Boxplot grid across datasets — one subplot per metric, models on x-axis.

    Each box shows the distribution across datasets for that model/metric pair.
    Individual dataset values are overlaid as scatter points.
    Layout: 3 columns, as many rows as needed (mirrors SDMBench benchmark figure).

    If output_dirs is provided, the Moran's I subplot also shows two extra
    hatched boxes for MGGP conditional posteriors (per-dataset mean across
    groups × factors).
    """
    # Stack all datasets into a long-form DataFrame
    records = []
    for dataset_name, df in all_dfs.items():
        for model in df.index:
            row = {"dataset": dataset_name, "model": model}
            for col in df.columns:
                row[col] = df.loc[model, col]
            records.append(row)
    long_df = pd.DataFrame(records)

    # MGGP conditional per-dataset means (for extra Moran's I boxes)
    conditional_per_dataset = (
        _load_moran_conditional_per_dataset_mean(output_dirs) if output_dirs else {}
    )

    # All metrics to plot — row 0: accuracy, row 1: spatial quality + continuity
    all_metrics = ["ARI", "NMI", "HOM", "COM", "moran_i_mean", "CHAOS", "PAS", "ASW"]
    available_metrics = [m for m in all_metrics if m in long_df.columns]

    # Metric display names and category prefixes
    metric_titles = {
        "moran_i_mean": "Spatial Quality: Moran's I",
        "ARI": "Accuracy: ARI",
        "NMI": "Accuracy: NMI",
        "HOM": "Accuracy: HOM",
        "COM": "Accuracy: COM",
        "CHAOS": "Continuity: CHAOS",
        "PAS": "Continuity: PAS",
        "ASW": "Continuity: ASW",
    }

    # Canonical model order (only those present)
    models_present = [m for m in MODEL_ORDER if m in long_df["model"].unique()]
    n_datasets = len(all_dfs)

    ncols = 4
    nrows = int(np.ceil(len(available_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Benchmark Comparison (n={n_datasets} datasets)", fontsize=14, fontweight="bold", y=1.01)

    for idx, metric in enumerate(available_metrics):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Collect data per model in canonical order
        box_data = []
        box_colors = []
        box_labels = []
        for model in models_present:
            vals = long_df.loc[long_df["model"] == model, metric].dropna().values.astype(float)
            box_data.append(vals)
            box_colors.append(MODEL_COLORS.get(model, "#666666"))
            box_labels.append(MODEL_LABELS.get(model, model))

        # For Moran's I panel: append MGGP conditional boxes (one value per dataset,
        # mean across groups × factors from groupwise_moran_i.csv)
        n_marginal = len(models_present)
        if metric == "moran_i_mean":
            for mggp_name in _GROUPWISE_MODELS:
                per_ds = conditional_per_dataset.get(mggp_name, {})
                if per_ds:
                    box_data.append(np.array(list(per_ds.values()), dtype=float))
                    box_colors.append(MODEL_COLORS.get(mggp_name, "#666666"))
                    box_labels.append(MODEL_LABELS.get(mggp_name, mggp_name) + "\nconditional")

        positions = np.arange(len(box_data))
        bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                       markeredgecolor="black", markersize=4),
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker="o", markerfacecolor="none",
                                        markeredgecolor="black", markersize=5))

        for i, (patch, color) in enumerate(zip(bp["boxes"], box_colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            if i >= n_marginal:
                patch.set_hatch("///")
                patch.set_edgecolor("black")

        # Overlay individual data points
        for i, vals in enumerate(box_data):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(positions[i] + jitter, vals, color=box_colors[i],
                       alpha=0.5, s=15, zorder=3, edgecolors="gray", linewidths=0.5)

        # Dashed separator between marginal and conditional boxes
        if len(box_data) > n_marginal:
            ax.axvline(n_marginal - 0.5, color="black",
                       linewidth=0.8, linestyle="--", alpha=0.4)

        ax.set_xticks(positions)
        ax.set_xticklabels([f"n={n_datasets}\n{lbl}" for lbl in box_labels],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(metric_titles.get(metric, metric), fontsize=11)
        ax.set_ylabel("Value")
        ax.set_xlabel("Method")

    # Hide unused subplots
    for idx in range(len(available_metrics), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "benchmark_aggregate.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_aggregate_sdmbench(
    all_dfs: Dict[str, pd.DataFrame],
    figures_dir: Path,
    baselines_df: pd.DataFrame,
):
    """3×3 boxplot grid: our models alongside 14 SDMBench baselines.

    Row 0 — Accuracy (shared):   NMI, HOM, COM       (our models + SDMBench)
    Row 1 — Continuity (shared): CHAOS, PAS, ASW     (our models + SDMBench)
    Row 2 — Ours only:           ARI, Moran's I, [empty]
    """
    # --- build our long-form data ---
    our_records = []
    for dataset_name, df in all_dfs.items():
        slide = dataset_name.split("/")[-1]
        for model in df.index:
            row = {"dataset": slide, "model": model}
            for col in df.columns:
                row[col] = df.loc[model, col]
            our_records.append(row)
    our_df = pd.DataFrame(our_records)

    baselines = baselines_df.copy()
    baselines["dataset"] = baselines["dataset"].astype(str)

    # Which slides appear in our data (to filter baselines to the same set)
    our_slides = set(our_df["dataset"].unique())

    our_models_present = [m for m in MODEL_ORDER if m in our_df["model"].unique()]
    # Sort SDMBench methods by median NMI (ascending: worst to best)
    sdmbench_in_data = [m for m in SDMBENCH_METHODS if m in baselines["model"].unique()]
    nmi_medians = {
        m: baselines.loc[baselines["model"] == m, "NMI"].median()
        for m in sdmbench_in_data
    }
    sdmbench_present = sorted(sdmbench_in_data, key=lambda m: nmi_medians[m])

    metric_layout = [
        ["NMI", "HOM", "COM"],
        ["CHAOS", "PAS", "ASW"],
    ]

    metric_titles = {
        "NMI": "Accuracy: NMI",
        "HOM": "Accuracy: HOM",
        "COM": "Accuracy: COM",
        "CHAOS": "Continuity: CHAOS",
        "PAS": "Continuity: PAS",
        "ASW": "Continuity: ASW",
    }

    n_datasets = len(all_dfs)
    nrows, ncols = 2, 3
    # Wide enough for ~20 boxes per subplot
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4.5 * nrows))
    fig.suptitle(
        f"Benchmark vs SDMBench Baselines (n={n_datasets} datasets)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    rng = np.random.default_rng(42)

    for r, metric_row in enumerate(metric_layout):
        for c, metric in enumerate(metric_row):
            ax = axes[r, c]

            if metric is None:
                ax.set_visible(False)
                continue

            methods = our_models_present + sdmbench_present

            box_data, box_colors, box_labels = [], [], []
            for method in methods:
                if method in our_models_present:
                    vals = (
                        our_df.loc[our_df["model"] == method, metric]
                        .dropna().values.astype(float)
                    )
                    color = MODEL_COLORS.get(method, "#666666")
                    label = MODEL_LABELS.get(method, method)
                else:
                    sub = baselines[
                        (baselines["model"] == method) &
                        (baselines["dataset"].isin(our_slides))
                    ]
                    vals = sub[metric].dropna().values.astype(float) if metric in sub.columns else np.array([])
                    color = SDMBENCH_COLOR
                    label = method

                box_data.append(vals)
                box_colors.append(color)
                box_labels.append(label)

            positions = np.arange(len(methods))

            # Only plot boxes where we have data
            valid = [i for i, v in enumerate(box_data) if len(v) > 0]
            if not valid:
                ax.set_visible(False)
                continue

            bp = ax.boxplot(
                [box_data[i] for i in valid],
                positions=positions[valid],
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="white",
                               markeredgecolor="black", markersize=4),
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor="none",
                                markeredgecolor="black", markersize=4),
            )

            for patch, i in zip(bp["boxes"], valid):
                patch.set_facecolor(box_colors[i])
                patch.set_alpha(0.7)

            for i in valid:
                jitter = rng.uniform(-0.15, 0.15, size=len(box_data[i]))
                ax.scatter(
                    positions[i] + jitter, box_data[i],
                    color=box_colors[i], alpha=0.5, s=12, zorder=3,
                    edgecolors="gray", linewidths=0.4,
                )

            # Vertical separator between our models and SDMBench
            if our_models_present and sdmbench_present:
                sep = len(our_models_present) - 0.5
                ax.axvline(sep, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

            ax.set_xticks(positions)
            ax.set_xticklabels(box_labels, rotation=60, ha="right", fontsize=8)
            ax.set_title(metric_titles.get(metric, metric), fontsize=11)
            ax.set_ylabel("Value")

            if metric in LOWER_IS_BETTER:
                ax.text(0.02, 0.98, "lower = better", transform=ax.transAxes,
                        fontsize=7, va="top", color="gray")

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "benchmark_aggregate_sdmbench.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_pca_factors_spatial(output_dir: Path, coords: np.ndarray) -> None:
    """Spatial plot of PCA factors using percentile-based color scale."""
    factors_path = output_dir / "pca" / "factors.npy"
    if not factors_path.exists():
        return

    factors = np.load(factors_path)  # (N, L)
    L = factors.shape[1]
    ncols = 5
    nrows = int(np.ceil(L / ncols))
    figsize_per = 3.0

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per * ncols + 1.0, figsize_per * nrows),
        squeeze=False,
    )

    vmin = float(np.percentile(factors, 1))
    vmax = float(np.percentile(factors, 99))

    for i in range(L):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=factors[:, i], vmin=vmin, vmax=vmax,
            alpha=0.8, cmap="turbo", s=0.5,
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")
        ax.set_title(f"PC {i+1}", fontsize=9)

    for i in range(L, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="PC value")

    figures_dir = output_dir / "pca" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "factors_spatial.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def run_single(output_dir: Path, dataset_name: str = ""):
    """Generate benchmark figure for one dataset."""
    benchmark_dir = output_dir / "benchmark"
    figures_dir = output_dir / "figures"
    plot_single(benchmark_dir, figures_dir, dataset_name=dataset_name)

    # PCA spatial factors plot
    pca_factors_path = output_dir / "pca" / "factors.npy"
    if pca_factors_path.exists():
        from ..datasets.base import load_preprocessed
        coords = load_preprocessed(output_dir).X.numpy()
        plot_pca_factors_spatial(output_dir, coords)

    # Groupwise Moran's I breakdown (LCGP only)
    plot_groupwise_moran_breakdown(output_dir)


def plot_groupwise_moran_breakdown(output_dir: Path):
    """Groupwise conditional Moran's I breakdown: per-cell-type and per-factor box plots (LCGP only)."""
    model = "mggp_lcgp"
    color = MODEL_COLORS[model]

    csv_path = output_dir / model / "groupwise_moran_i.csv"
    if not csv_path.exists():
        print(f"  No groupwise_moran_i.csv for {model}, skipping")
        return
    df = pd.read_csv(csv_path)
    marginal_median = pd.read_csv(output_dir / model / "moran_i.csv")["moran_i"].median()

    group_means = df.groupby("group_name")["moran_i"].mean().sort_values(ascending=False)
    groups = group_means.index.tolist()
    factors = sorted(df["factor_idx"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    rng = np.random.default_rng(42)

    # Per cell type
    ax = axes[0]
    data_by_group = [df[df["group_name"] == g]["moran_i"].values for g in groups]
    positions = np.arange(len(groups))
    bp = ax.boxplot(data_by_group, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                   markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markerfacecolor="none",
                                    markeredgecolor="black", markersize=5))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(data_by_group):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, color=color,
                   alpha=0.5, s=15, zorder=3, edgecolors="gray", linewidths=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels(groups, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title("Per Cell Type", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(marginal_median, color="gray", linestyle=":", linewidth=1.5, zorder=0,
               label=f"Marginal median = {marginal_median:.2f}")
    ax.legend(fontsize=9, loc="lower right")

    # Per factor
    ax = axes[1]
    data_by_factor = [df[df["factor_idx"] == f]["moran_i"].values for f in factors]
    positions = np.arange(len(factors))
    bp = ax.boxplot(data_by_factor, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                   markeredgecolor="black", markersize=4),
                    medianprops=dict(color="black", linewidth=1.5),
                    flierprops=dict(marker="o", markerfacecolor="none",
                                    markeredgecolor="black", markersize=5))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for i, vals in enumerate(data_by_factor):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(positions[i] + jitter, vals, color=color,
                   alpha=0.5, s=15, zorder=3, edgecolors="gray", linewidths=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"Factor {f+1}" for f in factors], fontsize=10)
    ax.set_ylabel("Moran's I", fontsize=12)
    ax.set_title("Per Factor", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(marginal_median, color="gray", linestyle=":", linewidth=1.5, zorder=0,
               label=f"Marginal median = {marginal_median:.2f}")
    ax.legend(fontsize=9, loc="lower right")

    plt.suptitle(f"Groupwise Conditional Moran's I — {output_dir.name} ({MODEL_LABELS[model]})", fontsize=15, y=1.02)
    plt.tight_layout()
    out = output_dir / "figures" / "groupwise_moran_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def run_from_cli(config: str, config_name: str = "general.yaml"):
    """Entry point from CLI. Handles directory scoping."""
    config_path = Path(config)
    config_paths = _resolve_configs(config_path, config_name)

    if not config_paths:
        print("No configs found.")
        return

    targets = _unique_output_dirs(config_paths)
    all_dfs = {}
    output_dirs_by_label: Dict[str, Path] = {}

    for output_dir, cfg_path in targets:
        dataset_label = "/".join(output_dir.parts[1:]) if len(output_dir.parts) > 1 else str(output_dir)
        bm_csv = output_dir / "benchmark" / "benchmark_results.csv"

        if not bm_csv.exists():
            print(f"  Skipping {dataset_label} (no benchmark_results.csv)")
            continue

        run_single(output_dir, dataset_name=dataset_label)
        all_dfs[dataset_label] = pd.read_csv(bm_csv, index_col="model")
        output_dirs_by_label[dataset_label] = output_dir

    if len(all_dfs) > 1:
        # Aggregate figure goes in common parent of all output dirs
        benchmarked_dirs = [od for od, _ in targets if (od / "benchmark" / "benchmark_results.csv").exists()]
        agg_figures_dir = _common_output_parent(benchmarked_dirs) / "figures"
        plot_aggregate(all_dfs, agg_figures_dir, output_dirs=output_dirs_by_label)

        # SDMBench comparison plot: auto-detect baselines CSV next to figures dir
        baselines_csv = agg_figures_dir.parent / "benchmarks" / "sdmbench_baselines.csv"
        if baselines_csv.exists():
            baselines_df = pd.read_csv(baselines_csv)
            plot_aggregate_sdmbench(all_dfs, agg_figures_dir, baselines_df)
        else:
            print(f"  No SDMBench baselines found at {baselines_csv}, skipping comparison plot.")
