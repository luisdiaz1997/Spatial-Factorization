"""Generate publication figures (Stage 4).

Generates:
- factors_spatial.png: Spatial factor visualization (sorted by Moran's I)
- scales_spatial.png: Spatial factor uncertainty visualization (sigma values)
- points.png: Group spatial map (+ inducing points for spatial models)
- elbo_curve.png: Training ELBO convergence
- top_genes.png: Top genes per factor by loading magnitude
- factors_with_genes.png: Factors alongside top gene expression (like GPzoo notebooks)
- gene_enrichment.png: Gene enrichment per factor by cell-type (group vs global loadings)

Outputs saved to: outputs/{dataset}/{model}/figures/
"""

import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
from scipy import sparse

from ..config import Config
from ..datasets.base import load_preprocessed
from .analyze import _normalize_loadings


def _load_analysis_results(model_dir: Path) -> dict:
    """Load analysis results from model directory."""
    results = {}

    # Load factors
    factors_path = model_dir / "factors.npy"
    if factors_path.exists():
        results["factors"] = np.load(factors_path)  # (N, L)

    # Load scales (factor uncertainty)
    scales_path = model_dir / "scales.npy"
    if scales_path.exists():
        results["scales"] = np.load(scales_path)  # (N, L)

    # Load global loadings
    loadings_path = model_dir / "loadings.npy"
    if loadings_path.exists():
        results["loadings"] = np.load(loadings_path)  # (D, L)

    # Load Moran's I
    moran_path = model_dir / "moran_i.csv"
    if moran_path.exists():
        moran_df = pd.read_csv(moran_path)
        results["moran_idx"] = moran_df["factor_idx"].to_numpy()
        results["moran_values"] = moran_df["moran_i"].to_numpy()

    # Load metrics
    metrics_path = model_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)

    # Load ELBO history
    elbo_path = model_dir / "elbo_history.csv"
    if elbo_path.exists():
        elbo_df = pd.read_csv(elbo_path)
        results["elbo_history"] = elbo_df["elbo"].to_numpy()

    # Load group-specific loadings
    group_loadings = {}
    for p in model_dir.glob("loadings_group_*.npy"):
        group_id = int(p.stem.split("_")[-1])
        group_loadings[group_id] = np.load(p)  # (D, L)
    if group_loadings:
        results["group_loadings"] = group_loadings

    # Load gene enrichment (computed in analyze stage)
    enrichment_path = model_dir / "gene_enrichment.json"
    if enrichment_path.exists():
        with open(enrichment_path) as f:
            results["gene_enrichment"] = json.load(f)

    # Load PCA gene order (computed in analyze stage)
    pca_gene_order_path = model_dir / "pca_gene_order.npy"
    if pca_gene_order_path.exists():
        results["pca_gene_order"] = np.load(pca_gene_order_path)

    pca_gene_order_by_celltype_path = model_dir / "pca_gene_order_by_celltype.npy"
    if pca_gene_order_by_celltype_path.exists():
        results["pca_gene_order_by_celltype"] = np.load(pca_gene_order_by_celltype_path)

    # Load groupwise conditional posterior factors (MGGP models)
    groupwise_dir = model_dir / "groupwise_factors"
    if groupwise_dir.exists():
        gw = {}
        for p in sorted(groupwise_dir.glob("group_*.npy"),
                        key=lambda p: int(p.stem.split("_")[1])):
            g = int(p.stem.split("_")[1])
            gw[g] = np.load(p)  # (N, L)
        if gw:
            results["groupwise_factors"] = gw

    # Load Lu data (SVGP: Lu.pt, LCGP: Lu.npy)
    lu_pt_path = model_dir / "Lu.pt"
    lu_npy_path = model_dir / "Lu.npy"

    if lu_pt_path.exists():
        import torch
        results["Lu"] = torch.load(lu_pt_path, map_location="cpu", weights_only=False)
        results["is_lcgp"] = False
    elif lu_npy_path.exists():
        results["Lu"] = np.load(lu_npy_path)  # (L, M, K)
        results["is_lcgp"] = True

    # Load Z (inducing point locations)
    z_path = model_dir / "Z.npy"
    if z_path.exists():
        results["Z"] = np.load(z_path)  # (M, 2)

    # Load groupsZ (inducing point group assignments)
    groupsz_path = model_dir / "groupsZ.npy"
    if groupsz_path.exists():
        results["groupsZ"] = np.load(groupsz_path)  # (M,)

    return results


def plot_factors_spatial(
    factors: np.ndarray,
    coords: np.ndarray,
    moran_idx: Optional[np.ndarray] = None,
    moran_values: Optional[np.ndarray] = None,
    figsize_per_factor: float = 3.0,
    ncols: int = 5,
    s: float = 0.5,
    alpha: float = 0.8,
    cmap: str = "turbo",
) -> plt.Figure:
    """Plot spatial distribution of factors.

    Args:
        factors: (N, L) array of factor values
        coords: (N, 2) spatial coordinates
        moran_idx: Optional sort order by Moran's I (descending)
        moran_values: Optional Moran's I values for titles
        figsize_per_factor: Size per subplot
        ncols: Number of columns
        s: Scatter point size
        alpha: Transparency
        cmap: Colormap

    Returns:
        matplotlib Figure
    """
    L = factors.shape[1]

    # Reorder by Moran's I if provided
    if moran_idx is not None:
        factors = factors[:, moran_idx]
        if moran_values is not None:
            moran_values = moran_values  # Already sorted

    nrows = int(np.ceil(L / ncols))
    # Make figure wider to accommodate colorbar
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_factor * ncols + 1.0, figsize_per_factor * nrows),
        squeeze=False
    )

    # Global color scale
    vmin = 0.0 #np.percentile(factors, 2.5)
    vmax = np.percentile(factors, 99)

    for i in range(L):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=factors[:, i],
            vmin=vmin, vmax=vmax,
            alpha=alpha, cmap=cmap, s=s
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")

        # Title with Moran's I
        if moran_values is not None:
            title = f"Factor {moran_idx[i]+1}\nI={moran_values[i]:.3f}"
        else:
            title = f"Factor {i+1}"
        ax.set_title(title, fontsize=9)

    # Hide empty subplots
    for i in range(L, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Factor value")

    return fig


def plot_scales_spatial(
    scales: np.ndarray,
    coords: np.ndarray,
    moran_idx: Optional[np.ndarray] = None,
    moran_values: Optional[np.ndarray] = None,
    figsize_per_factor: float = 3.0,
    ncols: int = 5,
    s: float = 0.5,
    alpha: float = 0.8,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot spatial distribution of factor uncertainty (scales/sigma).

    The scale parameter represents the standard deviation of the variational
    distribution q(F) = Normal(mu, scale^2). Higher values indicate greater
    uncertainty in the latent factor at that spatial location.

    Args:
        scales: (N, L) array of scale values (std dev)
        coords: (N, 2) spatial coordinates
        moran_idx: Optional sort order by Moran's I (descending)
        moran_values: Optional Moran's I values for titles
        figsize_per_factor: Size per subplot
        ncols: Number of columns
        s: Scatter point size
        alpha: Transparency
        cmap: Colormap (viridis is good for uncertainty - low=dark, high=bright)

    Returns:
        matplotlib Figure
    """
    L = scales.shape[1]

    # Reorder by Moran's I if provided
    if moran_idx is not None:
        scales = scales[:, moran_idx]
        if moran_values is not None:
            moran_values = moran_values  # Already sorted

    nrows = int(np.ceil(L / ncols))
    # Make figure wider to accommodate colorbar
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_factor * ncols + 1.0, figsize_per_factor * nrows),
        squeeze=False
    )

    # Color scale: 0 to 98th percentile for better contrast
    vmin = 0.0
    vmax = np.percentile(scales, 98)

    for i in range(L):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=scales[:, i],
            vmin=vmin, vmax=vmax,
            alpha=alpha, cmap=cmap, s=s
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")

        # Title with Moran's I
        if moran_values is not None:
            title = f"Factor {moran_idx[i]+1}\nI={moran_values[i]:.3f}"
        else:
            title = f"Factor {i+1}"
        ax.set_title(title, fontsize=9)

    # Hide empty subplots
    for i in range(L, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    # Add colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="Scale (σ)")

    return fig


def plot_elbo_curve(elbo_history: np.ndarray, figsize: tuple = (8, 5)) -> plt.Figure:
    """Plot ELBO training curve on log-log scale.

    Args:
        elbo_history: Array of ELBO values per iteration
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Use 1-indexed iterations for log scale (avoid log(0))
    iterations = np.arange(1, len(elbo_history) + 1)
    ax.plot(iterations, elbo_history, linewidth=1.5, color="steelblue")

    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1e-10)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("ELBO", fontsize=12)
    ax.set_title("Training Convergence (log-log)", fontsize=14)
    ax.grid(True, alpha=0.3, which="both")

    # Set y-limits to focus on data range (all negative)
    ymin = np.min(elbo_history) * 1.1
    ymax = np.max(elbo_history) * 0.9
    ax.set_ylim(ymin, ymax)

    # Mark final value
    final_elbo = elbo_history[-1]
    ax.axhline(final_elbo, color="red", linestyle="--", alpha=0.5, label=f"Final: {final_elbo:.2e}")
    ax.legend(loc="upper right")

    fig.tight_layout()
    return fig


def plot_top_genes(
    loadings: np.ndarray,
    gene_names: List[str],
    moran_idx: Optional[np.ndarray] = None,
    n_top: int = 10,
    figsize_per_factor: float = 2.5,
    ncols: int = 5,
) -> plt.Figure:
    """Plot top genes per factor by loading magnitude.

    Args:
        loadings: (D, L) loadings matrix
        gene_names: List of gene names
        moran_idx: Optional sort order for factors
        n_top: Number of top genes to show
        figsize_per_factor: Size per subplot
        ncols: Number of columns

    Returns:
        matplotlib Figure
    """
    D, L = loadings.shape
    gene_names = np.array(gene_names)

    # Reorder factors by Moran's I if provided
    if moran_idx is not None:
        loadings = loadings[:, moran_idx]

    nrows = int(np.ceil(L / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_factor * ncols, figsize_per_factor * nrows * 1.5),
        squeeze=False
    )

    for i in range(L):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        # Get top genes for this factor
        factor_loadings = loadings[:, i]
        top_idx = np.argsort(factor_loadings)[-n_top:][::-1]

        top_genes = gene_names[top_idx]
        top_values = factor_loadings[top_idx]

        # Horizontal bar plot
        y_pos = np.arange(n_top)
        ax.barh(y_pos, top_values, color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Loading", fontsize=8)

        if moran_idx is not None:
            title = f"Factor {moran_idx[i]+1}"
        else:
            title = f"Factor {i+1}"
        ax.set_title(title, fontsize=10)

    # Hide empty subplots
    for i in range(L, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.tight_layout()
    return fig


def plot_top_enriched_depleted_genes(
    gene_enrichment: dict,
    group_name: str,
    moran_idx: Optional[np.ndarray] = None,
    n_top: int = 10,
    figsize_per_factor: float = 2.5,
    ncols: int = 5,
) -> plt.Figure:
    """Plot top enriched and depleted genes per factor for a specific group.

    Shows n_top enriched genes (positive LFC, green bars to right) and
    n_top depleted genes (negative LFC, red bars to left) for each factor.

    Args:
        gene_enrichment: Dict from gene_enrichment.json with per-factor/group LFC data
        group_name: Name of the group to plot
        moran_idx: Optional sort order for factors (by Moran's I)
        n_top: Number of top genes to show per direction (total 2*n_top genes)
        figsize_per_factor: Size per subplot
        ncols: Number of columns

    Returns:
        matplotlib Figure
    """
    n_factors = gene_enrichment["n_factors"]

    # Determine factor order
    if moran_idx is not None:
        factor_order = moran_idx
    else:
        factor_order = np.arange(n_factors)

    nrows = int(np.ceil(n_factors / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_factor * ncols, figsize_per_factor * nrows * 2.0),
        squeeze=False
    )

    for plot_idx, factor_idx in enumerate(factor_order):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row, col]

        factor_key = f"factor_{factor_idx}"
        factor_data = gene_enrichment["factors"][factor_key]

        # Get data for this specific group
        group_data = factor_data["groups"].get(group_name, None)
        if group_data is None:
            ax.set_visible(False)
            continue

        # Get top enriched and depleted genes for this group
        top_enriched = [(e["gene"], e["lfc"]) for e in group_data["top_enriched"][:n_top]]
        top_depleted = [(e["gene"], e["lfc"]) for e in group_data["top_depleted"][:n_top]]

        # Combine: enriched on top (descending), depleted on bottom (ascending magnitude)
        # Reverse depleted so it reads: ..., -1, -2, -3 from top to bottom
        combined = top_enriched + top_depleted[::-1]
        genes = [g for g, _ in combined]
        values = [v for _, v in combined]

        # Color by sign: green for enriched, red for depleted
        colors = ["forestgreen" if v > 0 else "tomato" for v in values]

        # Horizontal bar plot
        y_pos = np.arange(len(combined))
        ax.barh(y_pos, values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(genes, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("log2(group/global)", fontsize=7)

        title = f"Factor {factor_idx + 1}"
        ax.set_title(title, fontsize=10)

    # Hide empty subplots
    for i in range(n_factors, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    # Add group name as suptitle
    fig.suptitle(f"{group_name}", fontsize=14, y=1.01)
    fig.tight_layout()
    return fig


def plot_gene_enrichment_heatmap(
    global_loadings: np.ndarray,
    group_loadings: dict,
    group_names: List[str],
    moran_idx: Optional[np.ndarray] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot heatmap of mean relative LFC per factor per group.

    Uses normalized loadings to compute relative enrichment.

    Args:
        global_loadings: (D, L) global loadings
        group_loadings: dict mapping group_id -> (D, L) loadings
        group_names: List of group names
        moran_idx: Optional sort order for factors
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    D, L = global_loadings.shape
    n_groups = len(group_loadings)

    # Normalize global loadings
    global_norm = _normalize_loadings(global_loadings)

    # Compute mean absolute LFC for each factor x group
    lfc_matrix = np.zeros((n_groups, L))
    eps = 1e-10

    sorted_group_ids = sorted(group_loadings.keys())

    for i, group_id in enumerate(sorted_group_ids):
        # Normalize group loadings
        group_norm = _normalize_loadings(group_loadings[group_id])
        for j in range(L):
            lfc = np.log2((group_norm[:, j] + eps) / (global_norm[:, j] + eps))
            # Use mean of absolute LFC as measure of deviation from global
            lfc_matrix[i, j] = np.mean(np.abs(lfc))

    # Reorder factors by Moran's I if provided
    if moran_idx is not None:
        lfc_matrix = lfc_matrix[:, moran_idx]
        factor_labels = [f"F{moran_idx[j]+1}" for j in range(L)]
    else:
        factor_labels = [f"F{j+1}" for j in range(L)]

    # Truncate group names for display
    group_labels = [
        group_names[g][:20] + "..." if len(group_names[g]) > 20 else group_names[g]
        for g in sorted_group_ids
    ]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(lfc_matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(np.arange(L))
    ax.set_xticklabels(factor_labels, fontsize=9)
    ax.set_yticks(np.arange(n_groups))
    ax.set_yticklabels(group_labels, fontsize=9)

    ax.set_xlabel("Factor (sorted by Moran's I)", fontsize=11)
    ax.set_ylabel("Cell Type", fontsize=11)
    ax.set_title("Relative Enrichment: Mean |log2(group/global)| (normalized)", fontsize=12)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean |LFC|", fontsize=10)

    fig.tight_layout()
    return fig


def plot_celltype_gene_loadings(
    group_loadings: dict,
    group_names: List[str],
    gene_names: List[str],
    factor_idx: int,
    pca_gene_order: np.ndarray,
    global_loadings: Optional[np.ndarray] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot cell-type x genes heatmap of LFC (group vs global) for one factor.

    Normalizes both group and global loadings per gene across factors, then
    computes log2(group_norm / global_norm). This shows relative enrichment
    or depletion of each gene in each cell type for this factor, regardless
    of the factor's overall loading magnitude.

    Args:
        group_loadings: dict mapping group_id -> (D, L) loadings
        group_names: List of group names
        gene_names: List of gene names (length D)
        factor_idx: Which factor (column) to plot
        pca_gene_order: (D,) gene index order from per-factor PCA on LFC across groups
        global_loadings: (D, L) global loadings for normalization reference.
            If None, uses the mean across groups as global reference.
        figsize: Figure size (defaults to auto based on D and G)

    Returns:
        matplotlib Figure
    """
    eps = 1e-10
    sorted_group_ids = sorted(group_loadings.keys())
    G = len(sorted_group_ids)
    D = len(gene_names)

    # Build global reference: normalize per gene across factors
    if global_loadings is not None:
        ref = np.maximum(global_loadings, eps)
    else:
        stacked = np.stack([group_loadings[g] for g in sorted_group_ids], axis=0)  # (G, D, L)
        ref = np.maximum(stacked.mean(axis=0), eps)  # (D, L)
    ref_norm = ref / ref.sum(axis=1, keepdims=True)  # (D, L), rows sum to 1
    global_col = ref_norm[:, factor_idx]  # (D,)

    # Build (G, D) LFC matrix
    lfc_rows = []
    for g in sorted_group_ids:
        grp = np.maximum(group_loadings[g], eps)  # (D, L)
        grp_norm = grp / grp.sum(axis=1, keepdims=True)
        grp_col = grp_norm[:, factor_idx]  # (D,)
        lfc_rows.append(np.log2(grp_col / global_col))
    mat = np.stack(lfc_rows, axis=0)  # (G, D)

    # Reorder genes by PCA
    mat = mat[:, pca_gene_order]
    ordered_gene_names = [gene_names[i] for i in pca_gene_order]

    # Symmetric color scale
    abs_max = np.nanmax(np.abs(mat))
    abs_max = max(abs_max, 0.1)  # avoid degenerate range

    if figsize is None:
        fig_w = max(12, D * 0.12)
        fig_h = max(3, G * 0.5 + 1.5)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="bwr", vmin=-abs_max, vmax=abs_max)

    ax.set_yticks(np.arange(G))
    ax.set_yticklabels(
        [group_names[g] for g in sorted_group_ids], fontsize=9
    )
    ax.set_xticks(np.arange(D))
    ax.set_xticklabels(ordered_gene_names, fontsize=6, rotation=90)
    ax.set_xlabel("Genes (ordered by PC1 of stacked loadings)", fontsize=10)
    ax.set_ylabel("Cell Type", fontsize=10)
    ax.set_title(f"Factor {factor_idx + 1} — Cell-type loadings", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("log2(group / global)", fontsize=9)

    fig.tight_layout()
    return fig


def plot_factor_gene_loadings(
    group_loadings: dict,
    group_names: List[str],
    gene_names: List[str],
    group_idx: int,
    pca_gene_order: np.ndarray,
    global_loadings: Optional[np.ndarray] = None,
    figsize: Optional[tuple] = None,
) -> plt.Figure:
    """Plot factors x genes heatmap of LFC for one cell type, genes ordered by per-celltype PCA.

    For cell type g, computes log2(group_norm / global_norm) for each (gene, factor) pair,
    then orders genes by PC1 of the (D, L) LFC matrix for that cell type.

    Args:
        group_loadings: dict mapping group_id -> (D, L) loadings
        group_names: List of group names
        gene_names: List of gene names (length D)
        group_idx: Which cell type (index into sorted group keys) to plot
        pca_gene_order: (D,) gene index order from per-celltype PCA
        global_loadings: (D, L) global loadings for normalization reference.
            If None, uses the mean across groups as global reference.
        figsize: Figure size (defaults to auto based on D and L)

    Returns:
        matplotlib Figure
    """
    eps = 1e-10
    sorted_group_ids = sorted(group_loadings.keys())
    g = sorted_group_ids[group_idx]
    L = next(iter(group_loadings.values())).shape[1]
    D = len(gene_names)

    # Normalize global reference
    if global_loadings is not None:
        ref = np.maximum(global_loadings, eps)
    else:
        stacked = np.stack([group_loadings[gi] for gi in sorted_group_ids], axis=0)
        ref = np.maximum(stacked.mean(axis=0), eps)
    ref_norm = ref / ref.sum(axis=1, keepdims=True)  # (D, L)

    # Compute LFC for this group: (D, L)
    grp = np.maximum(group_loadings[g], eps)
    grp_norm = grp / grp.sum(axis=1, keepdims=True)
    lfc = np.log2(grp_norm / ref_norm)  # (D, L)

    # Reorder genes by per-celltype PCA, then transpose to (L, D) for imshow
    lfc = lfc[pca_gene_order, :]  # (D, L)
    mat = lfc.T  # (L, D)
    ordered_gene_names = [gene_names[i] for i in pca_gene_order]

    # Symmetric color scale
    abs_max = np.nanmax(np.abs(mat))
    abs_max = max(abs_max, 0.1)

    if figsize is None:
        fig_w = max(12, D * 0.12)
        fig_h = max(3, L * 0.5 + 1.5)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat, aspect="auto", cmap="bwr", vmin=-abs_max, vmax=abs_max)

    ax.set_yticks(np.arange(L))
    ax.set_yticklabels([f"Factor {l + 1}" for l in range(L)], fontsize=9)
    ax.set_xticks(np.arange(D))
    ax.set_xticklabels(ordered_gene_names, fontsize=6, rotation=90)
    ax.set_xlabel("Genes (ordered by PC1 of cell-type LFC)", fontsize=10)
    ax.set_ylabel("Factor", fontsize=10)
    group_name = group_names[g] if g < len(group_names) else f"Group {g}"
    ax.set_title(f"{group_name} — Factor loadings", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("log2(group / global)", fontsize=9)

    fig.tight_layout()
    return fig


def plot_factors_with_top_genes(
    factors: np.ndarray,
    Y: np.ndarray,
    loadings: np.ndarray,
    coords: np.ndarray,
    gene_names: List[str],
    moran_idx: Optional[np.ndarray] = None,
    n_genes: int = 5,
    size: float = 2.0,
    s: float = 0.2,
    alpha: float = 0.9,
    cmap: str = "turbo",
) -> plt.Figure:
    """Plot factors alongside spatial expression of top genes.

    Reproduces the plot_top_genes pattern from GPzoo notebooks:
    Each row = one factor, columns = [factor, gene1, gene2, ..., geneN]

    Args:
        factors: (N, L) factor values
        Y: (N, D) count matrix (spots x genes)
        loadings: (D, L) loadings matrix
        coords: (N, 2) spatial coordinates
        gene_names: List of gene names
        moran_idx: Optional sort order for factors by Moran's I
        n_genes: Number of top genes to show per factor
        size: Size per subplot
        s: Scatter point size
        alpha: Transparency
        cmap: Colormap

    Returns:
        matplotlib Figure
    """
    N, L = factors.shape
    gene_names = np.array(gene_names)

    # Reorder factors by Moran's I if provided
    if moran_idx is not None:
        factors = factors[:, moran_idx]
        loadings = loadings[:, moran_idx]

    # Global color scale for factors
    vmin_factors = np.percentile(factors, 1)
    vmax_factors = np.percentile(factors, 99)

    fig, axes = plt.subplots(
        L, n_genes + 1,
        figsize=(size * (n_genes + 1), size * L),
        squeeze=False
    )

    for i in range(L):
        # Get top genes for this factor (by loading magnitude)
        factor_loadings = loadings[:, i]
        top_gene_idx = np.argsort(factor_loadings)[-n_genes:][::-1]
        top_gene_names = gene_names[top_gene_idx]

        # Column 0: Factor
        ax = axes[i, 0]
        ax.scatter(
            coords[:, 0], coords[:, 1],
            c=factors[:, i],
            vmin=vmin_factors, vmax=vmax_factors,
            alpha=alpha, cmap=cmap, s=s
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")

        if moran_idx is not None:
            factor_label = f"Factor {moran_idx[i] + 1}"
        else:
            factor_label = f"Factor {i + 1}"
        ax.set_ylabel(factor_label, rotation=0, fontsize=10, labelpad=40)

        # Columns 1+: Top genes
        for j, gene_idx in enumerate(top_gene_idx):
            ax = axes[i, j + 1]

            # Get gene expression
            gene_expr = Y[:, gene_idx]
            if sparse.issparse(gene_expr):
                gene_expr = gene_expr.toarray().flatten()

            ax.scatter(
                coords[:, 0], coords[:, 1],
                c=gene_expr,
                alpha=alpha, cmap=cmap, s=s
            )
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("gray")

            # Gene name with outline for visibility
            title_text = ax.set_title(
                top_gene_names[j],
                x=0.03, y=0.88,
                fontsize="small", color="white",
                ha="left", va="top"
            )
            title_text.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground='black'),
                path_effects.Normal()
            ])

    fig.tight_layout()
    return fig


def plot_lu_scales_at_inducing(
    Lu: "torch.Tensor",
    Z: "torch.Tensor",
    moran_idx: Optional[np.ndarray] = None,
    moran_values: Optional[np.ndarray] = None,
    figsize_per_factor: float = 3.0,
    ncols: int = 5,
    s: float = 2.0,
    alpha: float = 0.8,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot inducing-point uncertainty from the Cholesky factor Lu.

    Computes diag(Lu @ Lu^T).sqrt() per factor, i.e. the marginal standard
    deviation of the variational distribution q(u) at each inducing point,
    and scatter-plots the result at the inducing locations Z.

    Args:
        Lu: (L, M, M) constrained lower-triangular Cholesky factor
        Z: (M, 2) inducing point locations
        moran_idx: Optional sort order by Moran's I (descending)
        moran_values: Optional Moran's I values for titles
        figsize_per_factor: Size per subplot
        ncols: Number of columns
        s: Scatter point size
        alpha: Transparency
        cmap: Colormap

    Returns:
        matplotlib Figure
    """
    # diag(Lu @ Lu^T) = sum of squares along last dim of each row
    # Lu is (L, M, M) for SVGP or (L, M, K) for LCGP, result is (L, M)
    if isinstance(Lu, np.ndarray):
        lu_scales = np.sqrt(np.sum(Lu ** 2, axis=-1))  # (L, M)
    else:
        import torch as _torch
        lu_scales = _torch.sum(Lu ** 2, dim=-1).sqrt().detach().cpu().numpy()  # (L, M)
    Z_np = Z.detach().cpu().numpy() if hasattr(Z, 'detach') else np.asarray(Z)

    L, M = lu_scales.shape

    # Reorder by Moran's I if provided
    if moran_idx is not None:
        lu_scales = lu_scales[moran_idx]

    nrows = int(np.ceil(L / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_factor * ncols + 1.0, figsize_per_factor * nrows),
        squeeze=False,
    )

    vmin = 0.0
    vmax = np.percentile(lu_scales, 98)

    for i in range(L):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        sc = ax.scatter(
            Z_np[:, 0], Z_np[:, 1],
            c=lu_scales[i],
            vmin=vmin, vmax=vmax,
            alpha=alpha, cmap=cmap, s=s,
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")

        if moran_idx is not None and moran_values is not None:
            title = f"Factor {moran_idx[i]+1}\nI={moran_values[i]:.3f}"
        elif moran_idx is not None:
            title = f"Factor {moran_idx[i]+1}"
        else:
            title = f"Factor {i+1}"
        ax.set_title(title, fontsize=9)

    # Hide empty subplots
    for i in range(L, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label="diag(LuLu⊤)^½")

    return fig


def plot_groups(
    coords: np.ndarray,
    groups: np.ndarray,
    group_names: List[str],
    Z: Optional[np.ndarray] = None,
    groupsZ: Optional[np.ndarray] = None,
    s_data: float = 0.3,
    s_inducing: float = 5.0,
    alpha: float = 0.8,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Plot cell-type spatial map, with inducing-point groups side-by-side for spatial models.

    Left subplot: all data points colored by cell-type at spatial coordinates.
    Right subplot (spatial only): inducing points colored by their group assignment.
    A shared legend on the right lists all group names.

    Args:
        coords: (N, 2) spatial coordinates of data points
        groups: (N,) integer group codes (0-indexed)
        group_names: List of group names (length = n_groups)
        Z: (M, 2) inducing point locations (optional, spatial models only)
        groupsZ: (M,) inducing point group assignments (optional, spatial models only)
        s_data: Scatter point size for data points
        s_inducing: Scatter point size for inducing points
        alpha: Transparency
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    has_inducing_groups = Z is not None and groupsZ is not None
    has_inducing_points = Z is not None
    n_groups = len(group_names)

    # Use a categorical colormap
    if n_groups <= 10:
        cmap = plt.cm.tab10
    elif n_groups <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]

    ncols = 2 if has_inducing_points else 1
    fig, axes = plt.subplots(
        1, ncols,
        figsize=figsize if has_inducing_points else (figsize[0] / 2 + 2, figsize[1]),
        squeeze=False,
    )

    # --- Left subplot: data cell-types ---
    ax_data = axes[0, 0]
    for g in range(n_groups):
        mask = groups == g
        ax_data.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[g]], s=s_data, alpha=alpha,
            label=group_names[g], rasterized=True,
        )
    ax_data.invert_yaxis()
    ax_data.set_xticks([])
    ax_data.set_yticks([])
    ax_data.set_facecolor("gray")
    ax_data.set_title("Cell Types (data)", fontsize=12)

    # --- Right subplot: inducing points ---
    if has_inducing_groups:
        # MGGP models: color inducing points by group assignment
        ax_ind = axes[0, 1]
        for g in range(n_groups):
            mask = groupsZ == g
            if mask.any():
                ax_ind.scatter(
                    Z[mask, 0], Z[mask, 1],
                    c=[colors[g]], s=s_inducing, alpha=alpha,
                    label=group_names[g], rasterized=True,
                )
        ax_ind.invert_yaxis()
        ax_ind.set_xticks([])
        ax_ind.set_yticks([])
        ax_ind.set_facecolor("gray")
        ax_ind.set_title("Cell Types (inducing points)", fontsize=12)
    elif has_inducing_points:
        # Non-group spatial models: show inducing point distribution
        ax_ind = axes[0, 1]
        ax_ind.scatter(
            Z[:, 0], Z[:, 1],
            c="steelblue", s=s_inducing, alpha=alpha, rasterized=True,
        )
        ax_ind.invert_yaxis()
        ax_ind.set_xticks([])
        ax_ind.set_yticks([])
        ax_ind.set_facecolor("gray")
        ax_ind.set_title(f"Inducing Points (M={Z.shape[0]})", fontsize=12)

    # --- Shared legend on the right, outside the subplots ---
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title="Cell Type",
        title_fontsize=10,
        markerscale=5,
        frameon=True,
    )
    fig.subplots_adjust(right=0.78)

    return fig


def plot_top_enriched_genes_per_group(
    gene_enrichment: dict,
    factor_idx: int,
    n_top: int = 10,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Plot top enriched genes per group for a specific factor.

    Uses pre-computed enrichment data from gene_enrichment.json.

    Args:
        gene_enrichment: Dict from gene_enrichment.json with per-factor/group LFC data
        factor_idx: Which factor to visualize (0-indexed)
        n_top: Number of top genes per group
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    factor_key = f"factor_{factor_idx}"
    factor_data = gene_enrichment["factors"][factor_key]
    group_names = list(factor_data["groups"].keys())
    n_groups = len(group_names)

    ncols = min(4, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for i, group_name in enumerate(group_names):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        group_data = factor_data["groups"][group_name]

        # Get top enriched genes from pre-computed data
        top_enriched = group_data["top_enriched"][:n_top]
        top_genes = [e["gene"] for e in top_enriched]
        top_lfc = [e["lfc"] for e in top_enriched]

        # Color by sign
        colors = ["forestgreen" if v > 0 else "tomato" for v in top_lfc]

        y_pos = np.arange(len(top_genes))
        ax.barh(y_pos, top_lfc, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("log2(group/global)", fontsize=8)

        ax.set_title(f"{group_name[:25]}", fontsize=9)

    # Hide empty subplots
    for i in range(n_groups, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(f"Top Enriched Genes - Factor {factor_idx + 1}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def _draw_group_loc_panel(
    ax, coords: np.ndarray, groups: np.ndarray, g: int,
    s: float = 0.5, alpha: float = 0.9, cmap: str = "gray",
) -> None:
    """Scatter all cells with group g highlighted (1.0), others dimmed (0.0).

    Single scatter call with a binary value array and grayscale colormap,
    matching the notebook pattern.
    """
    values = np.where(groups == g, 1.0, 0.0)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, vmin=0.0, vmax=1.0,
        cmap=cmap, s=s, alpha=alpha,
        edgecolors="none", rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("gray")


def _draw_factor_3d_shared(
    ax, coords: np.ndarray, values: np.ndarray,
    vmin: float, vmax: float, z_floor: float, z_ceil: float,
    cmap: str = "turbo", s: float = 0.3, alpha: float = 0.9,
    elev: float = 25.0, azim: float = -90.0,
) -> None:
    """3D surface scatter with externally supplied color scale and z-limits."""
    x1, x2 = coords[:, 0], coords[:, 1]

    transparent = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(transparent)
    ax.yaxis.set_pane_color(transparent)
    ax.zaxis.set_pane_color(transparent)
    ax.xaxis.line.set_color(transparent)
    ax.yaxis.line.set_color(transparent)
    ax.zaxis.line.set_color(transparent)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_ticklabels([])
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis._axinfo["grid"]["color"] = (0, 0, 0, 0)
    ax.grid(False)
    ax.set_axis_off()

    # 2D factor map at z_floor (same colormap as 3D surface)
    ax.scatter(x1, x2, np.full_like(values, z_floor),
               c=values, cmap=cmap, vmin=vmin, vmax=vmax,
               s=s, alpha=0.9, edgecolors="none", zorder=1)

    # Factor surface
    ax.scatter(x1, x2, values,
               c=values, cmap=cmap, s=s, alpha=alpha,
               edgecolors="none", zorder=2, vmin=vmin, vmax=vmax)

    ax.set_xlim(float(x1.min()), float(x1.max()))
    ax.set_ylim(float(x2.max()), float(x2.min()))  # inverted y
    ax.set_zlim(z_floor, z_ceil)
    ax.margins(0, 0, 0)
    ax.view_init(elev=elev, azim=azim)
    ax.dist = 4

    pos = ax.get_position()
    pad = 0.18
    ax.set_position([pos.x0 - pad * pos.width,
                     pos.y0 - pad * pos.height,
                     pos.width  * (1 + 2 * pad),
                     pos.height * (1 + 2 * pad)])


def plot_groupwise_factors(
    factors: np.ndarray,
    groupwise_factors: dict,
    coords: np.ndarray,
    groups: np.ndarray,
    group_names: List[str],
    s: float = 0.5,
    panel_size: float = 2.5,
    cmap: str = "turbo",
) -> plt.Figure:
    """Grid figure of groupwise conditional posterior factor maps.

    Layout: (G+1) rows × (L+1) cols
    - Row 0:    [off] [F0 uncond] [F1 uncond] ... [FL-1 uncond]
    - Rows 1..G: [group_loc] [F0 cond_g] [F1 cond_g] ... [FL-1 cond_g]

    Shared vmin/vmax from 99th percentile of unconditional factors.

    Args:
        factors:           (N, L) unconditional factor means
        groupwise_factors: dict {g: (N, L)} conditional factor means
        coords:            (N, 2) spatial coordinates
        groups:            (N,) integer group codes
        group_names:       list of G group name strings
        s:                 scatter point size
        panel_size:        inches per panel (both axes)
        cmap:              colormap for factor panels

    Returns:
        matplotlib Figure
    """
    N, L = factors.shape
    G = len(group_names)
    n_rows = G + 1
    n_cols = L + 1

    # Shared color scale from unconditional factors
    vmin = 0.0
    vmax = float(np.percentile(factors, 99))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )

    # (0, 0): empty / off
    axes[0, 0].set_visible(False)

    # Row 0, cols 1..L: unconditional factor maps + column headers
    for l in range(L):
        ax = axes[0, l + 1]
        ax.scatter(
            coords[:, 0], coords[:, 1], c=factors[:, l],
            vmin=vmin, vmax=vmax, cmap=cmap, s=s, alpha=0.8,
            edgecolors="none", rasterized=True,
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")
        ax.set_title(f"Factor {l + 1}", fontsize=8)

    # Rows 1..G
    for g in range(G):
        # Col 0: group location panel
        ax_loc = axes[g + 1, 0]
        _draw_group_loc_panel(ax_loc, coords, groups, g, s=s)
        display_name = " ".join(group_names[g].split("_")[:2])
        ax_loc.set_ylabel(display_name, rotation=0, fontsize=8, labelpad=50,
                          ha="right", va="center")

        # Cols 1..L: conditional factor maps
        factors_g = groupwise_factors.get(g)
        for l in range(L):
            ax = axes[g + 1, l + 1]
            if factors_g is not None:
                ax.scatter(
                    coords[:, 0], coords[:, 1], c=factors_g[:, l],
                    vmin=vmin, vmax=vmax, cmap=cmap, s=s, alpha=0.8,
                    edgecolors="none", rasterized=True,
                )
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("gray")

    fig.tight_layout(pad=0.3)
    return fig


def plot_celltype_summary(
    factors: np.ndarray,
    groupwise_factors: dict,
    Y: np.ndarray,
    coords: np.ndarray,
    groups: np.ndarray,
    group_id: int,
    group_name: str,
    gene_names: List[str],
    gene_enrichment: dict,
    n_top: int = 3,
    s: float = 0.5,
    panel_size: float = 2.5,
    cmap_factors: str = "turbo",
    cmap_genes: str = "turbo",
) -> plt.Figure:
    """Per-cell-type summary: global factors, conditional factors, top/bottom enriched genes.

    Layout: (2 + 2*n_top) rows × (L+1) cols  [default: 8 rows with n_top=3]
    - Row 0:          [empty]        [Factor 1] ... [Factor L]   — global factors
    - Row 1:          [cell density] [cond F1]  ... [cond FL]    — conditional factors
    - Rows 2..n_top+1:   [label]     [top-k gene for F1] ...     — top enriched genes per factor
    - Rows n_top+2..end: [label]     [bot-k gene for F1] ...     — top depleted genes per factor

    Gene panels show log1p(Y[:, gene_idx]) spatial expression.
    Genes come from gene_enrichment.json top_enriched / top_depleted for this cell type.

    Args:
        factors:           (N, L) global factor means
        groupwise_factors: {g: (N, L)} conditional factor means (MGGP models)
        Y:                 (N, D) gene expression (raw counts)
        coords:            (N, 2) spatial coordinates
        groups:            (N,) integer group codes
        group_id:          cell type integer key into groupwise_factors
        group_name:        display name for this cell type
        gene_names:        list of D gene names
        gene_enrichment:   dict from gene_enrichment.json with per-factor/group LFC data
        n_top:             number of top/bottom genes per factor (default 3 → 8 rows total)
        s:                 scatter point size
        panel_size:        inches per panel
        cmap_factors:      colormap for factor panels
        cmap_genes:        colormap for gene expression panels
    """
    N, L = factors.shape
    n_rows = 2 + 2 * n_top
    n_cols = L + 1

    vmin_f = 0.0
    vmax_f = float(np.percentile(factors, 99))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )

    # Turn off all panels upfront; enable selectively below
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_visible(False)

    def _scatter(ax, values, vmin, vmax, cmap, title=None):
        ax.set_visible(True)
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], c=values,
            vmin=vmin, vmax=vmax, cmap=cmap,
            s=s, alpha=0.8, edgecolors="none", rasterized=True,
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")
        if title:
            txt = ax.text(
                0.03, 0.95, title,
                transform=ax.transAxes,
                fontsize=11, color="white",
                ha="left", va="top",
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ])
        return sc

    def _label_panel(ax, label):
        ax.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(label, rotation=0, fontsize=8, labelpad=50,
                      ha="right", va="center")

    # Row 0: global factors
    axes[0, 0].set_visible(False)
    for l in range(L):
        _scatter(axes[0, l + 1], factors[:, l], vmin_f, vmax_f, cmap_factors,
                 title=f"Factor {l + 1}")

    # Row 1: cell density + conditional factors
    _draw_group_loc_panel(axes[1, 0], coords, groups, group_id, s=s)
    axes[1, 0].set_visible(True)
    display_name = " ".join(group_name.split("_")[:2])
    axes[1, 0].set_ylabel(display_name, rotation=0, fontsize=8, labelpad=50,
                          ha="right", va="center")
    factors_g = groupwise_factors.get(group_id)
    for l in range(L):
        ax = axes[1, l + 1]
        if factors_g is not None:
            _scatter(ax, factors_g[:, l], vmin_f, vmax_f, cmap_factors)
        else:
            ax.set_visible(True)
            ax.set_facecolor("gray")
            ax.set_xticks([])
            ax.set_yticks([])

    # Rows 2..(n_top+1): top enriched; Rows (n_top+2)..(2*n_top+1): top depleted
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}

    for kind_idx, kind in enumerate(["top_enriched", "top_depleted"]):
        row_offset = 2 + kind_idx * n_top
        for rank in range(n_top):
            r = row_offset + rank
            label = f"Enriched {rank + 1}" if kind == "top_enriched" else f"Depleted {rank + 1}"
            _label_panel(axes[r, 0], label)
            for l in range(L):
                factor_key = f"factor_{l}"
                group_data = gene_enrichment["factors"].get(factor_key, {}).get("groups", {}).get(group_name)
                if group_data is None:
                    continue
                entries = group_data.get(kind, [])
                if rank >= len(entries):
                    continue
                gene = entries[rank]["gene"]
                gene_idx = gene_name_to_idx.get(gene)
                if gene_idx is None:
                    continue
                expr = np.log1p(Y[:, gene_idx].astype(float))
                vmax_g = float(np.percentile(expr, 99)) or 1.0
                _scatter(axes[r, l + 1], expr, 0.0, vmax_g, cmap_genes, title=gene)

    fig.suptitle(f"{group_name} — factors & top/bottom enriched genes",
                 fontsize=10, y=1.005)
    fig.tight_layout(rect=[0, 0, 0.93, 1], pad=0.3)

    # Single reference colorbar for gene expression (turbo, no tick values)
    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors
    sm = _cm.ScalarMappable(cmap=cmap_genes, norm=_mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.05, 0.012, 0.4])
    cb = fig.colorbar(sm, cax=cbar_ax, ticks=[])
    cb.set_label("gene expression\n(log1p)", fontsize=9, labelpad=8)
    cb.outline.set_linewidth(0.5)

    return fig


def plot_celltype_summary_loadings(
    factors: np.ndarray,
    groupwise_factors: dict,
    Y: np.ndarray,
    coords: np.ndarray,
    groups: np.ndarray,
    group_id: int,
    group_name: str,
    gene_names: List[str],
    group_loadings: dict,
    n_top: int = 5,
    s: float = 0.5,
    panel_size: float = 2.5,
    cmap_factors: str = "turbo",
    cmap_genes: str = "turbo",
) -> plt.Figure:
    """Same layout as plot_celltype_summary but genes ranked by normedW_c directly.

    For each factor l, top genes are those with the highest normalized loading
    for this cell type: normedW_c[:, l] (row-normalized W_c, no division by global).

    Layout: (2 + n_top) rows × (L+1) cols
    - Row 0:              [empty]        [Factor 1] ... [Factor L]
    - Row 1:              [cell density] [cond F1]  ... [cond FL]
    - Rows 2..n_top+1:   [label]        [top gene for F1] ...
    - Rows n_top+2..end: [label]        [bot gene for F1] ...

    Args:
        factors:        (N, L) global factor means
        groupwise_factors: {g: (N, L)} conditional factor means
        Y:              (N, D) gene expression (raw counts)
        coords:         (N, 2) spatial coordinates
        groups:         (N,) integer group codes
        group_id:       cell type integer key
        group_name:     display name
        gene_names:     list of D gene names
        group_loadings: dict {group_id: (D, L)} raw group-specific loadings
        n_top:          number of top/bottom genes per factor (default 3)
        s:              scatter point size
        panel_size:     inches per panel
        cmap_factors:   colormap for factor panels
        cmap_genes:     colormap for gene expression panels
    """
    eps = 1e-10
    N, L = factors.shape
    n_rows = 2 + n_top
    n_cols = L + 1

    # Row-normalize group loadings for this cell type
    grp_W = np.maximum(group_loadings[group_id], eps)   # (D, L)
    grp_norm = grp_W / grp_W.sum(axis=1, keepdims=True)  # (D, L)

    vmin_f = 0.0
    vmax_f = float(np.percentile(factors, 99))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
    )

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_visible(False)

    def _scatter(ax, values, vmin, vmax, cmap, title=None):
        ax.set_visible(True)
        ax.scatter(
            coords[:, 0], coords[:, 1], c=values,
            vmin=vmin, vmax=vmax, cmap=cmap,
            s=s, alpha=0.8, edgecolors="none", rasterized=True,
        )
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("gray")
        if title:
            txt = ax.text(
                0.03, 0.95, title,
                transform=ax.transAxes,
                fontsize=11, color="white",
                ha="left", va="top",
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=2, foreground="black"),
                path_effects.Normal(),
            ])

    def _label_panel(ax, label):
        ax.set_visible(True)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_ylabel(label, rotation=0, fontsize=8, labelpad=50,
                      ha="right", va="center")

    # Row 0: global factors
    axes[0, 0].set_visible(False)
    for l in range(L):
        _scatter(axes[0, l + 1], factors[:, l], vmin_f, vmax_f, cmap_factors,
                 title=f"Factor {l + 1}")

    # Row 1: cell density + conditional factors
    axes[1, 0].set_visible(True)
    _draw_group_loc_panel(axes[1, 0], coords, groups, group_id, s=s)
    display_name = " ".join(group_name.split("_")[:2])
    axes[1, 0].set_ylabel(display_name, rotation=0, fontsize=8, labelpad=50,
                          ha="right", va="center")
    factors_g = groupwise_factors.get(group_id)
    for l in range(L):
        ax = axes[1, l + 1]
        if factors_g is not None:
            _scatter(ax, factors_g[:, l], vmin_f, vmax_f, cmap_factors)
        else:
            ax.set_visible(True)
            ax.set_facecolor("gray")
            ax.set_xticks([])
            ax.set_yticks([])

    # Rows 2..end: top genes by normedW_c per factor
    for rank in range(n_top):
        r = 2 + rank
        _label_panel(axes[r, 0], f"Top {rank + 1}")
        for l in range(L):
            order = np.argsort(grp_norm[:, l])[::-1]
            gene_idx = order[rank]
            expr = np.log1p(Y[:, gene_idx].astype(float))
            vmax_g = float(np.percentile(expr, 99)) or 1.0
            _scatter(axes[r, l + 1], expr, 0.0, vmax_g, cmap_genes,
                     title=gene_names[gene_idx])

    fig.suptitle(f"{group_name} — factors & top W_c genes",
                 fontsize=10, y=1.005)
    fig.tight_layout(rect=[0, 0, 0.93, 1], pad=0.3)

    import matplotlib.cm as _cm
    import matplotlib.colors as _mcolors
    sm = _cm.ScalarMappable(cmap=cmap_genes, norm=_mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.94, 0.05, 0.012, 0.4])
    cb = fig.colorbar(sm, cax=cbar_ax, ticks=[])
    cb.set_label("gene expression\n(log1p)", fontsize=9, labelpad=8)
    cb.outline.set_linewidth(0.5)

    return fig


def plot_groupwise_factors_3d(
    factors: np.ndarray,
    groupwise_factors: dict,
    coords: np.ndarray,
    groups: np.ndarray,
    group_names: List[str],
    top_groups: np.ndarray,
    s: float = 0.5,
    cmap: str = "turbo",
    w_col: float = 3.5,
    h_header: float = 2.5,
    h_factor: float = 4.5,
    n_factors: int = 2,
) -> plt.Figure:
    """Compact figure: n_factors factors × top-3 groups with 3D surface panels.

    Layout:
        Row 0 (header): [empty] [group A loc] [group B loc] [group C loc]
        Row 1 (F0):     [F0 uncond 3D] [cond gA F0] [cond gB F0] [cond gC F0]
        ...

    Shared vmin/vmax and z-limits across all 3D panels derived from
    unconditional factor percentiles.

    Args:
        factors:           (N, L) unconditional factor means (already Moran-sorted)
        groupwise_factors: dict {g: (N, L)} conditional factor means
        coords:            (N, 2) spatial coordinates
        groups:            (N,) integer group codes
        group_names:       list of group name strings
        top_groups:        indices of top-3 groups by cell count
        s:                 base scatter point size
        cmap:              colormap for factor panels
        n_factors:         number of factors to show (default 2; pass factors.shape[1] for all)

    Returns:
        matplotlib Figure
    """
    from matplotlib.gridspec import GridSpec

    N = coords.shape[0]
    base = _auto_point_size(N)
    s_2d = base * 0.5
    s_3d = base * 0.6

    n_factors = min(n_factors, factors.shape[1])
    n_groups_show = len(top_groups)  # 3
    n_rows = 1 + n_factors   # header + 2 factor rows
    n_cols = 1 + n_groups_show  # uncond + 3 groups

    # Shared color scale from unconditional top-2 factors
    uncond_vals = factors[:, :n_factors].ravel()
    vmin = float(np.percentile(uncond_vals, 1))
    vmax = float(np.percentile(uncond_vals, 99))
    z_range = vmax - vmin
    z_floor = vmin - 0.4 * z_range

    fig = plt.figure(figsize=(n_cols * w_col, h_header + n_factors * h_factor + 0.5))
    gs = GridSpec(
        n_rows, n_cols, figure=fig,
        height_ratios=[h_header] + [h_factor] * n_factors,
        left=0.02, right=0.99, top=0.94, bottom=0.01,
        wspace=0.05, hspace=0.08,
    )

    # Row 0, col 0: empty
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.set_visible(False)

    # Row 0, cols 1..3: group location panels (2D)
    for i, g in enumerate(top_groups):
        ax = fig.add_subplot(gs[0, i + 1])
        _draw_group_loc_panel(ax, coords, groups, int(g), s=s_2d)
        ax.set_title(group_names[int(g)], fontsize=9)

    # Rows 1..n_factors: factor rows
    for fi in range(n_factors):
        # Col 0: unconditional 3D
        ax_uncond = fig.add_subplot(gs[fi + 1, 0], projection="3d")
        _draw_factor_3d_shared(
            ax_uncond, coords, factors[:, fi],
            vmin=vmin, vmax=vmax, z_floor=z_floor, z_ceil=vmax,
            cmap=cmap, s=s_3d,
        )
        ax_uncond.set_title(f"Factor {fi + 1}\n(unconditional)", fontsize=8)

        # Cols 1..3: conditional 3D
        for i, g in enumerate(top_groups):
            ax = fig.add_subplot(gs[fi + 1, i + 1], projection="3d")
            factors_g = groupwise_factors.get(int(g))
            if factors_g is not None:
                _draw_factor_3d_shared(
                    ax, coords, factors_g[:, fi],
                    vmin=vmin, vmax=vmax, z_floor=z_floor, z_ceil=vmax,
                    cmap=cmap, s=s_3d,
                )

    fig.suptitle(
        "Groupwise Conditional Posterior  (top-2 factors, top-3 groups)",
        fontsize=12, y=0.98,
    )
    return fig


def _auto_point_size(N: int) -> float:
    """Scale point size as 100 / sqrt(N) so visual density stays consistent."""
    return 100.0 / np.sqrt(N)


# Shared vmax for training animation GIFs: max 99th-percentile across variants (slideseq, 20k iters)
_PNMF_VIDEO_VMAX = 9.29    # max(pnmf_scaled, pnmf_unscaled) p99
_SVGP_VIDEO_VMAX = 15.91   # max(svgp_scaled, svgp_unscaled) p99


def _render_training_gif(
    frames_path: Path, moran_path: Path, iters_path: Path,
    coords: np.ndarray, model_dir: Path, spatial: bool,
) -> None:
    """Render training animation GIF from saved video frames.

    Frames must already be Moran-ordered (done by analyze stage).
    Uses a fixed palette (253 turbo + gray/white/black) for fast quantization.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as mcm
    from PIL import Image

    frames_arr = np.load(frames_path)          # (n_frames, N, L) — already ordered
    moran_values = np.load(moran_path)         # (L,) sorted descending
    frame_iters = np.load(iters_path).tolist()
    n_frames, N, L = frames_arr.shape

    vmax = _SVGP_VIDEO_VMAX if spatial else _PNMF_VIDEO_VMAX

    ncols, figsize_per = 5, 3.0
    nrows = int(np.ceil(L / ncols))
    s = 100 / np.sqrt(N)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per * ncols, figsize_per * nrows),
                             squeeze=False)
    axes_flat = axes.flatten()
    scatters = []
    for i in range(L):
        ax = axes_flat[i]
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=frames_arr[0, :, i], s=s, cmap="turbo",
                        vmin=0.0, vmax=vmax, alpha=0.8, rasterized=True)
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("gray")
        ax.set_title(f"Factor {i+1}\nI={moran_values[i]:.3f}", fontsize=9)
        scatters.append(sc)
    for i in range(L, nrows * ncols):
        axes_flat[i].set_visible(False)
    fig.suptitle(f"iter={frame_iters[0]}", y=0.98)
    fig.tight_layout()

    # Fixed palette: 253 turbo + gray (axes bg) + white (fig bg) + black (text)
    turbo_colors = (mcm.turbo(np.linspace(0, 1, 253))[:, :3] * 255).astype(np.uint8)
    extras = np.array([[128, 128, 128], [255, 255, 255], [0, 0, 0]], dtype=np.uint8)
    palette_img = Image.new("P", (1, 1))
    palette_img.putpalette(np.vstack([turbo_colors, extras]).flatten().tolist())

    gif_frames = []
    for fi in range(n_frames):
        for i, sc in enumerate(scatters):
            sc.set_array(frames_arr[fi, :, i])
        fig.suptitle(f"iter={frame_iters[fi]}", y=0.98)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        gif_frames.append(Image.fromarray(buf[:, :, :3]).quantize(palette=palette_img, dither=0))
        if fi % 20 == 0:
            print(f"  {fi + 1}/{n_frames} frames rendered...")

    plt.close("all")

    gif_path = model_dir / "training_animation.gif"
    gif_frames[0].save(str(gif_path), save_all=True, append_images=gif_frames[1:],
                       loop=0, duration=100, optimize=False)
    print(f"  Saved: {gif_path} ({gif_path.stat().st_size // 1024 // 1024}MB)")


def run(config_path: str, no_heatmap: bool = False):
    """Generate publication figures.

    Reads gene_enrichment.json from analyze stage (not computed here).

    Output files (outputs/{dataset}/{model}/figures/):
        factors_spatial.png       - Spatial factor visualization (sorted by Moran's I)
        scales_spatial.png        - Factor uncertainty (sigma) spatial visualization
        points.png               - Cell-type spatial map (+ inducing points for spatial models)
        elbo_curve.png            - Training convergence
        top_genes.png             - Top genes per factor (bar chart)
        factors_with_genes.png    - Factors + top gene expression (spatial, like GPzoo)
        gene_enrichment.png       - LFC heatmap (group vs global loadings)
        enrichment_factor_{i}.png - Top enriched genes per group for each factor
        celltype_gene_loadings_factor_{i}.png - Cell-type x genes heatmap per factor (genes ordered by PCA)
    """
    config = Config.from_yaml(config_path)
    output_dir = Path(config.output_dir)

    # Determine model directory
    spatial = config.model.get("spatial", False)
    model_dir = output_dir / config.model_name

    # Create figures directory
    figures_dir = model_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed data
    print(f"Loading preprocessed data from: {output_dir}/preprocessed/")
    data = load_preprocessed(output_dir)
    print(f"  Spots (N): {data.n_spots}, Genes (D): {data.n_genes}")

    # Load analysis results
    print(f"Loading analysis results from: {model_dir}/")
    results = _load_analysis_results(model_dir)

    factors = results.get("factors")
    scales = results.get("scales")
    loadings = results.get("loadings")
    moran_idx = results.get("moran_idx")
    moran_values = results.get("moran_values")
    elbo_history = results.get("elbo_history")
    group_loadings = results.get("group_loadings")
    gene_enrichment = results.get("gene_enrichment")

    coords = data.X.numpy()
    gene_names = data.gene_names
    group_names = data.group_names or [f"Group {i}" for i in range(data.n_groups)]

    N = data.n_spots
    s = _auto_point_size(N)
    print(f"  Auto point size: s={s:.3f} (N={N})")

    # 1. Spatial factor plot
    if factors is not None:
        print("Generating spatial factor plot...")
        fig = plot_factors_spatial(
            factors, coords,
            moran_idx=moran_idx,
            moran_values=moran_values,
            s=s,
        )
        fig.savefig(figures_dir / "factors_spatial.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/factors_spatial.png")

    # 1b. Scales (uncertainty) spatial plot
    if scales is not None:
        print("Generating factor uncertainty (scales) spatial plot...")
        fig = plot_scales_spatial(
            scales, coords,
            moran_idx=moran_idx,
            moran_values=moran_values,
            s=s,
        )
        fig.savefig(figures_dir / "scales_spatial.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/scales_spatial.png")

    # 1c. Lu inducing-point uncertainty (spatial models only)
    is_lcgp = results.get("is_lcgp", False)
    Lu_data = results.get("Lu")
    Z_data = results.get("Z")
    if spatial and Z_data is not None and Lu_data is not None:
        print("Generating Lu inducing-point uncertainty plot...")
        fig = plot_lu_scales_at_inducing(
            Lu_data, Z_data,
            moran_idx=moran_idx,
            moran_values=moran_values,
        )
        fig.savefig(figures_dir / "lu_scales_inducing.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/lu_scales_inducing.png")

    # 1d. Cell-type spatial plot (data groups + inducing points side-by-side)
    if data.groups is not None:
        groups_np = data.groups.numpy()
        # Load inducing-point data if available (skip for LCGP since M=N)
        groupsZ_data = results.get("groupsZ")
        Z_celltype = None
        groupsZ_celltype = None
        if spatial and Z_data is not None and not is_lcgp:
            # Only show inducing points for SVGP (LCGP uses all data points)
            Z_celltype = Z_data
            groupsZ_celltype = groupsZ_data
        fig = plot_groups(
            coords, groups_np, group_names,
            Z=Z_celltype, groupsZ=groupsZ_celltype,
            s_data=s,
        )
        plot_name = "points.png"
        print(f"Generating {plot_name}...")
        fig.savefig(figures_dir / plot_name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/{plot_name}")

    # 2. ELBO curve
    if elbo_history is not None:
        print("Generating ELBO curve...")
        fig = plot_elbo_curve(elbo_history)
        fig.savefig(figures_dir / "elbo_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/elbo_curve.png")

    # 3. Top genes per factor (bar chart)
    if loadings is not None and gene_names is not None:
        print("Generating top genes plot...")
        fig = plot_top_genes(loadings, gene_names, moran_idx=moran_idx)
        fig.savefig(figures_dir / "top_genes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/top_genes.png")

    # 3b. Top enriched/depleted genes per factor per group (from gene enrichment)
    if gene_enrichment is not None:
        enrichment_dir = figures_dir / "enrichment_by_group"
        enrichment_dir.mkdir(parents=True, exist_ok=True)
        print(f"Generating top enriched/depleted genes plots ({gene_enrichment['n_groups']} groups)...")
        for group_name in gene_enrichment["group_names"]:
            fig = plot_top_enriched_depleted_genes(
                gene_enrichment, group_name, moran_idx=moran_idx
            )
            # Sanitize group name for filename
            safe_name = group_name.replace("/", "_").replace(" ", "_")[:50]
            fig.savefig(
                enrichment_dir / f"top_genes_{safe_name}.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
        print(f"  Saved: {enrichment_dir}/top_genes_*.png ({gene_enrichment['n_groups']} files)")

    # 4. Factors with top genes spatial plot (like GPzoo notebooks)
    if factors is not None and loadings is not None and gene_names is not None:
        print("Generating factors + top genes spatial plot...")
        # Load Y for gene expression visualization
        Y = data.Y.numpy()  # (N, D) after transpose or (D, N) - check shape
        if Y.shape[0] != factors.shape[0]:
            Y = Y.T  # Transpose to (N, D) if needed

        fig = plot_factors_with_top_genes(
            factors, Y, loadings, coords, gene_names,
            moran_idx=moran_idx, n_genes=5, s=s,
        )
        fig.savefig(figures_dir / "factors_with_genes.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/factors_with_genes.png")

    # 5. Gene enrichment visualization (computed in analyze stage)
    if gene_enrichment is not None:
        print(f"Using pre-computed gene enrichment ({gene_enrichment['n_groups']} groups)...")

        # 5a. Heatmap of LFC deviation (still needs raw loadings for now)
        if group_loadings and loadings is not None:
            print("Generating gene enrichment heatmap...")
            fig = plot_gene_enrichment_heatmap(
                loadings, group_loadings, group_names, moran_idx=moran_idx
            )
            fig.savefig(figures_dir / "gene_enrichment.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {figures_dir}/gene_enrichment.png")

        # 5b. Per-factor enrichment plots (top 3 factors by Moran's I)
        L = gene_enrichment["n_factors"]
        top_factors = moran_idx[:min(3, L)] if moran_idx is not None else range(min(3, L))

        for factor_idx in top_factors:
            print(f"Generating enrichment plot for Factor {factor_idx + 1}...")
            fig = plot_top_enriched_genes_per_group(
                gene_enrichment,
                factor_idx=factor_idx
            )
            fig.savefig(
                figures_dir / f"enrichment_factor_{factor_idx + 1}.png",
                dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            print(f"  Saved: {figures_dir}/enrichment_factor_{factor_idx + 1}.png")

    # 5c. Cell-type x genes heatmaps (one per factor, genes ordered by PCA)
    pca_gene_order = results.get("pca_gene_order")
    if not no_heatmap and group_loadings and pca_gene_order is not None:
        L = next(iter(group_loadings.values())).shape[1]
        celltype_dir = figures_dir / "celltype_gene_loadings"
        celltype_dir.mkdir(exist_ok=True)
        print(f"Generating cell-type x gene loading heatmaps ({L} factors)...")
        for factor_idx in range(L):
            fig = plot_celltype_gene_loadings(
                group_loadings, group_names, gene_names, factor_idx, pca_gene_order[factor_idx],
                global_loadings=loadings,
            )
            fig.savefig(
                celltype_dir / f"factor_{factor_idx + 1}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        print(f"  Saved: {celltype_dir}/factor_*.png ({L} files)")

    # 5d. Factor x genes heatmaps (one per cell type, genes ordered by per-celltype PCA)
    pca_gene_order_by_celltype = results.get("pca_gene_order_by_celltype")
    if not no_heatmap and group_loadings and pca_gene_order_by_celltype is not None:
        sorted_group_ids = sorted(group_loadings.keys())
        G_ct = len(sorted_group_ids)
        factor_gene_dir = figures_dir / "factor_gene_loadings"
        factor_gene_dir.mkdir(exist_ok=True)
        print(f"Generating factor x gene loading heatmaps ({G_ct} cell types)...")
        for group_idx, g in enumerate(sorted_group_ids):
            group_name = group_names[g] if g < len(group_names) else f"group_{g}"
            safe_name = group_name.replace(" ", "_").replace("/", "-")
            fig = plot_factor_gene_loadings(
                group_loadings, group_names, gene_names, group_idx,
                pca_gene_order_by_celltype[group_idx],
                global_loadings=loadings,
            )
            fig.savefig(
                factor_gene_dir / f"celltype_{safe_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        print(f"  Saved: {factor_gene_dir}/celltype_*.png ({G_ct} files)")

    # 6. Groupwise conditional posterior figures (MGGP models only)
    groupwise_factors = results.get("groupwise_factors")
    if groupwise_factors is not None and data.groups is not None:
        groups_np = data.groups.numpy()
        G = data.n_groups

        print("Generating groupwise factors figure (2D grid)...")
        fig = plot_groupwise_factors(
            factors, groupwise_factors, coords, groups_np, group_names, s=s,
        )
        fig.savefig(figures_dir / "groupwise_factors.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/groupwise_factors.png")

        # 3D compact figure: top-2 factors × top-3 groups
        if factors.shape[1] >= 2:
            print("Generating groupwise factors 3D figure...")
            counts = np.bincount(groups_np, minlength=G)
            top_groups = np.argsort(counts)[::-1][:3]
            fig = plot_groupwise_factors_3d(
                factors, groupwise_factors, coords, groups_np, group_names,
                top_groups=top_groups, s=s,
            )
            fig.savefig(figures_dir / "groupwise_factors_3d.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {figures_dir}/groupwise_factors_3d.png")

            # Complete version: all factors × all groups
            all_groups = np.argsort(counts)[::-1]  # all groups, sorted by cell count
            print("Generating groupwise factors 3D figure (complete)...")
            fig = plot_groupwise_factors_3d(
                factors, groupwise_factors, coords, groups_np, group_names,
                top_groups=all_groups, s=s, n_factors=factors.shape[1],
            )
            fig.savefig(figures_dir / "groupwise_factors_3d_complete.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {figures_dir}/groupwise_factors_3d_complete.png")

        # 7. Per-cell-type summary: factors + conditional factors + top/bottom enriched genes
        if gene_enrichment is not None and gene_names is not None:
            # Load Y (raw counts) for gene expression panels
            Y_ct = data.Y.numpy()
            if Y_ct.shape[0] != factors.shape[0]:
                Y_ct = Y_ct.T  # ensure (N, D)

            sorted_group_ids = sorted(groupwise_factors.keys())
            celltype_summary_dir = figures_dir / "celltype_summary"
            celltype_summary_dir.mkdir(exist_ok=True)
            print(f"Generating per-cell-type summary figures ({len(sorted_group_ids)} cell types)...")
            for g in sorted_group_ids:
                gname = group_names[g] if g < len(group_names) else f"group_{g}"
                safe_name = gname.replace(" ", "_").replace("/", "-")
                fig = plot_celltype_summary(
                    factors, groupwise_factors, Y_ct, coords, groups_np,
                    group_id=g, group_name=gname,
                    gene_names=gene_names,
                    gene_enrichment=gene_enrichment,
                    s=s,
                )
                fig.savefig(
                    celltype_summary_dir / f"celltype_{safe_name}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
            print(f"  Saved: {celltype_summary_dir}/celltype_*.png ({len(sorted_group_ids)} files)")

        # 8. Per-cell-type summary: top/bottom genes by normedW_c (no enrichment ratio)
        if group_loadings is not None and gene_names is not None:
            Y_wc = data.Y.numpy()
            if Y_wc.shape[0] != factors.shape[0]:
                Y_wc = Y_wc.T
            sorted_group_ids = sorted(groupwise_factors.keys())
            celltype_summary_wc_dir = figures_dir / "celltype_summary_Wc"
            celltype_summary_wc_dir.mkdir(exist_ok=True)
            print(f"Generating per-cell-type Wc summary figures ({len(sorted_group_ids)} cell types)...")
            for g in sorted_group_ids:
                gname = group_names[g] if g < len(group_names) else f"group_{g}"
                safe_name = gname.replace(" ", "_").replace("/", "-")
                fig = plot_celltype_summary_loadings(
                    factors, groupwise_factors, Y_wc, coords, groups_np,
                    group_id=g, group_name=gname,
                    gene_names=gene_names,
                    group_loadings=group_loadings,
                    s=s,
                )
                fig.savefig(
                    celltype_summary_wc_dir / f"celltype_{safe_name}.png",
                    dpi=150, bbox_inches="tight",
                )
                plt.close(fig)
            print(f"  Saved: {celltype_summary_wc_dir}/celltype_*.png ({len(sorted_group_ids)} files)")

    # Training animation GIF (only if video frames were captured during training)
    video_frames_path = model_dir / "video_frames.npy"
    video_moran_path = model_dir / "video_moran_values.npy"
    video_iters_path = model_dir / "video_frame_iters.npy"
    if video_frames_path.exists() and video_moran_path.exists() and video_iters_path.exists():
        print("\nRendering training animation GIF...")
        _render_training_gif(
            video_frames_path, video_moran_path, video_iters_path,
            coords, model_dir, spatial=config.spatial,
        )

    print("\nFigures generation complete!")
    print(f"  All figures saved to: {figures_dir}/")
