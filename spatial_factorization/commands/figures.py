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

    # Load Lu data (SVGP: Lu.pt, LCGP: Lu_diag.npy + Lu_V.npy)
    lu_path = model_dir / "Lu.pt"
    lu_diag_path = model_dir / "Lu_diag.npy"
    lu_v_path = model_dir / "Lu_V.npy"

    if lu_path.exists():
        import torch
        results["Lu"] = torch.load(lu_path, map_location="cpu", weights_only=False)
        results["is_lcgp"] = False
    elif lu_diag_path.exists() and lu_v_path.exists():
        results["Lu_diag"] = np.load(lu_diag_path)  # (L, M)
        results["Lu_V"] = np.load(lu_v_path)        # (L, M, R)
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

    # Fixed color scale for uncertainty
    vmin = 0.0
    vmax = 1.0

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
    import torch as _torch

    # diag(Lu @ Lu^T) = sum of squares along last dim of each row
    # Lu is (L, M, M), result is (L, M)
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
    vmax = 1.0

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


def run(config_path: str):
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

    # 1. Spatial factor plot
    if factors is not None:
        print("Generating spatial factor plot...")
        fig = plot_factors_spatial(
            factors, coords,
            moran_idx=moran_idx,
            moran_values=moran_values
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
            moran_values=moran_values
        )
        fig.savefig(figures_dir / "scales_spatial.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {figures_dir}/scales_spatial.png")

    # 1c. Lu inducing-point uncertainty (spatial models only)
    is_lcgp = results.get("is_lcgp", False)
    Lu_data = results.get("Lu")
    Z_data = results.get("Z")
    if spatial and Z_data is not None:
        if Lu_data is not None:
            # SVGP: Full Cholesky factor
            print("Generating Lu inducing-point uncertainty plot...")
            fig = plot_lu_scales_at_inducing(
                Lu_data, Z_data,
                moran_idx=moran_idx,
                moran_values=moran_values,
            )
            fig.savefig(figures_dir / "lu_scales_inducing.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {figures_dir}/lu_scales_inducing.png")
        elif is_lcgp:
            # LCGP: Skip since M=N (all data points are inducing)
            # The scales_spatial.png already shows uncertainty at all data points
            print("  Skipping Lu inducing plot for LCGP (M=N, see scales_spatial.png instead)")
            pass

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
            moran_idx=moran_idx, n_genes=5
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

    print("\nFigures generation complete!")
    print(f"  All figures saved to: {figures_dir}/")
