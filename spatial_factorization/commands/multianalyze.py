"""Multi-model factor comparison (Stage: multianalyze).

Compares factors across two trained models by:
1. Computing pairwise normalized L2 distance between all factor pairs
2. Greedily finding the best N matched pairs
3. Generating a 2-row comparison figure:
   - Row 0: matched factors from model_a (2D spatial + 3D surface per pair)
   - Row 1: matched factors from model_b (2D spatial + 3D surface per pair)
4. Generating a distance heatmap with matched pairs highlighted

The 3D view reveals spatial continuity gained from GP priors.

Usage:
    spatial_factorization multianalyze -c configs/slideseq/general.yaml pnmf svgp
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from ..config import Config
from ..datasets.base import load_preprocessed


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_factors(model_dir: Path) -> np.ndarray:
    """Load factors.npy from a model output directory. Returns (N, L) array."""
    path = model_dir / "factors.npy"
    if not path.exists():
        raise FileNotFoundError(f"factors.npy not found in {model_dir}")
    return np.load(path)  # (N, L)


# ---------------------------------------------------------------------------
# Distance + matching
# ---------------------------------------------------------------------------

def compute_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise normalized L2 distance between factor columns.

    Each factor column is normalized to unit L2 norm before comparison so
    that scale differences between model types do not dominate.

    Args:
        A: (N, L_a) factor matrix
        B: (N, L_b) factor matrix

    Returns:
        D: (L_a, L_b) pairwise squared distance matrix
    """
    eps = 1e-10
    A_unit = A / (np.linalg.norm(A, axis=0, keepdims=True) + eps)  # (N, L_a)
    B_unit = B / (np.linalg.norm(B, axis=0, keepdims=True) + eps)  # (N, L_b)

    # Transpose to (L, N) for vectorized distance
    A_T = A_unit.T  # (L_a, N)
    B_T = B_unit.T  # (L_b, N)

    # ||A_i - B_j||^2 = ||A_i||^2 - 2 A_i·B_j + ||B_j||^2
    A_norms_sq = (A_T ** 2).sum(axis=1)       # (L_a,)
    B_norms_sq = (B_T ** 2).sum(axis=1)       # (L_b,)
    D = A_norms_sq[:, None] - 2.0 * (A_T @ B_T.T) + B_norms_sq[None, :]  # (L_a, L_b)
    return np.clip(D, 0.0, None)


def find_best_matches(D: np.ndarray, n: int = 2) -> List[Tuple[int, int]]:
    """Greedy factor matching: find n pairs with smallest pairwise distance.

    Each factor index is used at most once (no re-use across rows or columns).

    Args:
        D: (L_a, L_b) pairwise distance matrix
        n: Number of matched pairs to return

    Returns:
        List of (i, j) index pairs, sorted by distance (best first)
    """
    D_work = D.copy().astype(float)
    matches = []
    for _ in range(min(n, min(D.shape))):
        idx = np.unravel_index(np.argmin(D_work), D_work.shape)
        matches.append(idx)
        D_work[idx[0], :] = np.inf
        D_work[:, idx[1]] = np.inf
    return matches


# ---------------------------------------------------------------------------
# Subplot helpers
# ---------------------------------------------------------------------------

def _draw_factor_2d(ax, coords: np.ndarray, values: np.ndarray,
                    title: str, cmap: str = "plasma",
                    s: float = 0.5, alpha: float = 0.8) -> None:
    """Scatter plot of a factor on spatial coordinates (2D top-down view)."""
    vmin = np.percentile(values, 1)
    vmax = np.percentile(values, 99)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        c=values, vmin=vmin, vmax=vmax,
        cmap=cmap, s=s, alpha=alpha,
        edgecolors="none", rasterized=True,
    )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("gray")
    ax.set_title(title, fontsize=9)


def _draw_factor_3d(ax, coords: np.ndarray, values: np.ndarray,
                    title: str, cmap: str = "plasma",
                    s: float = 0.3, alpha: float = 0.9,
                    elev: float = 25.0, azim: float = -90.0) -> None:
    """3D surface scatter: x1, x2 as the plane; factor value as z-height.

    Styled to match create_figure_styled.py:
      - Blue-tinted pane, white grid lines, hidden spines/ticks
      - Shadow offset = 0.5 × value range (opaque gray disc at bottom)
      - Axis label arrows (x1, x2, y_i) in 2D axes-fraction coordinates
      - elev=25, azim=-45 for the characteristic "wavy landscape" view
    """
    x1, x2 = coords[:, 0], coords[:, 1]

    # Transparent panes — let the figure background show through
    transparent = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(transparent)
    ax.yaxis.set_pane_color(transparent)
    ax.zaxis.set_pane_color(transparent)

    # Hide axes spines, ticks, grid lines entirely
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

    # Shadow: opaque gray disc below the data
    z_range = values.max() - values.min()
    z_min = values.min() - 0.4 * z_range
    ax.scatter(x1, x2, np.full_like(values, z_min),
               c="#9e9e9e", s=s, alpha=0.9,
               edgecolors="none", zorder=1)

    # Factor surface colored by value
    ax.scatter(x1, x2, values,
               c=values, cmap=cmap,
               s=s, alpha=alpha,
               edgecolors="none", zorder=2,
               vmin=values.min(), vmax=values.max())

    # Tight data bounds — no auto-padding
    ax.set_xlim(float(x1.min()), float(x1.max()))
    ax.set_ylim(float(x2.max()), float(x2.min()))   # inverted y
    ax.set_zlim(z_min, float(values.max()))
    ax.margins(0, 0, 0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9, pad=2)
    # Zoom camera in to fill the panel (default dist=10 leaves large margins)
    ax.dist = 4
    # Expand the axes beyond its GridSpec cell to use up the whitespace
    pos = ax.get_position()
    pad = 0.12
    ax.set_position([pos.x0 - pad * pos.width,
                     pos.y0 - pad * pos.height,
                     pos.width  * (1 + 2 * pad),
                     pos.height * (1 + 2 * pad)])


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def plot_comparison(
    coords: np.ndarray,
    factors_a: np.ndarray,
    factors_b: np.ndarray,
    matches: List[Tuple[int, int]],
    name_a: str,
    name_b: str,
    n_pairs: int = 2,
    cmap: str = "plasma",
    s_3d: float = 0.3,
    w_2d: float = 4.0,   # inches per 2D panel
    w_3d: float = 6.0,   # inches per 3D panel (matches standalone ≈ 10")
    h_row: float = 4.5,  # inches per row (≈ w_2d so 2D panels are square)
) -> plt.Figure:
    """Two-row comparison figure for matched factor pairs.

    Uses GridSpec with wider 3D columns so each 3D panel is ~6 inches — large
    enough to show the wavy landscape effect from the GP prior.

    Layout (each pair = one 2D col + one 3D col):

        Row 0  [A-f1 2D] [A-f1 3D] [A-f2 2D] [A-f2 3D] ...
        Row 1  [B-f1 2D] [B-f1 3D] [B-f2 2D] [B-f2 3D] ...
    """
    from matplotlib.gridspec import GridSpec

    # Match figures.py visual density: s=0.5 at figsize_per_factor=3.0"
    # Marker area scales with panel area so density stays constant.
    _FIGURES_PY_S = 0.5
    _FIGURES_PY_W = 3.0
    s_2d = _FIGURES_PY_S * (w_2d / _FIGURES_PY_W) ** 2

    n_pairs = min(n_pairs, len(matches))
    # Column pattern per pair: [2D, 3D], repeated n_pairs times
    col_widths = [w_2d, w_3d] * n_pairs
    n_cols = len(col_widths)
    total_w = sum(col_widths) + 0.4   # small margin for row labels
    total_h = h_row * 2 + 0.5

    fig = plt.figure(figsize=(total_w, total_h))
    gs = GridSpec(2, n_cols, figure=fig, width_ratios=col_widths,
                  left=0.04, right=0.99, top=0.93, bottom=0.02,
                  wspace=0.05, hspace=0.18)

    for k, (i, j) in enumerate(matches[:n_pairs]):
        fa = factors_a[:, i]
        fb = factors_b[:, j]

        col_2d = k * 2
        col_3d = k * 2 + 1

        # ---- Row 0: model A ----
        ax_a2d = fig.add_subplot(gs[0, col_2d])
        _draw_factor_2d(ax_a2d, coords, fa,
                        title=f"{name_a}  F{i + 1}",
                        cmap=cmap, s=s_2d)

        ax_a3d = fig.add_subplot(gs[0, col_3d], projection="3d")
        _draw_factor_3d(ax_a3d, coords, fa,
                        title=f"{name_a}  F{i + 1}  (3D)",
                        cmap=cmap, s=s_3d, )

        # ---- Row 1: model B ----
        ax_b2d = fig.add_subplot(gs[1, col_2d])
        _draw_factor_2d(ax_b2d, coords, fb,
                        title=f"{name_b}  F{j + 1}",
                        cmap=cmap, s=s_2d)

        ax_b3d = fig.add_subplot(gs[1, col_3d], projection="3d")
        _draw_factor_3d(ax_b3d, coords, fb,
                        title=f"{name_b}  F{j + 1}  (3D)",
                        cmap=cmap, s=s_3d, )

    # Row labels on the left margin
    fig.text(0.005, 0.74, name_a.upper(), fontsize=12, fontweight="bold",
             va="center", ha="left", rotation=90)
    fig.text(0.005, 0.26, name_b.upper(), fontsize=12, fontweight="bold",
             va="center", ha="left", rotation=90)

    fig.suptitle(
        f"Factor comparison: {name_a} vs {name_b}  "
        f"(top {n_pairs} matched pairs)",
        fontsize=13, y=0.97,
    )
    return fig


def plot_distance_heatmap(
    D: np.ndarray,
    name_a: str,
    name_b: str,
    matches: Optional[List[Tuple[int, int]]] = None,
) -> plt.Figure:
    """Heatmap of pairwise factor distances with matched pairs boxed."""
    L_a, L_b = D.shape
    fig, ax = plt.subplots(figsize=(max(4.0, L_b * 0.8 + 1.5),
                                    max(3.5, L_a * 0.7 + 1.5)))

    im = ax.imshow(D, cmap="viridis_r", aspect="auto")

    ax.set_xlabel(f"{name_b} factors", fontsize=11)
    ax.set_ylabel(f"{name_a} factors", fontsize=11)
    ax.set_xticks(np.arange(L_b))
    ax.set_xticklabels([f"F{j + 1}" for j in range(L_b)])
    ax.set_yticks(np.arange(L_a))
    ax.set_yticklabels([f"F{i + 1}" for i in range(L_a)])
    ax.set_title(f"Pairwise factor distance ({name_a} vs {name_b})", fontsize=12)

    # Highlight matched pairs
    if matches:
        highlight_colors = ["red", "orange", "yellow", "cyan"]
        for k, (i, j) in enumerate(matches):
            color = highlight_colors[k % len(highlight_colors)]
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(j, i, f"#{k + 1}", color=color, ha="center", va="center",
                    fontsize=8, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Normalized L2 distance")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: str, model_names: List[str],
        n_pairs: int = 2, output_path: Optional[str] = None) -> None:
    """Compare and match factors between two trained models.

    Args:
        config_path: Path to any config YAML in the dataset (used to resolve
                     output_dir and load preprocessed coordinates).
        model_names: Exactly two model names, e.g. ['pnmf', 'svgp'].
        n_pairs:     Number of best-matched factor pairs to include in the
                     comparison figure (default: 2).
        output_path: Directory to save figures (default: output_dir/figures/).
    """
    if len(model_names) != 2:
        raise ValueError(
            f"multianalyze expects exactly 2 model names, got {len(model_names)}: "
            f"{model_names}"
        )

    config = Config.from_yaml(config_path)
    output_dir = Path(config.output_dir)

    name_a, name_b = model_names
    dir_a = output_dir / name_a
    dir_b = output_dir / name_b

    print(f"Loading factors from {dir_a}/factors.npy")
    factors_a = _load_factors(dir_a)  # (N, L_a)
    print(f"Loading factors from {dir_b}/factors.npy")
    factors_b = _load_factors(dir_b)  # (N, L_b)
    print(f"  {name_a}: {factors_a.shape}  {name_b}: {factors_b.shape}")

    # Spatial coordinates from preprocessed data
    data = load_preprocessed(output_dir)
    coords = data.X.numpy()  # (N, 2)

    # Pairwise distance matrix and greedy matching
    print("Computing pairwise factor distances...")
    D = compute_distance_matrix(factors_a, factors_b)  # (L_a, L_b)

    matches = find_best_matches(D, n=n_pairs)
    print(f"Best {len(matches)} matches:")
    for k, (i, j) in enumerate(matches):
        print(f"  #{k + 1}:  {name_a} F{i + 1}  ↔  {name_b} F{j + 1}"
              f"  (dist = {D[i, j]:.4f})")

    # Output directory
    figs_dir = Path(output_path) if output_path else output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Comparison figure
    print("Generating factor comparison figure...")
    fig = plot_comparison(
        coords, factors_a, factors_b, matches,
        name_a, name_b, n_pairs=n_pairs,
    )
    comp_path = figs_dir / f"comparison_{name_a}_vs_{name_b}.png"
    fig.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {comp_path}")

    # Distance heatmap
    print("Generating distance heatmap...")
    fig_hm = plot_distance_heatmap(D, name_a, name_b, matches=matches)
    hm_path = figs_dir / f"distance_heatmap_{name_a}_vs_{name_b}.png"
    fig_hm.savefig(hm_path, dpi=150, bbox_inches="tight")
    plt.close(fig_hm)
    print(f"  Saved: {hm_path}")

    print("\nMulti-analyze complete.")
