"""Multi-model factor comparison (Stage: multianalyze).

Algorithm
---------
1. Find the single best-matched factor pair between model_names[0] (reference,
   typically pnmf) and model_names[1] (typically svgp) via normalised L2 distance.
2. Use the reference factor (from model 0) as a template and find its single
   best match in each remaining model (mggp_svgp, lcgp, mggp_lcgp, …).
3. Generate a 2-row figure:
     Row 0 (top):    2D spatial scatter for each model's matched factor
     Row 1 (bottom): 3D surface scatter for each model's matched factor
4. Generate a distance heatmap between model_names[0] and model_names[1].

Usage
-----
    spatial_factorization multianalyze \\
        -c configs/slideseq/general.yaml \\
        pnmf svgp mggp_svgp lcgp mggp_lcgp
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
    """Pairwise normalised L2 distance between factor columns.

    Each factor column is normalised to unit L2 norm so that scale differences
    between model types do not dominate.

    Args:
        A: (N, L_a) factor matrix
        B: (N, L_b) factor matrix

    Returns:
        D: (L_a, L_b) pairwise squared distance matrix
    """
    eps = 1e-10
    A_unit = A / (np.linalg.norm(A, axis=0, keepdims=True) + eps)
    B_unit = B / (np.linalg.norm(B, axis=0, keepdims=True) + eps)

    A_T = A_unit.T  # (L_a, N)
    B_T = B_unit.T  # (L_b, N)

    A_norms_sq = (A_T ** 2).sum(axis=1)
    B_norms_sq = (B_T ** 2).sum(axis=1)
    D = A_norms_sq[:, None] - 2.0 * (A_T @ B_T.T) + B_norms_sq[None, :]
    return np.clip(D, 0.0, None)


def find_best_matches(D: np.ndarray, n: int = 2) -> List[Tuple[int, int]]:
    """Greedy factor matching: find n pairs with smallest pairwise distance.

    Each factor index is used at most once.

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


def find_best_match_for_factor(
    ref_factor: np.ndarray, factors_b: np.ndarray
) -> Tuple[int, float]:
    """Find the single factor in factors_b closest to ref_factor.

    Args:
        ref_factor: (N,) reference factor vector
        factors_b:  (N, L_b) factor matrix to search

    Returns:
        (j, dist): index of best matching factor and its normalised L2 distance
    """
    D = compute_distance_matrix(ref_factor.reshape(-1, 1), factors_b)  # (1, L_b)
    j = int(np.argmin(D[0]))
    return j, float(D[0, j])


# ---------------------------------------------------------------------------
# Point-size scaling
# ---------------------------------------------------------------------------

def _auto_point_size(N: int) -> float:
    """Scale point size as 100/sqrt(N) — matches figures.py visual density."""
    return 100.0 / np.sqrt(N)


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
                    cmap: str = "plasma",
                    s: float = 0.3, alpha: float = 0.9,
                    elev: float = 25.0, azim: float = -90.0) -> None:
    """3D surface scatter: x1/x2 as the plane, factor value as z-height."""
    x1, x2 = coords[:, 0], coords[:, 1]

    # Transparent panes
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

    # Shadow disc
    z_range = values.max() - values.min()
    z_min = values.min() - 0.4 * z_range
    ax.scatter(x1, x2, np.full_like(values, z_min),
               c="#9e9e9e", s=s, alpha=0.9,
               edgecolors="none", zorder=1)

    # Factor surface
    ax.scatter(x1, x2, values,
               c=values, cmap=cmap,
               s=s, alpha=alpha,
               edgecolors="none", zorder=2,
               vmin=values.min(), vmax=values.max())

    ax.set_xlim(float(x1.min()), float(x1.max()))
    ax.set_ylim(float(x2.max()), float(x2.min()))   # inverted y
    ax.set_zlim(z_min, float(values.max()))
    ax.margins(0, 0, 0)
    ax.view_init(elev=elev, azim=azim)
    ax.dist = 4
    pos = ax.get_position()
    pad = 0.18
    ax.set_position([pos.x0 - pad * pos.width,
                     pos.y0 - pad * pos.height,
                     pos.width  * (1 + 2 * pad),
                     pos.height * (1 + 2 * pad)])


# ---------------------------------------------------------------------------
# Two-model comparison figure (2 models, n_pairs matched pairs)
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
    w_2d: float = 4.0,
    w_3d: float = 6.0,
    h_row: float = 4.5,
) -> plt.Figure:
    """Two-row comparison figure for matched factor pairs (2-model case).

    Layout (each pair = one 2D col + one 3D col):
        Row 0  [A-f1 2D] [A-f1 3D] [A-f2 2D] [A-f2 3D] ...
        Row 1  [B-f1 2D] [B-f1 3D] [B-f2 2D] [B-f2 3D] ...
    """
    from matplotlib.gridspec import GridSpec

    N = len(coords)
    base = _auto_point_size(N)
    s_2d = base * (w_2d / 3.0) ** 2
    s_3d = base * 0.6

    n_pairs = min(n_pairs, len(matches))
    col_widths = [w_2d, w_3d] * n_pairs
    n_cols = len(col_widths)
    total_w = sum(col_widths) + 0.4
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

        ax_a2d = fig.add_subplot(gs[0, col_2d])
        _draw_factor_2d(ax_a2d, coords, fa,
                        title=f"{name_a}  F{i + 1}",
                        cmap=cmap, s=s_2d)

        ax_a3d = fig.add_subplot(gs[0, col_3d], projection="3d")
        _draw_factor_3d(ax_a3d, coords, fa, cmap=cmap, s=s_3d)

        ax_b2d = fig.add_subplot(gs[1, col_2d])
        _draw_factor_2d(ax_b2d, coords, fb,
                        title=f"{name_b}  F{j + 1}",
                        cmap=cmap, s=s_2d)

        ax_b3d = fig.add_subplot(gs[1, col_3d], projection="3d")
        _draw_factor_3d(ax_b3d, coords, fb, cmap=cmap, s=s_3d)

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


# ---------------------------------------------------------------------------
# Main figure: all models, two rows
# ---------------------------------------------------------------------------

def plot_comparison_all(
    coords: np.ndarray,
    factors_list: List[np.ndarray],
    model_names: List[str],
    factor_indices: List[int],
    cmap: str = "plasma",
    w_col: float = 3.5,   # inches per column
    h_2d: float = 4.0,    # inches for 2D row
    h_3d: float = 4.5,    # inches for 3D row
) -> plt.Figure:
    """Two-row figure: 2D spatial (top) and 3D surface (bottom) per model.

    Columns = one per model, all showing their matched factor.
    Point sizes are scaled automatically via 100/sqrt(N).
    """
    from matplotlib.gridspec import GridSpec

    n_models = len(model_names)
    N = len(coords)
    base = _auto_point_size(N)
    s_2d = base * (w_col / 3.0) ** 2
    s_3d = base * 0.6   # 0.6 * (100/sqrt(41783)) ≈ 0.3 for slideseq

    fig = plt.figure(figsize=(w_col * n_models, h_2d + h_3d + 0.6))
    gs = GridSpec(2, n_models, figure=fig,
                  height_ratios=[h_2d, h_3d],
                  left=0.02, right=0.99, top=0.92, bottom=0.01,
                  wspace=0.05, hspace=0.06)

    for k, (name, factors, fi) in enumerate(
            zip(model_names, factors_list, factor_indices)):
        vals = factors[:, fi]

        # Row 0: 2D
        ax_2d = fig.add_subplot(gs[0, k])
        _draw_factor_2d(ax_2d, coords, vals,
                        title=f"{name}  F{fi + 1}",
                        cmap=cmap, s=s_2d)

        # Row 1: 3D
        ax_3d = fig.add_subplot(gs[1, k], projection="3d")
        _draw_factor_3d(ax_3d, coords, vals, cmap=cmap, s=s_3d)

    fig.suptitle("Factor comparison across models", fontsize=13, y=0.97)
    return fig


# ---------------------------------------------------------------------------
# Distance heatmap (reference vs second model)
# ---------------------------------------------------------------------------

def plot_distance_heatmap(
    D: np.ndarray,
    name_a: str,
    name_b: str,
    matches: Optional[List[Tuple[int, int]]] = None,
) -> plt.Figure:
    """Heatmap of pairwise factor distances with matched pairs highlighted."""
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

    if matches:
        highlight_colors = ["red", "orange", "yellow", "cyan"]
        for k, (i, j) in enumerate(matches):
            color = highlight_colors[k % len(highlight_colors)]
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(rect)
            ax.text(j, i, f"#{k + 1}", color=color, ha="center", va="center",
                    fontsize=8, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Normalised L2 distance")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(config_path: str, model_names: List[str],
        n_pairs: int = 2,
        match_against: Optional[str] = None,
        output_path: Optional[str] = None) -> None:
    """Compare factors across two or more trained models.

    With exactly 2 models
    ---------------------
    Finds the top n_pairs matched factor pairs (greedy) and plots them in a
    2-row × (n_pairs × 2-col) figure: each pair gets a [2D | 3D] column block,
    model A on row 0 and model B on row 1.

    With 3+ models
    --------------
    Finds the single best match between model_names[0] (reference) and
    model_names[1], then matches that reference factor against every remaining
    model. Plots a 2-row figure: 2D spatial on the top row, 3D surface on the
    bottom row, one column per model.

    Args:
        config_path:  Path to any config YAML for the dataset.
        model_names:  Two or more model names; model_names[0] is the reference.
        n_pairs:      Number of pairs to show (only used for the 2-model case).
        output_path:  Directory to save figures (default: output_dir/figures/).
    """
    if len(model_names) < 2:
        raise ValueError(
            f"multianalyze expects at least 2 model names, got {len(model_names)}"
        )

    config = Config.from_yaml(config_path)
    output_dir = Path(config.output_dir)

    # Load factors for all models
    factors_list = []
    for name in model_names:
        path = output_dir / name
        print(f"Loading factors from {path}/factors.npy")
        factors_list.append(_load_factors(path))
    for name, f in zip(model_names, factors_list):
        print(f"  {name}: {f.shape}")

    # Spatial coordinates
    data = load_preprocessed(output_dir)
    coords = data.X.numpy()  # (N, 2)

    name_a = model_names[0]

    # Output directory
    figs_dir = Path(output_path) if output_path else output_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    stem = "_vs_".join(model_names)

    if len(model_names) == 2:
        # ---- Original 2-model comparison ----
        name_b = model_names[1]
        print(f"\nComputing pairwise distances: {name_a} vs {name_b} ...")
        D_ab = compute_distance_matrix(factors_list[0], factors_list[1])

        matches = find_best_matches(D_ab, n=n_pairs)
        print(f"Best {len(matches)} matches:")
        for k, (i, j) in enumerate(matches):
            print(f"  #{k + 1}:  {name_a} F{i + 1}  ↔  {name_b} F{j + 1}"
                  f"  (dist = {D_ab[i, j]:.4f})")

        print("\nGenerating factor comparison figure...")
        fig = plot_comparison(
            coords, factors_list[0], factors_list[1], matches,
            name_a, name_b, n_pairs=n_pairs,
        )
        comp_path = figs_dir / f"comparison_{stem}.png"
        fig.savefig(comp_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {comp_path}")

        print("Generating distance heatmap...")
        fig_hm = plot_distance_heatmap(D_ab, name_a, name_b, matches=matches)
        hm_path = figs_dir / f"distance_heatmap_{name_a}_vs_{name_b}.png"
        fig_hm.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close(fig_hm)
        print(f"  Saved: {hm_path}")

    else:
        # ---- Multi-model comparison ----
        # Determine which model to use for the initial reference matching
        name_ref = match_against if match_against is not None else model_names[1]
        if name_ref not in model_names:
            raise ValueError(
                f"--match-against {name_ref!r} is not in the model list: {model_names}"
            )
        if name_ref == name_a:
            raise ValueError("--match-against cannot be the same as the reference model")
        ref_idx = model_names.index(name_ref)

        print(f"\nComputing pairwise distances: {name_a} vs {name_ref} ...")
        D_ref = compute_distance_matrix(factors_list[0], factors_list[ref_idx])
        idx = np.unravel_index(np.argmin(D_ref), D_ref.shape)
        i_ref, j_ref = int(idx[0]), int(idx[1])
        print(f"  Reference: {name_a} F{i_ref + 1}  ↔  {name_ref} F{j_ref + 1}"
              f"  (dist = {D_ref[i_ref, j_ref]:.4f})")

        # Match reference factor against every other model
        factor_indices = [None] * len(model_names)
        factor_indices[0] = i_ref
        factor_indices[ref_idx] = j_ref
        ref_factor = factors_list[0][:, i_ref]

        for k in range(1, len(model_names)):
            if k == ref_idx:
                continue
            name_k = model_names[k]
            j_k, dist_k = find_best_match_for_factor(ref_factor, factors_list[k])
            print(f"  {name_k}: best match F{j_k + 1}  (dist = {dist_k:.4f})")
            factor_indices[k] = j_k

        print("\nGenerating comparison figure...")
        fig = plot_comparison_all(
            coords, factors_list, model_names, factor_indices,
        )
        comp_path = figs_dir / f"comparison_{stem}.png"
        fig.savefig(comp_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {comp_path}")

        print("Generating distance heatmap...")
        fig_hm = plot_distance_heatmap(
            D_ref, name_a, name_ref,
            matches=[(i_ref, j_ref)],
        )
        hm_path = figs_dir / f"distance_heatmap_{name_a}_vs_{name_ref}.png"
        fig_hm.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close(fig_hm)
        print(f"  Saved: {hm_path}")

    print("\nMulti-analyze complete.")
