"""Panel 2: tissue cell-type map + LCGP KNN neighborhood visualization.

Left panel:  All cells colored by cell type.
Right panel: All cells in gray; selected query cell + its K nearest neighbors highlighted.

The KNN is computed from the spatial coordinates Z (= X for LCGP/MGGP_LCGP,
since all data points are inducing points), matching what the model uses internally.

Usage:
    conda run -n factorization python paper/panel_2/plot_knn.py -c configs/osmfish/general.yaml
    conda run -n factorization python paper/panel_2/plot_knn.py -c configs/osmfish/general.yaml --model mggp_lcgp
    conda run -n factorization python paper/panel_2/plot_knn.py -c configs/osmfish/general.yaml --query-idx 4136
    conda run -n factorization python paper/panel_2/plot_knn.py -c configs/osmfish/mggp_lcgp.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from spatial_factorization.config import Config
from spatial_factorization.datasets.base import load_preprocessed
from spatial_factorization.commands.figures import _auto_point_size, _build_colormap

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_K(model_dir: str, config: Config) -> int:
    """Get K from the saved training.json, falling back to config, then 50."""
    training_json = os.path.join(model_dir, "training.json")
    if os.path.exists(training_json):
        with open(training_json) as f:
            t = json.load(f)
        return t.get("model_config", {}).get("K", config.model.get("K", 50))
    return config.model.get("K", 50)


def find_knn(Z: np.ndarray, query_idx: int, K: int) -> np.ndarray:
    """Return indices of K nearest neighbors of Z[query_idx] (excluding itself)."""
    nbrs = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(Z)
    _, indices = nbrs.kneighbors(Z[query_idx : query_idx + 1])
    return indices[0][indices[0] != query_idx][:K]


def default_query_idx(X: np.ndarray) -> int:
    """Pick a cell near the median distance from the tissue centroid."""
    dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
    return int(dists.argsort()[len(dists) // 2])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel(
    X: np.ndarray,
    C: np.ndarray,
    group_names: list,
    query_idx: int,
    neighbor_idxs: np.ndarray,
    s: float = 4.0,
    alpha: float = 0.8,
    figsize: tuple = (14, 6),
    neighbor_color: str = "#F4A23A",
    query_color: str = "#E03030",
    query_size_mult: float = 6.0,
) -> plt.Figure:
    n_groups = len(group_names)
    colors = _build_colormap(n_groups)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: cell types ---
    ax_left = axes[0]
    for g in range(n_groups):
        mask = C == g
        ax_left.scatter(
            X[mask, 0], X[mask, 1],
            c=[colors[g]], s=s, alpha=alpha,
            label=group_names[g], rasterized=True,
        )
    ax_left.invert_yaxis()
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.set_facecolor("#222222")
    ax_left.set_title("Cell types", fontsize=12)

    handles = [mpatches.Patch(color=colors[g], label=group_names[g]) for g in range(n_groups)]
    ax_left.legend(
        handles=handles, title="Cell type",
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        fontsize=7, title_fontsize=8, framealpha=0.7,
    )

    # --- Right: KNN highlight ---
    neighbor_set = set(neighbor_idxs.tolist())
    non_neighbor_mask = np.array(
        [i != query_idx and i not in neighbor_set for i in range(len(X))]
    )
    neighbor_mask = np.array([i in neighbor_set for i in range(len(X))])

    ax_right = axes[1]
    ax_right.scatter(
        X[non_neighbor_mask, 0], X[non_neighbor_mask, 1],
        c="#888888", s=s, alpha=0.5, rasterized=True,
    )
    ax_right.scatter(
        X[neighbor_mask, 0], X[neighbor_mask, 1],
        c=neighbor_color, s=s * 1.5, alpha=0.95,
        label=f"KNN (K={len(neighbor_idxs)})", rasterized=True, zorder=3,
    )
    ax_right.scatter(
        X[query_idx, 0], X[query_idx, 1],
        c=query_color, s=s * query_size_mult, alpha=1.0,
        label="Query cell", marker="*", zorder=4,
    )
    ax_right.invert_yaxis()
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_facecolor("#222222")
    ax_right.set_title(f"KNN neighborhood (K={len(neighbor_idxs)})", fontsize=12)
    ax_right.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0,
        fontsize=8, framealpha=0.7,
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot tissue cell-type map + KNN neighborhood")
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--model", default=None,
        help="Model subdirectory name (e.g. mggp_lcgp). Defaults to config.model_name, "
             "or 'mggp_lcgp' for general configs.",
    )
    parser.add_argument(
        "--query-idx", type=int, default=None,
        help="Index of the query cell (default: cell near tissue centroid)",
    )
    parser.add_argument(
        "--K", type=int, default=None,
        help="Number of neighbors to highlight (default: K from training.json)",
    )
    parser.add_argument("--out", type=str, default=None, help="Output path for the figure")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.model:
        model_name = args.model
    elif Config.is_general_config(args.config):
        model_name = "mggp_lcgp"
    else:
        model_name = config.model_name

    output_dir = config.output_dir
    model_dir = os.path.join(output_dir, model_name)

    data = load_preprocessed(output_dir)
    X = data.X.numpy()         # (N, 2)
    C = data.groups.numpy()    # (N,)
    group_names = data.group_names

    Z = np.load(os.path.join(model_dir, "Z.npy"))

    K = args.K if args.K is not None else resolve_K(model_dir, config)
    query_idx = args.query_idx if args.query_idx is not None else default_query_idx(X)

    print(f"Dataset:    {config.dataset}  (N={len(X)}, groups={len(group_names)})")
    print(f"Model:      {model_name}  (K={K})")
    print(f"Query idx:  {query_idx}  → cell type: {group_names[C[query_idx]]}")

    neighbor_idxs = find_knn(Z, query_idx, K)

    fig = plot_panel(
        X, C, group_names,
        query_idx=query_idx,
        neighbor_idxs=neighbor_idxs,
        s=_auto_point_size(len(X)),
    )

    if args.out:
        out_path = args.out
    else:
        dataset_tag = os.path.basename(os.path.normpath(output_dir))
        out_path = os.path.join(OUT_DIR, f"knn_panel_{dataset_tag}_{model_name}.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved:      {out_path}")


if __name__ == "__main__":
    main()
