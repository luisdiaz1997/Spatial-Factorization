"""MGGP kernel-weighted neighbor sampling — 5×3 grid, one panel per cell type.

Uses the same fixed query point (tissue centroid) across all panels.
Each panel samples K neighbors using the MGGP kernel with query_group=g,
showing how the selected neighborhood shifts by cell type.

Usage:
    conda run -n factorization python paper/panel_2/plot_samples.py \
        -c configs/slideseq/general.yaml
    conda run -n factorization python paper/panel_2/plot_samples.py \
        -c configs/slideseq/general.yaml --model mggp_svgp --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PANEL2_DIR)

from spatial_factorization.config import Config
from spatial_factorization.datasets.base import load_preprocessed
from spatial_factorization.commands.figures import _auto_point_size
from gpzoo.kernels import batched_MGGP_RBF
from knn_strategies import select_kernel_mggp, build_mggp_kernel

OUT_DIR = os.path.join(PANEL2_DIR, "samples")

NEIGHBOR_COLOR = "#F4A23A"
QUERY_COLOR    = "#E03030"


def draw_panel(ax, X, C, query_idx, neighbor_idxs, group_idx, s, title):
    neighbor_set = set(neighbor_idxs.tolist())
    group_mask = C == group_idx
    is_query   = np.arange(len(X)) == query_idx
    is_nbr     = np.array([i in neighbor_set for i in range(len(X))])
    is_bg      = ~group_mask & ~is_nbr & ~is_query

    # Binary grayscale: group cells → 1.0 (white), others → 0.0 (black)
    values = np.where(group_mask, 1.0, 0.0)
    ax.scatter(X[:, 0], X[:, 1], c=values, vmin=0.0, vmax=1.0,
               cmap="gray", s=s, alpha=0.9, edgecolors="none",
               rasterized=True, zorder=1)
    # Sampled neighbors — orange
    ax.scatter(X[is_nbr, 0], X[is_nbr, 1],
               c=NEIGHBOR_COLOR, s=max(s * 1.5, 4), alpha=0.95,
               rasterized=True, zorder=3)
    # Query point — red star
    ax.scatter(X[query_idx, 0], X[query_idx, 1],
               c=QUERY_COLOR, s=max(s * 4, 20), alpha=1.0,
               marker="*", zorder=4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("gray")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8, pad=4)


def main():
    parser = argparse.ArgumentParser(
        description="MGGP kernel sampling: 5×3 grid, one panel per cell type"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--kernel-type", choices=["matern32", "rbf"], default="matern32",
                        help="MGGP kernel: matern32 or rbf (both freshly constructed)")
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--lengthscale", type=float, default=None)
    parser.add_argument("--group-diff-param", type=float, default=None)
    parser.add_argument("--query-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.model:
        model_name = args.model
    elif Config.is_general_config(args.config):
        model_name = "mggp_lcgp"
    else:
        model_name = config.model_name

    output_dir = config.output_dir
    data = load_preprocessed(output_dir)
    X = data.X.numpy()
    C = data.groups.numpy()
    group_names = data.group_names
    N = len(X)
    n_groups = len(group_names)

    lengthscale = args.lengthscale if args.lengthscale is not None \
        else float(config.model.get("lengthscale", 8.0))
    group_diff_param = args.group_diff_param if args.group_diff_param is not None \
        else float(config.model.get("group_diff_param", 1.0))
    K = args.K if args.K is not None else int(config.model.get("K", 50))

    kernel = build_mggp_kernel(args.kernel_type, n_groups, lengthscale, group_diff_param)

    # Fixed query: cell closest to tissue centroid
    if args.query_idx is not None:
        query_idx = args.query_idx
    else:
        dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
        query_idx = int(dists.argmin())

    print(f"Dataset:   {config.dataset}  (N={N}, groups={n_groups})")
    print(f"K={K}")
    print(f"Kernel:    mggp_{args.kernel_type}  lengthscale={lengthscale}  group_diff_param={group_diff_param}")
    print(f"Query idx: {query_idx}  coord: {X[query_idx]}")
    print(f"Seed:      {args.seed}")

    s = _auto_point_size(N)
    n_cols, n_rows = 5, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes_flat = axes.flatten()

    for g, gname in enumerate(group_names):
        rng = np.random.default_rng(args.seed)
        neighbor_idxs = select_kernel_mggp(X, C, query_idx, g, K, kernel, rng)
        print(f"  [{g:2d}] {gname}")
        draw_panel(axes_flat[g], X, C, query_idx, neighbor_idxs, g, s, gname)

    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    fig.tight_layout()

    dataset_tag = os.path.basename(os.path.normpath(output_dir))
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, f"samples_mggp_{args.kernel_type}_{dataset_tag}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
