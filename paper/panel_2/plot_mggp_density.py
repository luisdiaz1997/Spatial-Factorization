"""MGGP kernel density plot — one panel per cell type in a single 5×3 figure.

Uses the same fixed query point (tissue centroid) for all panels, varying
query_group across panels. Shows how the MGGP kernel re-weights cells
differently depending on which cell type is being queried.

Usage:
    conda run -n factorization python paper/panel_2/plot_mggp_density.py \
        -c configs/slideseq/general.yaml
    conda run -n factorization python paper/panel_2/plot_mggp_density.py \
        -c configs/slideseq/general.yaml --log
    conda run -n factorization python paper/panel_2/plot_mggp_density.py \
        -c configs/slideseq/general.yaml --model mggp_svgp
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PANEL2_DIR)

from spatial_factorization.config import Config
from spatial_factorization.datasets.base import load_preprocessed
from spatial_factorization.commands.figures import _auto_point_size
from knn_strategies import kernel_weights_mggp, default_query_idx, build_mggp_kernel

OUT_DIR = os.path.join(PANEL2_DIR, "density", "mggp")

QUERY_COLOR = "#111111"
CMAP = "bwr"
N_LEVELS = 10


def draw_panel(ax, X, weights, query_coord, s, norm, cmap, title):
    ax.scatter(X[:, 0], X[:, 1], c=weights, cmap=cmap, norm=norm,
               s=s, alpha=0.85, rasterized=True)
    ax.scatter(query_coord[0], query_coord[1],
               c=QUERY_COLOR, s=30, alpha=1.0, marker="*", zorder=4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#222222")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=8, pad=4)


def main():
    parser = argparse.ArgumentParser(
        description="MGGP kernel density — 5×3 grid, one panel per cell type"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--kernel-type", choices=["matern32", "rbf"], default="matern32")
    parser.add_argument("--lengthscale", type=float, default=None)
    parser.add_argument("--group-diff-param", type=float, default=None)
    parser.add_argument("--query-idx", type=int, default=None,
                        help="Fixed query cell index (default: tissue centroid)")
    parser.add_argument("--levels", type=int, default=N_LEVELS,
                        help="Number of quantization levels for quasi-contour")
    parser.add_argument("--log", action="store_true", help="Log color scale")
    parser.add_argument("--log-vmin", type=float, default=1e-3)
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

    kernel = build_mggp_kernel(args.kernel_type, n_groups, lengthscale, group_diff_param)

    print(f"Dataset:   {config.dataset}  (N={N}, groups={n_groups})")
    print(f"Kernel:    mggp_{args.kernel_type}  lengthscale={lengthscale}  group_diff_param={group_diff_param}")

    if args.query_idx is not None:
        query_idx = args.query_idx
    else:
        # Cell closest to the tissue centroid
        dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
        query_idx = int(dists.argmin())
    query_coord = X[query_idx]
    print(f"Query idx: {query_idx}  coord: {query_coord}")

    if args.log:
        boundaries = np.geomspace(args.log_vmin, 1.0, args.levels + 1)
    else:
        boundaries = np.linspace(0.0, 1.0, args.levels + 1)
    cmap = plt.get_cmap(CMAP, args.levels)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=args.levels)
    s = _auto_point_size(N)

    n_cols, n_rows = 5, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes_flat = axes.flatten()

    for g, gname in enumerate(group_names):
        w = kernel_weights_mggp(X, C, query_coord, g, kernel)
        w_max = w.max()
        if w_max > 0:
            w = w / w_max
        print(f"  [{g:2d}] {gname}")
        draw_panel(axes_flat[g], X, w, query_coord, s, norm, cmap, gname)

    # Hide unused panels
    for ax in axes_flat[n_groups:]:
        ax.set_visible(False)

    # Shared colorbar
    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(mappable, ax=axes_flat[:n_groups], fraction=0.02, pad=0.02)
    cb.set_label("Kernel weight", fontsize=11)
    cb.set_ticks(boundaries)
    if args.log:
        cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())

    dataset_tag = os.path.basename(os.path.normpath(output_dir))
    scale_tag = "_log" if args.log else ""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(
        OUT_DIR, f"mggp_density_{args.kernel_type}_{dataset_tag}{scale_tag}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
