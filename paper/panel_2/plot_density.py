"""Kernel density plot over the tissue for a query point.

Cells are colored by the kernel value k(x_i, x_query) normalized to [0, 1].

Usage:
    conda run -n factorization python paper/panel_2/plot_density.py \
        -c configs/slideseq/general.yaml --kernel rbf --lengthscale 8.0
    conda run -n factorization python paper/panel_2/plot_density.py \
        -c configs/slideseq/general.yaml --kernel matern32 --lengthscale 8.0
    conda run -n factorization python paper/panel_2/plot_density.py \
        -c configs/slideseq/general.yaml --kernel rbf --lengthscale 8.0 --log
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
from spatial_factorization.commands.analyze import _load_model
from spatial_factorization.commands.figures import _auto_point_size
from knn_strategies import default_query_idx, kernel_weights

OUT_DIR = os.path.join(PANEL2_DIR, "density")

QUERY_COLOR = "#111111"
CMAP = "bwr"
N_LEVELS = 10

KERNEL_LABELS = {
    "rbf": "RBF",
    "matern32": "Matérn 3/2",
}


def plot_panel(ax, X, weights, query_coord, s, norm, cmap, title):
    ax.scatter(X[:, 0], X[:, 1], c=weights, cmap=cmap, norm=norm,
               s=s, alpha=0.85, rasterized=True)
    ax.scatter(query_coord[0], query_coord[1],
               c=QUERY_COLOR, s=40, alpha=1.0, marker="*", zorder=4)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#222222")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=12, pad=6)


def main():
    parser = argparse.ArgumentParser(
        description="Plot kernel density over tissue for a query point"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--lengthscale", type=float, default=None)
    parser.add_argument("--query-idx", type=int, default=None)
    parser.add_argument("--kernel", choices=["rbf", "matern32"], default="rbf")
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
    model_dir  = os.path.join(output_dir, model_name)

    data = load_preprocessed(output_dir)
    X = data.X.numpy()
    N = len(X)

    if args.lengthscale is not None:
        lengthscale = args.lengthscale
    else:
        try:
            model = _load_model(Path(model_dir))
            ls = model._prior.kernel.lengthscale
            lengthscale = float(ls.detach() if hasattr(ls, "detach") else ls)
        except Exception:
            lengthscale = float(config.model.get("lengthscale", 8.0))

    query_idx = args.query_idx if args.query_idx is not None else default_query_idx(X)
    query_coord = X[query_idx]

    print(f"Dataset:     {config.dataset}  (N={N})")
    print(f"lengthscale: {lengthscale:.2f}")
    print(f"Kernel:      {args.kernel}")
    print(f"Query idx:   {query_idx}  coord: {query_coord}")

    if args.log:
        boundaries = np.geomspace(args.log_vmin, 1.0, args.levels + 1)
    else:
        boundaries = np.linspace(0.0, 1.0, args.levels + 1)
    cmap = plt.get_cmap(CMAP, args.levels)
    norm = mcolors.BoundaryNorm(boundaries, ncolors=args.levels)
    s = _auto_point_size(N)
    w = kernel_weights(X, query_coord, lengthscale, args.kernel)

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_panel(ax, X, w, query_coord, s, norm, cmap, KERNEL_LABELS[args.kernel])

    mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Kernel weight", fontsize=10)
    cb.set_ticks(boundaries)
    if args.log:
        cb.ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())

    fig.tight_layout()

    dataset_tag = os.path.basename(os.path.normpath(output_dir))
    scale_tag = "_log" if args.log else ""
    out_path = args.out or os.path.join(
        OUT_DIR, f"density_{dataset_tag}_{args.kernel}{scale_tag}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
