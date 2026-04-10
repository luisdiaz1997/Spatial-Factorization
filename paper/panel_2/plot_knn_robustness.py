"""Robustness plot: KNN neighborhood radius vs. data density.

4 rows (strategies) × 3 columns (N=5000, 10000, 40000).
Each column is a random subsample of the full dataset.
The same spatial location (tissue centroid) is used as the query across all columns,
so we can visually compare whether the neighborhood radius changes with N.

Usage:
    conda run -n factorization python paper/panel_2/plot_knn_robustness.py \
        -c configs/slideseq/general.yaml
    conda run -n factorization python paper/panel_2/plot_knn_robustness.py \
        -c configs/slideseq/general.yaml --N-values 5000 10000 40000 --K 50
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PANEL2_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, PANEL2_DIR)

from spatial_factorization.config import Config
from spatial_factorization.datasets.base import load_preprocessed
from spatial_factorization.commands.analyze import _load_model
from spatial_factorization.commands.figures import _auto_point_size
from knn_strategies import (
    select_baseline,
    select_kernel,
    default_query_idx,
)

OUT_DIR = os.path.join(PANEL2_DIR, "robustness")

KERNEL_LABELS = {
    "rbf": "RBF",
    "matern32": "Matérn 3/2",
}
NEIGHBOR_COLOR = "#F4A23A"
QUERY_COLOR    = "#E03030"


# ---------------------------------------------------------------------------
# Per-subsample KNN computation
# ---------------------------------------------------------------------------

def compute_all_strategies(
    X_sub: np.ndarray,
    query_in_sub: int,
    K: int,
    lengthscale: float,
    rng: np.random.Generator,
    kernel: str = "gaussian",
) -> list:
    """Return [baseline, kernel] neighbor index arrays for query_in_sub."""
    baseline  = select_baseline(X_sub, query_in_sub, K)
    kernel_sel = select_kernel(X_sub, query_in_sub, K, lengthscale, rng=rng, kernel=kernel)
    return [baseline, kernel_sel]


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_robustness(
    X_full: np.ndarray,
    N_values: list,
    K: int,
    lengthscale: float,
    rng: np.random.Generator,
    query_coord: np.ndarray,
    kernel: str = "gaussian",
) -> plt.Figure:
    strategy_labels = ["Baseline\n(K-nearest)", KERNEL_LABELS[kernel]]
    n_rows = len(strategy_labels)
    n_cols = len(N_values)
    fig = plt.figure(figsize=(4.5 * n_cols + 1.2, 4.5 * n_rows + 1.0))
    gs = gridspec.GridSpec(
        n_rows, n_cols,
        figure=fig,
        hspace=0.08, wspace=0.08,
        left=0.03, right=0.97, top=0.92, bottom=0.03,
    )

    for col, N in enumerate(N_values):
        print(f"  N={N} ...", flush=True)

        # Subsample: always include the central query point at index 0
        full_query_idx = int(np.argmin(np.linalg.norm(X_full - query_coord, axis=1)))
        other_idxs = np.delete(np.arange(len(X_full)), full_query_idx)
        sub_other  = rng.choice(other_idxs, size=N - 1, replace=False)
        sub_idxs   = np.concatenate([[full_query_idx], sub_other])
        X_sub      = X_full[sub_idxs]
        query_in_sub = 0   # query is always first in the subsample

        neighbors_list = compute_all_strategies(
            X_sub, query_in_sub, K, lengthscale, rng=rng, kernel=kernel,
        )
        s = _auto_point_size(N)

        for row, (label, neighbor_idxs) in enumerate(
            zip(strategy_labels, neighbors_list)
        ):
            ax = fig.add_subplot(gs[row, col])
            neighbor_set = set(neighbor_idxs.tolist())
            non_nbr = np.array(
                [i != query_in_sub and i not in neighbor_set for i in range(N)]
            )
            nbr = np.array([i in neighbor_set for i in range(N)])

            ax.scatter(X_sub[non_nbr, 0], X_sub[non_nbr, 1],
                       c="#888888", s=s, alpha=0.4, rasterized=True)
            ax.scatter(X_sub[nbr, 0], X_sub[nbr, 1],
                       c=NEIGHBOR_COLOR, s=8, alpha=0.95,
                       rasterized=True, zorder=3)
            ax.scatter(X_sub[query_in_sub, 0], X_sub[query_in_sub, 1],
                       c=QUERY_COLOR, s=40, alpha=1.0,
                       marker="*", zorder=4)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#222222")
            ax.set_aspect("equal", adjustable="box")

            # Column title on first row
            if row == 0:
                ax.set_title(f"N = {N:,}", fontsize=28, pad=12)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot KNN neighborhood robustness across dataset sizes"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--N-values", type=int, nargs="+", default=[5000, 10000, 40000])
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--lengthscale", type=float, default=None)
    parser.add_argument("--kernel", choices=["rbf", "matern32"], default="rbf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

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
    X_full = data.X.numpy()
    N_full = len(X_full)

    model = _load_model(Path(model_dir))
    K = args.K if args.K is not None else model._prior.K

    if args.lengthscale is not None:
        lengthscale = args.lengthscale
    else:
        try:
            ls = model._prior.kernel.lengthscale
            lengthscale = float(ls.detach() if hasattr(ls, "detach") else ls)
        except AttributeError:
            lengthscale = float(config.model.get("lengthscale", 8.0))

    N_values = [min(n, N_full) for n in args.N_values]

    # Central query coord (from the full dataset)
    query_idx_full = default_query_idx(X_full)
    query_coord    = X_full[query_idx_full]

    print(f"Dataset:     {config.dataset}  (N_full={N_full})")
    print(f"Model:       {model_name}")
    print(f"K={K}  lengthscale={lengthscale:.2f}")
    print(f"N_values:    {N_values}")
    print(f"Query coord: {query_coord}")
    print("Building panels...")

    fig = plot_robustness(
        X_full, N_values, K, lengthscale, rng, query_coord, kernel=args.kernel,
    )

    dataset_tag = os.path.basename(os.path.normpath(output_dir))

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(OUT_DIR, f"knn_robustness_{dataset_tag}_{model_name}_{args.kernel}.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
