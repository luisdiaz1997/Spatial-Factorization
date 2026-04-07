"""KNN selection strategy comparison: K-nearest baseline vs Gaussian.

  1. Baseline:  K-nearest neighbors via FAISS
  2. Gaussian:  sample K from all N points proportional to exp(-0.5*(dist/lengthscale)^2)

Motivation: with standard K-nearest, dense regions produce very local neighborhoods.
The Gaussian approach keeps the spatial radius stable as N grows (radius ~ lengthscale).

Usage:
    conda run -n factorization python paper/panel_2/knn_strategies.py \
        -c configs/osmfish/general.yaml
    conda run -n factorization python paper/panel_2/knn_strategies.py \
        -c configs/slideseq/general.yaml --query-idx 1234 --K 50
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import faiss
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from spatial_factorization.config import Config
from spatial_factorization.datasets.base import load_preprocessed
from spatial_factorization.commands.analyze import _load_model
from spatial_factorization.commands.figures import _auto_point_size

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_query_idx(X: np.ndarray) -> int:
    """Pick a cell near the median distance from the tissue centroid."""
    dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
    return int(dists.argsort()[len(dists) // 2])


# ---------------------------------------------------------------------------
# KNN construction
# ---------------------------------------------------------------------------

def build_faiss_pool(X: np.ndarray, K: int) -> tuple:
    """FAISS flat-L2 search over all N points.

    Returns:
        indices:   (N, K+1) int64 — col 0 is the point itself
        distances: (N, K+1) float32 — squared L2 distances
    """
    X32 = X.astype(np.float32)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X32)
    distances, indices = index.search(X32, K + 1)
    return indices, distances


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_baseline(X: np.ndarray, query_idx: int, K: int) -> np.ndarray:
    """K-nearest neighbors via FAISS (single query)."""
    idxs, _ = build_faiss_pool(X, K)
    return idxs[query_idx, 1:]   # exclude self


def select_gaussian(
    X: np.ndarray,
    query_idx: int,
    K: int,
    lengthscale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample K neighbors from all N points proportional to Gaussian(||x_i - x_q||; 0, lengthscale).

    Sampling from all N (not a count-based pool) means the spatial radius is
    determined by lengthscale, not by data density — so it stays stable as N grows.
    """
    N = len(X)
    K = min(K, N - 1)
    dists = np.linalg.norm(X - X[query_idx], axis=1)
    dists[query_idx] = np.inf   # exclude self
    weights = np.exp(-0.5 * (dists / lengthscale) ** 2)
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        candidates = np.delete(np.arange(N), query_idx)
        return rng.choice(candidates, size=K, replace=False)
    weights /= total
    return rng.choice(N, size=K, replace=False, p=weights)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_strategies(
    X: np.ndarray,
    query_idx: int,
    neighbor_sets: dict,
    s: float = 4.0,
    neighbor_color: str = "#F4A23A",
    query_color: str = "#E03030",
    query_size_mult: float = 6.0,
) -> plt.Figure:
    n = len(neighbor_sets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (label, neighbor_idxs) in zip(axes, neighbor_sets.items()):
        neighbor_set = set(neighbor_idxs.tolist())
        non_nbr_mask = np.array(
            [i != query_idx and i not in neighbor_set for i in range(len(X))]
        )
        nbr_mask = np.array([i in neighbor_set for i in range(len(X))])

        ax.scatter(X[non_nbr_mask, 0], X[non_nbr_mask, 1],
                   c="#888888", s=s, alpha=0.5, rasterized=True)
        ax.scatter(X[nbr_mask, 0], X[nbr_mask, 1],
                   c=neighbor_color, s=s * 1.5, alpha=0.95, rasterized=True,
                   label=f"K={len(neighbor_idxs)}", zorder=3)
        ax.scatter(X[query_idx, 0], X[query_idx, 1],
                   c=query_color, s=s * query_size_mult, alpha=1.0,
                   marker="*", label="Query", zorder=4)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#222222")
        ax.set_title(label, fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                  borderaxespad=0, fontsize=8, framealpha=0.7)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare KNN baseline vs Gaussian selection"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--model", default=None, help="Model subdirectory (default: mggp_lcgp)")
    parser.add_argument("--query-idx", type=int, default=None,
                        help="Query cell index (default: cell near centroid)")
    parser.add_argument("--K", type=int, default=None,
                        help="Number of neighbors (default: from model)")
    parser.add_argument("--lengthscale", type=float, default=None,
                        help="Gaussian lengthscale (default: from model kernel or config)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default=None, help="Output path for figure")
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
    X = data.X.numpy()
    N = len(X)

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

    query_idx = args.query_idx if args.query_idx is not None else default_query_idx(X)

    print(f"Dataset:     {config.dataset}  (N={N})")
    print(f"Model:       {model_name}")
    print(f"K={K}  lengthscale={lengthscale:.2f}")
    print(f"Query idx:   {query_idx}")

    print()
    print("Timing full N×K graph construction:")

    t0 = time.perf_counter()
    baseline_all, _ = build_faiss_pool(X, K)
    baseline_all = baseline_all[:, 1:]   # (N, K)
    t_baseline = time.perf_counter() - t0
    print(f"  Baseline (FAISS K={K}):       {t_baseline:.4f}s")

    t0 = time.perf_counter()
    BATCH = 2000
    gauss_all = np.empty((N, K), dtype=np.int64)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        dists_batch = np.linalg.norm(
            X[start:end, np.newaxis, :] - X[np.newaxis, :, :], axis=2
        )
        for bi, i in enumerate(range(start, end)):
            dists_batch[bi, i] = np.inf
            w = np.exp(-0.5 * (dists_batch[bi] / lengthscale) ** 2)
            w /= w.sum()
            gauss_all[i] = rng.choice(N, size=K, replace=False, p=w)
    t_gauss = time.perf_counter() - t0
    print(f"  Gaussian  (all-N, batched):   {t_gauss:.4f}s")

    neighbor_sets = {
        f"Baseline\n(K-nearest, K={K})": baseline_all[query_idx],
        f"Gaussian\n(σ={lengthscale:.1f}, K={K})": gauss_all[query_idx],
    }

    s = _auto_point_size(N)
    fig = plot_strategies(X, query_idx, neighbor_sets, s=s)

    dataset_tag = os.path.basename(os.path.normpath(output_dir))
    fig.suptitle(
        f"{dataset_tag} — KNN strategies (query={query_idx})",
        fontsize=11, y=1.01,
    )

    if args.out:
        out_path = args.out
    else:
        out_path = os.path.join(OUT_DIR, f"knn_strategies_{dataset_tag}_{model_name}.png")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
