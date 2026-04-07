"""KNN selection strategy comparison for LCGP inducing points.

Three strategies for selecting K neighbors from a large candidate pool (K_large):
  1. Random:        uniformly random K from K_large nearest neighbors
  2. Evenly spaced: every (K_large / K)-th neighbor by distance rank
  3. Gaussian:      sample proportional to exp(-0.5 * (dist / lengthscale)^2)

Motivation: with standard K-nearest, dense regions produce very local neighborhoods.
Sampling from a larger pool with spatial spread gives better coverage.

Usage:
    conda run -n factorization python paper/panel_2/knn_strategies.py \
        -c configs/osmfish/general.yaml
    conda run -n factorization python paper/panel_2/knn_strategies.py \
        -c configs/slideseq/general.yaml --query-idx 1234 --K 50 --K-large 1000
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
# KNN pool construction
# ---------------------------------------------------------------------------

def build_faiss_pool(X: np.ndarray, K_large: int) -> tuple:
    """FAISS flat-L2 search over all N points.

    Returns:
        indices:   (N, K_large+1) int64 — col 0 is the point itself
        distances: (N, K_large+1) float32 — squared L2 distances
    """
    X32 = X.astype(np.float32)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X32)
    distances, indices = index.search(X32, K_large + 1)
    return indices, distances


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_random(pool: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random K from pool without replacement."""
    K = min(K, len(pool))
    return rng.choice(pool, size=K, replace=False)


def select_evenly_spaced(pool: np.ndarray, K: int) -> np.ndarray:
    """Pick K evenly spaced neighbors by distance rank (pool sorted by distance)."""
    K = min(K, len(pool))
    step = len(pool) / K
    return pool[[int(i * step) for i in range(K)]]


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
        description="Compare KNN selection strategies (random / evenly-spaced / Gaussian)"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to YAML config")
    parser.add_argument("--model", default=None, help="Model subdirectory (default: mggp_lcgp)")
    parser.add_argument("--query-idx", type=int, default=None,
                        help="Query cell index (default: cell near centroid)")
    parser.add_argument("--K", type=int, default=None,
                        help="Target number of neighbors (default: from model)")
    parser.add_argument("--K-large", type=int, default=None,
                        help="Candidate pool size (default: min(1000, N))")
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
    model_dir = os.path.join(output_dir, model_name)

    data = load_preprocessed(output_dir)
    X = data.X.numpy()
    N = len(X)

    # Load model to get K and lengthscale
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

    K_large = args.K_large if args.K_large is not None else min(1000, N)
    K_large = max(K_large, K)  # pool must be at least as large as K

    query_idx = args.query_idx if args.query_idx is not None else default_query_idx(X)

    print(f"Dataset:     {config.dataset}  (N={N})")
    print(f"Model:       {model_name}")
    print(f"K={K}  K_large={K_large}  lengthscale={lengthscale:.2f}")
    print(f"Query idx:   {query_idx}")

    print()
    print("Timing full N×K graph construction for each strategy:")

    # --- Baseline: FAISS directly with K ---
    t0 = time.perf_counter()
    baseline_all, _ = build_faiss_pool(X, K)   # (N, K+1)
    baseline_all = baseline_all[:, 1:]          # (N, K) exclude self
    t_baseline = time.perf_counter() - t0
    print(f"  Baseline   (FAISS K={K}):          {t_baseline:.4f}s")

    # --- Build shared large pool (used by Random, Evenly spaced) ---
    t0 = time.perf_counter()
    pool_all, _ = build_faiss_pool(X, K_large)  # (N, K_large+1)
    pool_all = pool_all[:, 1:]                   # (N, K_large) exclude self
    t_pool = time.perf_counter() - t0
    print(f"  FAISS pool (K_large={K_large}):      {t_pool:.4f}s  [shared by Random/Even]")

    # --- Random: shuffle each row, take first K ---
    t0 = time.perf_counter()
    rand_all = rng.permuted(pool_all, axis=1)[:, :K]   # (N, K)
    t_rand = time.perf_counter() - t0
    print(f"  Random     (pool + select):        {t_pool + t_rand:.4f}s  (select={t_rand:.4f}s)")

    # --- Evenly spaced: fixed index stride, same for all rows ---
    t0 = time.perf_counter()
    step = K_large / K
    even_cols = [int(i * step) for i in range(K)]
    even_all = pool_all[:, even_cols]                   # (N, K)
    t_even = time.perf_counter() - t0
    print(f"  Even spaced(pool + select):        {t_pool + t_even:.4f}s  (select={t_even:.4f}s)")

    # --- Gaussian: sample from ALL N points with Gaussian weights (no pool) ---
    # Radius is determined by lengthscale, not data density — stable across N.
    # Process in batches to avoid N×N distance matrix.
    t0 = time.perf_counter()
    BATCH = 2000
    gauss_all = np.empty((N, K), dtype=np.int64)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        # distances from batch queries to all N points: (batch, N)
        dists_batch = np.linalg.norm(
            X[start:end, np.newaxis, :] - X[np.newaxis, :, :], axis=2
        )
        for bi, i in enumerate(range(start, end)):
            dists_batch[bi, i] = np.inf   # exclude self
            w = np.exp(-0.5 * (dists_batch[bi] / lengthscale) ** 2)
            w /= w.sum()
            gauss_all[i] = rng.choice(N, size=K, replace=False, p=w)
    t_gauss = time.perf_counter() - t0
    print(f"  Gaussian   (all-N, batched):       {t_gauss:.4f}s")

    # Pull per-query rows for visualization
    baseline  = baseline_all[query_idx]
    rand_sel  = rand_all[query_idx]
    even_sel  = even_all[query_idx]
    gauss_sel = gauss_all[query_idx]

    neighbor_sets = {
        f"Baseline\n(K-nearest, K={K})": baseline,
        f"Random\n(pool={K_large}, K={K})": rand_sel,
        f"Evenly spaced\n(pool={K_large}, K={K})": even_sel,
        f"Gaussian\n(σ={lengthscale:.1f}, all-N, K={K})": gauss_sel,
    }

    s = _auto_point_size(N)
    fig = plot_strategies(X, query_idx, neighbor_sets, s=s)

    dataset_tag = os.path.basename(os.path.normpath(output_dir))
    fig.suptitle(
        f"{dataset_tag} — KNN selection strategies (query={query_idx})",
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
