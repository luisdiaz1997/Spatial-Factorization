"""KNN selection strategies for LCGP inducing points.

Utility functions imported by plot_knn.py and plot_knn_robustness.py.

  - select_baseline:  K-nearest neighbors for a single query (FAISS)
  - select_gaussian:  sample K from all N points with Gaussian distance weights

Motivation: with standard K-nearest, dense regions produce very local neighborhoods.
The Gaussian approach keeps the spatial radius stable as N grows (radius ~ lengthscale).
"""

import numpy as np
import faiss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def default_query_idx(X: np.ndarray) -> int:
    """Pick a cell near the median distance from the tissue centroid."""
    dists = np.linalg.norm(X - X.mean(axis=0), axis=1)
    return int(dists.argsort()[len(dists) // 2])


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_baseline(X: np.ndarray, query_idx: int, K: int) -> np.ndarray:
    """K-nearest neighbors via FAISS (single query)."""
    X32 = X.astype(np.float32)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X32)
    _, idxs = index.search(X32[query_idx:query_idx + 1], K + 1)
    return idxs[0, 1:]   # exclude self


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
