"""KNN selection strategies for LCGP inducing points.

Utility functions imported by plot_knn.py and plot_knn_robustness.py.

  - select_baseline:  K-nearest neighbors for a single query (FAISS)
  - select_kernel:    sample K from all N points with GPzoo kernel weights

Motivation: with standard K-nearest, dense regions produce very local neighborhoods.
The kernel-weighted approach keeps the spatial radius stable as N grows (radius ~ lengthscale).
"""

import numpy as np
import faiss
import torch

from gpzoo.kernels import batched_RBF, batched_Matern32, batched_MGGP_RBF, batched_MGGP_Matern32


# ---------------------------------------------------------------------------
# Kernel functions (backed by GPzoo)
# ---------------------------------------------------------------------------

def kernel_weights(X: np.ndarray, query_coord: np.ndarray, lengthscale: float, kernel: str) -> np.ndarray:
    """Compute k(x_i, query) for all x_i in X using GPzoo kernels.

    Args:
        X:           (N, D) spatial coordinates
        query_coord: (D,) query point coordinates
        lengthscale: kernel lengthscale
        kernel:      "rbf" (batched_RBF) or "matern32" (batched_Matern32)

    Returns:
        (N,) array of weights in (0, 1], with weight=1 where x_i == query.
    """
    if kernel == "rbf":
        k = batched_RBF(sigma=1.0, lengthscale=float(lengthscale))
    elif kernel == "matern32":
        k = batched_Matern32(sigma=1.0, lengthscale=float(lengthscale))
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    q_t = torch.from_numpy(np.asarray(query_coord, dtype=np.float32)).unsqueeze(0)  # (1, D)

    with torch.no_grad():
        w = k(X_t, q_t)  # (N, 1)
    return w.squeeze(-1).numpy()


def build_mggp_kernel(kernel_type: str, n_groups: int, lengthscale: float, group_diff_param: float):
    """Construct a fresh (untrained) MGGP kernel.

    Args:
        kernel_type:     "rbf" or "matern32"
        n_groups:        number of groups (for embedding init)
        lengthscale:     spatial lengthscale
        group_diff_param: cross-group penalty parameter

    Returns:
        A batched_MGGP_RBF or batched_MGGP_Matern32 instance.
    """
    if kernel_type == "rbf":
        return batched_MGGP_RBF(sigma=1.0, lengthscale=lengthscale,
                                 group_diff_param=group_diff_param, n_groups=n_groups)
    elif kernel_type == "matern32":
        return batched_MGGP_Matern32(sigma=1.0, lengthscale=lengthscale,
                                      group_diff_param=group_diff_param, n_groups=n_groups)
    else:
        raise ValueError(f"Unknown MGGP kernel type: {kernel_type!r}")


def kernel_weights_mggp(
    X: np.ndarray,
    C: np.ndarray,
    query_coord: np.ndarray,
    query_group: int,
    kernel,  # pre-loaded batched_MGGP_RBF or batched_MGGP_Matern32 from GPzoo
) -> np.ndarray:
    """Compute MGGP kernel weights k((x_i, c_i), (query, query_group)) for all cells.

    Args:
        X:           (N, 2) spatial coordinates
        C:           (N,) integer group assignments
        query_coord: (2,) query point coordinates
        query_group: group index of the query cell
        kernel:      trained batched_MGGP_* kernel loaded from model._prior.kernel

    Returns:
        (N,) array of weights; max=1 at the query cell itself.
    """
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    q_t = torch.from_numpy(np.asarray(query_coord, dtype=np.float32)).unsqueeze(0)
    groupsX_t = torch.from_numpy(np.asarray(C, dtype=np.int64))
    groupsZ_t = torch.tensor([query_group], dtype=torch.long)
    with torch.no_grad():
        w = kernel(X_t, q_t, groupsX_t, groupsZ_t)  # (N, 1)
    return w.squeeze(-1).numpy()


def select_kernel_mggp(
    X: np.ndarray,
    C: np.ndarray,
    query_idx: int,
    query_group: int,
    K: int,
    kernel,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample K neighbors weighted by the MGGP kernel centered at the query.

    Args:
        X:           (N, 2) spatial coordinates
        C:           (N,) integer group assignments
        query_idx:   index of the query cell in X
        query_group: group index of the query cell
        K:           number of neighbors to sample
        kernel:      trained batched_MGGP_* kernel from model._prior.kernel
        rng:         numpy random generator

    Returns:
        (K,) array of sampled neighbor indices (excluding query).
    """
    N = len(X)
    K = min(K, N - 1)
    weights = kernel_weights_mggp(X, C, X[query_idx], query_group, kernel)
    weights[query_idx] = 0.0
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        candidates = np.delete(np.arange(N), query_idx)
        return rng.choice(candidates, size=K, replace=False)
    weights /= total
    return rng.choice(N, size=K, replace=False, p=weights)


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


def select_kernel(
    X: np.ndarray,
    query_idx: int,
    K: int,
    lengthscale: float,
    rng: np.random.Generator,
    kernel: str = "rbf",
) -> np.ndarray:
    """Sample K neighbors from all N points weighted by a GPzoo kernel centered at the query.

    Sampling from all N (not a count-based pool) means the spatial radius is
    determined by lengthscale, not by data density — so it stays stable as N grows.

    Args:
        kernel: "rbf" (batched_RBF) or "matern32" (batched_Matern32)
    """
    N = len(X)
    K = min(K, N - 1)
    weights = kernel_weights(X, X[query_idx], lengthscale, kernel)
    weights[query_idx] = 0.0   # exclude self
    total = weights.sum()
    if total == 0 or not np.isfinite(total):
        candidates = np.delete(np.arange(N), query_idx)
        return rng.choice(candidates, size=K, replace=False)
    weights /= total
    return rng.choice(N, size=K, replace=False, p=weights)


# Keep old name as alias for backward compatibility
def select_gaussian(X, query_idx, K, lengthscale, rng):
    return select_kernel(X, query_idx, K, lengthscale, rng, kernel="rbf")
