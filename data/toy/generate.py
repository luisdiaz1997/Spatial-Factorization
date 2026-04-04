"""
Generate a toy spatial transcriptomics dataset.

Grid:     100x100 = 10,000 points in [-100, 100]^2
Groups:   A, B, C assigned by pie slices (120° each);
          1/2 of points are randomly reassigned to break hard boundaries
Factors:  L=10 GP samples from MGGP Matern3/2 (lengthscale=40.0, group_diff_param=8.0)
          via Cholesky: F = L @ randn(N, 10)
Loadings: W ~ Uniform(0, 1), shape (D=200, L=10)
Counts:   Y ~ Poisson(exp(F) @ W.T / 10), shape (N=10000, D=200)

Saved files (in same directory as this script):
  X.npy                           (N, 2)  spatial coordinates
  C.npy                           (N,)    integer group codes (0=A, 1=B, 2=C)
  Y.npz                           (N, D)  sparse Poisson counts
  metadata.json                           gene_names, group_names, etc.
  ground_truth_factors.npy        (N, L)  raw GP samples (before exp)
  ground_truth_loadings.npy       (D, L)  W matrix
  ground_truth_factors_group_0.npy (N, L) conditional posterior mean, all points as group A
  ground_truth_factors_group_1.npy (N, L) conditional posterior mean, all points as group B
  ground_truth_factors_group_2.npy (N, L) conditional posterior mean, all points as group C
"""

import json
import os

import numpy as np
import torch
from scipy.sparse import csr_matrix, save_npz

from gpzoo.kernels import batched_MGGP_Matern32
from gpzoo.utilities import add_jitter

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 97
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── constants ─────────────────────────────────────────────────────────────────
N_SIDE   = 100       # grid side → N = N_SIDE^2 = 10,000
COORD_LO = -100.0
COORD_HI =  100.0
L        = 10        # number of latent factors
D        = 200       # number of genes
N_GROUPS = 3
NOISE_FRAC   = 1/2   # fraction of points with randomly reassigned group
LENGTHSCALE  = 100.0
GROUP_DIFF   = 1e-1
JITTER       = 7e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 1. Build spatial grid ─────────────────────────────────────────────────────
coords_1d = np.linspace(COORD_LO, COORD_HI, N_SIDE)
xx, yy = np.meshgrid(coords_1d, coords_1d)
X_np = np.column_stack([xx.ravel(), yy.ravel()])   # (N, 2)
N = X_np.shape[0]
print(f"Grid: {N} points ({N_SIDE}x{N_SIDE})")

# ── 2. Assign cell types via pie slices (angle from centre) ───────────────────
angle = np.arctan2(X_np[:, 1], X_np[:, 0])   # [-π, π]
C_np = np.zeros(N, dtype=np.int64)
# A: [-π,  -π/3)   (bottom-left 120° sector)
# B: [-π/3,  π/3)  (right 120° sector)
# C: [ π/3,  π)    (top-left 120° sector)
C_np[angle >= -np.pi / 3] = 1
C_np[angle >=  np.pi / 3] = 2

# ── 3. Randomly reassign 1/10 of points ──────────────────────────────────────
n_noise = int(N * NOISE_FRAC)
noise_idx = np.random.choice(N, size=n_noise, replace=False)
C_np[noise_idx] = np.random.randint(0, N_GROUPS, size=n_noise)

group_counts = [(C_np == g).sum() for g in range(N_GROUPS)]
print(f"Group sizes: A={group_counts[0]}, B={group_counts[1]}, C={group_counts[2]}")

# ── 4. Build MGGP Matern3/2 kernel and sample L factors ──────────────────────
X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
C_t = torch.tensor(C_np, dtype=torch.long,    device=device)

kernel = batched_MGGP_Matern32(
    sigma=1.0,
    lengthscale=LENGTHSCALE,
    group_diff_param=GROUP_DIFF,
    n_groups=N_GROUPS,
).to(device)

print(f"Computing {N}x{N} MGGP Matern3/2 kernel matrix …")
with torch.no_grad():
    K = kernel(X_t, X_t, C_t, C_t).contiguous()  # (N, N)
    add_jitter(K, jitter=JITTER)                   # in-place diagonal jitter

    print("Cholesky decomposition …")
    L_chol = torch.linalg.cholesky(K)        # (N, N) lower triangular
    del K

    z = torch.randn(N, L, device=device)     # (N, L)
    F_t = L_chol @ z                         # (N, L)  raw GP samples
    del z

    # flip any factor whose samples sum to negative so exp(F) stays large
    signs = torch.sign(F_t.sum(dim=0))       # (L,)  +1 or -1 per factor
    signs[signs == 0] = 1
    F_t = F_t * signs.unsqueeze(0)           # broadcast over N

    # ── conditional posterior mean per group ─────────────────────────────────
    # μ_g = K(X_g, X_train) @ K_train⁻¹ @ F
    # K_train = L @ L^T  →  K⁻¹ @ F = L^{-T} @ (L^{-1} @ F)
    print("Computing conditional posterior means per group …")
    alpha = torch.linalg.solve_triangular(
        L_chol.T, torch.linalg.solve_triangular(L_chol, F_t, upper=False),
        upper=True,
    )  # (N, L)  K⁻¹ @ F via two triangular solves

    cond_means = {}
    for g in range(N_GROUPS):
        C_g = torch.full((N,), g, dtype=torch.long, device=device)
        K_g = kernel(X_t, X_t, C_g, C_t).contiguous()  # (N, N): test all-group-g vs true groups
        cond_means[g] = (K_g @ alpha).cpu().numpy()     # (N, L)
        del K_g
        print(f"  Group {g} done")

    del L_chol, alpha
    if device.type == "cuda":
        torch.cuda.empty_cache()

F_np = F_t.cpu().numpy()                     # (N, L) ground-truth factors
print(f"Factors shape: {F_np.shape}")

# ── 5. Loadings W ~ Uniform(0, 1) ────────────────────────────────────────────
W_np = np.random.rand(D, L).astype(np.float32)   # (D, L), all positive

# ── 6. Build Poisson rate and sample counts ───────────────────────────────────
exp_F = np.exp(F_np).astype(np.float32)           # (N, L)
rate  = (exp_F @ W_np.T) / 10.0                    # (N, D) = (N,L) @ (L,D)
print(f"Rate matrix shape: {rate.shape}, mean rate: {rate.mean():.4f}")

Y_np = np.random.poisson(rate).astype(np.float32) # (N, D)
print(f"Y shape: {Y_np.shape}, sparsity: {(Y_np == 0).mean():.3f}")

# ── 7. Save outputs ───────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "X.npy"), X_np.astype(np.float32))
np.save(os.path.join(OUT_DIR, "C.npy"), C_np)
np.save(os.path.join(OUT_DIR, "ground_truth_factors.npy"),  F_np)
np.save(os.path.join(OUT_DIR, "ground_truth_loadings.npy"), W_np)
for g, cond_mean in cond_means.items():
    np.save(os.path.join(OUT_DIR, f"ground_truth_factors_group_{g}.npy"), cond_mean.astype(np.float32))

Y_sparse = csr_matrix(Y_np)
save_npz(os.path.join(OUT_DIR, "Y.npz"), Y_sparse)

gene_names  = [f"gene_{i:03d}"  for i in range(D)]
group_names = ["A", "B", "C"]
metadata = {
    "n_spots":   N,
    "n_genes":   D,
    "n_groups":  N_GROUPS,
    "gene_names": gene_names,
    "group_names": group_names,
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print("\nSaved files:")
group_files = [f"ground_truth_factors_group_{g}.npy" for g in range(N_GROUPS)]
for fname in ["X.npy", "C.npy", "Y.npz", "metadata.json",
              "ground_truth_factors.npy", "ground_truth_loadings.npy"] + group_files:
    path = os.path.join(OUT_DIR, fname)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  {fname:<35s}  {size_mb:.2f} MB")

print("\nDone.")
