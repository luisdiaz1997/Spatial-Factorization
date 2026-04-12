# Chunked Expanded-K Posterior for Groupwise Factors (LCGP/MGGP_LCGP)

## Motivation

The current groupwise factor computation (`_get_groupwise_factors_batched` in `analyze.py:584-623`) uses the **training K** (K=50) to compute posteriors. During training, keeping K small is necessary for speed, but at inference time we can afford more neighbors since we are not backpropagating. More neighbors means the posterior conditions on more data, giving smoother and more accurate groupwise factor maps.

Note: K=50 is the **neighbor count** (number of conditioning points per query). The Lu parameter has shape `(L, M, R)` where R=250 is the **rank** of the VNNGP-style covariance factor, not the neighbor count.

We want to increase K to ~2000-5000 for the posterior pass. This creates memory challenges that require a chunked approach analogous to what GPzoo already does for probabilistic KNN selection (`gpzoo/knn_utilities.py:38-109`).

---

## Current Flow (K=50 neighbors, R=250 rank)

### 1. Entry Point: `_get_groupwise_factors_batched` (`analyze.py:584-623`)

For each group `g` in `{0, ..., G-1}`:
- Loops over cells in batches of 10000
- For each batch, calls `_get_spatial_qF(model, coords_batch, groups_batch=full(g))`
- Collects `exp(qF.mean)` per batch, concatenates to `(N, L)`

### 2. `_get_spatial_qF` (`PNMF/transforms.py:24-78`)

- Computes KNN for the batch: `calculate_knn(model._prior, coords_batch, ...)[:, :-1]`
  - Returns `(batch_size, K)` index tensor
  - Sets `model._prior.knn_idx = knn_idx`
- Calls `model._prior(X=coords_batch, groupsX=groups_batch)` which goes to...

### 3. LCGP.forward (inherits WSVGP.forward, `gpzoo/gp.py:324-394`)

Given `knn_idx` of shape `(N_batch, K)`:

1. **forward_kernels** (`gp.py:943-969` for MGGP, `gp.py:100-120` for non-MGGP):
   - `Kxx`: `(L, N_batch)` -- diagonal only
   - `Kzx`: `(L, K, N_batch)` -- cross-covariance between K neighbors and batch points
   - `Kzz`: `(L, K, K)` -- covariance among K neighbors

2. **reshape_parameters** (`gp.py:775-817`):
   - Gathers `mu[..., knn_idx]` -> `(L, N_batch, K)`
   - Gathers `Lu[:, knn_idx]` -> `(L, N_batch, K, R)` (R=250 rank from stored Lu)
   - Computes `Su_knn = Lu_knn @ Lu_knn.T` -> `(L, N_batch, K, K)`
   - **Cholesky**: `torch.linalg.cholesky(Su_knn)` -> `(L, N_batch, K, K)`

3. **Posterior** (`gp.py:341-381`):
   - `L_chol = cholesky(Kzz)` -> `(L, K, K)`
   - `Wt = solve_triangular(L_chol, Kzx)` -> `(L, K, N_batch)`
   - `W = Wt.T` -> `(L, N_batch, K)`
   - `mean = W @ mu` -> `(L, N_batch)`
   - `cov_diag = Kxx - sum(W^2) + sum((W @ Lu)^2)` -> `(L, N_batch)`

### Memory Profile at K=50 (current)

The dominant tensor is `Su_knn` after `Lu_knn @ Lu_knn.T`: `(L, N_batch, K, K)`.
With L=10, N_batch=10000, K=50: `10 * 10000 * 50 * 50 * 4 bytes = 1 GB` -- manageable.
The intermediate `Lu_knn` is `(L, N_batch, K, R)` = `(10, 10000, 50, 250)` = 5 GB, which is the actual bottleneck at current settings.

---

## Proposed: Chunked Posterior with K_post >> K_train (50)

### Core Observation: What's Shared Across Groups

In the MGGP kernel (`batched_MGGP_Matern32`, `gpzoo/kernels.py:115-170`), the group assignment only affects the **kernel matrices** (Kxx, Kzx, Kzz). The variational parameters (mu, Lu) are group-independent.

For a given set of query points and their K_post neighbors:

**Shared across all G groups (compute ONCE per chunk):**
- `knn_idx` -- neighbor indices
- `mu_knn` -- variational means gathered at neighbors
- `Lu_knn`, `Su_knn = Lu_knn @ Lu_knn.T`, `L_su = cholesky(Su_knn)` -- variational covariance
- `Kzz` -- kernel among neighbors (neighbors have **fixed training-time groups**)
- `L_kzz = cholesky(Kzz)` -- Cholesky of neighbor kernel
- `mu_transformed`, `Lu_transformed` -- whitened variational params via `transform_variables`

**Per-group (only kernel cross-terms change):**
- `Kxx(g)`: diagonal kernel at query points with group g -- `(L, C)`, cheap
- `Kzx(g)`: cross-kernel, neighbors (fixed groups) -> queries (group g) -- `(L, K_post, C)`, cheap
- One triangular solve + matrix-vector products for posterior mean/variance

This means **G groups add G kernel evaluations and G triangular solves, but ZERO additional Cholesky decompositions**.

### Algorithm

```python
def _get_groupwise_factors_expanded_K(
    model, coords, n_groups, sort_order,
    K_post=5000,       # expanded neighbor count for posterior
    mem_gb=8.0,        # GPU memory budget in GB
):
    N = coords.shape[0]
    L = model._prior.mu.shape[0]
    R = model._prior.Lu.shape[-1]  # stored Lu is (L, M, R) where R=250 is rank
    
    # Adaptive chunk size: need ~4 live (L, C, K_post, K_post) tensors
    C = max(1, int(mem_gb * 1e9 / (L * K_post**2 * 4 * 4)))
    
    # Build FAISS index once (M=N for LCGP)
    index = faiss.IndexFlatL2(2)
    index.add(model._prior.Z.cpu().numpy())
    
    result = {g: np.empty((N, L), dtype=np.float32) for g in range(n_groups)}
    
    for start in range(0, N, C):
        end = min(start + C, N)
        X_chunk = coords[start:end]                     # (C, 2)
        
        # ---- Step 1: K_post neighbors (ONCE) ----
        _, knn_idx = index.search(X_chunk.numpy(), K_post)  # (C, K_post)
        knn_idx = torch.tensor(knn_idx, dtype=torch.long)
        
        # ---- Step 2: Gather variational params (ONCE) ----
        mu = model._prior.mu                             # (L, M)
        Lu_raw = model._prior.Lu                         # (L, M, R) where R=250 is rank
        
        mu_knn = mu[:, knn_idx]                          # (L, C, K_post)
        Lu_knn = Lu_raw[:, knn_idx]                      # (L, C, K_post, R)
        
        # (L, C, K_post, R) @ (L, C, R, K_post) -> (L, C, K_post, K_post)
        Su_knn = Lu_knn @ Lu_knn.transpose(-2, -1)
        Su_knn = add_jitter(Su_knn)
        L_su = torch.linalg.cholesky(Su_knn)             # (L, C, K_post, K_post)
        
        # ---- Step 3: Neighbor kernel Kzz (ONCE, fixed groups) ----
        Z_neighbors = model._prior.Z[knn_idx]            # (C, K_post, 2)
        groupsZ_nbr = model._prior.groupsZ[knn_idx]      # (C, K_post)
        
        Kzz = eval_kernel_local(kernel, Z_neighbors, groupsZ_nbr)  # (L, C, K_post, K_post)
        Kzz = add_jitter(Kzz)
        L_kzz = torch.linalg.cholesky(Kzz)               # (L, C, K_post, K_post)
        
        # Whiten: mu_t = L_kzz^-1 @ mu_knn, Lu_t = L_kzz^-1 @ L_su
        mu_t, Lu_t = transform_variables(mu_knn, L_su, L_kzz)
        
        # ---- Step 4: Per-group posterior (CHEAP per group) ----
        for g in range(n_groups):
            groups_q = torch.full((end - start,), g, dtype=torch.long)
            
            Kxx_g = eval_kernel_diag(kernel, X_chunk, groups_q)     # (L, C)
            Kzx_g = eval_kernel_cross(kernel, Z_neighbors, X_chunk,
                                       groupsZ_nbr, groups_q)       # (L, C, K_post, 1) or similar
            
            Wt_g = torch.linalg.solve_triangular(L_kzz, Kzx_g, upper=False)
            W_g = Wt_g.transpose(-2, -1)                             # (L, C, K_post)
            
            mean_g = (W_g @ mu_t.unsqueeze(-1)).squeeze(-1)          # (L, C)
            # variance if needed:
            # var_g = Kxx_g - sum(W_g**2, -1) + sum((W_g @ Lu_t)**2, -1)
            
            factors_g = torch.exp(mean_g).T.cpu().numpy()            # (C, L)
            result[g][start:end] = factors_g[:, sort_order]
    
    return result
```

---

## Memory Analysis

### Per-Chunk Peak Memory

The dominant tensors are `(L, C, K_post, K_post)` shaped. We need ~4 of them live simultaneously (Su_knn, L_su, Kzz, L_kzz).

**Formula:** `C = floor(mem_gb * 1e9 / (L * K_post^2 * 4_bytes * 4_tensors))`

| K_post | mem_gb=8 GB | C (chunk) | Total chunks (N=41783) | Su/Kzz tensor size |
|--------|-------------|-----------|----------------------|-------------------|
| 5000 | 8 GB | 2 | ~20,892 | 1 GB each |
| 2000 | 8 GB | 12 | ~3,482 | 160 MB each |
| 1000 | 8 GB | 50 | ~836 | 40 MB each |
| 500 | 8 GB | 200 | ~209 | 10 MB each |

**Additional per-chunk memory:**
- `Lu_knn`: `(L, C, K_post, R)` -- with R=250 (rank), this is `R/K_post` times smaller than K_post^2 tensors
- `knn_idx`: `(C, K_post)` -- negligible
- Per-group `Kzx_g`: `(L, K_post, C)` -- negligible
- Per-group `W_g`: `(L, C, K_post)` -- negligible

### Recommended Defaults

- **K_post = 2000**: Good balance. C=12 chunks, ~3500 iterations for N=41K. Each `(L, 12, 2000, 2000)` tensor is ~1.9 GB. Peak ~8 GB.
- **K_post = 5000**: Maximum quality. C=2, ~21K iterations. Peak ~8 GB but slower due to more iterations and larger solves.

---

## KNN Considerations

### FAISS for Spatial Neighbors

For the expanded posterior, use **deterministic FAISS KNN** based on spatial distance alone:
- Build index once: `faiss.IndexFlatL2(2)` on all M=N inducing points
- Query per chunk: `index.search(X_chunk, K_post)`
- Group attenuation happens in the kernel evaluation, not neighbor selection

### KNN Index Memory

Full `knn_idx` at `(N, K_post)`: `41783 * 5000 * 8 = 1.67 GB`. Do NOT store all at once. Query FAISS per chunk instead -- the index object itself is tiny (just the 2D coordinates).

---

## Implementation Plan

### Where to Implement

New function `_get_groupwise_factors_expanded_K` in `analyze.py`, called when `posterior_K` is set in the config and `posterior_K > K` (training K=50). Falls back to existing `_get_groupwise_factors_batched` otherwise.

### Steps

1. **`analyze.py`**: Add `_get_groupwise_factors_expanded_K` function that:
   - Builds FAISS index once on `model._prior.Z`
   - Computes adaptive chunk size from memory budget
   - For each chunk: find neighbors, gather params, one Cholesky, loop over groups
   - Need helper functions to evaluate kernel locally on sliced neighbor sets (separate from the full `forward_kernels` which expects the model's stored Z)

2. **GPzoo helper** (or inline in analyze.py): Function to evaluate the kernel on arbitrary `(Z_subset, X_query)` pairs with given group assignments. The current `forward_kernels` in `gp.py` always uses `self.Z` -- we need a version that takes Z as an argument. Could use the kernel directly: `model._prior.kernel(Z_subset, X_query, groupsZ_subset, groupsX_query)`.

3. **Config**: Add optional `posterior_K` to training config section. Default: `null` (use training K). Example: `posterior_K: 2000`.

4. **CLI**: Optionally expose via `--posterior-k` flag on `analyze` command.

### Code References

| What | Where | Lines |
|------|-------|-------|
| Current groupwise factors | `analyze.py` | 584-623 |
| `_get_spatial_qF` (LCGP KNN + forward) | `PNMF/transforms.py` | 24-78 |
| LCGP `reshape_parameters` (gather + Cholesky) | `gpzoo/gp.py` | 775-817 |
| WSVGP.forward (full posterior math) | `gpzoo/gp.py` | 324-394 |
| `transform_variables` (whitening step) | `gpzoo/gp.py` | 353 (called in forward) |
| MGGP `forward_kernels` | `gpzoo/gp.py` | 943-969 |
| Non-MGGP `forward_kernels` | `gpzoo/gp.py` | 100-120 |
| Chunked probabilistic KNN | `gpzoo/knn_utilities.py` | 38-109 |
| MGGP kernel (group attenuation formula) | `gpzoo/kernels.py` | 129-149 |
| LCGP `forward_train` (marginal q(U)) | `gpzoo/gp.py` | 824-840 |
| LCGP `kl_divergence_full` | `gpzoo/gp.py` | 842-913 |
| Kernel `forward` (vmap over pairs) | `gpzoo/kernels.py` | 152-170 |

### Slideseq MGGP_LCGP Dimensions (reference)

| Parameter | Value |
|-----------|-------|
| N (cells) | 41,783 |
| L (factors) | 10 |
| K_train (neighbor count) | 50 |
| R (Lu rank, last dim of Lu) | 250 |
| G (groups) | 14 |
| Lu shape | (10, 41783, 250) = 418 MB |
| groupwise output | 14 files, each (41783, 10) |

---

## Open Questions

1. **K_post vs full N:** Using all N=41783 as neighbors would give the exact GP posterior (no local approximation). But `(L, 1, N, N)` = `(10, 1, 41783, 41783)` = 70 GB per tensor -- not feasible even for 1 point. K_post=5000 is a practical ceiling.

2. **Do we also want expanded-K for the unconditional factors?** Currently `_get_factors_batched` also uses K=50. Same approach would apply, but it's a separate code path.

3. **CPU vs GPU:** The Cholesky on `(L, C, K_post, K_post)` is compute-heavy. GPU is ~10x faster for large Cholesky. Should default to GPU if available, with CPU fallback.

4. **transform_variables location:** This function is a method on WSVGP (`gp.py:353`). We need to either call it through the model or extract the math inline. The transform is: `mu_t = L_kzz^{-1} @ mu_knn` and `Lu_t = L_kzz^{-1} @ L_su`. Just two triangular solves.
