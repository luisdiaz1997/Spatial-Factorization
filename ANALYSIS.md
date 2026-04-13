# ANALYSIS: EXPANDED-K POSTERIOR VS _get_groupwise_factors_batched

## QUESTION 1: IN THE OLD BATCHING, ARE K NEIGHBORS DRAWN FROM THE BATCH OR FROM ALL N?

### ANSWER: FROM ALL N (FULL Z)

In `_get_groupwise_factors_batched`, for each group g and each batch chunk:

```python
coords_b = coords[start:end]         # (batch_size, 2) — query points only
groups_b = np.full(batch_size, g)    # (batch_size,)   — hypothetical group g
qF = _get_spatial_qF(model, coordinates=coords_b, groups=groups_b)
```

Inside `_get_spatial_qF`:
```python
knn_idx = calculate_knn(model._prior, coords_b, ...)[:, :-1]
```

`calculate_knn` calls `_faiss_knn(X=coords_b, Z=model._prior.Z, K=K_train)`.
`model._prior.Z` contains ALL M=N training points.
So neighbors are always searched over the FULL Z, not just the batch.

The batch is only for X (query points). The neighbor pool is always the full dataset.
OUR NEW FUNCTION DOES THE SAME — KNN computed against full Z for all N, then indexed per chunk.

---

## QUESTION 2: DOES THE LCGP FORWARD DO SOMETHING DIFFERENT FROM OUR MANUAL COMPUTATION?

### TRACED CALL CHAIN FOR MGGP_LCGP FORWARD

When `_get_spatial_qF` calls `model._prior(X=coords_b, groupsX=groups_b)`:

**Step A — `MGGP.forward_kernels`** (gp.py:943-969):
Calls `self.reshape_input_data(X=coords_b, Z=self.Z, groupsX=groups_b, groupsZ=self.groupsZ, knn_idx=knn_idx)`.

**Step B — `MGGP.reshape_input_data`** (gp.py:928-939):
1. Calls `LCGP.reshape_input_data(X=coords_b, Z=self.Z, knn_idx=knn_idx)`:
   - Base: ensures 2D → coords stays (N,2), Z stays (M,2)
   - `X = X.unsqueeze(-2)` → (N, 1, 2)
   - `Z = Z[knn_idx]`     → (N, K, 2)
2. Calls `LCGP.reshape_input_data(X=groupsX, Z=groupsZ, knn_idx=knn_idx)` (treating groups as "coordinates"):
   - Base: (N,) → (N,1), (M,) → (M,1)
   - `groupsX = groupsX.unsqueeze(-2)` → (N, 1, 1)
   - `groupsZ = groupsZ[knn_idx]`      → (N, K, 1)
   - `.squeeze(-1)` → groupsX=(N,1), groupsZ=(N,K)
3. Returns: X=(N,1,2), Z=(N,K,2), groupsX=(N,1), groupsZ=(N,K)

**Step C — vmap kernel calls** (gp.py:954-962):
```python
Kzz = vmap(lambda z, gz: kernel(z, z, gz, gz))(Z_flat, groupsZ_flat)   # (N, K, K)
Kzx = vmap(lambda z, x, gz, gx: kernel(z, x, gz, gx))(Z_flat, X_flat, groupsZ_flat, groupsX_flat)  # (N, K, 1)
# groupsX_flat = (N, 1) where each row = g (hypothetical group)
```

**Step D — `LCGP.reshape_parameters`** (gp.py:775-817):
```python
mu_knn = mu[:, knn_idx]          # (L, N, K)
Lu_knn = Lu[:, knn_idx]          # (L, N, K, R)
Su_knn = Lu_knn @ Lu_knn.T      # (L, N, K, K)
add_jitter(Su_knn, jitter)
L_su = cholesky(Su_knn)          # (L, N, K, K)
add_jitter(Kzz, jitter)
```

**Step E — `WSVGP.forward`** (gp.py:324-394):
```python
L_kzz = cholesky(Kzz)           # (N, K, K)
stacked = cat([mu_knn.unsqueeze(-1), L_su], dim=-1)  # (L, N, K, K+1)
X_sol = solve_triangular(L_kzz, stacked)             # broadcasts (N,K,K) x (L,N,K,K+1)
mu_t = X_sol[..., 0]            # (L, N, K)
W = solve_triangular(L_kzz, Kzx).T  # (N, 1, K)
mean = (W @ mu_t.unsqueeze(-1)).squeeze()  # (L, N)
```

### OUR FUNCTION DOES THE SAME STEPS

Steps D and E are replicated exactly in `_get_groupwise_factors_expanded_K`.
The vmap calls in Step C are also identical.

**CONCLUSION: THE ALGORITHMS ARE IDENTICAL FOR KNN STRATEGY.**

---

## QUESTION 3: WHY DO RESULTS DIFFER AT K=50?

### CANDIDATE A: PROBABILISTIC KNN — DIFFERENT groupsX FOR NEIGHBOR SELECTION

This is the most likely cause of a real difference.

**OLD CODE (`_get_groupwise_factors_batched`) — groupsX = [g, g, g, ..., g]:**
```python
# For group g=2, for a batch of 5 points:
groups_b = np.full(batch_size, g)   # → [2, 2, 2, 2, 2]
knn_idx = calculate_knn(model._prior, coords_b, strategy=neighbors,
                         groupsX=groups_b, ...)
```
Every query point is treated as belonging to group g for KNN weight computation.
Kernel weights: `w(x_i, z_j) = kernel(x_i, z_j, gX=2, gZ=actual_group_j)` for all i.
Each group g gets a DIFFERENT set of neighbors, biased toward same-group training points.

**NEW CODE (`_get_groupwise_factors_expanded_K`) — groupsX = [3, 0, 2, 1, 3, ...]:**
```python
# groups_t contains actual group of each point, e.g. [3, 0, 2, 1, 3, ...]
knn_idx_all = _probabilistic_knn(coords_t, Z, K_post, kernel, multigroup=True,
                                  groupsX=groups_t,  # → [3, 0, 2, 1, 3, ...]
                                  groupsZ=groupsZ.cpu())
```
Every query point uses its ACTUAL group for KNN weight computation.
Kernel weights: `w(x_i, z_j) = kernel(x_i, z_j, gX=actual_group_i, gZ=actual_group_j)`.
ONE shared set of neighbors for all groups g — neighbors do NOT change per group.

**For `strategy='knn'` (FAISS):** groupsX is ignored entirely (L2 distance only)
→ results should be IDENTICAL at K=50 regardless of which groupsX is used.
**For `strategy='probabilistic'`:** groupsX changes kernel weights → different neighbors
→ old and new give legitimately different neighbors, hence different posterior maps.

### CANDIDATE B: FLOATING-POINT ACCUMULATION ORDER

Even for `strategy='knn'` at K=50, the old code calls the full `model._prior(X=..., groupsX=...)` forward pass, while our code replicates the same math manually. Different accumulation order → tiny floating-point differences. Should be negligible (< 1e-5).

### CANDIDATE C: `Kzz` JITTER APPLICATION ORDER

In `WSVGP.forward` at line 341: `L_kzz = cholesky(Kzz)` uses `Kzz` AFTER `reshape_parameters` already applied `add_jitter(Kzz, jitter)`.

In our function: `add_jitter(Kzz, gp.jitter)` then `L_kzz = cholesky(Kzz)`. Same order. ✓

---

## KEY FINDING: FIX NEEDED FOR PROBABILISTIC STRATEGY

The expanded-K function should compute KNN PER GROUP when strategy='probabilistic',
using the hypothetical group g as groupsX for each group's neighbor selection.

For strategy='knn' (FAISS), compute once — groups don't affect L2 distance.

### CORRECTED DESIGN

```python
# For 'knn': compute once (groups don't affect L2 distance)
if strategy == 'knn':
    knn_idx_all = _faiss_knn(coords_t, Z, K_post)[:, :-1]  # (N, K_post)

# For 'probabilistic': must compute PER GROUP (groupsX affects kernel weights)
# knn_idx_all_per_group[g] = (N, K_post)
else:
    knn_idx_per_group = {}
    for g in range(n_groups):
        groups_g = torch.full((N,), g, dtype=torch.long)
        knn_idx_per_group[g] = _probabilistic_knn(
            coords_t, Z, K_post, kernel,
            multigroup=True, groupsX=groups_g, groupsZ=groupsZ.cpu()
        )[:, :-1]
```

Then per chunk, per group: use the appropriate knn_idx.

---

## SUMMARY TABLE

| Aspect | `_get_groupwise_factors_batched` | `_get_groupwise_factors_expanded_K` (current) | Fix needed? |
|--------|----------------------------------|-----------------------------------------------|-------------|
| Neighbor pool | Full Z (all M) | Full Z (all M) | No |
| K for neighbors | K_train (50) | K_post (2000) | By design |
| KNN per group g (knn) | FAISS, groups ignored | FAISS, groups ignored | No |
| KNN per group g (probabilistic) | groupsX=full(g) per group | groupsX=actual groups (WRONG) | YES |
| Kzz computation | vmap over K neighbors | vmap over K neighbors | No |
| Kzx computation | groupsX=full(g) per group | groupsX=full(g) per group | No |
| transform_variables | solve_triangular stack | solve_triangular stack | No |
| Mean formula | W @ mu_t | W @ mu_t | No |
