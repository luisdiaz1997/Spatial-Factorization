# Chunked Expanded-K Posterior for Groupwise Factors (MGGP_LCGP only)

**Scope:** Only MGGP_LCGP (and optionally LCGP). The existing MGGP_SVGP groupwise path (`_get_groupwise_factors_batched` at `analyze.py:584-623`) stays untouched.

---

## Motivation

The current groupwise factor computation uses **K_train=50** neighbors (the training neighbor count). At inference we don't backpropagate, so we can afford K_post=2000+ neighbors for smoother, more accurate groupwise factor maps.

**Key distinction:** K=50 is the **neighbor count**. R=250 is the **Lu rank** (last dim of Lu parameter). They are independent:
- `Lu` shape: `(L=10, M=41783, R=250)` — stored in `outputs/.../Lu.npy`
- `K_train`: stored in `model.pth` hyperparameters as `K: 50`
- Changing K_post only changes the size of the gathered/Cholesky matrices, not R

---

## Verified Shapes (tested with real slideseq MGGP_LCGP model)

```
Prior state dict keys: Z, mu, groupsZ, Lu, kernel.sigma, kernel.lengthscale, 
                       kernel.group_diff_param, kernel.embedding

Lu:                 (10, 41783, 250)    — (L, M, R)
mu:                 (10, 41783)         — (L, M)
Z:                  (41783, 2)          — (M, D)
groupsZ:            (41783,)            — (M,)
kernel.sigma:       scalar (1.0)        — NOT L-batched
kernel.lengthscale: scalar (8.0)        — NOT L-batched
kernel.embedding:   (14, 14)            — (G, G)
K_train:            50                  — from hyperparameters
```

**Critical:** The kernel is NOT batched across L factors. Kernel matrices have NO L dimension. The L dimension is introduced only through `mu` and `Lu` broadcasting in matmuls.

---

## How LCGP Forward Pass Actually Works

Understanding this is essential — the LCGP `reshape_input_data` (`gpzoo/gp.py:758-768`) transforms Z into per-point neighbor subsets, making all kernel matrices K×K (not M×M):

```python
# LCGP.reshape_input_data (gp.py:758-768):
X = X.unsqueeze(-2)   # (N_batch, 2) → (N_batch, 1, 2)
Z = Z[knn_idx]         # (M, 2) → (N_batch, K, 2)   ← KEY: Z is subsetted!
```

This means `forward_kernels` vmaps over N_batch, computing ONE K×K kernel per query point:
- `Kzz`: `(N_batch, K, K)` — no L dim!
- `Kzx`: `(N_batch, K, 1)` — no L dim!
- `Kxx`: `(N_batch, 1)` → squeezed to `(N_batch,)` — no L dim!

Then `reshape_parameters` (`gp.py:775-817`) gathers variational params which DO have L:
- `mu_knn = mu[:, knn_idx]` → `(L, N_batch, K)`
- `Lu_knn = Lu[:, knn_idx]` → `(L, N_batch, K, R)`
- `Su_knn = Lu_knn @ Lu_knn.T` → `(L, N_batch, K, K)`
- `L_su = cholesky(Su_knn)` → `(L, N_batch, K, K)`

In `WSVGP.forward` (`gp.py:324-394`), L broadcasts with kernel dims:
```
L_kzz = cholesky(Kzz)        # (N_batch, K, K)          — no L
Wt = solve_triangular(L_kzz, Kzx)  # (N_batch, K, 1)    — no L
W = Wt.T                           # (N_batch, 1, K)    — no L
# transform_variables (gp.py:496-511, else branch):
stacked = cat([mu.unsqueeze(-1), L_su], dim=-1)  # (L, N_batch, K, K+1)
X = solve_triangular(L_kzz, stacked)   # (N_batch,K,K) broadcasts with (L,N_batch,K,K+1)
mu_t = X[..., 0]                       # (L, N_batch, K) → then (L, N_batch) after W@
```

---

## The Expanded-K Posterior Algorithm

### What's Shared vs Per-Group

For MGGP kernel (`batched_MGGP_Matern32`, `gpzoo/kernels.py:115-170`), neighbor groups are FIXED (training-time assignments). Only the **query point's hypothetical group** changes per group g.

**Shared across all G groups (compute ONCE per chunk):**
1. `knn_idx` — FAISS neighbor indices `(C, K_post)`
2. `mu_knn = mu[:, knn_idx]` → `(L, C, K_post)`
3. `Lu_knn = Lu[:, knn_idx]` → `(L, C, K_post, R)`
4. `Su_knn = Lu_knn @ Lu_knn.T` → `(L, C, K_post, K_post)` + Cholesky → `L_su`
5. `Kzz` via vmap kernel on neighbor coords/groups → `(C, K_post, K_post)` + Cholesky → `L_kzz`
6. `transform_variables` → `mu_t (L, C, K_post)`, `Lu_t (L, C, K_post, K_post)`

**Per-group g (CHEAP — just one kernel eval + two matmuls):**
7. `Kzx_g` = kernel(neighbors, query, fixed_groups, group_g) → `(C, K_post, 1)`
8. `Wt_g = solve_triangular(L_kzz, Kzx_g)` → `(C, K_post, 1)`
9. `W_g = Wt_g.T` → `(C, 1, K_post)`
10. `mean_g = W_g @ mu_t.unsqueeze(-1)` → broadcasts → `(L, C, 1, 1)` → squeeze → `(L, C)`
11. `factors_g = exp(mean_g).T` → `(C, L)`

**Kxx is always sigma^2 regardless of group** (the `diag=True` path in `kernels.py:154-156` returns `sigma**2` expanded, ignoring groups). So it doesn't need per-group computation.

---

## Verified Working Code (tested with torch)

This exact code runs successfully. The key gotcha is `.contiguous()` after `torch.vmap` — `add_jitter` uses `.view()` which requires contiguous memory.

```python
import torch, faiss, numpy as np
from gpzoo.utilities import add_jitter

def _get_groupwise_factors_expanded_K(
    model, coords, n_groups, sort_order,
    K_post=2000, mem_gb=8.0,
):
    """Expanded-K posterior for MGGP_LCGP groupwise factors.
    
    Only for LCGP/MGGP_LCGP models. Uses K_post >> K_train neighbors 
    for smoother posterior. MGGP_SVGP should use _get_groupwise_factors_batched.
    """
    gp = model._prior
    kernel = gp.kernel
    mu = gp.mu                    # (L, M)
    Lu_raw = gp.Lu                # (L, M, R)
    Z = gp.Z                      # (M, 2)
    groupsZ = gp.groupsZ          # (M,)
    
    L = mu.shape[0]
    N = coords.shape[0]
    device = mu.device
    
    # Adaptive chunk size: ~4 live (L, C, K_post, K_post) tensors
    C = max(1, int(mem_gb * 1e9 / (L * K_post**2 * 4 * 4)))
    
    # Build FAISS index once (M=N for LCGP, just 2D coords)
    Z_np = Z.detach().cpu().numpy().astype(np.float32)
    index = faiss.IndexFlatL2(2)
    index.add(Z_np)
    
    # Ensure coords is numpy for FAISS, tensor for torch
    if isinstance(coords, np.ndarray):
        coords_np = coords.astype(np.float32)
        coords_t = torch.from_numpy(coords_np).to(device)
    else:
        coords_np = coords.cpu().numpy().astype(np.float32)
        coords_t = coords.to(device)
    
    result = {g: np.empty((N, L), dtype=np.float32) for g in range(n_groups)}
    n_chunks = (N + C - 1) // C
    
    with torch.no_grad():
        for chunk_i, start in enumerate(range(0, N, C)):
            end = min(start + C, N)
            c = end - start
            print(f"  Chunk {chunk_i+1}/{n_chunks} ({c} points, K_post={K_post})...", 
                  end="\r", flush=True)
            
            # --- Step 1: K_post neighbors via FAISS (ONCE) ---
            _, knn_idx_np = index.search(coords_np[start:end], K_post)  # (c, K_post)
            knn_idx = torch.tensor(knn_idx_np, dtype=torch.long, device=device)
            
            # --- Step 2: Gather variational params (ONCE) ---
            mu_knn = mu[:, knn_idx]                           # (L, c, K_post)
            Lu_knn = Lu_raw[:, knn_idx]                       # (L, c, K_post, R)
            Su_knn = Lu_knn @ Lu_knn.transpose(-2, -1)        # (L, c, K_post, K_post)
            add_jitter(Su_knn, gp.jitter)
            L_su = torch.linalg.cholesky(Su_knn)              # (L, c, K_post, K_post)
            del Lu_knn, Su_knn  # free memory
            
            # --- Step 3: Neighbor kernel Kzz (ONCE, fixed groups) ---
            Z_nbr = Z[knn_idx]                                # (c, K_post, 2)
            gZ_nbr = groupsZ[knn_idx]                         # (c, K_post)
            
            # vmap kernel over c points: each computes (K_post, K_post)
            Kzz = torch.vmap(
                lambda z, gz: kernel(z, z, gz, gz)
            )(Z_nbr, gZ_nbr).contiguous()                     # (c, K_post, K_post)  ← .contiguous()!
            add_jitter(Kzz, gp.jitter)
            L_kzz = torch.linalg.cholesky(Kzz)                # (c, K_post, K_post)
            del Kzz
            
            # --- Step 4: Transform variables (ONCE) ---
            # Replicate transform_variables logic from gp.py:496-511 (else branch)
            stacked = torch.cat([mu_knn.unsqueeze(-1), L_su], dim=-1)  # (L, c, K_post, K_post+1)
            X = torch.linalg.solve_triangular(L_kzz, stacked, upper=False)
            # L_kzz (c, K_post, K_post) broadcasts with stacked (L, c, K_post, K_post+1)
            mu_t = X[..., 0]     # (L, c, K_post)
            Lu_t = X[..., 1:]    # (L, c, K_post, K_post) — only needed if computing variance
            del stacked, X, L_su, mu_knn
            
            # --- Step 5: Per-group posterior (CHEAP per group) ---
            X_chunk = coords_t[start:end].unsqueeze(1)         # (c, 1, 2)
            
            for g in range(n_groups):
                gX_g = torch.full((c, 1), g, dtype=torch.long, device=device)
                
                # Cross-kernel: neighbors (fixed groups) → query (group g)
                Kzx_g = torch.vmap(
                    lambda z, x, gz, gx: kernel(z, x, gz, gx)
                )(Z_nbr, X_chunk, gZ_nbr, gX_g)               # (c, K_post, 1)
                
                Wt_g = torch.linalg.solve_triangular(L_kzz, Kzx_g, upper=False)  # (c, K_post, 1)
                W_g = Wt_g.transpose(-2, -1)                   # (c, 1, K_post)
                
                # Posterior mean: W_g @ mu_t broadcasts (c,1,K) @ (L,c,K,1) → (L,c,1,1)
                mean_g = (W_g @ mu_t.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (L, c)
                
                factors_g = torch.exp(mean_g).T.cpu().numpy()  # (c, L)
                result[g][start:end] = factors_g[:, sort_order]
            
            del L_kzz, Z_nbr, gZ_nbr, mu_t
    
    print()  # newline after \r progress
    return result
```

---

## Memory Analysis

### Dominant Tensors Per Chunk

| Tensor | Shape | Size formula |
|--------|-------|-------------|
| `Su_knn` | `(L, C, K_post, K_post)` | `L * C * K_post^2 * 4` bytes |
| `L_su` | `(L, C, K_post, K_post)` | same |
| `Kzz` | `(C, K_post, K_post)` | `C * K_post^2 * 4` (no L!) |
| `L_kzz` | `(C, K_post, K_post)` | same |
| `stacked` | `(L, C, K_post, K_post+1)` | ~same as Su_knn |
| `X` (solve result) | `(L, C, K_post, K_post+1)` | ~same |
| `Lu_knn` (temporary) | `(L, C, K_post, R)` | `L * C * K_post * R * 4` |

Peak is ~4 tensors of `(L, C, K_post, K_post)` live simultaneously.

**Formula:** `C = floor(mem_gb * 1e9 / (L * K_post^2 * 4 * 4))`

| K_post | mem=8 GB | C (chunk) | Total chunks (N=41783) |
|--------|----------|-----------|------------------------|
| 5000 | 8 GB | 2 | ~20,892 |
| 2000 | 8 GB | 12 | ~3,482 |
| 1000 | 8 GB | 50 | ~836 |
| 500 | 8 GB | 200 | ~209 |

### Recommended Default: K_post=2000

C=12, ~3500 chunks for slideseq. Each `(10, 12, 2000, 2000)` tensor is ~1.9 GB.

---

## Gotchas Discovered During Testing

1. **`.contiguous()` after `torch.vmap`**: `add_jitter` (`gpzoo/utilities.py:694-706`) uses `.view()` which requires contiguous memory. Tensors returned by `torch.vmap` are often non-contiguous. Always call `.contiguous()` on vmap output before `add_jitter`.

2. **Kernel returns NO L dimension**: `batched_MGGP_Matern32` with scalar sigma returns `(K, K)` per query point, NOT `(L, K, K)`. The L dimension comes only from variational params (mu, Lu) via broadcasting in `solve_triangular` and matmuls.

3. **Kernel call convention**: `kernel(Z, X, gZ, gX)` → `forward(X=Z, Z=X, ...)` internally, which vmaps outer over Z arg, inner over X arg. Returns `(|Z|, |X|)` after transpose. So `kernel(Z_nbr, X_query, gZ, gX)` where Z_nbr=(K,2) and X_query=(1,2) returns `(K, 1)`.

4. **`add_jitter` is in-place**: `gpzoo/utilities.py:694-706` modifies the tensor in-place via `K.view(B, N*N)[:, ::N+1] += jitter`. No need to capture return value, but tensor must be contiguous.

5. **FAISS needs float32 numpy**: `index.add()` and `index.search()` require `float32` numpy arrays, not tensors.

---

## Implementation Steps

### 1. Add `_get_groupwise_factors_expanded_K` to `analyze.py`

**Where:** After `_get_groupwise_factors_batched` (after line 623).

**Function:** Exactly the code in the "Verified Working Code" section above.

### 2. Modify the call site in `analyze.py:790-800`

**Current code** (`analyze.py:790-800`):
```python
if config.groups and spatial:
    print(f"\nComputing groupwise conditional posterior ({data.n_groups} groups)...")
    groupwise_dir = model_dir / "groupwise_factors"
    groupwise_dir.mkdir(exist_ok=True)
    groupwise = _get_groupwise_factors_batched(
        model, coords, data.n_groups, sort_order, analyze_batch_size
    )
```

**New code:**
```python
if config.groups and spatial:
    print(f"\nComputing groupwise conditional posterior ({data.n_groups} groups)...")
    groupwise_dir = model_dir / "groupwise_factors"
    groupwise_dir.mkdir(exist_ok=True)
    
    # Use expanded-K posterior for LCGP models when posterior_K is set
    posterior_K = config.training.get("posterior_K")
    is_lcgp = not hasattr(model._prior.Lu, '_raw')  # same check as line 770
    
    if is_lcgp and posterior_K and posterior_K > config.model.get("K", 50):
        mem_gb = config.training.get("posterior_mem_gb", 8.0)
        groupwise = _get_groupwise_factors_expanded_K(
            model, coords, data.n_groups, sort_order,
            K_post=posterior_K, mem_gb=mem_gb,
        )
    else:
        groupwise = _get_groupwise_factors_batched(
            model, coords, data.n_groups, sort_order, analyze_batch_size
        )
```

### 3. Add CLI flag `--posterior-k` to analyze command

**File:** `cli.py:48-55`

Add `--posterior-k` option to the `analyze` command:
```python
@cli.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True))
@click.option("--probabilistic", is_flag=True, default=False, ...)
@click.option("--posterior-k", default=None, type=int,
              help="Expanded K for LCGP posterior (overrides config posterior_K)")
def analyze(config, probabilistic, posterior_k):
    from .commands import analyze as cmd
    cmd.run(config, probabilistic=probabilistic, posterior_k=posterior_k)
```

### 4. Update `analyze.run()` signature

**File:** `analyze.py:653`

```python
def run(config_path: str, probabilistic: bool = False, posterior_k: int | None = None):
```

If `posterior_k` is passed from CLI, override `config.training["posterior_K"]` before the call site.

### 5. Config support

**File:** `config.py` — no changes needed. `posterior_K` is just read from `config.training` dict via `.get()`.

**Example YAML:**
```yaml
training:
  max_iter: 20000
  posterior_K: 2000        # expanded K for LCGP posterior (optional)
  posterior_mem_gb: 8.0    # GPU memory budget (optional, default 8.0)
```

### 6. Wire `--posterior-k` through `run` pipeline and runner

**File:** `cli.py:149-259` — add `--posterior-k` to `run_pipeline` command, pass through to analyze stage.

**File:** `runner.py` — pass `posterior_k` when spawning analyze subprocess (add to command args if set).

---

## Code References (with verified line numbers)

### Files to MODIFY

| File | Lines | What to change |
|------|-------|----------------|
| `spatial_factorization/commands/analyze.py` | After 623 | Add `_get_groupwise_factors_expanded_K` function |
| `spatial_factorization/commands/analyze.py` | 653 | Add `posterior_k` param to `run()` |
| `spatial_factorization/commands/analyze.py` | 790-800 | Branch: expanded-K for LCGP, batched for SVGP |
| `spatial_factorization/cli.py` | 48-55 | Add `--posterior-k` to `analyze` command |
| `spatial_factorization/cli.py` | 149-259 | Add `--posterior-k` to `run_pipeline`, pass to analyze |

### Files to READ (context only, do NOT modify)

| File | Lines | What it contains |
|------|-------|------------------|
| `gpzoo/gp.py` | 758-768 | `LCGP.reshape_input_data` — how Z gets subsetted to `Z[knn_idx]` |
| `gpzoo/gp.py` | 775-817 | `LCGP.reshape_parameters` — gather mu/Lu at knn_idx, Su=LuLu^T, cholesky |
| `gpzoo/gp.py` | 324-394 | `WSVGP.forward` — full posterior: cholesky(Kzz), solve, transform, W@mu |
| `gpzoo/gp.py` | 496-511 | `SVGP.transform_variables` (else branch) — `stacked = cat([mu, Lu])`, solve_triangular, split |
| `gpzoo/gp.py` | 770-773 | `LCGP.apply_constraints` — returns raw mu, Lu (no CholeskyParameter transform) |
| `gpzoo/gp.py` | 943-969 | `MGGP.forward_kernels` — vmapped kernel with groupsX/groupsZ |
| `gpzoo/kernels.py` | 129-149 | `batched_MGGP_Matern32.covariance` — scalar-valued, group attenuation formula |
| `gpzoo/kernels.py` | 152-170 | `batched_MGGP_Matern32.forward` — vmap over Z × X, returns `(|Z|, |X|)` after transpose |
| `gpzoo/kernels.py` | 154-156 | `diag=True` branch — returns `sigma**2` expanded, group-independent |
| `gpzoo/utilities.py` | 694-706 | `add_jitter` — in-place, requires contiguous tensor (uses `.view()`) |
| `PNMF/transforms.py` | 24-78 | `_get_spatial_qF` — current KNN + forward for LCGP (K=K_train) |
| `PNMF/models.py` | 718-764 | LCGP initialization — K, R=estimate_lcgp_rank, Lu shape (L,M,R) |
| `analyze.py` | 770 | `is_lcgp = not hasattr(gp.Lu, '_raw')` — how to detect LCGP vs SVGP |

### Dependencies

| Import | Source | Notes |
|--------|--------|-------|
| `faiss` | `pip install faiss-cpu` | Already in env (v1.13.2). Use `faiss.IndexFlatL2(2)` |
| `gpzoo.utilities.add_jitter` | GPzoo | In-place jitter on diagonal, needs `.contiguous()` input |
| `torch.linalg.cholesky` | PyTorch | Broadcasts leading dims |
| `torch.linalg.solve_triangular` | PyTorch | Broadcasts: `(C,K,K)` with `(L,C,K,K+1)` works |
| `torch.vmap` | PyTorch | Nested vmap OK (kernel already uses internal vmap) |

---

## Open Questions

1. **Expanded-K for unconditional factors too?** Currently `_get_factors_batched` (`analyze.py:692-694`) also uses K_train=50. Same approach would apply but is a separate code path. Could be a follow-up.

2. **CPU vs GPU:** Cholesky on `(L, C, K_post, K_post)` is compute-heavy. Current code runs on whatever device the model is on. GPU ~10x faster for large Cholesky.

3. **Default K_post:** Plan says 2000. Could also be `null` (disabled) by default and only enabled via CLI `--posterior-k 2000` or config `posterior_K: 2000`.

---

## Testing

After implementation, test on the already-trained slideseq MGGP_LCGP model:

```bash
# Run analyze with expanded-K posterior (K_post=2000)
spatial_factorization analyze -c configs/slideseq/mggp_lcgp.yaml --posterior-k 2000

# Then regenerate figures (--no-heatmap to skip slow heatmaps)
spatial_factorization figures -c configs/slideseq/mggp_lcgp.yaml --no-heatmap
```

**What to verify:**
- Groupwise factor files are written to `outputs/slideseq/mggp_lcgp/groupwise_factors/group_{0..13}.npy`
- Each file has shape `(41783, 10)`
- Factors are non-negative (exp-space)
- Spatial plots in figures stage look smoother than the K=50 baseline
- No OOM on GPU (should stay within ~8 GB budget with C=12 chunks)
