# Probabilistic KNN: Batched Kernel Evaluation

## Problem

`_probabilistic_knn` computes the full `(N, M)` kernel matrix to sample K neighbors per point. For LCGP where `M = N`, this becomes `(N, N)`:

| Dataset | N | Matrix size | Memory |
|---------|---|-------------|--------|
| slideseq | 41K | 41K x 41K | ~6.7 GB |
| liver healthy | 90K | 90K x 90K | ~32 GB |
| liver diseased | 310K | 310K x 310K | ~384 GB |
| colon | 1.2M | 1.2M x 1.2M | ~5.7 TB |

Each element is float32 (4 bytes), so memory = `N * M * 4`.

## Goal

Chunk the kernel evaluation row-wise so we never exceed ~50 GB of memory, while avoiding a Python-level per-row loop.

## Strategy: Row-Batched Kernel + Sampling

For each chunk of B rows:
1. Compute `kernel(X_chunk, Z)` -> `(B, M)` weights
2. Zero out self-indices if applicable
3. Normalize rows
4. Sample K neighbors per row (Gumbel-max)
5. Concatenate results

This keeps peak memory at `B * M * 4` bytes instead of `N * M * 4`.

## Batch Size Formula

Given a memory budget `mem_gb` (default 50.0 GB):

```
B = floor(mem_gb * 2.5e8 / N)
B = clamp(B, min=1, max=N)
```

Derivation: each row of the `(B, M)` kernel matrix costs `M * 4` bytes = `N * 4` bytes (since M = N for LCGP). So `B = budget_bytes / (N * 4)` = `(mem_gb * 1e9) / (N * 4)` = `mem_gb * 2.5e8 / N`.

| Dataset | N | B (50 GB budget) | Chunks |
|---------|---|-------------------|--------|
| slideseq | 41K | ~305K (1 chunk) | 1 |
| liver healthy | 90K | ~139K (1 chunk) | 1 |
| liver diseased | 310K | ~40K | 8 |
| colon | 1.2M | ~10K | 120 |

## Implementation Plan

1. Add `mem_gb: float = 50.0` parameter to `_probabilistic_knn`
2. Compute `B = clamp(floor(mem_gb * 2.5e8 / M), 1, N)` where M = Z.shape[0]
3. If `B >= N`, fall through to current single-shot path (no overhead for small datasets)
4. Otherwise, loop over chunks of size B:
   - `X_chunk = X[start:end]`
   - `weights_chunk = kernel(X_chunk, Z, ...)` -> `(B, M)`
   - Zero self-indices for rows in `[start, end)` if `N == M`
   - Normalize + Gumbel-max sample -> `(B, K)` indices
   - Append to results list
5. `torch.cat` all chunks, prepend self-index column -> `(N, K+1)`

## Files to Modify

### Primary change: `GPzoo/gpzoo/knn_utilities.py`

The only file that needs code changes. Everything is self-contained here.

**`_probabilistic_knn()` (line 38-84)** — add `mem_gb` param, implement chunked loop:
- Compute `B = clamp(floor(mem_gb * 2.5e8 / M), 1, N)`
- If `B >= N`: current single-shot path (no change for small datasets)
- Else: loop over `ceil(N/B)` chunks, each computing `kernel(X[start:end], Z)` -> `(chunk_size, M)`, then normalize + Gumbel-max sample -> `(chunk_size, K)`, then `torch.cat` all chunks

**`_sample_without_replacement()` (line 30-35)** — no change needed, already works on arbitrary `(B, M)` input

**`calculate_knn()` (line 87-112)** — thread `mem_gb` param through to `_probabilistic_knn`

### No changes needed (callers)

These all call `calculate_knn()` which dispatches internally. The `mem_gb` default handles everything:

| File | Line(s) | Strategy | Why no change |
|------|---------|----------|---------------|
| `GPzoo/gpzoo/training_utilities.py` | 301, 567, 712, 983 | `"knn"` | FAISS path, unaffected |
| `GPzoo/gpzoo/utilities.py` | 221 | `"knn"` | FAISS path, unaffected |
| `GPzoo/gpzoo/models/nsf.py` | 246, 248 | `"knn"` | FAISS path, unaffected |
| `GPzoo/gpzoo/datasets/tenxvisium/generate_annotations.py` | 166, 170, 200 | `"knn"` | FAISS path, unaffected |
| `PNMF/models.py` | 758 | `self.neighbors` | Uses `calculate_knn`, gets `mem_gb` default |
| `PNMF/models.py` | 1314 | `self.neighbors` | Uses `calculate_knn`, gets `mem_gb` default |
| `PNMF/transforms.py` | 67 | `neighbors` | Uses `calculate_knn`, gets `mem_gb` default |
| `Spatial-Factorization/commands/analyze.py` | 155 | `neighbors_strategy` | Uses `calculate_knn`, gets `mem_gb` default |

### Notebooks (legacy, use old `model.prior.calculate_knn()`)

Many notebooks in `GPzoo/notebooks/` still call the removed method directly. These are historical and won't be updated as part of this fix.

## Notes

- The kernel `forward()` uses `torch.vmap` internally, so chunking at the caller level avoids the vmap trying to allocate the full matrix
- For non-LCGP (SVGP where M << N), the matrix is `(N, M)` which is already small -- the formula naturally gives B = N (single chunk)
- `mem_gb` exposed in `calculate_knn` signature so callers can tune if needed
- No changes needed to the `_faiss_knn` path (FAISS handles large datasets fine)
