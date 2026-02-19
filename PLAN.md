# Bug Fixes: Multi-Dataset Run Failures

Findings from the first `run all -c configs/ --config-name general_test.yaml` run
across all datasets. Five distinct bugs were identified.

---

## Bug 1 — Analyze OOM: SVGP forward pass on full N

**Affected:** `merfish_svgp`, `colon_svgp`, `liver_svgp` (diseased)
**Stage:** analyze

**Root cause:** `analyze.py` calls `model.get_factors(Y, coordinates)` which runs the
SVGP forward on all N points at once. This materializes an `(L, N, M)` or `(M, N)`
tensor (and the full K(Z,X) kernel matrix) that exceeds GPU memory for large N:

| Dataset | N | Error |
|---------|---|-------|
| merfish | 73 K | tried to allocate 9.88 GiB — `gp.py:376` `(W @ Lu) ** 2` |
| colon | 120 K | tried to allocate 16.08 GiB — same line |
| liver (diseased) | 310 K | tried to allocate 3.46 GiB — `kernels.py:53` vmap K(Z,X) |

**Fix:** Chunk the `get_factors` / `_get_spatial_qF` call in `analyze.py` over N
with a configurable `analyze_batch_size` (e.g., default 10 000). Concatenate
factors and scales across chunks. The `y_batch_size` and model training batching
are irrelevant here — this is inference-time memory.

---

## Bug 2 — Analyze OOM: LCGP forward pass on full N

**Affected:** `merfish_lcgp`, `colon_lcgp`
**Stage:** analyze

**Root cause:** LCGP `reshape_parameters` materializes `[L, N, K, K]` for `Su_knn`
before the Cholesky decomposition:

| Dataset | N | K | Shape | Error |
|---------|---|---|-------|-------|
| merfish | 73 K | 50 | [10, 73655, 50, 50] | 8.23 GiB — `gp.py:825` cholesky |
| colon | 120 K | 50 | [10, 119906, 50, 50] | 13.40 GiB — `gp.py:821` Lu_knn @ Lu_knn.T |

**Fix:** Same batching solution as Bug 1 — chunk N in the analyze forward pass.
Both SVGP and LCGP paths use `get_factors`, so a single batching loop covers both.

---

## Bug 3 — Train crash: MGGP_SVGP inducing allocation mismatch

**Affected:** `colon_mggp_svgp`
**Stage:** train

**Root cause:** Colon has many small groups. The `inducing_allocation='derived'`
method fails for 5 groups (`[2, 8, 38, 43, 48]`) and falls back to `'proportional'`,
but the actual allocated inducing points (2796) is less than `num_inducing=3000`.
This causes a shape mismatch in `transform_variables`:

```
RuntimeError: linalg.solve_triangular: Incompatible shapes of A and B
(2796×2796 and 3000×36012)
```

The Cholesky `L` has shape `(2796, 2796)` but `stacked` is `(3000, 36012)`.

**Applied patch (PNMF `models.py`):** After the `derived` allocation call, update
`M = Z.shape[0]` so the rest of training uses the actual returned count. This stops
the crash but does not fix the under-allocation itself.

**Root cause of the under-allocation (found in `gpzoo/model_utilities.py`
`mggp_kmeans_inducing_points`, lines ~293–340):**

When `derived` fails to assign any k-means centers to some groups (the "missing
groups"), the fallback logic computes:

```python
target_per_group = (total_points - n_replace) * group_counts / total_available
```

`group_counts` and `total_available` include **all** groups, even the missing ones.
The missing groups have `derived_group_counts = 0` but a positive `target_per_group`,
so they don't appear in `excess_mask`. Their "allocation" is silently subtracted from
the budget of the non-missing groups, making those groups look like they have more
excess than they do. The removal loop therefore strips out:

```
n_remove_total ≈ n_replace + floor(total_points * missing_fraction)
```

…but only `n_replace` points are added back. The deficit is roughly
`floor(M * sum(missing_group_cells) / N)`. For colon with 5 missing groups
this was 3000 − 2796 = 204.

**Proper fix (to be done in GPzoo in a separate session):** Compute
`target_per_group` only over the non-missing groups so their excess is calculated
correctly, then allocate the full remaining budget among them. The missing groups
get their points exclusively from the fallback k-means step.

---

## Bug 4 — Analyze crash: K not saved to checkpoint (LCGP/MGGP_LCGP)

**Affected:** `osmfish_mggp_lcgp`
**Stage:** analyze

**Root cause:** `_load_model()` in `analyze.py` reconstructs the LCGP/MGGP_LCGP
prior using a hardcoded or config-derived `K=50`, but the saved `model.pth` has
`Lu` shaped `[10, 4839, 20]` (K=20, from a prior run with different config).
The `state_dict` load then fails:

```
RuntimeError: size mismatch for Lu: copying a param with shape
torch.Size([10, 4839, 20]) from checkpoint,
the shape in current model is torch.Size([10, 4839, 50]).
```

**Fix:**
1. In `train.py` `_save_model()`, add `K` to the saved hyperparameters when
   `config.local` is True.
2. In `analyze.py` `_load_model()`, read `K` from `state["hyperparameters"]`
   when reconstructing an LCGP or MGGP_LCGP prior.

---

## Bug 5 — Runner job name collision: liver_healthy vs liver_diseased

**Affected:** All liver jobs (healthy silently overwritten by diseased in status tracking)
**Stage:** all

**Root cause:** Both `configs/liver/healthy/general_test.yaml` and
`configs/liver/diseased/general_test.yaml` have `dataset: liver`. When
`generate_configs` runs on each, it sets `model_config.name = f"{config.dataset}_{variant['name']}"`,
producing identical names (`liver_pnmf`, `liver_svgp`, etc.) for both.

In the runner:
- `self.run_status.jobs[job.name]` is a dict → healthy entries overwritten by diseased
- `StatusManager.jobs` is a dict keyed by `f"{config.name}_train"` → same collision

The result: liver_healthy jobs run but their status is tracked under the diseased
names, making the final status report incorrect and potentially causing GPU slot
accounting bugs.

**Fix:** Derive a unique job name from the output directory rather than from
`config.name`. E.g., in Phase 3 of `runner.py`:

```python
# Use output_dir to derive a unique prefix (avoids dataset name collisions)
out_path = Path(config.output_dir)
# e.g. "outputs/liver_test/healthy" → "liver_test_healthy"
unique_prefix = "_".join(out_path.parts[1:])  # skip "outputs/"
job_name = f"{unique_prefix}_{config.model_name}"
```

This is independent of `config.dataset` and guaranteed unique across all configs.

---

## Summary Table

| # | Bug | Datasets | Stage | Fix location |
|---|-----|----------|-------|-------------|
| 1 | SVGP analyze OOM (full N) | merfish, colon, liver_dis | analyze | `analyze.py` — batch `get_factors` over N |
| 2 | LCGP analyze OOM (Su_knn) | merfish, colon | analyze | same fix as #1 |
| 3 | MGGP_SVGP train crash (alloc mismatch) | colon | train | **Patched** in PNMF (`M = Z.shape[0]`); root cause in GPzoo `mggp_kmeans_inducing_points` — fix `target_per_group` to exclude missing groups |
| 4 | LCGP/MGGP_LCGP K not saved | osmfish | analyze | `train.py` save K; `analyze.py` load K |
| 5 | Liver job name collision | liver healthy+diseased | runner | `runner.py` — use output_dir for job name |

Bugs 1+2 share the same fix (batched analyze). Bug 4 is a one-line save + one-line
load. Bug 5 is a naming change in the runner. Bug 3 may require a GPzoo fix.
