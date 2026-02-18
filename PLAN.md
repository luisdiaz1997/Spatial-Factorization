# LCGP Integration Plan

## Context

The PNMF package's `LCGP` branch adds support for Locally Conditioned GPs (LCGP and MGGP_LCGP) as alternatives to SVGP/MGGP_SVGP. LCGP uses all data points as inducing points with a low-rank+diagonal covariance approximation (O(MR) instead of O(M²)), and locally conditioned KL divergence. GPzoo already has `LCGP` and `MGGP_LCGP` classes on `main`.

The key PNMF API change: `local=True` alongside `spatial=True` selects LCGP instead of SVGP. PNMF auto-derives the prior type from `spatial`, `multigroup`, and `local` flags:
- `spatial=True, local=False, multigroup=False` → SVGP
- `spatial=True, local=False, multigroup=True` → MGGP_SVGP
- `spatial=True, local=True, multigroup=False` → LCGP
- `spatial=True, local=True, multigroup=True` → MGGP_LCGP

This plan adds LCGP support to the Spatial-Factorization pipeline so that `generate`, `train`, `analyze`, and `figures` all work with `lcgp` and `mggp_lcgp` model variants.

---

## Prerequisites

- [x] Switch PNMF to `LCGP` branch: `cd ../Probabilistic-NMF && git checkout LCGP`
- [x] Create branch: `git checkout -b LCGP` (in Spatial-Factorization)

---

## 1. Update `generate.py` — Add LCGP variants

**File:** `spatial_factorization/generate.py`

### 1a. Add LCGP model variants (line 17-21)
Add `local` key to all existing entries (set `False`), and add two new entries with `local: True`:
- `{"name": "lcgp", "spatial": True, "groups": False, "prior": "LCGP", "local": True}`
- `{"name": "mggp_lcgp", "spatial": True, "groups": True, "prior": "LCGP", "local": True}`

### 1b. Add field sets (after line 48)
Add `SVGP_ONLY_FIELDS`: `{"num_inducing", "cholesky_mode", "diagonal_only"}`
Add `LCGP_MODEL_FIELDS`: `{"local", "K", "rank", "low_rank_mode", "precompute_knn"}`

### 1c. Update `_generate_model_config()` (line 99-152)
In the spatial branch (line 124-136):
- Check `variant["local"]` to decide which fields to include
- When `local=True`: set `model_dict["local"] = True`, include `LCGP_MODEL_FIELDS`, exclude `SVGP_ONLY_FIELDS`
- When `local=False`: include `SVGP_ONLY_FIELDS` as currently done, do NOT include LCGP fields

---

## 2. Update `config.py` — Pass LCGP params to PNMF

**File:** `spatial_factorization/config.py`

### 2a. Add `local` property (after line 81)
Returns `self.model.get("local", False)`.

### 2b. Update `model_name` property (line 84-96)
Currently uses `prior` field for naming. Since `local=True` means LCGP, update logic:
- `spatial=True, local=True, groups=False` → `"lcgp"`
- `spatial=True, local=True, groups=True` → `"mggp_lcgp"`

The existing logic already handles this via `prior.lower()`, but confirm `prior` is set to `"LCGP"` in LCGP configs.

### 2c. Update `to_pnmf_kwargs()` (line 98-140)
Inside the `if self.model.get("spatial", False):` block (line 110-122):
- Always pass: `kwargs["local"] = self.model.get("local", False)`
- When `local=True`, add LCGP-specific params: `K`, `rank`, `low_rank_mode`, `precompute_knn`
- When `local=True`, skip: `num_inducing`, `cholesky_mode`, `diagonal_only`, `inducing_allocation`
- When `local=False`, keep existing SVGP params as-is

---

## 3. Update `general.yaml` — Add LCGP parameters

**File:** `configs/slideseq/general.yaml`

Add LCGP section to model block (after line 31, before group params):
```yaml
  # LCGP params (lcgp/mggp_lcgp only)
  K: 50
  rank: null
  low_rank_mode: softplus
  precompute_knn: true
```

Note: `local` is NOT in general.yaml (it's set per-variant by `generate.py`).

---

## 4. Update `train.py` — LCGP-aware training

**File:** `spatial_factorization/commands/train.py`

### 4a. Update spatial info prints (line 110-112)
When `config.local`:
- Print LCGP-specific info: K, rank
- Skip printing `num_inducing` (irrelevant for LCGP since M=N)

### 4b. Save `local` flag in hyperparameters (line 42-46)
In `_save_model()`, when `config.spatial`, also save:
- `state["hyperparameters"]["local"] = config.local`

This is needed by `_load_model()` to reconstruct the correct GP class.

---

## 5. Update `analyze.py` — LCGP model loading and Lu saving

**File:** `spatial_factorization/commands/analyze.py`

### 5a. Update `_load_model()` (line 35-166)
Add LCGP detection in the spatial branch (line 75-127).

**Detection:** Check for `Lu.diag._raw` key in `prior_sd` — if present, it's LCGP; if `Lu._raw` is present, it's SVGP/MGGP_SVGP.

**LCGP reconstruction path** (new branch alongside the existing SVGP path):
- Import `LCGP`, `MGGP_LCGP` from `gpzoo.gp` and `LowRankPlusDiagonal` from `gpzoo.modules`
- Extract `rank` from `prior_sd["Lu.V._raw"]` shape: `(L, M, R)` → R is rank
- Extract `K` from `hyperparams.get("K", 50)`
- Detect multigroup via `"groupsZ" in prior_sd` (same as SVGP path)
- Create `LCGP`/`MGGP_LCGP` with appropriate kernel + params
- Replace `gp.Lu` with `LowRankPlusDiagonal(m=M, rank=R, batch_size=L)`
- Set `gp.mu = nn.Parameter(...)`
- Call `gp.load_state_dict(prior_sd)`
- Compute KNN indices: `gp.knn_idx = gp.calculate_knn(Z)[:, :-1]`; `gp.knn_idz = gp.knn_idx`
- Set `model.local = True` on the PNMF wrapper

### 5b. Update Lu saving (line 594-607)
LCGP's Lu is `LowRankPlusDiagonal` — calling `.data` would produce `(L, N, N)` which is huge. Save components separately instead:

Detect LCGP via `hasattr(gp.Lu, 'V')`:
- **LCGP:** Save `Lu_diag.npy` `(L, M)` and `Lu_V.npy` `(L, M, R)` (reordered by Moran's I)
- **SVGP:** Keep existing `Lu.pt` save `(L, M, M)`

Both paths still save `Z.npy` and optionally `groupsZ.npy`.

### 5c. `_get_spatial_qF` for LCGP
PNMF's `transforms._get_spatial_qF` (in `PNMF/transforms.py`) already handles LCGP — it checks `hasattr(model, 'local') and model.local` and sets KNN indices before calling forward. No changes needed here, just ensure `model.local = True` is set in `_load_model()`.

---

## 6. Update `figures.py` — LCGP-aware figure loading

**File:** `spatial_factorization/commands/figures.py`

### 6a. Update `_load_analysis_results()` (line 31-83)
Add loading of LCGP-specific Lu files:
- Check for `Lu_diag.npy` and `Lu_V.npy` as alternative to `Lu.pt`
- Store as `results["Lu_diag"]` and `results["Lu_V"]`

### 6b. Points plot — skip inducing point overlay for LCGP
LCGP uses ALL data points as inducing (M=N), so overlaying inducing points is uninformative. Detect LCGP by checking if `Z.npy` shape matches `factors.npy` shape (both have N rows). When detected, skip inducing point markers on `points.png`.

Find the `points.png` plotting function and add this check.

---

## 7. Runner — No changes needed

**File:** `spatial_factorization/runner.py`

Runner iterates over whatever `generate_configs()` returns. Since we add LCGP variants to `MODEL_VARIANTS`, they'll automatically be included in `run all`.

**Note:** LCGP uses M=N inducing points and may need more GPU memory. The current `GPU_MEMORY_BUDGET_GB = 11` (line 28) may need adjustment, but leave as-is for now and test.

---

## 8. Create test configs

### 8a. `configs/slideseq/lcgp_test.yaml`
Based on existing `svgp_no_groups_test.yaml` pattern:
- `spatial: true`, `local: true`, `groups: false`, `prior: LCGP`
- `max_iter: 10`, K: 50, rank: null, low_rank_mode: softplus
- Same preprocessing, training batch sizes, etc. as other test configs

### 8b. `configs/slideseq/mggp_lcgp_test.yaml`
Based on existing `svgp_test.yaml` pattern:
- `spatial: true`, `local: true`, `groups: true`, `prior: LCGP`
- `max_iter: 10`, same LCGP params

---

## 9. Delete old `configs/slideseq/lcgp.yaml`

The existing `lcgp.yaml` uses an outdated config format (non-standard fields like `gp_class`, `output.dir`). Remove it — `generate` will create a proper one.

---

## 10. Update `CLAUDE.md`

Add LCGP documentation:
- Add LCGP to the model variants table
- Document `local` config field
- Add LCGP-specific params to config reference
- Add lcgp/mggp_lcgp to output directory naming table
- Add lcgp_test.yaml and mggp_lcgp_test.yaml to available configs table

---

## File Change Summary

| File | Change |
|------|--------|
| `spatial_factorization/generate.py` | Add LCGP variants + field sets + config filtering |
| `spatial_factorization/config.py` | Add `local` property + LCGP params in `to_pnmf_kwargs()` |
| `configs/slideseq/general.yaml` | Add LCGP params section |
| `spatial_factorization/commands/train.py` | LCGP print info + save `local` in hyperparameters |
| `spatial_factorization/commands/analyze.py` | LCGP model loading + Lu component saving |
| `spatial_factorization/commands/figures.py` | LCGP Lu loading + skip inducing overlay |
| `configs/slideseq/lcgp_test.yaml` | New test config |
| `configs/slideseq/mggp_lcgp_test.yaml` | New test config |
| `configs/slideseq/lcgp.yaml` | Delete (outdated format) |
| `CLAUDE.md` | Add LCGP documentation |

---

## Verification

1. `spatial_factorization generate -c configs/slideseq/general.yaml` → produces 5 configs (pnmf, svgp, mggp_svgp, lcgp, mggp_lcgp)
2. `spatial_factorization run train analyze figures -c configs/slideseq/lcgp_test.yaml` → full pipeline on LCGP
3. `spatial_factorization run train analyze figures -c configs/slideseq/mggp_lcgp_test.yaml` → full pipeline on MGGP_LCGP
4. `spatial_factorization run all -c configs/slideseq/general.yaml --dry-run` → shows 5 jobs planned
5. Verify existing SVGP/MGGP_SVGP/PNMF pipelines still work unchanged

---

## Key PNMF LCGP API Reference (for implementation)

These are the PNMF changes on the LCGP branch relevant to our integration:

| What | Where | Details |
|------|-------|---------|
| `local` param | `PNMF.__init__()` (models.py:416) | New flag, default `False` |
| LCGP params | `PNMF.__init__()` (models.py:416-420) | `K=50, rank=None, low_rank_mode='softplus', precompute_knn=True` |
| Prior derivation | `PNMF.__init__()` (models.py:466-470) | `spatial + local → 'LCGP'`, `spatial + not local → 'SVGP'` |
| `forward_train()` | `LCGP` (gpzoo/gp.py:823-868) | Training-time forward, returns `(qF, None, None)` |
| `kl_divergence_full()` | `LCGP` (gpzoo/gp.py:870-951) | Locally conditioned KL |
| `LowRankPlusDiagonal` | gpzoo/modules.py:393-575 | `S = D + VV^T`, params: `diag._raw (L,M)`, `V._raw (L,M,R)` |
| LCGP state dict keys | `prior.Lu.diag._raw`, `prior.Lu.V._raw` | vs SVGP's `prior.Lu._raw` |
| KNN setup | PNMF models.py:940-943 | `_knn_idx = gp.knn_idz.clone()` stored after `_create_spatial_prior()` |
| Transform KNN | PNMF transforms.py:58-62 | `_get_spatial_qF` checks `hasattr(model, 'local') and model.local` |
