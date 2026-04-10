# Conditional Posterior Plan

## Overview

For MGGP models (`mggp_svgp`, `mggp_lcgp`), the GP prior is **group-aware**: the kernel encodes
inter-group correlations via `group_diff_param`. This means we can query the posterior
with **all cells assigned to a single group** — a conditional posterior that answers:
"what would the factor map look like if every cell were of cell type g?"

Non-MGGP models (`pnmf`, `svgp`, `lcgp`) have no group dimension in the prior, so this is not meaningful for them.

---

## Part 1 — What the Conditional Posterior Is

For each group `g` in `{0, ..., G-1}`:

```python
groupsX_g = torch.full_like(groupsX, fill_value=g)   # all cells forced to group g
qF_g, _, _ = model._prior.forward(X, groupsX=groupsX_g)
factors_g = torch.exp(qF_g.mean)                      # shape (L, N), exp-space
```

This is the GP predictive mean evaluated at all N spatial locations, **under the kernel
conditioned on group g**. The result is a full spatial factor map per group.

For `mggp_svgp`: standard SVGP predictive — call `model._prior.forward(X, groupsX=groupsX_g)`.

For `mggp_lcgp`: same call, but KNN must be precomputed against `gp.Z` (all N training
points) before calling forward. The batched GP forward pass already handles this in analyze;
the same batching logic applies here.

---

## Part 2 — Storage (Analyze Stage)

Computed and saved in the **analyze stage**, after the unconditional factors are already computed.

### Output files

```
outputs/<dataset>/mggp_svgp/
└── groupwise_factors/
    ├── group_0.npy     # (L, N) exp-space posterior means, all cells forced to group 0
    ├── group_1.npy
    ├── ...
    └── group_{G-1}.npy

outputs/<dataset>/mggp_lcgp/
└── groupwise_factors/
    ├── group_0.npy
    ├── ...
    └── group_{G-1}.npy
```

Each file is shape `(L, N)` float32 — same layout as `factors.npy` (which is `(N, L)`,
transposed). Match whatever convention `factors.npy` uses.

### When to compute

- Only when `config.groups == True` (i.e., model is an MGGP variant)
- After unconditional `(factors, scales)` are already in memory
- Use the same batched forward pass as unconditional factors (`analyze_batch_size`)
- Factor ordering (Moran's I sort) should be applied to groupwise factors too, so indices are consistent

### What to skip

- `pnmf`, `svgp`, `lcgp`: skip entirely — no group dimension in prior

---

## Part 3 — Figures

### Layout (from notebook pattern)

Grid: `(G + 1) rows × (L + 1) columns`

```
         [empty]  Factor 0   Factor 1  ...  Factor L-1
row 0:   [off]   [uncond]   [uncond]  ...  [uncond]     ← unconditional posterior (all N cells)
row 1:   [loc]   [cond g0]  [cond g0] ...  [cond g0]   ← group 0 conditional
row 2:   [loc]   [cond g1]  [cond g1] ...  [cond g1]   ← group 1 conditional
...
row G:   [loc]   [cond gG-1]...                         ← group G-1 conditional
```

- **Column 0, row 0**: empty / axis off
- **Column 0, rows 1..G**: group location panel — scatter of all N cells using a grayscale
  colormap; cells belonging to group `g` are highlighted (high value), all others dimmed
  (low value). Row label = group name (e.g. `"Astrocytes"`)
- **Columns 1..L, row 0**: unconditional factor means (same as `factors_spatial.png`)
- **Columns 1..L, rows 1..G**: conditional posterior factor means for group `g`, turbo colormap

Column headers = `"Factor 0"`, `"Factor 1"`, ...

### Output file

```
outputs/<dataset>/mggp_svgp/figures/groupwise_factors.png
outputs/<dataset>/mggp_lcgp/figures/groupwise_factors.png
```

### Color scale

- Shared `vmin`/`vmax` across all factor panels (rows 0..G, columns 1..L), derived from
  percentiles of the unconditional factors — so conditional and unconditional panels are
  directly comparable
- Group location panels use a separate binary-ish colormap (e.g. `copper` or `gray`),
  not shared with factor panels

### Point size

Use the same `_auto_point_size(N)` formula as other spatial plots: `s = 100 / sqrt(N)`.

### Only generated for MGGP models

Skip `groupwise_factors.png` entirely for `pnmf`, `svgp`, `lcgp`.

---

## Part 4 — 3D Posterior Figures (Top-2 Factors, Top-3 Groups)

A compact 3×4 figure showing the top-2 factors (by Moran's I) against the 3 most
abundant groups, using 3D surface scatter panels.

### Layout

```
(3 rows × 4 cols)

             [empty]         [Group A loc]   [Group B loc]   [Group C loc]
Factor 0:    [uncond 3D]     [cond gA f0]    [cond gB f0]    [cond gC f0]
Factor 1:    [uncond 3D]     [cond gA f1]    [cond gB f1]    [cond gC f1]
```

- **Row 0 (header)**: 2D scatter panels showing where each group lives spatially.
  Cell [0, 0] is empty/off. Cells [0, 1..3] each highlight one group: group cells
  in a distinct color, all others dimmed (same copper/gray style as Part 3).
  Column header = group name.

- **Column 0 (left)**: 3D surface scatter of the **unconditional** factor (loaded from
  `factors.npy`). One panel per factor row. Row label = `"Factor 0"`, `"Factor 1"`.

- **Inner 2×3 (rows 1-2, cols 1-3)**: 3D surface scatter of the **conditional posterior**
  for that factor under that group. Uses `_draw_factor_3d` from `multianalyze.py`.

All 3D panels (column 0 and inner 2×3) use the same `vmin`/`vmax` derived from the
unconditional factors, so all surfaces are on a comparable z-scale.

### Which factors

Top-2 by Moran's I — factors 0 and 1 after the Moran's I sort already applied in analyze.

### Which groups

Top-3 by cell count: `np.argsort(np.bincount(C))[::-1][:3]`, fixed across both rows.

### Output file

```
outputs/<dataset>/mggp_svgp/figures/groupwise_factors_3d.png
outputs/<dataset>/mggp_lcgp/figures/groupwise_factors_3d.png
```

### Data source

Reads from `groupwise_factors/group_{g}.npy` files saved in Part 2 — no re-computation
at figure time.

### Only generated for MGGP models

Same gate as Parts 2 and 3: skip for `pnmf`, `svgp`, `lcgp`.

---

## Code References (from notebooks)

| What | Notebook | Notes |
|------|----------|-------|
| `get_groupwise_factors()` | `liver_mggp_healthy_matern32_umap_init.ipynb` cell 41 | Loop over groups, force `groupsX_test = full(g)`, call `model.prior(X, groupsX=groupsX_test)`, return `exp(qF.mean)` |
| `plot_groupwise_factors()` | `liver_mggp_healthy_matern32_umap_init.ipynb` cell 42 | Grid layout, copper for location col, turbo for factors, shared vmin/vmax |
| Inline slideseq version | `Slideseqv2_MGGP_november.ipynb` cell 74 | Same idea, gray colormap for location col, uses `min_val`/`max_val` from unconditional factors |
