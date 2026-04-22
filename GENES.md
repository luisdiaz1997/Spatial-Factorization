# Gene-Level Analysis via Factor Loadings

> See `POSTERIOR_ANALYSIS.md` for the cell-type-conditional posterior framework, the L1-norm
> specificity metric, and the Shannon entropy taxonomy that classifies each factor as
> **universal**, **cell-type dependent**, or **cell-type specific**.

## Questions

The specificity analysis tells us *how* each factor's spatial signal responds to cell-type
conditioning. The complementary question — at the gene level — is:

1. **Which genes use which factors?** Given a factor `f`, which genes have the largest loadings?
2. **How concentrated is each factor across genes?** Is factor `f` driving a narrow gene program
   (few high-loading genes) or a diffuse one (many genes, each a little)?
3. **How specialized is each gene across factors?** Is gene `d` dominated by a single factor
   (factor-specific gene), or does it participate in many factors at once (multiplexed)?
4. **Which genes are associated with which cell types?** Under the multigroup model, per-group
   loadings `W_g` differ across cell types — which genes stand out in each?
5. **What proportion of a given gene's signal is attributable to each cell type?** A single gene
   may be driven entirely by one cell type or split across several.
6. **What do factor-specific genes look like spatially?** Genes with low per-gene entropy are
   dominated by one or two factors — reconstructing their spatial expression (marginal and
   per-cell-type) gives a clean picture of what the model claims that gene does.
7. **How do the answers depend on the factor's specificity class?** Cell-type-specific factors,
   universal factors, and dependent factors should produce qualitatively different gene-level
   patterns. Characterizing those differences is the point of the analysis.

## Architecture Template

Follow the same split as the existing conditional / Moran's I breakdown work:

- **Compute + save CSVs** in `spatial_factorization/commands/benchmark_analyze.py`.
- **Load CSVs + render multi-panel figure** in `spatial_factorization/commands/benchmark_figures.py`.

Concrete templates already in the tree (do not edit — borrow patterns):

- `benchmark_analyze.py:199` — `_compute_factor_specificity(model_dir, output_dir, group_names)` — loads marginal `factors.npy` + per-group `groupwise_factors/group_g.npy`, computes a per-(group, factor) metric, writes `factor_specificity.csv`; lines 233-248 show the Shannon-entropy loop pattern we already use for cell-type entropy. Extend this or add a sibling function for the new CSVs below.
- `benchmark_analyze.py:390-395` — where new per-MGGP-model analyses are wired into the pipeline (iterates `mggp_svgp`, `mggp_lcgp`).
- `benchmark_figures.py:613` — `plot_groupwise_moran_breakdown(output_dir)` — produces `groupwise_moran_breakdown.png` (the layout we want to mirror for new combined figures): loads CSVs, builds a multi-panel figure via `fig.add_gridspec`, saves to `output_dir / "figures" / ...png`.

## Data

Everything we need is already saved by the analyze stage:

| File | Shape | Meaning |
|------|-------|---------|
| `loadings.npy` | (D, L) | Global gene × factor loadings (Moran's I ordering) |
| `loadings_group_g.npy` | (D, L) | Per-cell-type loadings — one file per group `g` |
| `factors.npy` | (N, L) | Marginal spatial factor values (exp-space) |
| `groupwise_factors/group_g.npy` | (N, L) | Conditional factor values per cell type |
| `factor_specificity.csv` | G·L rows | L1 ratios from the conditional analysis |
| `factor_entropy.csv` | L rows | Cell-type Shannon entropy per factor |
| `metadata.json` | — | `gene_names` list of length D |

## Analyses

### A. Factor → Gene: which genes drive factor `f`?

For each factor `f`, rank genes by `loadings[:, f]` (descending). Report the top-K and visualize.

**Plot:** per-factor horizontal bar chart of top-K gene loadings. Annotate each factor panel with
its Shannon entropy `H_f` (from `factor_entropy.csv`) and specificity class label. Grid of L
panels, one per factor.

**What to look for per specificity class:**
- **Specific** (low H): top genes should be recognizable markers of the enriched cell type(s).
- **Universal** (high H): top genes are either housekeeping or broadly expressed structural genes;
  they should recur across many factors if truly generic.
- **Dependent** (mid H): top genes form a program that is spatially modulated by cell identity —
  the same genes appear in multiple per-group rankings below.

**Code to borrow from:**
- `figures.py:334` — `plot_top_genes(loadings, gene_names, moran_idx, n_top, figsize_per_factor, ncols)` — renders a `ceil(L/ncols) × ncols` grid of horizontal bar charts; each panel uses `np.argsort(loadings[:, i])[-n_top:][::-1]` + `ax.barh` + `ax.invert_yaxis()`. Pattern is ~65 lines; reuse almost verbatim but annotate each panel title with `H_f` and specificity class.

### B. Entropy of the Loadings Matrix — Two Complementary Views

The (D, L) loadings matrix admits two Shannon entropies, each summarizing a different structural
property of the factorization. They are computed along opposite axes and answer different
questions.

#### B1. Factor → Genes (how concentrated is a factor across genes)

For each factor `f`, normalize its column and compute entropy over genes:

```
p_d = loadings[d, f] / sum_d loadings[d, f]
H_f^(gene) = -sum_d p_d · log2(p_d)  /  log2(D)      (one value per factor, ∈ [0, 1])
```

- **Low `H_f^(gene)`** → factor is gene-specific (a narrow program dominated by few genes).
- **High `H_f^(gene)`** → factor is gene-broad (many genes participate roughly equally).

This resolves the ambiguity raised by universal factors: a universal factor with low gene entropy
is a housekeeping program (few shared genes); a universal factor with high gene entropy is truly
"everyone uses everything" (no gene-specific signal).

**Plot:** scatter of all L factors in the plane
`(cell-type entropy from factor_entropy.csv, gene entropy H_f^(gene))`, colored or labeled by
factor index. Quadrants:

| | Low gene entropy | High gene entropy |
|---|---|---|
| **Low cell-type entropy** | Cell-type-specific marker program | Broad transcriptional signature of one cell type |
| **High cell-type entropy** | Shared housekeeping on a narrow gene set | Truly universal — no specificity on either axis |

Save: `factor_gene_entropy.csv` with columns `factor_idx, gene_entropy`.

**Code to borrow from:**
- `benchmark_analyze.py:233-248` — the Shannon-entropy loop already used for cell-type entropy (clip → normalize → sum(-p·log₂p) → divide by `log₂(n)`). Same template, applied to column-normalized loadings instead of per-group L1 ratios.

#### B2. Gene → Factors (how specialized is a gene across factors)

For each gene `d`, normalize its row and compute entropy over factors:

```
q_f = loadings[d, f] / sum_f loadings[d, f]
H_d^(factor) = -sum_f q_f · log2(q_f)  /  log2(L)     (one value per gene, ∈ [0, 1])
```

- **Low `H_d^(factor)`** → gene is factor-specific (dominated by one or two factors).
- **High `H_d^(factor)`** → gene is multiplexed (participates in many factors roughly equally).

Factor-specific genes are the natural targets for visualization and marker discovery: if a gene's
signal is mostly routed through one factor, its spatial expression is well-approximated by
`loadings[d, f*] × factors[:, f*]` for its dominant factor `f*`, and the cell-type behavior of
that factor directly predicts the gene's cell-type behavior.

**Plot:** histogram of `H_d^(factor)` across all D genes, with annotations for the top-K lowest
(most factor-specific) and top-K highest (most multiplexed) genes.

Save: `gene_factor_entropy.csv` with columns `gene_idx, gene_name, factor_entropy,
dominant_factor` where `dominant_factor = argmax_f loadings[d, f]`.

**Code to borrow from:**
- `analyze.py:479` — `_normalize_loadings(loadings)` — exact row-normalization we need (`max(loadings, eps) / sum along axis=1`). Feed its output row-by-row into the B1 entropy loop (with `log₂(L)` normalization).
- 2D scatter of (cell-type H, factor-gene H): use a single `ax.scatter` with factor index labels; no existing analog to borrow, but the panel + annotation style from `benchmark_figures.py:613` onward is the reference.

### C. Cell-Type → Gene: which genes are associated with cell type `g`?

For each cell type `g`, form a gene score that integrates over all factors. A natural choice is the
L1-weighted per-group loading magnitude:

```
gene_score[d, g] = sum_f |loadings_group_g[d, f]|
```

Rank genes by `gene_score[:, g]` to find the top genes per cell type.

**Plot:** grid of G panels (one per cell type), each a bar chart of top-K genes by
`gene_score[:, g]`. Useful as a cell-type "transcriptional signature" summary that is not tied to
any single factor.

**Code to borrow from:**
- `figures.py:1039` — `plot_top_enriched_genes_per_group(gene_enrichment, factor_idx, n_top, figsize)` — per-group bar-panel grid (`ncols=min(4, n_groups)`, `nrows=ceil(n_groups/ncols)`), uses `ax.barh` + `ax.invert_yaxis` + per-panel title. Same grid template, different ranking source.
- `analyze.py:499` — `_compute_gene_enrichment(global_loadings, group_loadings, gene_names, group_names)` — produces `gene_enrichment.json` with top-10 enriched/depleted per (factor, group). Different math (LFC of row-normalized loadings, not `Σ_f |W_g[d, f]|`), but the iteration pattern over groups and factor keys is the template.
- `analyze.py:1056-1067` — saves `pca_gene_order_by_celltype.npy` (G, D): per-celltype PC1 ordering of genes. Useful if we later want to order the bars by PC1 instead of raw magnitude.

### D. Gene × Cell-Type Proportion

For a fixed gene `d`, compute the fraction of its total signal attributable to each cell type.
Using per-group loadings:

```
share[d, g] = |loadings_group_g[d, :]|_1  /  sum_g' |loadings_group_g'[d, :]|_1
```

Each row sums to 1.

**Plot:** stacked horizontal bar chart, one bar per gene (restricted to the top-M most variable
genes across groups, otherwise the plot is unreadable for large D). Ordering genes by dominant
cell type makes the stratification visible.

This directly answers "what proportion of a given gene is used by a certain cell type?" —
complementing the factor-level specificity analysis with a gene-level counterpart.

**Code to borrow from:**
- `figures.py:491` — `plot_gene_enrichment_heatmap(global_loadings, group_loadings, group_names, moran_idx, figsize)` — renders a G×L mean-|LFC| heatmap (`imshow` + colorbar + `xticks/yticks` from group/factor labels). Not the same math as the proportion share we want, but the existing (G × *) heatmap layout + color scaling pattern transfers.
- `analyze.py:479` — `_normalize_loadings` — row-normalize pattern; the share computation is a one-line extension (`|W_g[d, :]|_1 / Σ_g' |W_g'[d, :]|_1`).
- No existing stacked-bar template in the repo; use `ax.barh` with `left` offsets accumulated across groups. Point-size / color conventions: `benchmark_figures.py` `MODEL_COLORS` / `plt.cm.tab20` (see `benchmark_figures.py:704` for the per-group tab20 pattern already used in the specificity bars).

### E. Gene × Factor × Cell-Type: where the three views meet

For each gene `d`, we can summarize its participation in the factorization as a matrix
`loadings_group_g[d, :]` of shape (G, L). Selecting a small set of interesting genes (e.g. top
marker candidates from analysis C), plot each gene as a G × L heatmap.

**Plot:** small multiples — one heatmap per selected gene, rows = cell types (in a consistent
order), columns = factors (Moran's I order). Annotate the columns with specificity-class labels.
This reveals, for a single gene, *which* factors carry its signal and *how* that distribution
varies across cell types.

**Code to borrow from:**
- `figures.py:651` — `plot_factor_gene_loadings(group_loadings, group_names, gene_names, group_idx, pca_gene_order, global_loadings, figsize)` — per-cell-type (L, D) LFC heatmap. The `imshow(mat, cmap="bwr", vmin=-abs_max, vmax=abs_max)` + symmetric-colorbar pattern is what we want per-gene panel.
- `figures.py:566` — `plot_celltype_gene_loadings` — sibling helper with the same color scale + PCA-ordering pattern, but slicing differently (G × D for one factor). Either is fine as a visual reference.

### F. Spatial Reconstruction of Factor-Specific Genes

Using the per-gene entropy `H_d^(factor)` from **B2**, select the top-K genes with the lowest
entropy — these are the genes most cleanly explained by the factorization. For each such gene,
reconstruct and plot its spatial expression.

**Marginal reconstruction** (through the dominant factor `f* = argmax_f loadings[d, f]`):

```
Ŷ_marginal[d] = loadings[d, f*] × factors[:, f*]        (length-N spatial vector)
```

**Per-cell-type conditional reconstruction**:

```
Ŷ_g[d] = loadings_group_g[d, f*] × groupwise_factors_g[:, f*]     (length-N per group g)
```

**Plot:** for each selected gene, a row of spatial scatter panels:
- Panel 0: marginal reconstruction (all cells).
- Panels 1..G: conditional reconstructions, one per cell type.
- Annotate the gene name, dominant factor `f*`, the factor's cell-type entropy and specificity
  class, and the gene's own factor entropy.

**What the three classes predict:**
- If `f*` is **specific** (low cell-type H): most conditional panels will be near-zero; one or two
  show the gene's full expression. The gene is effectively a marker of those cell types.
- If `f*` is **universal** (high cell-type H): all conditional panels look similar — the gene is
  broadly expressed, and no single cell type claims it.
- If `f*` is **dependent** (mid H): different conditional panels show distinct spatial patterns
  for the same gene — spatial organization is cell-type-modulated.

This analysis operationalizes the user-level claim that "factor-specific genes should behave like
their dominant factor does under conditioning" — it is the visual counterpart to the entropy
classifications in **B**.

**Code to borrow from:**
- `figures.py:1600` — `plot_celltype_summary_loadings(factors, groupwise_factors, Y, coords, groups, group_id, group_name, gene_names, group_loadings, n_top, ...)` — the closest visual template: row 0 = global factors, row 1 = cell density + conditional factors, rows 2..N = top genes by row-normalized W_c plotted spatially. The `_scatter(ax, values, vmin, vmax, cmap, title)` inner helper (lines 1665-1686) and the row/col iteration structure are exactly what F needs. **Key difference:** that function plots raw `Y[:, gene_idx]`; F plots `loadings[d, f*] * factors[:, f*]` (marginal reconstruction) and `loadings_group_g[d, f*] * groupwise_factors_g[:, f*]` (conditional reconstruction).
- `figures.py:730` — `plot_factors_with_top_genes(factors, Y, loadings, coords, gene_names, moran_idx, n_genes, ...)` — simpler alternative template: one row per factor, column 0 = factor spatial, columns 1..N = top-gene spatial (again raw Y). Useful if we want a per-gene row (marginal + G conditionals) instead of a per-cell-type column layout.
- `figures.py:1102` — `_draw_group_loc_panel(ax, coords, groups, g, s, alpha, cmap)` — cell density panel helper (used to mark "this is the cell type" column). Direct reuse.
- `figures.py:2110` — `_auto_point_size(N)` — the repo-wide spatial-scatter point-size convention (`100 / sqrt(N)`).

## Context: What the Three Classes Predict for Slideseq Factors

The following reasoning motivates the analyses above and grounds them in concrete slideseq
observations (H values from `factor_entropy.csv`, visual patterns from
`groupwise_factors.png`).

If a gene is reconstructed through a factor, and that factor's spatial pattern changes under
cell-type conditioning, the reconstructed expression for that gene changes too. The three
specificity classes therefore imply three qualitatively different gene–cell-type stories:

- **Cell-type specific factors** (e.g. **F1**, low H). The factor's signal is depleted for most
  cell types but preserved or enriched for one or two. Genes loading on this factor are plausibly
  markers of those one or two enriched cell types — the factor acts as a transcriptional proxy
  for that lineage.

- **Universal factors** (e.g. **F3, F4, F7**, H near 1.0). Genes associated with these factors
  should show little change under conditioning. Two interpretations are compatible with this
  observation, and gene-level analysis is required to disambiguate them:
  1. The genes are genuinely shared across all cell types (housekeeping).
  2. The factor itself is not gene-specific — it is broadly used by many genes, each contributing
     a small amount, which is why no cell-type-level signal emerges.

  Analysis **B1** (per-factor entropy over genes) resolves this: low gene entropy favors
  interpretation 1, high gene entropy favors interpretation 2.

- **Cell-type dependent factors** (e.g. **F5, F8, F9, F10**, mid H). These change across cell
  types and, importantly, reveal spatial patterns that are only visible through conditioning — the
  marginal factor averages them out. Genes on these factors participate in cell-type-modulated
  spatial programs. The top-gene lists per cell type (analysis **C**) and the gene × cell-type
  proportion share (analysis **D**) are the natural tools here.

- **Hybrid case (F2, H=0.94)**. Statistically universal by entropy, but one cell type (CA1_CA2)
  shows a clear enriched pattern. The top-gene list is expected to be stable across cell types,
  with a single per-group loading vector shifted upward. Analysis **D** is the right lens for this
  pattern, which would otherwise be masked by a purely entropy-based classification.

## Tie-In With the Three Specificity Classes

The gene-level analyses above should behave predictably under the three classes documented in
`POSTERIOR_ANALYSIS.md`. The table summarizes expected outcomes:

| Class | Expected top-gene behavior (A) | Factor-over-genes entropy (B1) | Per-cell-type top genes (C) |
|-------|-------------------------------|------------------|------------------------------|
| **Specific** (low H, e.g. F1) | Top genes are markers of the enriched cell type(s) | Often low — a narrow marker program | The enriched cell type's top-gene list overlaps heavily with factor `f`'s top-gene list |
| **Universal** (H ≈ 1, e.g. F3) | Top genes are structural/housekeeping or appear across many factors | Informative: **low** = housekeeping subset, **high** = truly generic | Per-cell-type top-gene lists all overlap with factor `f`'s top-gene list |
| **Dependent** (mid H, e.g. F5, F8, F9, F10) | Top genes form a program expressed with different spatial patterns per cell type | Typically moderate | Multiple cell types contribute distinct top-gene subsets — same factor, different gene programs |

Note the **F2 pattern** flagged in the conditional analysis: mostly universal (`H=0.94`) with one
cell type showing a pronounced enrichment. In gene-level terms, this predicts a top-gene list that
is largely stable across cell types but with one per-group loading vector `loadings_group_g[:, f]`
showing a clear up-shift. Analysis D (proportion share) is the right lens for catching this kind
of hybrid.

## Suggested Output Artifacts

Following the existing convention (analyses compute and save CSVs, figures module plots them):

**In `benchmark_analyze.py`** (extend `_compute_factor_specificity` or add new functions):
- `factor_gene_entropy.csv` — columns `factor_idx, gene_entropy` (per-factor normalized Shannon
  entropy over genes; analysis B1).
- `gene_factor_entropy.csv` — columns `gene_idx, gene_name, factor_entropy, dominant_factor`
  (per-gene normalized Shannon entropy over factors; analysis B2).
- `gene_celltype_share.csv` — columns `gene_idx, gene_name, group_idx, group_name, share` where
  `share` is `|loadings_group_g[d, :]|_1 / sum_g'` (analysis D).
- `top_genes_per_factor.csv` — columns `factor_idx, rank, gene_idx, gene_name, loading` (top-K per
  factor, K configurable; analysis A).
- `top_genes_per_celltype.csv` — columns `group_idx, group_name, rank, gene_idx, gene_name, score`
  (analysis C).

**In `benchmark_figures.py`** (new plot functions, loading the CSVs above):
- `top_genes_per_factor.png` — grid of L bar charts (analysis A).
- `factor_entropy_2d.png` — scatter of cell-type vs factor-over-genes entropy (analysis B1).
- `gene_factor_entropy_hist.png` — histogram of per-gene factor entropy, annotated extremes
  (analysis B2).
- `top_genes_per_celltype.png` — grid of G bar charts (analysis C).
- `gene_celltype_share.png` — stacked bar over top-M variable genes (analysis D).
- `gene_factor_celltype_heatmaps.png` — small multiples for curated marker genes (analysis E).
- `factor_specific_gene_reconstructions.png` — spatial reconstruction panels (marginal + per
  cell-type) for top-K lowest-entropy genes (analysis F).
