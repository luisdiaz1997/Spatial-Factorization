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

### E. Gene × Factor × Cell-Type: where the three views meet

For each gene `d`, we can summarize its participation in the factorization as a matrix
`loadings_group_g[d, :]` of shape (G, L). Selecting a small set of interesting genes (e.g. top
marker candidates from analysis C), plot each gene as a G × L heatmap.

**Plot:** small multiples — one heatmap per selected gene, rows = cell types (in a consistent
order), columns = factors (Moran's I order). Annotate the columns with specificity-class labels.
This reveals, for a single gene, *which* factors carry its signal and *how* that distribution
varies across cell types.

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
