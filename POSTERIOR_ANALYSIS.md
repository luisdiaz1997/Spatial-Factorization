# Posterior Analysis Ideas

Cell-type-conditional posteriors (`groupwise_factors/group_g.npy`) enable analyses beyond Moran's I. Each is a (N, L) array — full-tissue posterior for cell type g.

## Done

- **Moran's I per cell type / per factor** (`groupwise_moran_breakdown.png`): Which cell types and factors are more spatially autocorrelated. Includes marginal median reference line.
- **Factor specificity via norm ratio** (`||conditional|| / ||marginal||`): Bar chart per factor showing which cell types preserve/deplete/enrich each factor. Distinguishes universal vs specific vs dependent.

## Cell-Type Specificity Taxonomy

When we condition on cell type, factors behave in three qualitatively different ways:

| Behavior | What happens on conditioning | Interpretation |
|----------|------------------------------|----------------|
| **Universal / cell-type independent** | Barely changes across any cell type — all conditionals ≈ marginal in magnitude and shape | Shared tissue structure, not driven by cell identity |
| **Cell-type specific** | Depleted everywhere *except* one (or few) cell type(s) that preserve/enrich it | That factor is essentially "owned" by that cell type |
| **Cell-type dependent** | Different cell types carry different amounts of the signal — same general shape but varying intensity/location | Factor means different things depending on which cell type you ask |

**Metric: `||conditional_g|| / ||marginal||`** (relative L2 norm)

Correlation is misleading because it's scale-invariant — a flattened-to-near-zero conditional can still correlate highly with the marginal. Norm ratio captures what the eye sees: **enrichment or depletion**.

- **~1.0** → conditional preserves marginal magnitude (not depleted)
- **<< 1.0** → conditional depleted (signal killed off)
- **> 1.0** → conditional enriched (amplified)

**Visualization:** Grouped bar chart per factor, one bar per cell type showing norm ratio. Dotted line at 1.0 = "preserved."

### Slideseq LCGP Visual Annotation

From `outputs/slideseq/mggp_lcgp/figures/groupwise_factors.png`:

| Factor | Visual Pattern | Classification |
|--------|---------------|----------------|
| F1 | Arc/curved band in Astrocytes & CA1; weaker in Dentate; most others depleted | **Dependent** |
| F2 | Tiny isolated hotspot only in CA1_CA2; everyone else near-zero | **Specific** to CA1 |
| F3 | Gradient/band visible across many rows (Astrocytes, CA1, Dentate, Endothelial, Interneurons, Oligo, Subiculum) | **Universal** |
| F4 | Hotspot in bottom-right for DentatePyramids & Endothelial_Tip; weak elsewhere | **Specific** to dentate/endothelial |
| F5 | One tiny red dot only in Astrocytes; dead everywhere else | **Highly specific** to Astrocytes |
| F6 | Bright arc/hotspot in DentatePyramids; faint or depleted elsewhere | **Specific** to DentatePyramids |
| F7 | Curved band across many rows with varying intensity | **Dependent** |
| F8 | Arc in CA1_CA2; very weak elsewhere | **Specific** to CA1 |
| F9 | Full-coverage yellow/green in Astrocytes; depleted all others | **Specific** to Astrocytes |
| F10 | Scattered noisy hotspots only in Microglia | **Specific** to Microglia |

## To Explore

### 1. Per-Cell-Type Reconstruction Error
Condition on cell type g, reconstruct expression for cells that actually *are* type g. Compare reconstruction quality (Poisson deviance) across cell types.

**How:** For each g, take `group_g[factors_for_cells_of_type_g]`, multiply by loadings, compare to observed Y for those cells.

### 2. Cross-Cell-Type Factor Correlation
Correlate conditional posteriors *between* cell types. High correlation = tissue-wide structure; low correlation = cell-type-specific organization.

**How:** N×L conditional per group → correlate column-by-column between groups → G×G correlation matrix per factor.

### 3. Conditional Posterior Uncertainty
Compute uncertainty (scales/variance) on the conditionals. Which combos is the model confident about?

**How:** If scales available for conditionals, plot per-cell-type uncertainty distribution.

### 4. Spatial Directionality / Gradient
For each cell-type/factor, compute principal axis of spatial trend. Tells you *which direction* the pattern runs.

**How:** Fit plane to coords vs values; report gradient magnitude/direction. Or first PC of (x, y, value).

## Priority

Top picks: **Specificity taxonomy** (done) and **Directionality** — visually compelling, support Results claims about "factor specificity."
