# Posterior Analysis Ideas

Cell-type-conditional posteriors (`groupwise_factors/group_g.npy`) enable analyses beyond Moran's I. Each is a (N, L) array — full-tissue posterior for cell type g.

## Done

- **Moran's I per cell type / per factor** (`groupwise_moran_breakdown.png`): Which cell types and factors are more spatially autocorrelated. Includes marginal median reference line.
- **Factor specificity via norm ratio** (`‖conditional‖₁ / ‖marginal‖₁`): Bar chart per factor showing which cell types preserve/deplete/enrich each factor. Distinguishes universal vs specific vs dependent.
- **Shannon entropy of specificity** (H per factor): Single number quantifying how concentrated a factor's signal is across cell types. Low H = specific, high H = universal. Saved to `factor_entropy.csv`.

## Cell-Type Specificity Taxonomy

When we condition on cell type, factors behave in three qualitatively different ways:

| Behavior | What happens on conditioning | Interpretation |
|----------|------------------------------|----------------|
| **Universal / cell-type independent** | Barely changes across any cell type — all conditionals ≈ marginal in magnitude and shape | Shared tissue structure, not driven by cell identity |
| **Cell-type specific** | Depleted everywhere *except* one (or few) cell type(s) that preserve/enrich it | That factor is essentially "owned" by that cell type |
| **Cell-type dependent** | Different cell types carry different amounts of the signal — same general shape but varying intensity/location | Factor means different things depending on which cell type you ask |

**Metric: `‖conditional_g‖₁ / ‖marginal‖₁`** (L1-norm ratio)

Correlation is misleading because it's scale-invariant — a flattened-to-near-zero conditional can still correlate highly with the marginal. Norm ratio captures what the eye sees: **enrichment or depletion**.

- **~1.0** → conditional preserves marginal magnitude (not depleted)
- **<< 1.0** → conditional depleted (signal killed off)
- **> 1.0** → conditional enriched (amplified)

We use L1 (not L2) because it's less sensitive to outlier-driven blowup from KNN noise in LCGP conditionals.

**Visualization:** Grouped bar chart per factor, one bar per cell type showing norm ratio on log₂ scale. Dotted line at 1.0 = "preserved."

### Shannon Entropy of Specificity

To quantify whether a factor is universal or specific in a single number, we compute the Shannon entropy of the (shifted) log-ratio distribution across cell types:

```
For each factor f:
  r_g   = ‖conditional_g[:,f]‖₁ / ‖marginal[:,f]‖₁          (raw ratio)
  ℓ_g   = log₂(r_g)                                          (log-ratio; negative = depleted)
  m_g   = clip(ℓ_g + 1, 0, ∞)                                (anchor at r=0.5, clamp negatives)
  p_g   = m_g / Σ m                                           (normalize to probability)
  H_f   = -Σ p_g · log₂(p_g) / log₂(G)                       (normalized entropy ∈ [0,1])
```

**Why `log₂(ratio) + 1`?** This anchors at ratio = 0.5 (`log₂(0.5) + 1 = 0`). Groups whose conditional is depleted below half the marginal contribute zero mass — they're treated as "silenced." Only groups that preserve or enrich the factor above this floor participate in the entropy calculation. This distinguishes:

- **min(ℓ) ≈ 0** → nobody was significantly depleted → **H ≈ 1** (universal)
- **min(ℓ) ≈ –1** → someone lost ≥50% of the signal → **H < 1** (specific/dependent)

The result is normalized by `log₂(G)` so it always lies in [0,1] regardless of the number of cell types.

| H value | Interpretation |
|---------|---------------|
| **≈ 1.0** | Universal — all cell types carry similar mass (no group stands out) |
| **≈ 0.5–0.8** | Dependent — several groups differ but no single owner |
| **≈ 0.3–0.5** | Specific — one or few groups dominate the signal |

### Slideseq LCGP Visual Annotation

From `outputs/slideseq/mggp_lcgp/figures/groupwise_factors.png`:

| Factor | Visual Pattern | Classification | H (entropy) |
|--------|---------------|----------------|-------------|
| F1 | Arc/curved band in Astrocytes & CA1; weaker in Dentate; most others depleted | **Dependent** | 0.44 |
| F2 | Tiny isolated hotspot only in CA1_CA2; everyone else near-zero | **Specific** to CA1 | 0.94 |
| F3 | Gradient/band visible across many rows (Astrocytes, CA1, Dentate, Endothelial, Interneurons, Oligo, Subiculum) | **Universal** | 0.96 |
| F4 | Hotspot in bottom-right for DentatePyramids & Endothelial_Tip; weak elsewhere | **Specific** to dentate/endothelial | 0.69 |
| F5 | One tiny red dot only in Astrocytes; dead everywhere else | **Highly specific** to Astrocytes | 0.80 |
| F6 | Bright arc/hotspot in DentatePyramids; faint or depleted elsewhere | **Specific** to DentatePyramids | 0.99 |
| F7 | Curved band across many rows with varying intensity | **Dependent** | 0.76 |
| F8 | Arc in CA1_CA2; very weak elsewhere | **Specific** to CA1 | 0.89 |
| F9 | Full-coverage yellow/green in Astrocytes; depleted all others | **Specific** to Astrocytes | 0.87 |
| F10 | Scattered noisy hotspots only in Microglia | **Specific** to Microglia | 0.87 |

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
