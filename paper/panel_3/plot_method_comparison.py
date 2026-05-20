"""Three-cell-type × N-factor side-by-side panel for smNSF method comparisons.

Mirrors the SF figures pipeline's groupwise_factors aesthetic (turbo cmap,
fixed vmin=0, vmax=exp(2.3263)≈10.24, gray facecolor), so the comparison
panels match the rest of the dissertation visually.

Two driver modes:
    --mode vnngp_vs_lcgp   compares KNN-based (VNNGP-style) vs probabilistic
                           neighbor selection, both at group_diff_param=1e6
                           (so MGGP is effectively turned off and the
                           comparison isolates neighbor-selection strategy).
    --mode mggp_gain       compares a≈∞ (~independent GPs per group) vs
                           regular MGGP-LCGP with cross-group sharing, both
                           with probabilistic neighbor selection.

Factors are MATCHED across methods by Pearson correlation on the gene
loadings: for each reference factor in the right-side (treatment) method,
the highest-correlated factor in the left-side (control) method is chosen.
This means the columns in the left and right panels show the SAME biological
program (up to model fit), not just the same numerical index.

Usage:
    conda run -n factorization python paper/panel_3/plot_method_comparison.py \\
        --mode vnngp_vs_lcgp \\
        --dataset slideseq \\
        --out paper/panel_3/figures/vnngp_vs_lcgp_slideseq.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from spatial_factorization.commands.figures import plot_groupwise_factors_subset, _auto_point_size  # noqa: E402


# Publication font defaults — match the rest of the dissertation figures.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


_MODE_PAIRS = {
    # mode: (left_method_path, right_method_path,
    #        left_label_png, right_label_png, suptitle_png,
    #        left_label_pdf, right_label_pdf, suptitle_pdf)
    # Two title forms: plain unicode for PIL/PNG (no LaTeX rendering) and
    # LaTeX-mathified for the matplotlib PDF re-render.
    "vnngp_vs_lcgp": (
        "group_diff_1000000_knn/mggp_lcgp",
        "group_diff_1000000_probabilistic/mggp_lcgp",
        "VNNGP-style (knn)", "LCGP (probabilistic)",
        "Neighbor-selection strategy (group_diff_param = ∞, MGGP effectively off)",
        "VNNGP-style (knn)", "LCGP (probabilistic)",
        "Neighbor-selection strategy (group_diff_param $= \\infty$, MGGP effectively off)",
    ),
    "mggp_gain": (
        "group_diff_1000000_probabilistic/mggp_lcgp",
        "mggp_lcgp",
        "Independent GPs (a -> infinity)", "MGGP-LCGP (finite a)",
        "Multi-group sharing (both probabilistic; only a varies)",
        "Independent GPs ($a\\to\\infty$)", "MGGP-LCGP (finite $a$)",
        "Multi-group sharing (both probabilistic; only $a$ varies)",
    ),
}


def _load_loadings(method_dir: Path) -> np.ndarray:
    """Read (G_genes, L_factors) loadings matrix."""
    p = method_dir / "loadings.npy"
    if not p.exists():
        raise FileNotFoundError(f"loadings.npy not found at {p}")
    return np.load(p)


def _match_factors(ref_loadings: np.ndarray, query_loadings: np.ndarray,
                   ref_factor_ids: list[int]) -> list[int]:
    """For each reference factor, return the best-matching query factor index.

    Match by Pearson correlation on gene loadings (z-scored per factor across
    genes). Each query factor can be selected at most once — if the top match
    is already taken, the next-best unused match is used.

    Args:
        ref_loadings: (G, L_ref) reference loadings.
        query_loadings: (G, L_query) query loadings.
        ref_factor_ids: factor indices in the reference whose matches we want.

    Returns:
        list of length ``len(ref_factor_ids)`` with the matched query factor
        index for each (no repeats).
    """
    def zscore(W):
        W = W - W.mean(axis=0, keepdims=True)
        W = W / (W.std(axis=0, keepdims=True) + 1e-12)
        return W

    Z_ref = zscore(ref_loadings)
    Z_query = zscore(query_loadings)
    G = Z_ref.shape[0]
    # (L_ref, L_query) correlation
    C = (Z_ref.T @ Z_query) / G

    used = set()
    matches = []
    for r in ref_factor_ids:
        order = np.argsort(-np.abs(C[r]))   # descending by |r|
        for q in order:
            q = int(q)
            if q in used:
                continue
            matches.append(q)
            used.add(q)
            break
    return matches


def _pick_top_spatial_factors(metrics_path: Path, k: int = 3) -> list[int]:
    """Top-k factors by Moran's I."""
    m = json.loads(metrics_path.read_text())
    mi = m["moran_i"]
    sorted_idx = mi.get("sorted_indices")
    if sorted_idx is None:
        sorted_idx = list(np.argsort(mi["values"])[::-1])
    return [int(i) for i in sorted_idx[:k]]


def _l1_ratio_matrix(method_dir: Path) -> np.ndarray:
    """Return the (n_groups, n_factors) L1 specificity-ratio matrix.

    Mirrors the benchmark pipeline (``benchmark_analyze.
    _compute_factor_specificity``): both marginal and per-group conditional
    factor maps are clipped at the per-factor 99th percentile of the
    marginal before L1 is taken, to prevent GP-extrapolation outliers from
    dominating. Row ``g``, column ``f`` is
    ``||cond[g, :, f]_clipped||_1 / ||marginal[:, f]_clipped||_1``.
    Benchmark calls factors with max-over-groups ratio > 1.5
    "celltype_enriched".
    """
    marginal = np.load(method_dir / "factors.npy")          # (N, L)
    p99 = np.percentile(marginal, 99, axis=0)
    marginal_clipped = np.minimum(marginal, p99[None, :])
    m_l1 = marginal_clipped.sum(axis=0)                     # (L,)

    gf_dir = method_dir / "groupwise_factors"
    gf_paths = sorted(gf_dir.glob("group_*.npy"),
                      key=lambda p: int(p.stem.split("_")[1]))
    G = len(gf_paths)
    L = marginal.shape[1]
    M = np.zeros((G, L), dtype=float)
    for row_g, gf_path in enumerate(gf_paths):
        g = int(gf_path.stem.split("_")[1])
        cond = np.load(gf_path)
        cond_clipped = np.minimum(cond, p99[None, :])
        c_l1 = cond_clipped.sum(axis=0)
        M[g] = c_l1 / (m_l1 + 1e-10)
    return M


def _pick_top_enriched_factors(method_dir: Path, k: int = 3) -> list[int]:
    """Top-k factors by max-over-groups L1 specificity ratio."""
    M = _l1_ratio_matrix(method_dir)
    max_ratio_per_factor = M.max(axis=0)
    order = np.argsort(-max_ratio_per_factor)
    return [int(f) for f in order[:k]]


def _pick_top_universal_factors(method_dir: Path, k: int = 3) -> list[int]:
    """Top-k factors classified as ``universal`` in ``factor_entropy.csv``.

    Uses the saved class assignment from benchmark_analyze (which writes
    factor_entropy.csv with shannon_entropy + class columns). Returns up
    to ``k`` universal-class factors ranked by descending Shannon entropy.
    """
    csv = method_dir / "factor_entropy.csv"
    if not csv.exists():
        raise FileNotFoundError(f"Missing factor_entropy.csv at {csv}")
    df = pd.read_csv(csv)
    universal = df[df["class"] == "universal"].sort_values(
        "shannon_entropy", ascending=False)
    if len(universal) == 0:
        raise ValueError(f"No universal-class factors found in {csv}")
    return [int(f) for f in universal["factor_idx"].tolist()[:k]]


def _pick_universal_with_argmin_celltype(
    method_dir: Path, k: int = 3, candidate_pool: int = 10,
) -> tuple[list[int], list[int]]:
    """Top-k universal factors paired with their most-depleted cell type.

    For each universal factor (top by Shannon entropy), use the cell type
    with the lowest L1 ratio for that factor (i.e. the cell type where the
    universal factor is most depleted) as the row. Cell-type indices are
    deduplicated; if two factors would share the same argmin cell type the
    second falls back to its next-most-depleted unused cell type.

    Returns:
        (factor_indices, celltype_indices) — each of length ``k``.
    """
    M = _l1_ratio_matrix(method_dir)
    candidates = _pick_top_universal_factors(method_dir, k=candidate_pool)

    used_cts: set[int] = set()
    selected_factors: list[int] = []
    selected_celltypes: list[int] = []
    for f in candidates:
        col = M[:, f]
        for ct in np.argsort(col):       # ascending: most depleted first
            ct = int(ct)
            if ct not in used_cts:
                selected_factors.append(int(f))
                selected_celltypes.append(ct)
                used_cts.add(ct)
                break
        if len(selected_factors) >= k:
            break
    return selected_factors, selected_celltypes


def _pick_universal_depleted_pairs(
    ref_dir: Path, other_dir: Path, k: int = 3,
    candidate_pool: int = 10,
    n_celltypes: int | None = None,
) -> tuple[list[int], list[int]]:
    """Universal factors in ``ref_dir`` whose matches in ``other_dir`` are
    most depleted.

    For each candidate universal factor in ``ref_dir`` (top by Shannon
    entropy), find the matched factor in ``other_dir`` via gene-loading
    Pearson correlation, then locate the cell type where the matched
    factor's L1 ratio in ``other_dir`` is lowest (most depleted). Returns
    the top-k pairs sorted by depletion severity, with cell-type indices
    deduplicated.

    The intent is to surface the failure mode of the ``other_dir`` method:
    factors that are universal under the reference model but collapse
    under the alternative.

    Returns:
        (ref_factor_indices, celltype_indices) — each of length ``k``.
    """
    candidates = _pick_top_universal_factors(ref_dir, k=candidate_pool)
    ref_loadings = _load_loadings(ref_dir)
    other_loadings = _load_loadings(other_dir)
    matched = _match_factors(
        ref_loadings=ref_loadings,
        query_loadings=other_loadings,
        ref_factor_ids=candidates,
    )
    other_M = _l1_ratio_matrix(other_dir)               # (G, L)

    # For each candidate, find its (cell_type, ratio) where ratio is minimal.
    candidate_pairs = []
    for f_ref, f_other in zip(candidates, matched):
        col = other_M[:, f_other]
        ct = int(np.argmin(col))
        candidate_pairs.append((f_ref, ct, float(col[ct])))

    # Sort by depletion severity (lowest ratio first).
    candidate_pairs.sort(key=lambda r: r[2])

    selected_factors: list[int] = []
    selected_celltypes: list[int] = []
    used_cts: set[int] = set()
    for f_ref, ct, ratio in candidate_pairs:
        if ct in used_cts:
            # Fall through to next-most-depleted cell type for this factor.
            col = other_M[:, matched[candidates.index(f_ref)]]
            for ct2 in np.argsort(col):
                ct2 = int(ct2)
                if ct2 not in used_cts:
                    ct = ct2
                    ratio = float(col[ct2])
                    break
            else:
                continue
        selected_factors.append(int(f_ref))
        selected_celltypes.append(ct)
        used_cts.add(ct)
        if len(selected_factors) >= k:
            break

    if n_celltypes is not None and n_celltypes != len(selected_celltypes):
        # Expand (or contract) cell-type rows: rank all groups by minimum L1
        # ratio across the selected factors (most-depleted across any chosen
        # universal factor wins).
        sel_cols = [matched[candidates.index(f)] for f in selected_factors]
        min_per_ct = other_M[:, sel_cols].min(axis=1)
        order = np.argsort(min_per_ct)
        selected_celltypes = [int(c) for c in order[:n_celltypes]]
    return selected_factors, selected_celltypes


def _pick_enriched_factor_celltype_pairs(
    method_dir: Path, k: int = 3
) -> tuple[list[int], list[int]]:
    """Top-k (factor, cell-type) pairs by L1 specificity, one row per factor.

    For each of the top-k factors (ranked by max-over-groups L1 ratio),
    return the cell-type index that maximizes that factor's L1 ratio.
    Cell-type indices are deduplicated: if two factors point to the same
    cell type, the later factor falls back to its next-best (unused)
    cell type.

    Returns:
        (factor_indices, celltype_indices) — both lists length ``k``,
        in factor-rank order.
    """
    M = _l1_ratio_matrix(method_dir)                          # (G, L)
    max_ratio = M.max(axis=0)
    factor_order = np.argsort(-max_ratio)

    selected_factors: list[int] = []
    selected_celltypes: list[int] = []
    used_celltypes: set[int] = set()
    for f in factor_order:
        if len(selected_factors) >= k:
            break
        # Cell-type order for this factor, descending by ratio
        ct_order = np.argsort(-M[:, f])
        for ct in ct_order:
            ct = int(ct)
            if ct not in used_celltypes:
                selected_factors.append(int(f))
                selected_celltypes.append(ct)
                used_celltypes.add(ct)
                break
    return selected_factors, selected_celltypes


def _draw_group_loc_panel(ax, coords, groups, group_idx, s):
    """Binary 'is-this-cell-type' panel (white = group, black = other)."""
    mask = groups == group_idx
    ax.scatter(coords[~mask, 0], coords[~mask, 1], c="#222", s=s, alpha=0.6,
               edgecolors="none", rasterized=True)
    ax.scatter(coords[mask, 0], coords[mask, 1], c="#f0f0f0", s=s, alpha=0.95,
               edgecolors="none", rasterized=True)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_block(method_dir: Path, coords: np.ndarray, groups: np.ndarray,
                  group_names: list[str], group_ids: list[int],
                  factor_ids: list[int], s: float,
                  show_group_loc: bool = True,
                  show_row_labels: bool = True) -> plt.Figure:
    """SF-style groupwise-factors panel for one method.

    Args:
        show_group_loc: if True, includes the 'is-this-cell-type' panel as
            the leftmost column of each row. When two blocks are placed
            side-by-side, the right block should set this False (the left
            block already shows the cell-type identities).
        show_row_labels: if True, prints the cell-type name to the left of
            the row. When show_group_loc is True the label sits in the
            margin to the left of the loc panel; when False it sits to the
            left of the first factor panel.
    """
    import textwrap
    factors = np.load(method_dir / "factors.npy")
    groupwise_dir = method_dir / "groupwise_factors"
    groupwise_factors = {}
    for p in sorted(groupwise_dir.glob("group_*.npy"),
                    key=lambda x: int(x.stem.split("_")[1])):
        g = int(p.stem.split("_")[1])
        groupwise_factors[g] = np.load(p)

    vmin = 0.0
    vmax = np.exp(2.3263)
    panel_size = 2.5
    n_rows = len(group_ids) + 1                          # +1 for unconditional row
    n_cols = len(factor_ids) + (1 if show_group_loc else 0)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * panel_size, n_rows * panel_size),
        squeeze=False,
        gridspec_kw=dict(wspace=0.05, hspace=0.05),
    )

    # Row 0: unconditional factor maps (no loc panel above the loc column)
    for col_idx, l in enumerate(factor_ids):
        ax = axes[0, col_idx + (1 if show_group_loc else 0)]
        ax.scatter(coords[:, 0], coords[:, 1], c=factors[:, l],
                   vmin=vmin, vmax=vmax, cmap="turbo", s=s, alpha=0.8,
                   edgecolors="none", rasterized=True)
        ax.invert_yaxis()
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("gray")
        ax.set_title(f"Factor {l + 1}", fontsize=10)
    if show_group_loc:
        axes[0, 0].set_visible(False)

    # Rows 1..: per-group
    for row_idx, g in enumerate(group_ids):
        raw_name = group_names[g] if g < len(group_names) else f"group_{g}"
        # Keep the label on a single line if possible by using a wider wrap.
        display_name = textwrap.fill(raw_name.replace("_", " "), width=22)

        first_factor_col = 1 if show_group_loc else 0
        if show_group_loc:
            ax_loc = axes[row_idx + 1, 0]
            _draw_group_loc_panel(ax_loc, coords, groups, g, s=s)
            if show_row_labels:
                ax_loc.set_ylabel(display_name, rotation=0, fontsize=11,
                                  labelpad=15, ha="right", va="center")

        factors_g = groupwise_factors.get(g)
        for col_idx, l in enumerate(factor_ids):
            ax = axes[row_idx + 1, col_idx + first_factor_col]
            if factors_g is not None:
                ax.scatter(coords[:, 0], coords[:, 1], c=factors_g[:, l],
                           vmin=vmin, vmax=vmax, cmap="turbo", s=s, alpha=0.8,
                           edgecolors="none", rasterized=True)
            ax.invert_yaxis()
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("gray")

            # When the loc panel is absent, hang the row label off the first
            # factor panel instead.
            if (not show_group_loc) and show_row_labels and col_idx == 0:
                ax.set_ylabel(display_name, rotation=0, fontsize=11,
                              labelpad=15, ha="right", va="center")

    return fig


def _fig_to_pil(fig: plt.Figure, dpi: int = 200) -> Image.Image:
    """Render a matplotlib figure to PIL via bbox_inches='tight'.

    Going through a temp PNG ensures matplotlib expands the figure bbox to
    include row labels / suptitles that fall outside the original axes
    rectangle. The simpler ``canvas.draw()`` path crops those off.
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = Image.open(buf).convert("RGB").copy()
    buf.close()
    plt.close(fig)
    return img


def _save_pub(img: Image.Image, fig_for_pdf: plt.Figure | None,
              output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Saved: {output_path}")
    if fig_for_pdf is not None and output_path.suffix.lower() == ".png":
        pdf_path = output_path.with_suffix(".pdf")
        fig_for_pdf.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")


def render(mode: str, dataset: str, out_path: Path,
           cell_type_indices: list[int] | None = None,
           ref_factor_indices: list[int] | None = None,
           rank_by: str = "moran_i",
           rank_from: str = "right",
           k: int = 3,
           n_celltypes: int | None = None,
           dpi: int = 200) -> None:
    if mode not in _MODE_PAIRS:
        raise ValueError(f"Unknown mode: {mode!r}; choose from {list(_MODE_PAIRS)}")
    (left_rel, right_rel,
     left_lab, right_lab, suptitle,
     left_lab_pdf, right_lab_pdf, suptitle_pdf) = _MODE_PAIRS[mode]
    dataset_root = REPO_ROOT / "outputs" / dataset
    left_dir = dataset_root / left_rel
    right_dir = dataset_root / right_rel
    for d in (left_dir, right_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing run directory: {d}")

    # Metadata
    meta = json.loads((dataset_root / "preprocessed" / "metadata.json").read_text())
    group_names = meta["group_names"]

    # Pick which side drives reference factor + cell-type selection.
    if rank_from not in ("left", "right"):
        raise ValueError(f"rank_from must be 'left' or 'right', got {rank_from!r}")
    ref_dir = left_dir if rank_from == "left" else right_dir

    # Reference factors + cell-type rows derived from the chosen ref side.
    if ref_factor_indices is None:
        if rank_by == "enrichment":
            ref_factor_indices, auto_celltypes = _pick_enriched_factor_celltype_pairs(
                ref_dir, k=k)
            if cell_type_indices is None:
                cell_type_indices = auto_celltypes
        elif rank_by == "universal":
            # Top universal factors paired with each one's most-depleted
            # cell type (argmin L1 ratio within that factor's distribution).
            # A universal factor is by definition near 1.0 across cell types;
            # the argmin row exposes where universality is weakest.
            ref_factor_indices, auto_celltypes = _pick_universal_with_argmin_celltype(
                ref_dir, k=k)
            if cell_type_indices is None:
                cell_type_indices = auto_celltypes
        elif rank_by == "universal_depleted":
            # Pick universal factors in ref_dir whose matched factor in the
            # OTHER side is most depleted (lowest L1 ratio for some cell
            # type). The chosen cell type for each row is that argmin
            # cell type from the other side — this is the failure mode
            # we want to expose.
            other_dir = right_dir if rank_from == "left" else left_dir
            ref_factor_indices, auto_celltypes = _pick_universal_depleted_pairs(
                ref_dir, other_dir, k=k, n_celltypes=n_celltypes)
            if cell_type_indices is None:
                cell_type_indices = auto_celltypes
        elif rank_by == "moran_i":
            ref_factor_indices = _pick_top_spatial_factors(
                ref_dir / "metrics.json", k=k)
        else:
            raise ValueError(f"Unknown rank_by={rank_by!r}; "
                             f"choose 'moran_i', 'enrichment', or 'universal'.")

    if cell_type_indices is None:
        n_ct = n_celltypes if n_celltypes is not None else k
        wanted = ["CA1_CA2_CA3_Subiculum", "Oligodendrocytes", "DentatePyramids"]
        cell_type_indices = [group_names.index(w) for w in wanted if w in group_names]
        if len(cell_type_indices) < n_ct:
            cell_type_indices = list(range(min(n_ct, len(group_names))))
        cell_type_indices = cell_type_indices[:n_ct]

    # Match the OTHER side's factors to the ref side via gene-loading correlation.
    right_loadings = _load_loadings(right_dir)
    left_loadings = _load_loadings(left_dir)
    if rank_from == "right":
        left_factor_indices = _match_factors(
            ref_loadings=right_loadings, query_loadings=left_loadings,
            ref_factor_ids=ref_factor_indices,
        )
        right_factor_indices = ref_factor_indices
    else:  # rank_from == "left"
        right_factor_indices = _match_factors(
            ref_loadings=left_loadings, query_loadings=right_loadings,
            ref_factor_ids=ref_factor_indices,
        )
        left_factor_indices = ref_factor_indices
    matched_left_factors = left_factor_indices  # kept for downstream compat

    # Spatial coords + groups (shared across methods — both fit the same data)
    coords = np.load(dataset_root / "preprocessed" / "X.npy")
    # groups vector: prefer the preprocessed C, fall back to either method's groupsZ
    C_path = dataset_root / "preprocessed" / "C.npy"
    if C_path.exists():
        groups_np = np.load(C_path)
    else:
        groups_np = np.load(right_dir / "groupsZ.npy")
    N = coords.shape[0]
    s = _auto_point_size(N)

    print(f"Reference side: {rank_from}; top-3 factors by {rank_by}: {ref_factor_indices}")
    other_side = "left" if rank_from == "right" else "right"
    other_idx = left_factor_indices if rank_from == "right" else right_factor_indices
    print(f"Matched {other_side}-side factor indices (gene-loading r): {other_idx}")
    print(f"Cell-type rows: {[group_names[g] for g in cell_type_indices]}")

    # Left block carries the cell-type location panel + row labels;
    # right block omits both to avoid redundancy when stitched together.
    fig_left = _render_block(left_dir, coords, groups_np, group_names,
                              cell_type_indices, left_factor_indices, s,
                              show_group_loc=True, show_row_labels=True)
    fig_right = _render_block(right_dir, coords, groups_np, group_names,
                               cell_type_indices, right_factor_indices, s,
                               show_group_loc=False, show_row_labels=False)

    img_left = _fig_to_pil(fig_left, dpi=dpi)
    img_right = _fig_to_pil(fig_right, dpi=dpi)

    # Stitch side-by-side with a small column gap + a top banner with labels.
    # Subplot titles only; no global suptitle (left/right block labels are
    # sufficient and the panel already conveys the comparison).
    gap_x = 40
    banner_h = 50
    W = img_left.size[0] + gap_x + img_right.size[0]
    H = banner_h + max(img_left.size[1], img_right.size[1])
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(img_left, (0, banner_h))
    canvas.paste(img_right, (img_left.size[0] + gap_x, banner_h))

    # Banner labels using matplotlib for consistent font
    from PIL import ImageDraw
    try:
        from PIL import ImageFont
        font_candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/google-droid/DroidSans-Bold.ttf",
        ]
        font = None
        for fp in font_candidates:
            if Path(fp).exists():
                font = ImageFont.truetype(fp, 30)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = None

    draw = ImageDraw.Draw(canvas)

    def _text_width(text: str) -> int:
        try:
            bbox = font.getbbox(text)
            return bbox[2] - bbox[0]
        except Exception:
            return 8 * len(text)  # rough fallback

    label_y = banner_h - 40
    # Left block label, centered over the factor columns only (the left block
    # has an extra cell-type-location column at column 0 that should not pull
    # the title leftward).
    n_left_cols = len(left_factor_indices) + 1  # cell-type-loc + factors
    left_factor_x0 = img_left.size[0] / n_left_cols  # start of factor area
    # ``img_left`` is rendered with bbox_inches="tight", so it includes the
    # row-label margin to the left of the location column. Compensate for that
    # extra margin when centering the title over the factor columns.
    left_cx = int(left_factor_x0 + (img_left.size[0] - left_factor_x0) / 2
                  + 0.025 * img_left.size[0])
    draw.text((left_cx - _text_width(left_lab) // 2, label_y),
              left_lab, fill="black", font=font)
    # Right block label, centered over the right subplot's factor columns
    # (no cell-type-location column on this side).
    right_cx = img_left.size[0] + gap_x + img_right.size[0] // 2
    draw.text((right_cx - _text_width(right_lab) // 2, label_y),
              right_lab, fill="black", font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    print(f"Saved: {out_path}")

    # Vector PDF: re-render the two blocks side-by-side via matplotlib
    # PDF: save the same PIL canvas to PDF so the PNG and PDF have identical
    # layout (titles, alignment, banner). Pillow can write RGB images to PDF
    # directly. This avoids the imshow-in-matplotlib-axes path, which added
    # aspect-ratio padding that misaligned the left title relative to the PNG.
    pdf_path = out_path.with_suffix(".pdf")
    canvas.save(pdf_path, "PDF", resolution=dpi)
    print(f"Saved: {pdf_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(_MODE_PAIRS.keys()))
    p.add_argument("--dataset", default="slideseq")
    p.add_argument("--out", required=True)
    p.add_argument("--cell-types", default=None,
                   help="Comma-separated cell-type names (default: hippocampus triad)")
    p.add_argument("--factors", default=None,
                   help="Comma-separated reference factor indices "
                        "(overrides --rank-by)")
    p.add_argument("--rank-by",
                   choices=["moran_i", "enrichment", "universal",
                            "universal_depleted"],
                   default="moran_i",
                   help="Auto-select reference factors. moran_i: highest "
                        "spatial autocorrelation. enrichment: highest "
                        "max-over-groups L1 specificity ratio (benchmark "
                        "celltype_enriched class). universal: highest "
                        "Shannon entropy of normalized L1 ratios (benchmark "
                        "universal class). universal_depleted: universal "
                        "factors in the ref side whose matched factor in "
                        "the OTHER side is most depleted; row cell types "
                        "are the argmin-L1 group on the other side.")
    p.add_argument("--rank-from", choices=["left", "right"], default="right",
                   help="Which method picks reference factors + cell-type "
                        "rows. The other method's columns are matched in via "
                        "gene-loading correlation.")
    p.add_argument("-k", "--n-factors", type=int, default=3,
                   help="Number of (factor, cell-type) pairs to show "
                        "(rows × columns). Default 3.")
    p.add_argument("--n-celltypes", type=int, default=None,
                   help="Number of cell-type rows (defaults to --n-factors). "
                        "When --rank-by=universal_depleted, the extra rows are "
                        "the next-most-depleted cell types across the chosen "
                        "universal factors.")
    args = p.parse_args()

    cell_type_indices = None
    if args.cell_types:
        dataset_root = REPO_ROOT / "outputs" / args.dataset
        meta = json.loads((dataset_root / "preprocessed" / "metadata.json").read_text())
        cell_type_indices = [meta["group_names"].index(c.strip())
                             for c in args.cell_types.split(",")]
    factor_indices = None
    if args.factors:
        factor_indices = [int(f.strip()) for f in args.factors.split(",")]

    render(args.mode, args.dataset, Path(args.out),
           cell_type_indices=cell_type_indices,
           ref_factor_indices=factor_indices,
           rank_by=args.rank_by,
           rank_from=args.rank_from,
           k=args.n_factors,
           n_celltypes=args.n_celltypes)


if __name__ == "__main__":
    main()
