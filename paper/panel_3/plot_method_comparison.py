"""Three-cell-type × N-factor side-by-side panel for smNSF method comparisons.

Two driver modes:
    --mode vnngp_vs_lcgp   compares KNN-based (VNNGP-style) vs probabilistic
                           neighbor selection, both at group_diff_param=1e6
                           (so MGGP is effectively turned off and the
                           comparison isolates neighbor-selection strategy).
    --mode mggp_gain       compares a≈∞ (~independent GPs per group) vs
                           regular MGGP-LCGP with cross-group sharing, both
                           with probabilistic neighbor selection.

Output layout matches presentation Panel2 conventions:
    Rows   = 3 showcase cell types (CA1 pyramidal, Oligodendrocytes,
             DentatePyramids — the standard hippocampus triad).
    Block A (left)  = 3 factor maps from method A.
    Block B (right) = 3 factor maps from method B.
    Cell type label on the row's left edge.
    Factor index labelled on each column.
    Small horizontal gap between block A and block B for visual separation.

Usage:
    conda run -n factorization python paper/panel_3/plot_method_comparison.py \\
        --mode vnngp_vs_lcgp \\
        --dataset slideseq \\
        --out paper/panel_3/vnngp_vs_lcgp_slideseq.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image


# Publication font defaults.
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

REPO_ROOT = Path(__file__).resolve().parents[2]


_MODE_PAIRS = {
    # mode: (left_method_path, right_method_path, left_label, right_label,
    #        suptitle, default_outname)
    "vnngp_vs_lcgp": (
        "group_diff_1000000_knn/mggp_lcgp",
        "group_diff_1000000_probabilistic/mggp_lcgp",
        "VNNGP-style (knn)",
        "LCGP (probabilistic)",
        "Neighbor-selection strategy (group_diff_param = $\\infty$, MGGP effectively off)",
        "vnngp_vs_lcgp",
    ),
    "mggp_gain": (
        "group_diff_1000000_probabilistic/mggp_lcgp",
        "mggp_lcgp",
        "Independent GPs ($a\\to\\infty$)",
        "MGGP-LCGP (finite $a$)",
        "Multi-group sharing (both probabilistic; only $a$ varies)",
        "mggp_gain",
    ),
}


def _auto_point_size(n_spots: int, tissue_area: float) -> float:
    """Heuristic point size scaling — denser tissue → smaller points."""
    # tissue_area is the bounding-box area; aim for ~2-3x spot coverage.
    density = n_spots / max(tissue_area, 1.0)
    base = 4.0 / max(density ** 0.5, 0.5)
    return max(0.15, min(base, 1.5))


def _panel_imshow(ax, X: np.ndarray, vals: np.ndarray, s: float,
                  vmin: float, vmax: float, cmap: str = "magma") -> None:
    """One factor-map panel."""
    ax.scatter(X[:, 0], X[:, 1], c=vals, vmin=vmin, vmax=vmax,
               cmap=cmap, s=s, alpha=0.85, edgecolors="none", rasterized=True)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")
    for spine in ax.spines.values():
        spine.set_linewidth(0.3)
        spine.set_color("#888")


def _per_column_limits(vals_left: np.ndarray, vals_right: np.ndarray,
                       lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    """Shared color limits per (cell-type, factor) cell across the two methods.

    Use a common percentile-based vmin/vmax so visual comparison is fair.
    """
    pooled = np.concatenate([vals_left, vals_right])
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(pooled, [lo, hi])
    if vmin == vmax:
        vmax = vmin + 1e-6
    return float(vmin), float(vmax)


def _save_pub(fig, output_path: Path, dpi: int = 300) -> None:
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        img = Image.open(output_path)
        if img.mode == "RGBA":
            img.convert("RGB").save(output_path)
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def _pick_top_spatial_factors(metrics_path: Path, k: int = 3) -> list[int]:
    """Top-k factors by Moran's I from the right-side method's metrics."""
    m = json.loads(metrics_path.read_text())
    mi = m["moran_i"]
    sorted_idx = mi.get("sorted_indices")
    if sorted_idx is None:
        # Compute from values
        vals = mi["values"]
        sorted_idx = list(np.argsort(vals)[::-1])
    return [int(i) for i in sorted_idx[:k]]


def render(mode: str, dataset: str, out_path: Path,
           cell_type_indices: list[int] | None = None,
           factor_indices: list[int] | None = None,
           cmap: str = "magma") -> None:
    """Render the comparison panel for one mode."""
    if mode not in _MODE_PAIRS:
        raise ValueError(f"Unknown mode: {mode!r}; choose from {list(_MODE_PAIRS)}")
    left_rel, right_rel, left_lab, right_lab, suptitle, _ = _MODE_PAIRS[mode]
    dataset_root = REPO_ROOT / "outputs" / dataset
    left_dir = dataset_root / left_rel
    right_dir = dataset_root / right_rel
    for d in (left_dir, right_dir):
        if not d.exists():
            raise FileNotFoundError(f"Missing run directory: {d}")

    # Cell-type metadata
    meta = json.loads((dataset_root / "preprocessed" / "metadata.json").read_text())
    group_names = meta["group_names"]
    if cell_type_indices is None:
        # Default hippocampus triad (presentation Panel2 convention)
        wanted = ["CA1_CA2_CA3_Subiculum", "Oligodendrocytes", "DentatePyramids"]
        cell_type_indices = [group_names.index(w) for w in wanted if w in group_names]
        if len(cell_type_indices) < 3:
            # Fall back to first three groups if the canonical names are absent.
            cell_type_indices = list(range(min(3, len(group_names))))

    # Factor indices: top-3 by Moran's I on the right (treatment) method
    if factor_indices is None:
        factor_indices = _pick_top_spatial_factors(right_dir / "metrics.json", k=3)

    # Tissue coordinates + size heuristic
    X = np.load(dataset_root / "preprocessed" / "X.npy")
    bbox = (X[:, 0].max() - X[:, 0].min()) * (X[:, 1].max() - X[:, 1].min())
    s = _auto_point_size(len(X), bbox)

    n_rows = len(cell_type_indices)
    n_factors = len(factor_indices)
    n_cols_per_block = n_factors

    # Figure geometry
    fig_w = 1.6 * 2 * n_cols_per_block + 1.0  # 2 blocks + left margin for labels
    fig_h = 1.6 * n_rows + 0.8                # rows + suptitle/footer
    fig = plt.figure(figsize=(fig_w, fig_h))

    outer = gridspec.GridSpec(
        n_rows, 2, wspace=0.10, hspace=0.10, figure=fig,
        left=0.07, right=0.99, bottom=0.06, top=0.90,
    )

    for row, g in enumerate(cell_type_indices):
        # Load conditional factor maps for this cell type.
        left_factors = np.load(left_dir / "groupwise_factors" / f"group_{g}.npy")
        right_factors = np.load(right_dir / "groupwise_factors" / f"group_{g}.npy")

        for block_idx, (block_factors, _block_dir) in enumerate(
            [(left_factors, left_dir), (right_factors, right_dir)]
        ):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, n_cols_per_block, subplot_spec=outer[row, block_idx],
                wspace=0.06,
            )
            for col, f_idx in enumerate(factor_indices):
                ax = fig.add_subplot(inner[0, col])
                vmin, vmax = _per_column_limits(
                    left_factors[:, f_idx], right_factors[:, f_idx]
                )
                _panel_imshow(ax, X, block_factors[:, f_idx], s,
                              vmin=vmin, vmax=vmax, cmap=cmap)
                if row == 0:
                    ax.set_title(f"Factor {f_idx}", fontsize=9, pad=3)
                if col == 0 and block_idx == 0:
                    ax.text(-0.18, 0.5, group_names[g].replace("_", " "),
                            transform=ax.transAxes, rotation=90,
                            ha="center", va="center", fontsize=9)

    # Block labels above each block
    for block_idx, lab in enumerate([left_lab, right_lab]):
        block_left = outer[0, block_idx].get_position(fig).x0
        block_right = outer[0, block_idx].get_position(fig).x1
        x_mid = 0.5 * (block_left + block_right)
        fig.text(x_mid, 0.93, lab, ha="center", va="bottom",
                 fontsize=11, fontweight="bold")

    fig.suptitle(suptitle, fontsize=12, y=0.98)

    _save_pub(fig, out_path)
    print(f"Saved: {out_path}")
    if out_path.suffix.lower() == ".png":
        print(f"Saved: {out_path.with_suffix('.pdf')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=list(_MODE_PAIRS.keys()))
    p.add_argument("--dataset", default="slideseq")
    p.add_argument("--out", required=True)
    p.add_argument("--cell-types", default=None,
                   help="Comma-separated cell-type names (default: hippocampus triad)")
    p.add_argument("--factors", default=None,
                   help="Comma-separated factor indices (default: top-3 by Moran's I)")
    p.add_argument("--cmap", default="magma")
    args = p.parse_args()

    cell_type_indices = None
    if args.cell_types:
        dataset_root = REPO_ROOT / "outputs" / args.dataset
        meta = json.loads((dataset_root / "preprocessed" / "metadata.json").read_text())
        group_names = meta["group_names"]
        cell_type_indices = [group_names.index(c.strip()) for c in args.cell_types.split(",")]

    factor_indices = None
    if args.factors:
        factor_indices = [int(f.strip()) for f in args.factors.split(",")]

    render(args.mode, args.dataset, Path(args.out),
           cell_type_indices=cell_type_indices,
           factor_indices=factor_indices,
           cmap=args.cmap)


if __name__ == "__main__":
    main()
