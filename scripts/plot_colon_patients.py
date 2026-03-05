"""Plot v11_top and v11_mid cell-type maps for Patient 1 and Patient 2.

Uses the same style as figures.py plot_groups():
  - tab10/tab20 colormap, per-group scatter loop
  - ax.invert_yaxis(), facecolor="gray", s=0.3, alpha=0.8
  - markerscale=5 in legend

Outputs (notebooks/colon_exploration/):
    patient1_v11_top.png, patient1_v11_mid.png
    patient2_v11_top.png, patient2_v11_mid.png
    comparison_v11_top.png, comparison_v11_mid.png

Usage:
    conda run -n factorization python scripts/plot_colon_patients.py
"""

from __future__ import annotations

import os

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

PLOT_H = 7.0      # inches: height of the tissue panel
LEGEND_W = 2.5    # inches: width reserved for legend
GAP_W = 0.2       # inches: gap between panels in comparison plots

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "colon_exploration"
)
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {
        "path": "/gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient1_v11_full.h5ad",
        "label": "Patient 1 (full cellpose, 500 genes)",
        "tag": "patient1",
        "coord": "obsm",  # obsm["spatial"]
    },
    {
        "path": "/gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient2_v11_full.h5ad",
        "label": "Patient 2 (full cellpose, 500 genes)",
        "tag": "patient2",
        "coord": "obsm",  # obsm["spatial"]
    },
]

LEVELS = ["v11_top", "v11_mid"]


# Divergent colors for the top 4 most abundant cell types (ColorBrewer Set1)
TOP4_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]  # red, blue, green, orange


def make_colors(n_groups: int) -> list:
    """Same colormap logic as figures.py plot_groups."""
    if n_groups <= 10:
        cmap = plt.cm.tab10
    elif n_groups <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    return [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]


def build_color_map(loaded, level):
    """Build a color map keyed by cell-type name.

    Top 4 most abundant (across all loaded datasets) get divergent colors.
    Remaining types get tab10/tab20 colors.
    """
    from collections import Counter
    counts: Counter = Counter()
    for _, adata in loaded:
        if level not in adata.obs.columns:
            continue
        counts.update(adata.obs[level].value_counts().to_dict())

    top4 = [c for c, _ in counts.most_common(4)]
    all_cats = sorted(counts.keys())
    rest = [c for c in all_cats if c not in top4]

    colors_rest = make_colors(len(rest)) if rest else []
    color_map = {c: TOP4_COLORS[i] for i, c in enumerate(top4)}
    color_map.update({c: colors_rest[i] for i, c in enumerate(rest)})
    return color_map


def get_coords(adata, coord_type: str) -> np.ndarray:
    """Return (N, 2) float32 array [x, y]."""
    if coord_type == "obs":
        return np.stack(
            [adata.obs["center_x"].to_numpy(dtype=np.float32),
             adata.obs["center_y"].to_numpy(dtype=np.float32)],
            axis=1,
        )
    spatial = adata.obsm["spatial"]
    if hasattr(spatial, "values"):
        return np.asarray(spatial.values, dtype=np.float32)
    return np.asarray(spatial, dtype=np.float32)


def _draw_groups(ax, coords, groups_int, colors,
                 group_names, s=None, alpha=0.8):
    if s is None:
        s = 20.0 / np.sqrt(len(coords))
    """Per-group scatter loop: most abundant first (background), rare last (foreground)."""
    n_groups = len(group_names)
    draw_order = sorted(range(n_groups), key=lambda g: -(groups_int == g).sum())
    for g in draw_order:
        mask = groups_int == g
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[colors[g]], s=s, alpha=alpha,
            label=group_names[g], rasterized=True,
        )
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("gray")


def plot_individual(adata, label, tag, coord_type, level, color_map):
    groups_cat = adata.obs[level].astype("category")
    group_names = list(groups_cat.cat.categories)
    groups_int = groups_cat.cat.codes.to_numpy()
    colors = [color_map[g] for g in group_names]
    coords = get_coords(adata, coord_type)

    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    plot_w = PLOT_H * (x_range / y_range)
    fig_w = plot_w + LEGEND_W

    s = 20.0 / np.sqrt(len(coords))

    fig = plt.figure(figsize=(fig_w, PLOT_H))
    ax = fig.add_axes([0, 0, plot_w / fig_w, 1.0])
    _draw_groups(ax, coords, groups_int, colors, group_names, s=s)
    ax.set_title(f"{level} | {label}", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right", bbox_to_anchor=(1.0, 0.5),
        fontsize=8, title="Cell Type", title_fontsize=10,
        markerscale=8 / np.sqrt(s), frameon=True,
    )

    out = os.path.join(OUT_DIR, f"{tag}_{level}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_comparison(loaded, level, color_map):
    """Two-panel side-by-side with a shared color map keyed by cell-type name."""
    all_cats = sorted(color_map.keys())

    # Compute per-panel aspect ratios from actual coordinate ranges
    coords_list = [get_coords(adata, ds["coord"]) for ds, adata in loaded]
    panel_widths = [
        PLOT_H * (c[:, 0].max() - c[:, 0].min()) / (c[:, 1].max() - c[:, 1].min())
        for c in coords_list
    ]
    w1, w2 = panel_widths
    fig_w = w1 + GAP_W + w2 + LEGEND_W

    fig = plt.figure(figsize=(fig_w, PLOT_H))
    ax1 = fig.add_axes([0 / fig_w,              0, w1 / fig_w, 1.0])
    ax2 = fig.add_axes([(w1 + GAP_W) / fig_w,  0, w2 / fig_w, 1.0])

    for ax, (ds, adata), coords in zip([ax1, ax2], loaded, coords_list):
        groups_cat = adata.obs[level].astype("category")
        local_cats = list(groups_cat.cat.categories)
        groups_int = groups_cat.cat.codes.to_numpy()
        local_colors = [color_map[c] for c in local_cats]
        _draw_groups(ax, coords, groups_int, local_colors, local_cats)
        ax.set_title(ds["label"], fontsize=11)

    # Shared legend from the union of all categories
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[c], markersize=8, label=c)
        for c in all_cats
    ]
    fig.legend(
        handles, all_cats,
        loc="center right", bbox_to_anchor=(1.0, 0.5),
        fontsize=8, title="Cell Type", title_fontsize=10,
        markerscale=1, frameon=True,
    )
    fig.suptitle(level, fontsize=13, y=1.01)

    out = os.path.join(OUT_DIR, f"comparison_{level}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ── Load ───────────────────────────────────────────────────────────────────────
loaded = []
for ds in DATASETS:
    if not os.path.exists(ds["path"]):
        print(f"Skipping {ds['tag']}: {ds['path']} not found")
        continue
    print(f"Loading {ds['tag']} ...", flush=True)
    adata = ad.read_h5ad(ds["path"])
    print(f"  {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    loaded.append((ds, adata))

# ── Build global color maps (top 4 get divergent colors, rest get tab colors) ──
COLOR_MAPS: dict = {level: build_color_map(loaded, level) for level in LEVELS}

# ── Individual plots ───────────────────────────────────────────────────────────
for ds, adata in loaded:
    for level in LEVELS:
        if level not in adata.obs.columns:
            print(f"  Skipping {level}: column not found")
            continue
        print(f"  {ds['tag']} {level} ...", flush=True)
        plot_individual(adata, ds["label"], ds["tag"], ds["coord"], level, COLOR_MAPS[level])

# ── Comparison plots ───────────────────────────────────────────────────────────
if len(loaded) == 2:
    for level in LEVELS:
        if all(level in adata.obs.columns for _, adata in loaded):
            print(f"  comparison {level} ...", flush=True)
            plot_comparison(loaded, level, COLOR_MAPS[level])

print("\nDone.")
