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

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "colon_exploration"
)
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = [
    {
        "path": "/gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient1_v11.h5ad",
        "label": "Patient 1 (2022 Pelka ingest)",
        "tag": "patient1",
        "coord": "obs",   # obs["center_x"], obs["center_y"]
    },
    {
        "path": "/gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient2_v11.h5ad",
        "label": "Patient 2 (scanpy ingest)",
        "tag": "patient2",
        "coord": "obsm",  # obsm["spatial"]
    },
]

LEVELS = ["v11_top", "v11_mid"]


def make_colors(n_groups: int) -> list:
    """Same colormap logic as figures.py plot_groups."""
    if n_groups <= 10:
        cmap = plt.cm.tab10
    elif n_groups <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    return [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]


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
                 group_names, s=0.3, alpha=0.8):
    """Per-group scatter loop exactly as in figures.py plot_groups."""
    n_groups = len(group_names)
    for g in range(n_groups):
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


def plot_individual(adata, label, tag, coord_type, level):
    groups_cat = adata.obs[level].astype("category")
    group_names = list(groups_cat.cat.categories)
    groups_int = groups_cat.cat.codes.to_numpy()
    n_groups = len(group_names)
    colors = make_colors(n_groups)
    coords = get_coords(adata, coord_type)

    fig, ax = plt.subplots(figsize=(14, 6))
    _draw_groups(ax, coords, groups_int, colors, group_names)
    ax.set_title(f"{level} | {label}", fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right", bbox_to_anchor=(1.0, 0.5),
        fontsize=8, title="Cell Type", title_fontsize=10,
        markerscale=5, frameon=True,
    )
    fig.subplots_adjust(right=0.78)

    out = os.path.join(OUT_DIR, f"{tag}_{level}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


def plot_comparison(loaded, level):
    """Two-panel side-by-side with a shared color map keyed by cell-type name."""
    # Build union of categories (same order = same color across both patients)
    all_cats: list = []
    seen: set = set()
    for _, adata in loaded:
        for c in adata.obs[level].astype("category").cat.categories:
            if c not in seen:
                all_cats.append(c)
                seen.add(c)

    n_cats = len(all_cats)
    colors_list = make_colors(n_cats)
    color_map = {c: colors_list[i] for i, c in enumerate(all_cats)}

    fig, axes = plt.subplots(1, 2, figsize=(26, 6))
    fig.subplots_adjust(wspace=0.04, right=0.82)

    for ax, (ds, adata) in zip(axes, loaded):
        groups_cat = adata.obs[level].astype("category")
        local_cats = list(groups_cat.cat.categories)
        groups_int = groups_cat.cat.codes.to_numpy()
        local_colors = [color_map[c] for c in local_cats]
        coords = get_coords(adata, ds["coord"])
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

# ── Individual plots ───────────────────────────────────────────────────────────
for ds, adata in loaded:
    for level in LEVELS:
        if level not in adata.obs.columns:
            print(f"  Skipping {level}: column not found")
            continue
        print(f"  {ds['tag']} {level} ...", flush=True)
        plot_individual(adata, ds["label"], ds["tag"], ds["coord"], level)

# ── Comparison plots ───────────────────────────────────────────────────────────
if len(loaded) == 2:
    for level in LEVELS:
        if all(level in adata.obs.columns for _, adata in loaded):
            print(f"  comparison {level} ...", flush=True)
            plot_comparison(loaded, level)

print("\nDone.")
