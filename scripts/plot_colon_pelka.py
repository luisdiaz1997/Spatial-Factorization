"""Plot pelka-annotated Patient 1 files: sisi_ingest (v11_top/v11_mid)
and vizgen_public_midlevel (cell_type/midlevel_type).

Same style as plot_colon_patients.py.

Outputs (notebooks/colon_exploration/pelka/):
    sisi_v11_top.png, sisi_v11_mid.png
    vizgen_cell_type.png, vizgen_midlevel_type.png
"""

from __future__ import annotations

import os
from collections import Counter

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "colon_exploration", "pelka"
)
os.makedirs(OUT_DIR, exist_ok=True)

TOP4_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]


def make_colors(n_groups: int) -> list:
    if n_groups <= 10:
        cmap = plt.cm.tab10
    elif n_groups <= 20:
        cmap = plt.cm.tab20
    else:
        cmap = plt.cm.gist_ncar
    return [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]


def build_color_map(obs_series) -> dict:
    counts = obs_series.value_counts()
    top4 = list(counts.index[:4])
    all_cats = sorted(counts.index)
    rest = [c for c in all_cats if c not in top4]
    colors_rest = make_colors(len(rest)) if rest else []
    color_map = {c: TOP4_COLORS[i] for i, c in enumerate(top4)}
    color_map.update({c: colors_rest[i] for i, c in enumerate(rest)})
    return color_map


def get_coords_obs(adata) -> np.ndarray:
    return np.stack(
        [adata.obs["center_x"].to_numpy(dtype=np.float32),
         adata.obs["center_y"].to_numpy(dtype=np.float32)],
        axis=1,
    )


def get_coords_obsm(adata) -> np.ndarray:
    spatial = adata.obsm["spatial"]
    if hasattr(spatial, "values"):
        return np.asarray(spatial.values, dtype=np.float32)
    return np.asarray(spatial, dtype=np.float32)


LEGEND_WIDTH_IN = 2.5  # inches reserved for legend on the right


def plot_one(adata, coords, col, title, out_path, s=None, alpha=0.8):
    groups_cat = adata.obs[col].astype("category")
    group_names = list(groups_cat.cat.categories)
    groups_int = groups_cat.cat.codes.to_numpy()
    color_map = build_color_map(adata.obs[col])
    colors = [color_map[g] for g in group_names]

    if s is None:
        s = 20.0 / np.sqrt(len(coords))

    # Derive figsize from actual coordinate ranges to preserve aspect ratio
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    aspect = x_range / y_range  # width / height
    plot_h = 7.0  # inches for the plot area height
    plot_w = plot_h * aspect
    fig_w = plot_w + LEGEND_WIDTH_IN
    fig_h = plot_h

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_order = sorted(range(len(group_names)), key=lambda g: -(groups_int == g).sum())
    for g in draw_order:
        mask = groups_int == g
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[g]], s=s, alpha=alpha,
                   label=group_names[g], rasterized=True)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("gray")
    ax.set_title(title, fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", bbox_to_anchor=(1.0, 0.5),
               fontsize=8, title="Cell Type", title_fontsize=10,
               markerscale=8 / np.sqrt(s), frameon=True)
    right_frac = plot_w / fig_w
    fig.subplots_adjust(right=right_frac)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── sisi_ingest (Patient 1, 844K cells, obs["center_x/y"], v11_top/v11_mid) ───
print("Loading sisi_ingest ...", flush=True)
sisi = ad.read_h5ad(
    "/gladstone/engelhardt/pelka-collaboration/"
    "HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427/"
    "sisi_tmp_ingest_ingest_cellpose_202202121342_"
    "HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_adata_viz_counts_ingest.h5ad"
)
print(f"  {sisi.n_obs:,} cells × {sisi.n_vars} genes")
sisi_coords = get_coords_obs(sisi)

for col, tag in [("v11_top", "sisi_v11_top"), ("v11_mid", "sisi_v11_mid")]:
    print(f"  Plotting {col} ...", flush=True)
    plot_one(sisi, sisi_coords, col,
             f"{col} | sisi_ingest Patient 1 (844K cells, 211 genes)",
             os.path.join(OUT_DIR, f"{tag}.png"))

# ── vizgen_public_midlevel (Patient 1, 754K cells, obsm["spatial"]) ─────────
print("\nLoading vizgen_public_midlevel ...", flush=True)
viz = ad.read_h5ad(
    "/gladstone/engelhardt/pelka-collaboration/"
    "HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427/"
    "HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427/"
    "vizgen_public_midlevel_annotated.h5ad"
)
print(f"  {viz.n_obs:,} cells × {viz.n_vars} genes")
viz_coords = get_coords_obsm(viz)

for col, tag in [("cell_type", "vizgen_cell_type"), ("midlevel_type", "vizgen_midlevel_type")]:
    print(f"  Plotting {col} ...", flush=True)
    plot_one(viz, viz_coords, col,
             f"{col} | vizgen_public Patient 1 (754K cells, 500 genes)",
             os.path.join(OUT_DIR, f"{tag}.png"))

print("\nDone.")
