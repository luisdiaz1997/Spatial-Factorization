"""Plot ground-truth toy factors and groups using the same functions as figures.py."""

import json
import os
import sys

import numpy as np

# Allow importing from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from spatial_factorization.commands.figures import (
    plot_factors_spatial, plot_groupwise_factors, plot_groupwise_factors_3d,
)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

X = np.load(os.path.join(OUT_DIR, "X.npy"))                       # (N, 2)
C = np.load(os.path.join(OUT_DIR, "C.npy"))                       # (N,)
F = np.load(os.path.join(OUT_DIR, "ground_truth_factors.npy"))    # (N, L) raw GP samples

with open(os.path.join(OUT_DIR, "metadata.json")) as f:
    meta = json.load(f)
group_names = meta["group_names"]

# --- groups plot (custom colors: red, blue, yellow) ---
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GROUP_COLORS = ["red", "blue", "yellow"]

fig_groups, ax = plt.subplots(figsize=(7, 6))
for g, (name, color) in enumerate(zip(group_names, GROUP_COLORS)):
    mask = C == g
    ax.scatter(X[mask, 0], X[mask, 1], c=color, s=2.0, alpha=0.8,
               label=name, rasterized=True)
ax.invert_yaxis()
ax.set_xticks([])
ax.set_yticks([])
ax.set_facecolor("gray")
ax.set_title("Cell Types (data)", fontsize=12)
legend_handles = [mpatches.Patch(color=c, label=n)
                  for n, c in zip(group_names, GROUP_COLORS)]
ax.legend(handles=legend_handles, title="Cell Type",
          loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
fig_groups.tight_layout()
out_groups = os.path.join(OUT_DIR, "groups.png")
fig_groups.savefig(out_groups, dpi=150, bbox_inches="tight")
print(f"Saved: {out_groups}")

# --- unconditional factors plot ---
exp_F = np.exp(F)
fig_factors = plot_factors_spatial(exp_F, X, s=2.0)
fig_factors.suptitle("Toy dataset — ground-truth factors (exp)", fontsize=12, y=1.01)
out_factors = os.path.join(OUT_DIR, "ground_truth_factors.png")
fig_factors.savefig(out_factors, dpi=150, bbox_inches="tight")
print(f"Saved: {out_factors}")

# --- groupwise conditional factors plot ---
groupwise = {
    g: np.exp(np.load(os.path.join(OUT_DIR, f"ground_truth_factors_group_{g}.npy")))
    for g in range(len(group_names))
}
fig_gw = plot_groupwise_factors(exp_F, groupwise, X, C, group_names, s=2.0)
fig_gw.suptitle("Toy dataset — conditional factor means per group", fontsize=12)
out_gw = os.path.join(OUT_DIR, "ground_truth_groupwise_factors.png")
fig_gw.savefig(out_gw, dpi=150, bbox_inches="tight")
print(f"Saved: {out_gw}")

# --- groupwise 3D complete (all factors × all groups sorted by cell count) ---
counts = np.bincount(C, minlength=len(group_names))
all_groups = np.argsort(counts)[::-1]  # all groups sorted by cell count descending
fig_3d = plot_groupwise_factors_3d(
    exp_F, groupwise, X, C, group_names,
    top_groups=all_groups, s=2.0, n_factors=exp_F.shape[1], cmap="bwr",
)
out_3d = os.path.join(OUT_DIR, "ground_truth_groupwise_factors_3d_complete.png")
fig_3d.savefig(out_3d, dpi=150, bbox_inches="tight")
print(f"Saved: {out_3d}")
