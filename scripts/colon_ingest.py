"""Scanpy ingest: transfer v11 labels from CRC atlas to Patient 1 or 2 (full cellpose).

Both patients use the full 500-gene cellpose segmentation (not the sisi_ingest/cropped files).
Gene overlap with atlas is used only for PCA projection; all 500 genes are kept in the output.

Usage:
    conda run -n factorization python scripts/colon_ingest.py --patient 1
    conda run -n factorization python scripts/colon_ingest.py --patient 2

Outputs:
    /gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient1_v11_full.h5ad
    /gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient2_v11_full.h5ad
"""

from __future__ import annotations

import argparse
import os
import sys

import scanpy as sc
import numpy as np

ATLAS_PATH = (
    "/gladstone/engelhardt/lab/cywu/CRC_atlas/"
    "CRC_atlas_min_genes100_min_cells20_counts/T1000/Objects/"
    "CRC_atlas_min_genes100_min_cells20_counts_scRNA_adata_"
    "mingenes100_mincells20.h5ad"
)
PATIENT_PATHS = {
    1: (
        "/gladstone/engelhardt/lab/cywu/HubID/HTHubID/"
        "HumanColonCancerPatient1/Objects/"
        "HumanColonCancerPatient1_cellpose_adata_mingenes0_mincounts10.h5ad"
    ),
    2: (
        "/gladstone/engelhardt/lab/cywu/HubID/HTHubID/"
        "HumanColonCancerPatient2/Objects/"
        "HumanColonCancerPatient2_cellpose_adata_mingenes0_mincounts10.h5ad"
    ),
}
OUT_DIR = "/gladstone/engelhardt/lab/lchumpitaz/datasets/colon"
OUT_PATHS = {
    1: os.path.join(OUT_DIR, "patient1_v11_full.h5ad"),
    2: os.path.join(OUT_DIR, "patient2_v11_full.h5ad"),
}

parser = argparse.ArgumentParser()
parser.add_argument("--patient", type=int, choices=[1, 2], required=True)
args = parser.parse_args()
patient = args.patient

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load atlas ─────────────────────────────────────────────────────────────────
print(f"Loading CRC atlas ...", flush=True)
atlas = sc.read_h5ad(ATLAS_PATH)
print(f"  Atlas: {atlas.n_obs:,} × {atlas.n_vars:,} genes", flush=True)

for col in ("clTopLevel", "clMidwayPr"):
    if col not in atlas.obs.columns:
        print(f"ERROR: atlas missing column '{col}'", file=sys.stderr)
        sys.exit(1)

# ── Load patient ───────────────────────────────────────────────────────────────
print(f"\nLoading Patient {patient} ...", flush=True)
p = sc.read_h5ad(PATIENT_PATHS[patient])
print(f"  Patient {patient}: {p.n_obs:,} × {p.n_vars:,} genes", flush=True)

# ── Gene overlap (for ingest only) ────────────────────────────────────────────
overlap = sorted(set(atlas.var_names) & set(p.var_names))
print(f"\nGene overlap: {len(overlap):,} / {p.n_vars} patient genes", flush=True)
if len(overlap) == 0:
    print("ERROR: no gene overlap", file=sys.stderr)
    sys.exit(1)

# Work on overlap subsets for ingest; keep full p intact for output
atlas_sub = atlas[:, overlap].copy()
p_sub = p[:, overlap].copy()

# ── Prepare atlas (PCA → neighbors → UMAP) ────────────────────────────────────
print("\nPreparing atlas (PCA → neighbors → UMAP) ...", flush=True)
print("  Running PCA ...", flush=True)
sc.pp.pca(atlas_sub, n_comps=50)
print("  Computing neighbors ...", flush=True)
sc.pp.neighbors(atlas_sub, use_rep="X_pca")
print("  Running UMAP ...", flush=True)
sc.tl.umap(atlas_sub)

# ── Normalize patient subset to match atlas ───────────────────────────────────
print(f"\nNormalizing Patient {patient} subset ...", flush=True)
sc.pp.normalize_total(p_sub, target_sum=1e4)
sc.pp.log1p(p_sub)

# ── Run ingest ────────────────────────────────────────────────────────────────
print(f"\nRunning sc.tl.ingest on Patient {patient} ...", flush=True)
sc.tl.ingest(p_sub, atlas_sub, obs=["clTopLevel", "clMidwayPr"])
print("  Ingest complete.", flush=True)

# ── Transfer labels to full patient adata (all 500 genes) ────────────────────
p.obs["v11_top"] = p_sub.obs["clTopLevel"].values
p.obs["v11_mid"] = p_sub.obs["clMidwayPr"].values

# ── Print distributions ───────────────────────────────────────────────────────
print(f"\nv11_top distribution:")
print(p.obs["v11_top"].value_counts().to_string())
print(f"\nv11_mid distribution:")
print(p.obs["v11_mid"].value_counts().to_string())

# ── Fix obs index name conflict ───────────────────────────────────────────────
if p.obs.index.name in p.obs.columns:
    p.obs.index.name = None

# ── Save full patient adata with labels ───────────────────────────────────────
out = OUT_PATHS[patient]
print(f"\nSaving to {out} ...", flush=True)
p.write_h5ad(out)
print(f"Done. Saved {p.n_obs:,} cells × {p.n_vars:,} genes → {out}", flush=True)
