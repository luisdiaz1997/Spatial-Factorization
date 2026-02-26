"""Scanpy ingest script: transfer v11 labels from CRC atlas to Patient 2.

Usage:
    conda run -n factorization python scripts/colon_ingest_patient2.py

Inputs:
    CRC atlas (18 GB):
        /gladstone/engelhardt/lab/cywu/CRC_atlas/CRC_atlas_min_genes100_min_cells20_counts/
        T1000/Objects/CRC_atlas_min_genes100_min_cells20_counts_scRNA_adata_mingenes100_mincells20.h5ad
        obs: clTopLevel (→ v11_top), clMidwayPr (→ v11_mid)

    Patient 2 cellpose (3.0 GB):
        /gladstone/engelhardt/lab/cywu/HubID/HTHubID/HumanColonCancerPatient2/Objects/
        HumanColonCancerPatient2_cellpose_adata_mingenes0_mincounts10.h5ad
        N=785,725 × D=500 genes

Output:
    /gladstone/engelhardt/lab/lchumpitaz/datasets/colon/patient2_v11.h5ad
    obs: v11_top, v11_mid (ingested from atlas)
    obsm: spatial (coordinates)
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scanpy as sc

# ── Paths ──────────────────────────────────────────────────────────────────────
ATLAS_PATH = (
    "/gladstone/engelhardt/lab/cywu/CRC_atlas/"
    "CRC_atlas_min_genes100_min_cells20_counts/T1000/Objects/"
    "CRC_atlas_min_genes100_min_cells20_counts_scRNA_adata_"
    "mingenes100_mincells20.h5ad"
)
PATIENT2_PATH = (
    "/gladstone/engelhardt/lab/cywu/HubID/HTHubID/"
    "HumanColonCancerPatient2/Objects/"
    "HumanColonCancerPatient2_cellpose_adata_mingenes0_mincounts10.h5ad"
)
OUT_DIR = "/gladstone/engelhardt/lab/lchumpitaz/datasets/colon"
OUT_PATH = os.path.join(OUT_DIR, "patient2_v11.h5ad")

os.makedirs(OUT_DIR, exist_ok=True)

# ── Load atlas ─────────────────────────────────────────────────────────────────
print(f"Loading CRC atlas from {ATLAS_PATH} ...", flush=True)
atlas = sc.read_h5ad(ATLAS_PATH)
print(f"  Atlas: {atlas.n_obs:,} cells × {atlas.n_vars:,} genes", flush=True)
print(f"  Atlas obs columns: {list(atlas.obs.columns)}", flush=True)

# Verify required label columns
for col in ("clTopLevel", "clMidwayPr"):
    if col not in atlas.obs.columns:
        print(f"ERROR: atlas missing column '{col}'", file=sys.stderr)
        sys.exit(1)

# ── Load Patient 2 ────────────────────────────────────────────────────────────
print(f"\nLoading Patient 2 from {PATIENT2_PATH} ...", flush=True)
p2 = sc.read_h5ad(PATIENT2_PATH)
print(f"  Patient 2: {p2.n_obs:,} cells × {p2.n_vars:,} genes", flush=True)

# ── Gene overlap ──────────────────────────────────────────────────────────────
atlas_genes = set(atlas.var_names)
p2_genes = set(p2.var_names)
overlap = sorted(atlas_genes & p2_genes)
print(f"\nGene overlap: {len(overlap):,} genes (atlas {len(atlas_genes):,}, patient2 {len(p2_genes):,})", flush=True)

if len(overlap) == 0:
    print("ERROR: no gene overlap between atlas and Patient 2", file=sys.stderr)
    sys.exit(1)

# Subset both to overlap
atlas = atlas[:, overlap].copy()
p2 = p2[:, overlap].copy()
print(f"  After subsetting: atlas {atlas.n_obs:,}×{atlas.n_vars:,}, p2 {p2.n_obs:,}×{p2.n_vars:,}", flush=True)

# ── Prepare atlas for ingest ──────────────────────────────────────────────────
# Atlas X is already log-transformed (do NOT re-normalize).
# After gene subsetting, stored X_pca was computed on all 27K genes → invalid.
# We must rerun PCA on the 497-gene overlap, then neighbors + UMAP.
# sc.tl.ingest requires sc.tl.umap() to have been called to store the model.
print("\nPreparing atlas on overlap genes (PCA → neighbors → UMAP) ...", flush=True)
print("  Running PCA ...", flush=True)
sc.pp.pca(atlas, n_comps=50)
print("  Computing neighbors ...", flush=True)
sc.pp.neighbors(atlas, use_rep="X_pca")
print("  Running UMAP (needed for ingest) ...", flush=True)
sc.tl.umap(atlas)

# ── Prepare Patient 2 (normalize + log1p to match atlas) ─────────────────────
# Save raw counts BEFORE normalization — NMF requires non-negative count data.
# After ingest, we restore X to raw counts so the saved h5ad has counts, not log values.
print("\nNormalizing Patient 2 for ingest (saving raw counts in layers) ...", flush=True)
import scipy.sparse as sp
p2.layers["counts"] = p2.X.copy()  # preserve raw counts
sc.pp.normalize_total(p2, target_sum=1e4)
sc.pp.log1p(p2)

# ── Run ingest ────────────────────────────────────────────────────────────────
print("\nRunning sc.tl.ingest (this may take a while) ...", flush=True)
sc.tl.ingest(p2, atlas, obs=["clTopLevel", "clMidwayPr"])
print("  Ingest complete.", flush=True)

# ── Rename columns ────────────────────────────────────────────────────────────
p2.obs["v11_top"] = p2.obs.pop("clTopLevel")
p2.obs["v11_mid"] = p2.obs.pop("clMidwayPr")

# ── Print label distributions ─────────────────────────────────────────────────
print("\nv11_top distribution:")
print(p2.obs["v11_top"].value_counts().to_string())

print("\nv11_mid distribution:")
print(p2.obs["v11_mid"].value_counts().to_string())

# ── Restore raw counts before saving ─────────────────────────────────────────
# NMF needs raw non-negative counts; ingest was done on log-normalized data.
print("\nRestoring raw counts to X ...", flush=True)
p2.X = p2.layers["counts"]

# ── Fix obs index name conflict before saving ─────────────────────────────────
# anndata raises ValueError if index.name matches a column with different values.
if p2.obs.index.name in p2.obs.columns:
    p2.obs.index.name = None

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_PATH} ...", flush=True)
p2.write_h5ad(OUT_PATH)
print(f"Done. Saved {p2.n_obs:,} cells × {p2.n_vars:,} genes.", flush=True)
