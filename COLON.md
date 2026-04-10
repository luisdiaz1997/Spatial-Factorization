# Colon Cancer MERFISH Dataset Notes

## ⚠️ GeneFormer is NOT an option

GeneFormer-based annotations (`cl46v1SubShort_ds` from `geneformer_anno.csv` files,
`geneformer_adata/` h5ads) were an experimental project by a postdoc (cywu) who has since
left the lab. That work was **never published**. Do not use these as group labels.

---

## Current Setup (Patient 1 only — to be replaced)

The existing pipeline uses a single colon cancer patient (VMSC02001) processed by jcai:

| Field | Value |
|-------|-------|
| h5ad | `/gladstone/engelhardt/lab/jcai/hdp/results/merfish/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_baysor_adata_mingenes0_mincounts10.h5ad` |
| Labels CSV | `/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427_prediction-labels.csv` |
| Group column | `cl46v1SubShort_ds` |
| N cells | ~1,199,060 (full) |
| D genes | 492 |

**Problems with current setup:**
1. `cl46v1SubShort_ds` labels are GeneFormer-generated (see above — not an option)
2. Only Patient 1
3. We want both donors with consistent, non-GeneFormer annotations

---

## Goal

Use **both donors (Patient 1 and Patient 2)** with **non-GeneFormer cell-type labels**
from the original Pelka lab / cywu pipeline.

---

## Key Finding: Who Owned This Data

- **cywu** was the original postdoc who processed both patients and built the CRC scRNA-seq atlas used for annotation
- **rlaursen** did a summer NNMF project on Patient 1 (her `cell_type` labels are derived from her NNMF method, not prior annotations)
- **jcai** ran the current Patient 1 baysor pipeline (used by existing `colon.py`)
- **Pelka lab** created the MERFISH dataset and the CRC scRNA-seq reference atlas

---

## The v11 Annotation System (Original Pelka/cywu Labels)

The **original, non-GeneFormer** annotation for Patient 1 was done in 2022 by cywu using
**scanpy `ingest`** (KNN label transfer) from the Pelka CRC scRNA-seq atlas.

### Patient 1 annotated file (sisi_ingest, 2022):
- **File:** `/gladstone/engelhardt/pelka-collaboration/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427/sisi_tmp_ingest_ingest_cellpose_202202121342_HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_adata_viz_counts_ingest.h5ad`
- Shape: 844,468 × 211 genes (cellpose, Patient 1 only)
- Spatial: `obs["center_x"]`, `obs["center_y"]`

**v11 group columns:**

| Column | Granularity | Values |
|--------|------------|--------|
| `v11_top` | Broad (7) | Epi (400K), Myeloid (154K), Mast (102K), Strom (81K), TNKILC (75K), Plasma (24K), B (8K) |
| `v11_mid` | Detailed (20) | EpiT, EpiN, Macro, Fibro, TCD4, Mono, Plasma, Endo, TCD8, Granulo, DC, B, TZBTB16, Peri, Tgd, SmoothMuscle, NK, ILC, Mast, Schwann |

**Note:** Only 211 genes (a subset), not the full 500-gene panel. Also cellpose segmentation (not baysor).

### CRC scRNA-seq reference atlas (used for ingest):
- **File:** `/gladstone/engelhardt/lab/cywu/CRC_atlas/CRC_atlas_min_genes100_min_cells20_counts/T1000/Objects/CRC_atlas_min_genes100_min_cells20_counts_scRNA_adata_mingenes100_mincells20.h5ad`
- Shape: 370,115 cells × 27,331 genes
- Labels: `obs["cl295v11SubShort"]`, `obs["cl295v11SubFull"]`, `obs["clTopLevel"]`, `obs["clMidwayPr"]`
- This is the Pelka CRC atlas (295 patients, v11 cell type scheme)

### Patient 2 v11 annotations: DO NOT EXIST YET

No pre-existing Patient 2 file with v11 labels has been found anywhere in the lab. Every
Patient 2 h5ad has only `n_counts` (raw) or `HDP_cluster` (integer, uninterpreted).

---

## All Available Files for Both Patients

### Patient 1

| File | N | D | Labels | Segmentation | Notes |
|------|---|---|--------|-------------|-------|
| sisi_ingest (2022, counts) | 844,468 | 211 | `v11_top`, `v11_mid` | Cellpose | **Original Pelka/cywu annotation** |
| `vizgen_public_annotated.h5ad` | 754,866 | 500 | `obs["cell_type"]` (kmeans) | Cellpose | rlaursen NNMF-derived labels, NOT prior annotations |
| `Vizgen_HuColonCa_20220427.h5ad` (Pelka) | 1,239,939 | 492 | `cl46v1SubShort_ds` | Cellpose | GeneFormer — NOT an option |
| `HumanColonCancerPatient1_cellpose_adata_mingenes0_mincounts10.h5ad` | 652,837 | 500 | `n_counts` only | Cellpose | Raw, no labels |
| `HumanColonCancerPatient1_baysor_adata_mingenes0_mincounts10.h5ad` | 893,863 | 500 | `n_counts` only | Baysor | Raw, no labels |
| `geneformer_adata/HuColonCancerPatient1.h5ad` | 893,863 | 500 | `cl46v1SubShort_ds` | Baysor | GeneFormer — NOT an option |
| jcai HDP baysor | ~1.2M | 492 | `cl46v1SubShort_ds` (ext CSV) | Baysor | GeneFormer — NOT an option (current) |

### Patient 2

| File | N | D | Labels | Segmentation | Notes |
|------|---|---|--------|-------------|-------|
| `HumanColonCancerPatient2_cellpose_adata_mingenes0_mincounts10.h5ad` (HTHubID) | 785,725 | 500 | `n_counts` only | Cellpose | Raw, no labels |
| `HumanColonCancerPatient2_baysor_adata_mingenes0_mincounts10.h5ad` (HTHubID) | 1,075,620 | 500 | `n_counts` only | Baysor | Raw, no labels |
| `HumanColonCancerPatient2_adata_HDP.h5ad` (cywu Objects) | 767,659 | 500 | `HDP_cluster` (int) | Cellpose | Uninterpreted clusters |
| `HumanColonCancerPatient2_baysor_adata_HDP.h5ad` (cywu Objects) | 1,075,620 | 500 | `HDP_cluster` (int) | Baysor | Uninterpreted clusters |
| `geneformer_adata/HuColonCancerPatient2.h5ad` | 1,075,620 | 500 | `cl46v1SubShort_ds` | Baysor | GeneFormer — NOT an option |
| Raw cellpose CSV | 817,589 | — | none | Cellpose | `/gladstone/.../old_files.../HumanColonCancerPatient2/cell_metadata.csv` |

**Patient 2 raw data location:**
`/gladstone/engelhardt/lab/cywu/wynton_sync/Projects/HubID/Vizgen/CRC1/`
- `HumanColonCancerPatient2_cell_by_gene.csv`
- `HumanColonCancerPatient2_cell_metadata.csv`
- `HumanColonCancerPatient2_detected_transcripts.csv`

### rlaursen NNMF annotated (both patients, but rlaursen-derived labels)

| File | N | D | Labels | Notes |
|------|---|---|--------|-------|
| `/gladstone/engelhardt/pelka-collaboration/old_files_10012025/NNMF_files/2025-03-05_adata_annotated.h5ad` | 1,964,907 | 500 | `KNN_Grouped`, `KNN_Mid`, `KNN_Top`, `predicted_celltype` | Both patients via `obs["sample_name"]`; labels from rlaursen's NNMF+KNN method |

Per-patient: `sample_name == "HuColonCancerPatient1"` (892,097) and `"HuColonCancerPatient2"` (1,072,810).
Coords in `obs["x"]`, `obs["y"]` (not obsm).
`KNN_Grouped` values: Epithelial, Fibroblast, Immune, Pericyte, Endothelial, SmoothMuscle.

---

## Strategies (in priority order)

### Strategy 1 (Best): Run scanpy ingest on Patient 2 → get v11 labels

Use the same pipeline cywu used for Patient 1 in 2022:
1. Load Patient 2 cellpose data (785K cells, 500 genes)
2. Run `scanpy.ingest` with the CRC atlas as reference
3. Get `v11_top` / `v11_mid` labels for Patient 2
4. Both patients have consistent labels from the same reference atlas

**Pros:** Consistent with Patient 1 original Pelka/cywu annotations; non-GeneFormer
**Cons:** Requires running a pipeline step (~30 min); 211-gene overlap between atlas and MERFISH panel needs checking

**Files needed:**
- Reference: CRC atlas h5ad (370K × 27K, has `cl295v11SubShort`, `clTopLevel`, etc.)
- Query: Patient 2 cellpose h5ad (785K × 500 genes)
- Reference PCA must be computed on the gene overlap between atlas and MERFISH panel

**Loader change:** new config param `h5ad_path` pointing to Patient 2 file post-ingest; `group_column: v11_top` (or `v11_mid`)

---

### Strategy 2 (Quick): Use rlaursen NNMF file, both patients, `KNN_Grouped`

Use `2025-03-05_adata_annotated.h5ad` split by `sample_name`.

**Pros:** Both patients ready now; 6 clean interpretable categories; 500 genes
**Cons:** Labels are rlaursen's NNMF-derived, not the original Pelka/cywu v11 system

**Loader changes needed:**
- Filter by `obs["sample_name"]`
- Use `obs["x"]`, `obs["y"]` for spatial coords
- Use `obs["KNN_Grouped"]` as group column

---

### Strategy 3 (Discarded): GeneFormer `cl46v1SubShort_ds`

❌ Not an option — experimental, unpublished, postdoc left.

---

## Spatial Plots (for reference)

Patient 1 v11 plots saved to `notebooks/colon_exploration/`:
- `colon_v11_top.png` — 7 broad cell types, clear spatial organization
- `colon_v11_mid.png` — 20 detailed types

Key observation: v11 labels show strong spatial structure (Strom dominates lower-left,
Epi forms gland-like patches, Myeloid scattered throughout tumor regions).

---

## Desired Config Structure

```
configs/colon/patient1/general.yaml
configs/colon/patient1/general_test.yaml
configs/colon/patient2/general.yaml
configs/colon/patient2/general_test.yaml
outputs/colon/patient1/
outputs/colon/patient2/
```

---

## Index Files (for reference)

rlaursen's `/gladstone/engelhardt/lab/rlaursen/HubID/data/` boolean masks:

| File | Rows | True | Indexes into |
|------|------|------|-------------|
| `cellidx_LH.csv` | 844,468 | 783,549 | Patient 1 cellpose (LH tissue section) |
| `cellidx_patient1.csv` | 677,451 | 640,524 | Patient 1 cellpose (different pipeline) |
| `cellidx_patient2.csv` | 817,588 | 767,087 | Patient 2 cellpose |
| `cellidx_baysor1.csv` | 1,260,830 | 1,137,456 | Patient 1 baysor |

`vizgen_public_annotated.h5ad` cell IDs (2–844,463) confirmed to be **Patient 1 ONLY**
(indexes into the 844,468-cell cellpose segmentation via `cellidx_LH`).
