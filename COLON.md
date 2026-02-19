# Colon Cancer MERFISH Dataset Notes

## Current Setup (Patient 1 only)

The existing pipeline uses a single colon cancer patient (VMSC02001) processed by jcai:

| Field | Value |
|-------|-------|
| h5ad | `/gladstone/engelhardt/lab/jcai/hdp/results/merfish/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_baysor_adata_mingenes0_mincounts10.h5ad` |
| Labels CSV | `/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427_prediction-labels.csv` |
| Group column | `cl46v1SubShort_ds` |
| N cells | ~1,199,060 (full) |
| D genes | 492 |
| Coords | `obsm["spatial"]` (pandas DataFrame — needs `.index.argsort()` sort) |

---

## Two-Patient Data: What Exists

There are **two colon cancer patients** from the Vizgen MERFISH public data (HuBMAP). Finding all
relevant files across the lab:

### Patient 1 (VMSC02001)

Multiple versions available:

| File | N cells | D genes | Groups | Source |
|------|---------|---------|--------|--------|
| `/gladstone/engelhardt/lab/jcai/hdp/results/merfish/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_baysor_adata_mingenes0_mincounts10.h5ad` | ~1.2M | 492 | external CSV `cl46v1SubShort_ds` | jcai HDP pipeline |
| `/gladstone/engelhardt/pelka-collaboration/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427/Vizgen_HuColonCa_20220427.h5ad` | 1,239,939 | 492 | `obs["cl46v1SubShort_ds"]` (embedded) | Pelka lab — cleaner, no external CSV needed |
| `/gladstone/engelhardt/lab/cywu/HubID/Objects/HumanColonCancerPatient1_adata_HDP.h5ad` | 640,867 | 500 | `obs["HDP_cluster"]` | cywu HDP pipeline |
| `/gladstone/engelhardt/lab/cywu/HubID/Objects/HumanColonCancerPatient1_baysor_adata_HDP.h5ad` | 893,863 | 500 | `obs["HDP_cluster"]` | cywu HDP (baysor) |
| `/gladstone/engelhardt/lab/rlaursen/HubID/data/LH_VMSC02001_sub.h5ad` | 6,620 | 492 | `obs["HDP_cluster"]` | rlaursen subset |

### Patient 2

| File | N cells | D genes | Groups | Source |
|------|---------|---------|--------|--------|
| `/gladstone/engelhardt/lab/cywu/HubID/Objects/HumanColonCancerPatient2_adata_HDP.h5ad` | 767,659 | 500 | `obs["HDP_cluster"]` | cywu HDP pipeline |
| `/gladstone/engelhardt/lab/cywu/HubID/Objects/HumanColonCancerPatient2_baysor_adata_HDP.h5ad` | 1,075,620 | 500 | `obs["HDP_cluster"]` | cywu HDP (baysor) |

**No Patient 2 file with 492 genes + `cl46v1SubShort_ds` labels has been found.** Patient 2
is only available in the cywu HDP format (500 genes, `HDP_cluster`).

---

## rlaursen's Combined Annotated Data

rlaursen's `/gladstone/engelhardt/lab/rlaursen/HubID/data/` directory contains a
combined/annotated dataset and patient-split index files:

### `vizgen_public_annotated.h5ad`
- Shape: **754,866 cells × 500 genes**
- `obs["cell_type"]` — annotated cell types (e.g. "Epithelial cells", "Neutrophils", "Plasma cells", …)
- `obs["kmeans11_anno"]`, `obs["kmeans11"]` — k-means cluster annotations
- `obsm["spatial"]` — spatial coordinates
- x range: 66 – 146,445 (very wide — may be two patients placed side-by-side)
- Same size as `/gladstone/engelhardt/pelka-collaboration/.../Merged_clustering_Anannotated.h5ad`
  (also 754,866 × 500)

Top cell types:
```
Epithelial cells                          517,349
0 CAF2                                     22,210
0 TREM2+ Macrophages_Stromal               21,002
Plasma cells                               20,767
Neutrophils                                18,298
...
```

### Patient split index files

| File | Rows | True count | False count |
|------|------|-----------|-------------|
| `cellidx_patient1.csv` | 677,451 | 640,524 | 36,927 |
| `cellidx_patient2.csv` | 817,588 | 767,087 | 50,501 |

- These are **boolean masks** (column `x`, values True/False)
- The mask lengths (677K and 817K) do **not** match the combined h5ad (754K), so they index
  into two **separate source files** whose locations are currently unknown
- `celltype_LH.csv` — 754,866-row cell type table matching `vizgen_public_annotated.h5ad`
- `under20genes.csv` — 76,596 integer indices of low-quality cells (< 20 genes detected)

---

## TODO: Unresolved Questions

1. **What are the two source files that `cellidx_patient1.csv` / `cellidx_patient2.csv` index into?**
   - Patient 1 source: 677,451 cells → filter to 640,524 (≈ cywu Patient1 HDP 640,867)
   - Patient 2 source: 817,588 cells → filter to 767,087 (≈ cywu Patient2 HDP 767,659)
   - Most likely: the source files are the raw Vizgen cellpose-segmented data for each patient

2. **Best h5ad for each patient going forward:**
   - Patient 1: use Pelka version `Vizgen_HuColonCa_20220427.h5ad` (labels embedded, 492 genes)?
     Or keep the jcai version?
   - Patient 2: use cywu `HumanColonCancerPatient2_adata_HDP.h5ad` (but different gene set
     and `HDP_cluster` groups)?

3. **Desired config structure:**
   - `configs/colon/patient1/general.yaml` + `general_test.yaml`
   - `configs/colon/patient2/general.yaml` + `general_test.yaml`
   - Output dirs: `outputs/colon/patient1/` and `outputs/colon/patient2/`

---

## Possible Next Steps

**Option A — Use Pelka h5ad for Patient 1 + cywu HDP for Patient 2**
- Pro: both patients available now; no new processing needed
- Con: different gene sets (492 vs 500) and different group labels (`cl46v1SubShort_ds` vs `HDP_cluster`)
- Requires updating the colon loader to handle the `HDP_cluster` group column for Patient 2

**Option B — Use rlaursen's combined h5ad split by cellidx**
- Pro: consistent annotation (`cell_type`), same gene set (500 genes), both patients from same pipeline
- Con: need to locate the two source files the cellidx masks index into; requires a new
  split-by-patient loader variant

**Option C — Request Patient 2 processed equivalently to Patient 1**
- Ask jcai or Pelka lab to run the same baysor + HDP pipeline on Patient 2
- Results in `cl46v1SubShort_ds` labels and 492 genes for both
