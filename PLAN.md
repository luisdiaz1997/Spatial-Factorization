# Datasets Integration Plan

## Goal

Integrate all datasets that GPzoo supports into the Spatial-Factorization pipeline.
Each dataset needs: a loader in `spatial_factorization/datasets/`, configs in `configs/`,
and a verified preprocess → train → analyze → figures run.

---

## Dataset Inventory

### 1. SlideseqV2 — DONE

**Status:** Fully implemented and working.

**GPzoo ref:** `../GPzoo/gpzoo/datasets/slideseq/common.py:27` — `load_slideseq_with_groups()`
**Notebook:** `../GPzoo/notebooks/Slideseqv2_MGGP_April_2025.ipynb`

| Key | Value |
|-----|-------|
| Source | `sq.datasets.slideseqv2()` |
| N × D | ~41,786 × 4,000 (after QC) |
| Coords | `adata.obsm["spatial"]` |
| Groups | `adata.obs["cluster"]` (Leiden) |
| Expression | `adata.raw.to_adata().X` → then filter MT genes |
| Loader | `spatial_factorization/datasets/slideseq.py` |
| Configs | `configs/slideseq/` (all 5 models + test variants) |

---

### 2. MERFISH squidpy (3D) — NOT IMPLEMENTED

**Status:** No loader. Squidpy dataset, 3D spatial coordinates.

**Notebook:** `../GPzoo/notebooks/merfish_squidpy.ipynb` — uses `sq.datasets.merfish()`

| Key | Value |
|-----|-------|
| Source | `sq.datasets.merfish()` |
| N × D | 73,655 × 161 |
| Coords 2D | `adata.obsm["spatial"]` (Centroid_X, Centroid_Y) |
| Coords 3D | `adata.obsm["spatial3d"]` (Centroid_X, Centroid_Y, Bregma) |
| Groups | `adata.obs["Cell_class"]` (16 cell classes) |
| Expression | `adata.X` (sparse, no `.raw`) |
| Notes | Bregma = Z axis (coronal slice position, 12 slices); notebook rescales Z: `X[:, 2] = X[:, 2]/50.0; X = X*50.0` |

**Notebook hyperparams:**
- `L=12`, `K=50`, `lengthscale=10.0`, `spatial_scale=50`
- `x_batch_size=15000`, `y_batch_size=D=161`

**Loader task:** Support both 2D (use `obsm["spatial"]`) and 3D (use `obsm["spatial3d"]`). The pipeline currently only handles 2D. For now, implement 2D only (one slice or full collapsed).

---

### 3. Liver Healthy MERFISH — NOT IMPLEMENTED

**Status:** No loader.

**Notebook:** `../GPzoo/notebooks/liver_mggp_healthy.ipynb` (and variants: `_matern32.ipynb`, `_matern32_umap_init.ipynb`, `_exploratory.ipynb`)
**SDMBench benchmark:** `../SDMBench/Benchmarks/liver.ipynb` — uses this data for benchmarking

| Key | Value |
|-----|-------|
| Path | `/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_merfish.h5ad` |
| N × D | 90,408 × 317 |
| Coords | `adata.obsm["X_spatial"]` (NOT `"spatial"`) |
| Groups | `adata.obs["Cell_Type"]` (9 cell types: Cholangiocyte, LSEC, HSC 1/2, Hepatocyte 1/2/3, Mac 1/2) |
| Expression | `adata.raw.X` (raw counts) |
| obsm keys | `X_pca`, `X_pca_harmony`, `X_spatial`, `X_umap` |

**GPzoo ref:** `../GPzoo/gpzoo/datasets/liver/common.py:26` — `load_liver_with_groups()`
**GPzoo hyperparams (config.py):** `L=12`, `K=50`, `spatial_scale=50`, `lengthscale=10`, `x_batch_size=13000`, `y_batch_size=317`

---

### 4. Liver Diseased MERFISH — NOT IMPLEMENTED

**Status:** No loader (can reuse the `LiverLoader` with a different path).

**Notebook:** `../GPzoo/notebooks/liver_mggp_desease.ipynb` (and `_init.ipynb`, `_exploratory.ipynb`)

| Key | Value |
|-----|-------|
| Path | `/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_diseased_merfish.h5ad` |
| N × D | 309,808 × 317 |
| Coords | `adata.obsm["X_spatial"]` |
| Groups | `adata.obs["Cell_Type_final"]` (note: different column name than healthy!) |
| Expression | `adata.raw.X` |

**Note:** The obs column for cell types is `Cell_Type_final` (not `Cell_Type`). The loader should take a `cell_type_column` preprocessing param.

---

### 5. Colon Cancer (Vizgen MERFISH) — NOT IMPLEMENTED

**Status:** No loader. Very large dataset (1.2M cells). Groups come from an external label CSV.

**Notebook:** `../GPzoo/notebooks/HuColonCa-FFPE_april2025.ipynb`

| Key | Value |
|-----|-------|
| h5ad path | `/gladstone/engelhardt/lab/jcai/hdp/results/merfish/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_baysor_adata_mingenes0_mincounts10.h5ad` |
| Alt h5ad | `/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427.h5ad` |
| Labels CSV | `/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427_prediction-labels.csv` |
| N × D | 1,199,060 × 492 |
| Coords | `adata.obsm["spatial"]` (a pandas DataFrame with x, y columns — index must be sorted) |
| Groups | From CSV column `cl46v1SubShort_ds` (50 cell types: gB1, gE1_d/n, gM1-7, gS01-15, gTNI01-15, etc.) |
| Expression | `adata.X` (no `.raw`) |
| obs | `obs["index"]`, `obs["n_counts"]` only — groups come from external CSV |

**Loading logic (from notebook):**
```python
adata = ad.read_h5ad(ad_path)
df = pd.read_csv(label_path, index_col=0)  # col: cl46v1SubShort_ds
order = adata.obsm['spatial'].index.argsort()
X = adata.obsm['spatial'].values[order]  # sort by cell index
X = rescale_spatial_coords(X) * 50
Y = adata.X[order].T  # genes × cells
groupsX = df.cl46v1SubShort_ds.values[::10].codes  # subsampled by 10 in notebook
```

**Notebook hyperparams:** `L=12`, `K=50`, `x_batch_size=13000`, `y_batch_size=D=492`

**Note:** Due to size (1.2M cells), the notebook subsampled `::10` to ~120K cells. Our pipeline should support this or use full data with LCGP.

---

### 6. 10x Visium squidpy — ALREADY WORKS

**Status:** The current `tenxvisium.py` loads `sq.datasets.visium_hne_adata()` and extracts groups from `obs["cluster"/"leiden"/"louvain"]`. This already works for the squidpy Visium demo (mouse brain H&E section).

**Notebook:** `../GPzoo/notebooks/Visium_MGGP.ipynb`, `../GPzoo/notebooks/Visium_VNNGP.ipynb`

| Key | Value |
|-----|-------|
| Source | `sq.datasets.visium_hne_adata()` |
| N × D | ~3,309 × 15,000 (after QC) |
| Coords | `adata.obsm["spatial"]` |
| Groups | `adata.obs["cluster"]` / `"leiden"` / `"louvain"]` |
| Expression | `adata.X` |
| Loader | `spatial_factorization/datasets/tenxvisium.py` (current) |
| Configs | `configs/tenxvisium/` — needs to be created |

---

### 7. 10x Visium LIBD DLPFC (SDMBench) — NEEDS PATH PARAM

**Status:** No dedicated loader. The SDMBench h5ad files are the LIBD human dorsolateral prefrontal cortex (DLPFC) dataset (Maynard et al. 2021, *Nature Neuroscience*, https://www.nature.com/articles/s41593-020-00787-0), distributed via the SDMBench benchmark (http://sdmbench.drai.cn/). Structure differs from squidpy Visium (groups in `obs["Region"]` = cortical layers, no cluster columns).

**Notebook:** `../SDMBench/Benchmarks/DLPFC.ipynb`, `../GPzoo/notebooks/Visium_MGGP.ipynb`, `../GPzoo/notebooks/151507_mggp.ipynb`

| Key | Value |
|-----|-------|
| Data dir | `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/` |
| Files | `151507-151676.h5ad` (12 DLPFC slides, 4 donors × 3 sections each) |
| N × D | 3,460–4,789 × 33,538 per slide |
| Coords | `adata.obsm["spatial"]` |
| Groups | `adata.obs["Region"]` (cortical layers: L1–L6 + WM, NaN = outside tissue) |
| Ground truth | `adata.obs["ground_truth"]` (manual layer annotations for benchmarking) |
| Expression | `adata.X` (csr_matrix, no `.raw`) |
| QC | None — SDMBench data is pre-processed |

**GPzoo ref:** `../GPzoo/gpzoo/datasets/tenxvisium/common.py:31` — `load_visium_with_regions()`
**GPzoo hyperparams:** `L=12`, `K=50`, `spatial_scale=50`, `lengthscale=10`, `x_batch_size=4221`, `SVGP_INDUCING=4000`

---

### 9. osmFISH (SDMBench) — NOT IMPLEMENTED

**Status:** No loader.

| Key | Value |
|-----|-------|
| Path | `/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/osmfish.h5ad` |
| N × D | 4,839 × 33 |
| Coords | `adata.obsm["spatial"]` |
| Groups | `adata.obs["ClusterName"]` or `adata.obs["Region"]` |
| Expression | `adata.X` (dense ndarray, NOT sparse) |
| Notes | Tiny gene panel (33 genes), mouse somatosensory cortex |

---

## Implementation Plan

### Step 1 — Keep `tenxvisium.py` (squidpy Visium) + add `sdmbench.py`

**`tenxvisium.py`** — keep as-is; already loads `sq.datasets.visium_hne_adata()` with cluster/leiden/louvain groups. Just create configs for it.

**`spatial_factorization/datasets/sdmbench.py`** — new loader for SDMBench h5ad slides:
- Accepts `path` preprocessing param (required)
- Uses `obs["Region"]` for groups, filters NaN rows
- No QC filtering (SDMBench data is pre-processed)
- Uses `adata.X` directly (no `.raw`)

```python
DEFAULTS = {
    "spatial_scale": 50.0,
    "region_column": "Region",
    # "path" is required — e.g. ".../SDMBench/Data/151507.h5ad"
}
```

### Step 2 — Create `liver.py` loader

**File:** `spatial_factorization/datasets/liver.py`

New `LiverLoader` supporting both healthy and diseased:
```python
DEFAULTS = {
    "spatial_scale": 50.0,
    "path": "/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_merfish.h5ad",
    "cell_type_column": "Cell_Type",  # use "Cell_Type_final" for diseased
}
```

Loading:
- `adata.obsm["X_spatial"]` → coords
- `adata.raw.X` → expression
- `adata.obs[cell_type_column]` → groups

### Step 3 — Create `merfish.py` loader (squidpy MERFISH, 2D)

**File:** `spatial_factorization/datasets/merfish.py`

New `MerfishLoader` using squidpy:
```python
DEFAULTS = {
    "spatial_scale": 50.0,
    "group_column": "Cell_class",   # 16 cell classes
    "use_3d": False,                 # if True, use obsm["spatial3d"]
}
```

Loading:
- `sq.datasets.merfish()`
- `adata.obsm["spatial"]` (2D) or `adata.obsm["spatial3d"]` (3D, X/Y/Bregma)
- `adata.obs["Cell_class"]` → groups
- `adata.X` → expression (no `.raw`)

For now implement 2D only; 3D is a future extension (pipeline currently assumes 2D coords).

### Step 4 — Create `colon.py` loader

**File:** `spatial_factorization/datasets/colon.py`

New `ColonLoader`:
```python
DEFAULTS = {
    "spatial_scale": 50.0,
    "h5ad_path": "/gladstone/engelhardt/lab/jcai/hdp/results/merfish/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_baysor_adata_mingenes0_mincounts10.h5ad",
    "labels_path": "/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427_prediction-labels.csv",
    "group_column": "cl46v1SubShort_ds",
}
```

Loading (from notebook pattern):
- Load h5ad + labels CSV
- Sort cells by `adata.obsm["spatial"].index.argsort()`
- Merge groups from CSV by position (already aligned)
- `adata.X` → expression
- No `.raw`

**Warning:** At 1.2M cells, full data may require large RAM. Consider `subsample` preprocessing param.

### Step 5 — Create `osmfish.py` loader

**File:** `spatial_factorization/datasets/osmfish.py`

New `OsmfishLoader`:
```python
DEFAULTS = {
    "spatial_scale": 50.0,
    "path": "/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/osmfish.h5ad",
    "group_column": "ClusterName",
}
```

Loading:
- `adata.obsm["spatial"]` → coords
- `adata.X` (dense ndarray) → expression (33 genes)
- `adata.obs["ClusterName"]` → groups

### Step 6 — Register all in `__init__.py`

```python
LOADERS = {
    "slideseq":   SlideseqLoader,     # existing
    "tenxvisium": TenxVisiumLoader,   # existing (squidpy Visium)
    "sdmbench":   SDMBenchLoader,     # new (SDMBench h5ad, path param)
    "liver":      LiverLoader,         # new
    "merfish":    MerfishLoader,       # new (squidpy MERFISH 3D)
    "colon":      ColonLoader,         # new
    "osmfish":    OsmfishLoader,       # new
}
```

### Step 7 — Create configs

One `configs/<dataset>/` directory per dataset with:
- `general.yaml` — superset for `generate` command (20k epochs)
- `general_test.yaml` — 10-epoch test

| Dataset | Config dir | `dataset:` key | `path:` needed |
|---------|-----------|----------------|----------------|
| SlideseqV2 | `configs/slideseq/` | `slideseq` | No |
| MERFISH squidpy | `configs/merfish/` | `merfish` | No |
| Liver healthy | `configs/liver/` | `liver` | Yes (healthy path) |
| Liver diseased | `configs/liver_diseased/` | `liver` | Yes (diseased path + `cell_type_column: Cell_Type_final`) |
| Visium squidpy | `configs/tenxvisium/` | `tenxvisium` | No |
| Visium SDMBench 151507 | `configs/sdmbench/` | `sdmbench` | Yes — SDMBench path |
| osmFISH | `configs/osmfish/` | `osmfish` | Yes |
| Colon | `configs/colon/` | `colon` | Yes |

---

## File Change Summary

| File | Action |
|------|--------|
| `spatial_factorization/datasets/tenxvisium.py` | Keep as-is (squidpy Visium, already works) |
| `spatial_factorization/datasets/sdmbench.py` | New: path-based, `obs["Region"]`, no QC |
| `spatial_factorization/datasets/liver.py` | New: `obsm["X_spatial"]`, `raw.X`, configurable group column |
| `spatial_factorization/datasets/merfish.py` | New: `sq.datasets.merfish()`, `obsm["spatial"]`, `obs["Cell_class"]` |
| `spatial_factorization/datasets/colon.py` | New: path-based, external labels CSV, `obsm["spatial"]` sorted by index |
| `spatial_factorization/datasets/osmfish.py` | New: path-based, `obsm["spatial"]`, `obs["ClusterName"]`, dense X |
| `spatial_factorization/datasets/__init__.py` | Register all new loaders |
| `configs/merfish/` | New: general.yaml + general_test.yaml |
| `configs/liver/` | New: general.yaml + general_test.yaml (healthy) |
| `configs/liver_diseased/` | New: general.yaml + general_test.yaml (diseased, Cell_Type_final) |
| `configs/tenxvisium/` | New: general.yaml + general_test.yaml (one slide, e.g. 151507) |
| `configs/osmfish/` | New: general.yaml + general_test.yaml |
| `configs/colon/` | New: general.yaml + general_test.yaml |

---

## Key Notes Per Dataset

### Coordinate conventions
| Dataset | obsm key | Notes |
|---------|----------|-------|
| SlideseqV2 | `obsm["spatial"]` | Standard |
| MERFISH squidpy | `obsm["spatial"]` (2D) or `obsm["spatial3d"]` (3D) | 3D has Bregma as Z |
| Liver healthy | `obsm["X_spatial"]` | Not `"spatial"` |
| Liver diseased | `obsm["X_spatial"]` | Not `"spatial"` |
| Colon | `obsm["spatial"]` (pandas DataFrame) | Must sort by `.index.argsort()` |
| Visium DLPFC | `obsm["spatial"]` | Standard |
| osmFISH | `obsm["spatial"]` | Standard |

### Expression source
| Dataset | Source | Format |
|---------|--------|--------|
| SlideseqV2 | `adata.raw.to_adata().X` then filter MT | sparse |
| MERFISH squidpy | `adata.X` | sparse |
| Liver (both) | `adata.raw.X` | dense |
| Colon | `adata.X` | dense |
| Visium DLPFC | `adata.X` | sparse |
| osmFISH | `adata.X` | dense ndarray |

### Groups column
| Dataset | Column | N groups |
|---------|--------|----------|
| SlideseqV2 | `obs["cluster"]` | ~20 |
| MERFISH squidpy | `obs["Cell_class"]` | 16 |
| Liver healthy | `obs["Cell_Type"]` | 9 |
| Liver diseased | `obs["Cell_Type_final"]` | TBD |
| Colon | external CSV `cl46v1SubShort_ds` | 50 |
| Visium DLPFC | `obs["Region"]` (filter NaN) | ~7 |
| osmFISH | `obs["ClusterName"]` | ~33 |

### Dataset sizes
| Dataset | N cells | D genes | Notes |
|---------|---------|---------|-------|
| SlideseqV2 | ~41K | ~4K | After QC |
| MERFISH squidpy | 73K | 161 | Squidpy built-in |
| Liver healthy | 90K | 317 | MERFISH panel |
| Liver diseased | 310K | 317 | Large — may need batching |
| Colon | 1.2M | 492 | Very large — consider subsampling |
| Visium 151507 | 4.2K | 33.5K | One slide; many genes |
| osmFISH | 4.8K | 33 | Tiny |

---

## Important Path Reference

```
# SDMBench Visium (12 slides)
/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/151507.h5ad  (4226 × 33538)
/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/151508.h5ad
...
/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/151676.h5ad
/gladstone/engelhardt/home/lchumpitaz/gitclones/SDMBench/Data/osmfish.h5ad (4839 × 33)

# Liver MERFISH
/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_merfish.h5ad          (90408 × 317)
/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/adata_healthy_diseased_merfish.h5ad (309808 × 317)
/gladstone/engelhardt/lab/lchumpitaz/datasets/liver/readme.txt

# Colon cancer MERFISH
/gladstone/engelhardt/lab/jcai/hdp/results/merfish/HuColonCa-FFPE-ImmuOnco-LH_VMSC02001_20220427_seed_2024_K100_T500_baysor_adata_mingenes0_mincounts10.h5ad  (1199060 × 492)
/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427.h5ad               (alt)
/gladstone/engelhardt/pelka-collaboration/Vizgen_HuColonCa_20220427_prediction-labels.csv  (groups)

# MERFISH squidpy (3D)
sq.datasets.merfish()  # downloaded automatically (73655 × 161, obsm["spatial3d"])

# GPzoo notebooks (reference)
../GPzoo/notebooks/merfish_squidpy.ipynb              # MERFISH 3D squidpy
../GPzoo/notebooks/MERFISH_MGGP.ipynb                 # MERFISH Moffitt CSV (older, different data)
../GPzoo/notebooks/HuColonCa-FFPE_april2025.ipynb     # Colon
../GPzoo/notebooks/liver_mggp_healthy.ipynb            # Liver healthy
../GPzoo/notebooks/liver_mggp_desease.ipynb            # Liver diseased
../SDMBench/Benchmarks/liver.ipynb                     # Liver SDMBench benchmarks
../SDMBench/Benchmarks/colon.ipynb                     # Colon SDMBench benchmarks
```
