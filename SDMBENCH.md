# SDMBench Integration Notes

Reference: http://sdmbench.drai.cn/ | GitHub: https://github.com/zhaofangyuan98/SDMBench  
Paper: Zhao et al., Nature Methods 2024 (doi: 10.1038/s41592-024-02215-8)

---

## What SDMBench Benchmarks

13 spatial clustering methods evaluated on 34 real datasets (10x Visium, Stereo-Seq, BaristaSeq, MERFISH, osmFISH, STARmap, STARmap*).

**Methods (13 total):**
Louvain, Leiden, SpaGCN, BayesSpace, stLearn, SEDR, CCST, SCAN-IT, STAGATE, SpaceFlow, conST, BASS, DeepST

**Metrics used (we exclude Marker Score):**
- Accuracy: ARI, NMI, HOM, COM
- Continuity: CHAOS, PAS, ASW
- *(Excluded: Moran's I, Geary's C, time, memory)*

---

## Datasets: 10x Visium DLPFC (our focus)

These are the same 12 slides we already have in `configs/sdmbench/`:

| SDMBench Data ID | Slide | Our Config |
|-----------------|-------|-----------|
| 1  | 151507 | `configs/sdmbench/151507/` |
| 2  | 151508 | `configs/sdmbench/151508/` |
| 3  | 151509 | `configs/sdmbench/151509/` |
| 4  | 151510 | `configs/sdmbench/151510/` |
| 5  | 151669 | `configs/sdmbench/151669/` |
| 6  | 151670 | `configs/sdmbench/151670/` |
| 7  | 151671 | `configs/sdmbench/151671/` |
| 8  | 151672 | `configs/sdmbench/151672/` |
| 9  | 151673 | `configs/sdmbench/151673/` |
| 10 | 151674 | `configs/sdmbench/151674/` |
| 11 | 151675 | `configs/sdmbench/151675/` |
| 12 | 151676 | `configs/sdmbench/151676/` |

---

## SDMBench Website API (Reverse-Engineered)

The site is a React (UMI framework) + Ant Design SPA. The backend API is routed via TCM proxy. The URL goes in an `API-URL` header, the service in `API-SERVICE`. All requests go to `POST /tcm/api/?`.

### Working endpoints

```bash
# Step 1: Get TSV path for a dataset (data_index 1–34)
curl -s "http://sdmbench.drai.cn/tcm/api/?" \
  -X POST \
  -H "Content-Type: application/json" \
  -H "API-URL: /sdmbench/merge" \
  -H "API-SERVICE: api" \
  -d '{"task_id_list": [], "data_index": 1}'
# Returns: {"data": {"out_tsv": "/mnt/JINGD/data/file/sdmbench/db/tsv/1.tsv"}, "status": true}

# Step 2: Fetch the TSV content
curl -s "http://sdmbench.drai.cn/file/content/?" \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/mnt/JINGD/data/file/sdmbench/db/tsv/1.tsv"}'
# Returns: {"message": "success", "data": [{Method_name, NMI, HOM, COM, CHAOS, PAS, ASW, Moran'I, Geary's C}, ...]}
```

Pre-computed TSV paths follow the pattern: `/mnt/JINGD/data/file/sdmbench/db/tsv/{data_index}.tsv`

```bash
# Other useful endpoints
curl -s "http://sdmbench.drai.cn/file/content/?" -X POST \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/mnt/JINGD/data/file/sdmbench/example/data.csv"}'   # dataset catalog
```

---

## How to Get Performance Data for All 13 Methods

### Download Script (fully automated)

```python
import requests, json, csv

BASE = "http://sdmbench.drai.cn"
HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "API-URL": "/sdmbench/merge",
    "API-SERVICE": "api",
}

slide_map = {1:"151507",2:"151508",3:"151509",4:"151510",5:"151669",6:"151670",
             7:"151671",8:"151672",9:"151673",10:"151674",11:"151675",12:"151676"}

rows = []
for data_index, slide in slide_map.items():
    r1 = requests.post(f"{BASE}/tcm/api/?", headers=HEADERS,
                       json={"task_id_list": [], "data_index": data_index})
    tsv_path = r1.json()["data"]["out_tsv"]
    r2 = requests.post(f"{BASE}/file/content/?",
                       headers={"Content-Type": "application/json"},
                       json={"file_path": tsv_path})
    for m in r2.json()["data"]:
        rows.append({"dataset": slide, "model": m["Method_name"],
                     "NMI": float(m["NMI"]), "HOM": float(m["HOM"]),
                     "COM": float(m["COM"]), "CHAOS": float(m["CHAOS"]),
                     "PAS": float(m["PAS"]), "ASW": float(m["ASW"])})

with open("outputs/sdmbench/sdmbench_baselines.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["dataset","model","NMI","HOM","COM","CHAOS","PAS","ASW"])
    w.writeheader(); w.writerows(rows)
```

**Status: already run** — results saved at `outputs/sdmbench/sdmbench_baselines.csv`  
14 methods × 12 slides = 167 rows (GraphST is an extra method on the website, not in the original paper's 13).

---

## Result File Format (TSV output from website)

When a benchmark run completes, the TSV contains one row per method with these columns:

```
Method  NMI  ARI  HOM  COM  CHAOS  PAS  ASW  Moran'I  Geary's C
```

The 13 pre-computed methods are always included. Your submitted method appears as an additional row.

---

## Plan: Adding SDMBench Baselines to Our Aggregate Plot

Goal: overlay 13 SDMBench method results on our `benchmark_aggregate.png` for the 12 DLPFC slides.

### Step 1: Collect data

For each of the 12 slides, submit a dummy prediction to the SDMBench website and download the comparison TSV. This gives ARI, NMI, HOM, COM, CHAOS, PAS, ASW for all 13 methods.

Alternatively: run SDMBench locally — the package is at https://github.com/zhaofangyuan98/SDMBench and takes prediction TXTs + h5ad files as input.

### Step 2: Save as CSV

Store results as `outputs/sdmbench/sdmbench_baselines.csv` with columns:
```
dataset, model, ARI, NMI, HOM, COM, CHAOS, PAS, ASW
```
where `model` is one of the 13 SDMBench method names.

### Step 3: Integrate into benchmark_figures.py

In `plot_aggregate()`, load the SDMBench baseline CSV and overlay it on the existing boxplots, or show as a separate set of boxplots in the same figure.

---

## Local Resources

The SDMBench gitclone at `~/gitclones/SDMBench/` has:

- **`download_10x_datasets.sh`** — downloads all 12 h5ad slides from the SDMBench server:
  ```
  http://sdmbench.drai.cn/tcm/download/?file_path=/mnt/JINGD/data/file/sdmbench/db/{sample}.h5ad
  ```
  All 12 h5ads are **already downloaded** to `SDMBench/Data/`.

- **`SpatialClustering/`** — scripts to run all 13 methods locally:
  `BASS_SequencingBased.R`, `BayesSpace_SequencingBased.R`, `CCST_SequencingBased.py`,
  `conST_SequencingBased.py`, `DeepST_SequencingBased.py`, `Leiden_SequencingBased.py`,
  `Louvain_SequencingBased.py`, `SCAN-IT_SequencingBased.py`, `SEDR_SequencingBased.py`,
  `SpaceFlow_SequencingBased.py`, `SpaGCN_SequencingBased.py`, `STAGATE_SequencingBased.py`,
  `stLearn_SequencingBased.py`

- **`Benchmarks/results/`** — our own model results submitted to SDMBench (mggp_vnngp, vnngp, nmf, pca); NOT the 13 SDMBench baselines.

- **`Benchmarks/DLPFC.ipynb`** — shows how to use `from SDMBench import sdmbench` to compute all metrics locally (ARI, NMI, HOM, COM, CHAOS, PAS, ASW, Moran's I, Geary's C).

## Notes

- The 12 DLPFC h5ad files used by SDMBench are already downloaded to `~/gitclones/SDMBench/Data/`.
- SDMBench Python package (`from SDMBench import sdmbench`) **does** compute ARI — `sdmbench.compute_ARI()` is available. Earlier note was wrong.
- CHAOS and PAS are lower-is-better (same convention as our benchmark).
- The 13 SDMBench methods all use fixed-resolution clustering (no spatial GP structure); they are reasonable baselines for comparing against our PNMF/SVGP/LCGP models.
- To get baseline results: run the scripts in `SpatialClustering/` to generate prediction `.txt` files, then score with the SDMBench Python package or submit to the website.
