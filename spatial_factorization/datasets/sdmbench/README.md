# SDMBench DLPFC Datasets

**Paper:** Zhao et al., Nature Methods 2024 (doi: 10.1038/s41592-024-02215-8)  
**Website:** http://sdmbench.drai.cn/  
**GitHub:** https://github.com/zhaofangyuan98/SDMBench

---

## Files

| Script | Output | Description |
|--------|--------|-------------|
| `download_datasets.sh` | `/gladstone/engelhardt/lab/lchumpitaz/datasets/sdmbench/*.h5ad` | 12 DLPFC h5ad files |
| `download_baselines.py` | `outputs/sdmbench/benchmarks/sdmbench_baselines.csv` | Pre-computed scores for 14 baseline methods |

---

## Datasets — 12 10x Visium DLPFC Slides

| SDMBench ID | Slide | Spots | Genes |
|-------------|-------|-------|-------|
| 1  | 151507 | 4226 | 33538 |
| 2  | 151508 | 4384 | 33538 |
| 3  | 151509 | 4789 | 33538 |
| 4  | 151510 | 4634 | 33538 |
| 5  | 151669 | 3661 | 33538 |
| 6  | 151670 | 3498 | 33538 |
| 7  | 151671 | 4110 | 33538 |
| 8  | 151672 | 4015 | 33538 |
| 9  | 151673 | 3639 | 33538 |
| 10 | 151674 | 3673 | 33538 |
| 11 | 151675 | 3592 | 33538 |
| 12 | 151676 | 3460 | 33538 |

Download endpoint:
```
http://sdmbench.drai.cn/tcm/download/?file_path=/mnt/JINGD/data/file/sdmbench/db/{slide}.h5ad
```

---

## Baselines — Pre-computed Method Scores

**Status:** Downloaded 2026-04-15, saved to `outputs/sdmbench/benchmarks/sdmbench_baselines.csv`

**Methods (14):** BASS, BayesSpace, CCST, GraphST, Leiden, Louvain, SCAN-IT, SEDR, STAGATE, SpaGCN, SpaGCN(HE), SpaceFlow, conST, stLearn

> GraphST is an extra method available on the website but not in the original SDMBench paper's 13.

**Metrics available:** NMI, HOM, COM, CHAOS, PAS, ASW  
**Not available:** ARI (not stored in pre-computed results), Moran's I / Geary's C (excluded — Marker Score)

### How the API Works

The SDMBench website is a React SPA that routes API calls via a TCM proxy. The trick is that the endpoint path goes in an `API-URL` request header, not in the URL:

**Step 1 — get the TSV path for a dataset:**
```
POST http://sdmbench.drai.cn/tcm/api/?
Headers:
  API-URL: /sdmbench/merge
  API-SERVICE: api
Body: {"task_id_list": [], "data_index": 1}

Response: {"data": {"out_tsv": "/mnt/JINGD/data/file/sdmbench/db/tsv/1.tsv"}, "status": true}
```

**Step 2 — fetch the TSV content:**
```
POST http://sdmbench.drai.cn/file/content/?
Body: {"file_path": "/mnt/JINGD/data/file/sdmbench/db/tsv/1.tsv"}

Response: {"message": "success", "data": [{Method_name, NMI, HOM, COM, CHAOS, PAS, ASW, ...}, ...]}
```

Pre-computed TSVs live at `/mnt/JINGD/data/file/sdmbench/db/tsv/{data_index}.tsv` on the server.
