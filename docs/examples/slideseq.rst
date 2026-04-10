Slide-seq V2 Walkthrough
========================

This example walks through the complete pipeline on the Slide-seq V2 mouse cerebellum
dataset (~41,000 spots, ~4,000 genes, 8 cell type clusters). The dataset is automatically
downloaded via squidpy on first use.

Dataset
-------

- **Source:** ``sq.datasets.slideseq_v2()``
- **N:** ~41,000 spots
- **D:** ~4,000 genes (after filtering)
- **Groups:** 8 cell type clusters (``obs["cluster"]``)
- **Technology:** Slide-seq V2 (10 µm bead resolution)
- **Tissue:** Mouse cerebellum

This is the reference dataset for the pipeline. All five model variants have been validated
on it.

Configuration
-------------

The reference config is at ``configs/slideseq/general.yaml``:

.. code-block:: yaml

   name: slideseq
   seed: 67
   dataset: slideseq
   output_dir: outputs/slideseq

   preprocessing:
     spatial_scale: 50.0
     filter_mt: true
     min_counts: 100
     min_cells: 10
     min_group_fraction: 0.01

   model:
     n_components: 10
     kernel: Matern32
     num_inducing: 3000
     lengthscale: 8.0
     K: 50
     group_diff_param: 1.0

   training:
     max_iter: 20000
     learning_rate: 2e-3
     device: gpu
     batch_size: 7000
     y_batch_size: 2000

Full Pipeline
-------------

.. code-block:: bash

   # Step 1: Generate per-model configs
   spatial_factorization generate -c configs/slideseq/general.yaml

   # Step 2: Run all 5 models in parallel (full training, ~hours on GPU)
   spatial_factorization run all -c configs/slideseq/general.yaml

   # Step 3 (quick test): Run 10 iterations to verify setup
   spatial_factorization run all -c configs/slideseq/general_test.yaml

During training, a live table shows ELBO progress for each model. Expect:

- **pnmf**: ~2–5 min on GPU
- **svgp**: ~1–2 hours on GPU
- **mggp_svgp**: ~1–2 hours on GPU (slightly slower due to group kernel)
- **lcgp**: ~1–3 hours on GPU (K=50 local structure)
- **mggp_lcgp**: ~2–4 hours on GPU

Individual Models
-----------------

.. code-block:: bash

   # Preprocess once (shared by all models)
   spatial_factorization preprocess -c configs/slideseq/svgp_test.yaml

   # Quick SVGP test (10 epochs)
   spatial_factorization run train analyze figures -c configs/slideseq/svgp_test.yaml

   # Full SVGP training
   spatial_factorization run train analyze figures -c configs/slideseq/svgp.yaml

   # Resume MGGP_SVGP from checkpoint
   spatial_factorization train --resume -c configs/slideseq/mggp_svgp.yaml
   spatial_factorization analyze -c configs/slideseq/mggp_svgp.yaml
   spatial_factorization figures -c configs/slideseq/mggp_svgp.yaml --no-heatmap

Reading Results
---------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   import json

   model_dir = "outputs/slideseq/mggp_svgp"
   pre_dir   = "outputs/slideseq/preprocessed"

   # Load outputs
   factors  = np.load(f"{model_dir}/factors.npy")    # (N, L) — sorted by Moran's I
   loadings = np.load(f"{model_dir}/loadings.npy")   # (D, L)
   moran    = pd.read_csv(f"{model_dir}/moran_i.csv")

   # Dataset metadata
   with open(f"{pre_dir}/metadata.json") as f:
       meta = json.load(f)
   gene_names  = meta["gene_names"]
   group_names = meta["group_names"]  # 8 cell type clusters

   # Top genes per factor
   L = factors.shape[1]
   for ell in range(L):
       top_idx = np.argsort(loadings[:, ell])[::-1][:5]
       top_genes = [gene_names[i] for i in top_idx]
       print(f"Factor {ell} (Moran's I={moran['moran_i'].iloc[ell]:.3f}): {top_genes}")

   # Reconstruct expression for one cell
   W   = loadings.T            # (L, D)
   F_0 = factors[0, :]        # (L,) — first cell's factor values
   Y_hat_0 = F_0 @ W          # (D,) — reconstructed expression

Comparing Models
----------------

.. code-block:: bash

   # SVGP vs MGGP_SVGP — find 3 best matched factor pairs
   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp --n-pairs 3

   # All 5 models
   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp pnmf lcgp mggp_lcgp

Factor Interpretation
---------------------

After training, factors are sorted by Moran's I. For the Slide-seq cerebellum:

- **Factor 0** (highest Moran's I): typically captures the Purkinje cell layer, which
  forms a sharp boundary with high spatial autocorrelation
- **Factors 1–3**: molecular layer, granule cell layer, white matter programs
- **Later factors**: finer-grained programs, cell type–specific expression

Look at ``figures/factors_with_genes.png`` to see which genes drive each spatial program,
and ``figures/gene_enrichment.png`` for enrichment across the 8 cell type clusters.

Resume + Probabilistic KNN (LCGP)
-----------------------------------

.. code-block:: bash

   # Continue training LCGP with probabilistic neighbor sampling
   spatial_factorization train --resume --probabilistic -c configs/slideseq/lcgp.yaml

   # Re-analyze with probabilistic KNN (overwrites analyze outputs in place)
   spatial_factorization analyze --probabilistic -c configs/slideseq/lcgp.yaml
   spatial_factorization figures -c configs/slideseq/lcgp.yaml
