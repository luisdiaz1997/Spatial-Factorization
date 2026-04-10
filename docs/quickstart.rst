Quick Start
===========

This page gets you from zero to trained model and figures in under 15 minutes using the
Slide-seq V2 dataset (loaded automatically via squidpy).

.. note::
   Use ``general_test.yaml`` for quick runs (10 iterations). Never use ``general.yaml``
   for testing — it trains for 20,000 iterations and takes hours.

5-Minute Pipeline
-----------------

**Step 1 — Generate per-model configs from a general config:**

.. code-block:: bash

   spatial_factorization generate -c configs/slideseq/general.yaml

This writes five per-model YAML files alongside ``general.yaml``:
``pnmf.yaml``, ``svgp.yaml``, ``mggp_svgp.yaml``, ``lcgp.yaml``, ``mggp_lcgp.yaml``.

**Step 2 — Run the full pipeline (all 5 models in parallel):**

.. code-block:: bash

   spatial_factorization run all -c configs/slideseq/general_test.yaml

A live status table shows training progress, ELBO values, and time estimates
for each model running in parallel across your GPUs and CPU.

**Step 3 — Explore outputs:**

.. code-block:: text

   outputs/slideseq/
   ├── preprocessed/          ← shared by all models (computed once)
   │   ├── X.npy              ← (N, 2) spatial coordinates
   │   ├── Y.npz              ← (D, N) count matrix
   │   └── metadata.json      ← gene names, group names
   ├── svgp/
   │   ├── factors.npy        ← (N, L) spatial factor values
   │   ├── loadings.npy       ← (D, L) gene loadings
   │   ├── moran_i.csv        ← spatial autocorrelation per factor
   │   └── figures/
   │       ├── factors_spatial.png
   │       ├── top_genes.png
   │       └── gene_enrichment.png
   └── mggp_svgp/ ...

Single Model Example
--------------------

To run one model through specific stages:

.. code-block:: bash

   # Preprocess once (shared across all models for this dataset)
   spatial_factorization preprocess -c configs/slideseq/svgp_test.yaml

   # Train → analyze → figures
   spatial_factorization run train analyze figures -c configs/slideseq/svgp_test.yaml

Stages are always executed in pipeline order regardless of the order you list them.

Reading Results in Python
--------------------------

.. code-block:: python

   import numpy as np
   import json

   model_dir = "outputs/slideseq/svgp"

   # Factor values: (N spots, L components), sorted by Moran's I descending
   # Factor 0 has the highest spatial autocorrelation
   factors = np.load(f"{model_dir}/factors.npy")   # (N, L)
   scales  = np.load(f"{model_dir}/scales.npy")    # (N, L) uncertainty

   # Gene loadings: (D genes, L components)
   loadings = np.load(f"{model_dir}/loadings.npy") # (D, L)

   # Moran's I per factor
   import pandas as pd
   moran = pd.read_csv(f"{model_dir}/moran_i.csv")
   print(moran.head())

   # Metrics: reconstruction error, Poisson deviance
   with open(f"{model_dir}/metrics.json") as f:
       metrics = json.load(f)

   # Dataset metadata: gene names, group names
   with open("outputs/slideseq/preprocessed/metadata.json") as f:
       meta = json.load(f)
   gene_names  = meta["gene_names"]   # list of D gene names
   group_names = meta["group_names"]  # list of G group names

Compare Models
--------------

After training multiple models, compare how they represent the same underlying biology:

.. code-block:: bash

   # Side-by-side [2D | 3D] factor comparison: SVGP vs MGGP_SVGP
   spatial_factorization multianalyze -c configs/slideseq/general.yaml svgp mggp_svgp

   # All 5 models — one column per model, 2D (top row) / 3D (bottom row)
   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp pnmf lcgp mggp_lcgp

Output is saved to ``outputs/slideseq/figures/comparison_*.png``.

Next Steps
----------

- :doc:`math` — understand the probabilistic model
- :doc:`models` — choose the right model for your dataset
- :doc:`configuration` — full YAML reference
- :doc:`cli` — all commands and flags
- :doc:`examples/slideseq` — detailed Slide-seq walkthrough
