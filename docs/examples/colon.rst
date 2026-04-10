Colon Cancer Example (Large Scale)
====================================

The Colon Cancer Vizgen MERFISH dataset presents the most challenging scale in this
pipeline: 1.2 million cells, 492 genes, and dozens of cell type groups. This example
covers configuration choices for extreme-scale data.

Dataset
-------

- **Source:** Vizgen MERFISH (h5ad + CSV labels)
- **N:** 1.2 million cells (full dataset)
- **D:** 492 genes
- **Groups:** Cell type labels from a separate CSV file (column ``cl46v1SubShort_ds``)
- **Technology:** MERFISH (Vizgen MERSCOPE)
- **Tissue:** Human colon cancer

Configuration
-------------

.. code-block:: yaml

   name: colon
   dataset: colon
   output_dir: outputs/colon

   preprocessing:
     path: /path/to/colon.h5ad
     labels_path: /path/to/labels.csv
     subsample: null         # null = all 1.2M cells
     min_group_fraction: 0.01

   model:
     n_components: 10
     kernel: Matern32
     num_inducing: 3000      # M = 3000 << N = 1.2M for SVGP
     lengthscale: 8.0
     group_diff_param: 1.0

   training:
     max_iter: 20000
     learning_rate: 2e-3
     device: gpu
     batch_size: 13000       # mini-batch size for 1.2M cells
     y_batch_size: 492       # full gene panel per batch

Model Choice
------------

.. warning::
   LCGP and MGGP_LCGP are **not feasible** at this scale. LCGP uses M=N=1.2M
   inducing points, which exceeds GPU memory for any realistic K. Use SVGP or
   MGGP_SVGP instead.

For 1.2M cells:

- **SVGP** with ``num_inducing=3000``: The 3000 inducing points are a tiny fraction
  of 1.2M cells but still provide a good global spatial approximation. Training time
  is dominated by the :math:`\mathcal{O}(NM)` kernel computation at each iteration.

- **MGGP_SVGP**: Recommended when cell type annotations are informative. The multi-group
  kernel allows the model to learn cross-cell-type spatial correlations (tumor vs. stroma
  vs. immune infiltrate, etc.).

Generated configs will include pnmf, svgp, and mggp_svgp. The lcgp and mggp_lcgp configs
are generated but training will OOM — skip them explicitly or reduce K dramatically.

Subsampling
-----------

For development and testing, subsample to a manageable size:

.. code-block:: yaml

   preprocessing:
     subsample: 50000         # Use 50K cells instead of 1.2M

.. code-block:: bash

   # Quick test with 10 iterations (full 1.2M cells)
   spatial_factorization run all -c configs/colon/general_test.yaml

   # Or subsample in a custom config
   spatial_factorization run train analyze figures -c configs/colon/svgp_50k.yaml

Memory and Batch Size Tuning
-----------------------------

At 1.2M cells with ``batch_size=13000``, each training step processes ~1% of the data.
The ELBO estimate is noisier but training is feasible.

If GPU OOM occurs during training:

- Reduce ``batch_size`` (13000 → 8000 → 5000)
- Reduce ``num_inducing`` (3000 → 2000 → 1000)

If GPU OOM occurs during analyze (GP forward pass):

- Reduce ``analyze_batch_size`` in the config:

  .. code-block:: yaml

     training:
       analyze_batch_size: 5000   # default is 10000

Running the Pipeline
--------------------

.. code-block:: bash

   # Generate per-model configs
   spatial_factorization generate -c configs/colon/general.yaml

   # Run SVGP and MGGP_SVGP only (skip LCGP)
   spatial_factorization run train analyze figures -c configs/colon/svgp.yaml
   spatial_factorization run train analyze figures -c configs/colon/mggp_svgp.yaml

   # Or run all (LCGP jobs will fail with OOM — use --failed to retry after config adjustment)
   spatial_factorization run all -c configs/colon/general.yaml

   # Resume after interruption
   spatial_factorization run all -c configs/colon/general.yaml --resume

Label Loading
-------------

The colon dataset requires a separate CSV file for cell type labels. The loader reads
the CSV and joins it to the AnnData by cell barcode. The relevant column is
``cl46v1SubShort_ds``.

The colon ``obsm["spatial"]`` coordinates are stored as a pandas DataFrame (unlike
other datasets where it is a numpy array). The loader sorts by index before extracting
coordinates — this is handled transparently.

Reading Results
---------------

.. code-block:: python

   import numpy as np
   import pandas as pd
   import json

   model_dir = "outputs/colon/svgp"
   pre_dir   = "outputs/colon/preprocessed"

   factors  = np.load(f"{model_dir}/factors.npy")    # (1_200_000, L)
   loadings = np.load(f"{model_dir}/loadings.npy")   # (492, L)

   with open(f"{pre_dir}/metadata.json") as f:
       meta = json.load(f)
   gene_names  = meta["gene_names"]   # 492 MERFISH genes
   group_names = meta["group_names"]  # cell type labels (after small-group filter)

   moran = pd.read_csv(f"{model_dir}/moran_i.csv")
   print(f"Factor 0 Moran's I: {moran['moran_i'].iloc[0]:.4f}")

   # Top genes per factor
   for ell in range(factors.shape[1]):
       top5 = np.argsort(loadings[:, ell])[::-1][:5]
       print(f"Factor {ell}: {[gene_names[i] for i in top5]}")

Comparing SVGP and MGGP_SVGP
------------------------------

.. code-block:: bash

   spatial_factorization multianalyze -c configs/colon/general.yaml \
       svgp mggp_svgp --n-pairs 3
