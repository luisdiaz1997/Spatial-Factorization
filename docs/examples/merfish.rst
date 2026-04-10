MERFISH Example
===============

This example covers the squidpy MERFISH mouse brain dataset: 73,000 cells, 161 genes,
12 cell type groups. MERFISH provides sub-cellular resolution with a targeted gene panel,
making it well-suited for studying cell type–specific spatial programs.

Dataset
-------

- **Source:** ``sq.datasets.merfish()`` (auto-downloaded)
- **N:** 73,000 cells
- **D:** 161 genes (targeted MERFISH panel)
- **Groups:** 12 cell type classes (``obs["Cell_class"]``)
- **Technology:** MERFISH (multiplexed error-robust FISH)
- **Tissue:** Mouse hypothalamus

Configuration
-------------

.. code-block:: yaml

   name: merfish
   seed: 67
   dataset: merfish
   output_dir: outputs/merfish

   preprocessing:
     spatial_scale: 50.0
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
     batch_size: 15000    # larger mini-batches for 73K cells
     y_batch_size: 161    # full gene panel fits in one batch (161 genes)

.. note::
   With only 161 genes, you can set ``y_batch_size: 161`` (full gene batch).
   This is usually faster than gene mini-batching for small D.

Running the Pipeline
--------------------

.. code-block:: bash

   # Generate per-model configs
   spatial_factorization generate -c configs/merfish/general.yaml

   # Full pipeline, all models in parallel
   spatial_factorization run all -c configs/merfish/general.yaml

   # Quick test (10 iterations)
   spatial_factorization run all -c configs/merfish/general_test.yaml

Why MGGP_SVGP?
--------------

MERFISH datasets typically come with high-quality cell type annotations based on
known marker gene expression. These annotations are informative: different cell types
occupy distinct spatial niches in the hypothalamus.

**MGGP_SVGP** with ``group_diff_param: 1.0`` learns a multi-group kernel that captures
both within-cell-type and cross-cell-type spatial correlations. This allows the model
to identify programs that are:

- **Within-group**: e.g., spatial gradients within a single cell type region
- **Cross-group**: e.g., programs shared between neighboring cell types

Compare MGGP_SVGP and SVGP with ``multianalyze`` to see whether multi-group modeling
reveals additional biological structure:

.. code-block:: bash

   spatial_factorization multianalyze -c configs/merfish/general.yaml svgp mggp_svgp

Gene Enrichment
---------------

With 161 targeted genes, gene enrichment is interpretable:

.. code-block:: python

   import json
   import numpy as np

   model_dir = "outputs/merfish/mggp_svgp"
   pre_dir   = "outputs/merfish/preprocessed"

   with open(f"{pre_dir}/metadata.json") as f:
       meta = json.load(f)
   gene_names  = meta["gene_names"]   # 161 MERFISH target genes
   group_names = meta["group_names"]  # 12 cell type classes

   with open(f"{model_dir}/gene_enrichment.json") as f:
       enrichment = json.load(f)

   # Top enriched genes for Factor 0 in group "Excitatory"
   factor_0_exc = enrichment["0"]["Excitatory"]  # list of 161 LFC values
   top_idx = np.argsort(factor_0_exc)[::-1][:10]
   print("Top enriched genes:", [gene_names[i] for i in top_idx])

Resume Training
---------------

For 73K cells, training SVGP for 20,000 iterations takes several hours. Use ``--resume``
to continue from a checkpoint if training was interrupted:

.. code-block:: bash

   # Resume all merfish models
   spatial_factorization run all -c configs/merfish/general.yaml --resume
