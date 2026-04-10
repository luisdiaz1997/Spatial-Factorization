Pipeline Stages
===============

The Spatial-Factorization pipeline has four sequential stages.
Each stage reads from and writes to a well-defined directory layout under ``output_dir``.

.. code-block:: text

   preprocess → train → analyze → figures

Stage 1: Preprocess
-------------------

**Command:** ``spatial_factorization preprocess -c CONFIG``

**Reads:** Raw dataset (downloaded or from ``preprocessing.path``)

**Writes:** ``{output_dir}/preprocessed/``

.. code-block:: text

   preprocessed/
   ├── X.npy          ← (N, 2)  float32 spatial coordinates (normalized)
   ├── Y.npz          ← (D, N)  float32 count matrix (sparse CSR)
   ├── C.npy          ← (N,)    int32 group codes, 0..G-1
   └── metadata.json  ← gene_names, group_names, N, D, G, filter stats

**Steps:**

1. Load the dataset with the appropriate loader (squidpy, h5ad, h5ad + CSV)
2. Extract 2D spatial coordinates from ``obsm``
3. Extract group codes from the configured ``obs`` column
4. Apply **NaN filter**: drop cells where coordinates, expression, or group is NaN
5. Apply **small-group filter**: drop groups below ``min_group_fraction`` or ``min_group_size``;
   re-encode surviving codes contiguously (0..G'-1)
6. Normalize coordinates: ``X = X_raw / spatial_scale``
7. Optionally subsample to ``preprocessing.subsample`` cells
8. Save outputs

.. note::
   Run preprocess **once per dataset**, not once per model. All five model variants
   share the same preprocessed data. The ``run all`` command automatically skips
   preprocessing if ``preprocessed/Y.npz`` already exists (use ``--force`` to re-run).

Stage 2: Train
--------------

**Command:** ``spatial_factorization train -c CONFIG [--resume] [--probabilistic]``

**Reads:** ``{output_dir}/preprocessed/``

**Writes:** ``{output_dir}/{model_name}/``

.. code-block:: text

   {model_name}/
   ├── model.pth          ← PyTorch state dict (always)
   ├── model.pkl          ← Full model pickle (non-MGGP models only)
   ├── training.json      ← Metadata: ELBO, iterations, time, config
   ├── elbo_history.csv   ← Per-iteration ELBO values
   ├── elbo_history.npy   ← Same, as numpy array
   └── config.yaml        ← Snapshot of the config used

**Steps:**

1. Load preprocessed data from ``{output_dir}/preprocessed/``
2. Clamp ``num_inducing``, ``batch_size``, ``y_batch_size`` to actual data dimensions
3. If ``--resume`` and checkpoint exists: warm-start from ``model.pth``
4. Construct ``PNMF(**config.to_pnmf_kwargs())``
5. Call ``model.fit(Y, coordinates=X, groups=C)`` (spatial models) or ``model.fit(Y)`` (PNMF)
6. Save model, ELBO history, metadata, and config snapshot

**Warm-start resume:**
``--resume`` uses ``_create_warm_start_pnmf``, a PNMF subclass that injects the
loaded prior and W matrix instead of random initialization, then runs the normal
``fit()`` training loop. After training, the subclass is reset to ``PNMF`` for
pickling. If no checkpoint exists, training proceeds from scratch (safe for batch runs).

**Pickling:**
MGGP variants (mggp_svgp, mggp_lcgp) use an internal wrapper class that cannot be
pickled — only ``model.pth`` is saved. All other models save both formats.

Stage 3: Analyze
----------------

**Command:** ``spatial_factorization analyze -c CONFIG [--probabilistic]``

**Reads:** ``{model_dir}/model.pth``, ``{output_dir}/preprocessed/``

**Writes:** ``{model_dir}/``

.. code-block:: text

   {model_name}/
   ├── factors.npy              ← (N, L)    exp(F) factor values, sorted by Moran's I
   ├── scales.npy               ← (N, L)    factor uncertainty (std of q(F))
   ├── loadings.npy             ← (D, L)    global gene loadings W
   ├── loadings_group_0.npy     ← (D, L)    per-group loadings, group 0
   ├── loadings_group_1.npy     ← (D, L)    per-group loadings, group 1
   ├── ...                                   (one file per group G)
   ├── Z.npy                    ← (M, 2)    inducing point coordinates
   ├── groupsZ.npy              ← (M,)      inducing group labels (MGGP only)
   ├── Lu.pt                    ← (L, M, M) Cholesky covariance (SVGP)
   ├── Lu.npy                   ← (L, N, K) VNNGP covariance (LCGP)
   ├── moran_i.csv              ← factor_idx, moran_i (sorted descending)
   ├── gene_enrichment.json     ← LFC per factor per group
   └── metrics.json             ← reconstruction_error, poisson_deviance, moran_i_mean

**Steps:**

1. Load model from ``model.pth`` (reconstruct GP class from state dict)
2. Compute factors and scales via batched GP forward pass (batch size: ``analyze_batch_size``)
3. Compute **Moran's I** per factor
4. **Reorder** all factor outputs by descending Moran's I
5. Compute **global loadings** :math:`W` via ``transform_W``
6. Compute **per-group loadings** for each group (groups from ``C.npy``)
7. Compute **gene enrichment** (log fold change per factor per group)
8. Compute **reconstruction error** and **Poisson deviance**
9. Save all outputs

**Batched GP forward pass:**
For large :math:`N`, computing the GP forward pass on all cells at once can cause OOM.
``_get_factors_batched()`` chunks cells into batches of ``analyze_batch_size`` (default 10,000),
runs ``_get_spatial_qF`` per batch, and concatenates results.

**Factor reuse:**
The ``(factors, scales)`` arrays computed in step 2 are reused across all downstream
computations (reconstruction error, Poisson deviance, group loadings). The GP forward
pass is run only once.

Stage 4: Figures
----------------

**Command:** ``spatial_factorization figures -c CONFIG [--no-heatmap]``

**Reads:** All ``.npy``, ``.csv``, ``.json`` files from analyze output

**Writes:** ``{model_dir}/figures/``

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Description
   * - ``factors_spatial.png``
     - :math:`L` panels of 2D scatter plots colored by factor value. Points sized
       by :math:`100/\sqrt{N}` for legibility across datasets.
   * - ``scales_spatial.png``
     - :math:`L` panels of 2D scatter plots colored by factor uncertainty (std of q(F)).
   * - ``elbo_curve.png``
     - ELBO convergence across training iterations from ``elbo_history.csv``.
   * - ``points.png``
     - Scatter of cells colored by group. SVGP: overlays inducing point locations.
       LCGP: shows data groups only (M=N, inducing points = training points).
   * - ``top_genes.png``
     - Top 10 genes per factor shown as horizontal bar charts of loading values.
   * - ``factors_with_genes.png``
     - Factor maps annotated with the names of the top loading genes.
   * - ``gene_enrichment.png``
     - Heatmap of log fold change across factors (rows) and groups (columns).
   * - ``enrichment_factor_*.png``
     - Per-factor bar chart of LFC across all groups.
   * - ``enrichment_by_group/*.png``
     - Per-group bar chart of LFC across all factors.
   * - ``lu_scales_inducing.png``
     - Variational covariance scale at inducing points (SVGP/MGGP_SVGP only).
   * - ``celltype_gene_loadings.png``
     - Heatmap of group-specific loadings (skipped with ``--no-heatmap``).
   * - ``factor_gene_loadings.png``
     - Factor × gene loadings heatmap (skipped with ``--no-heatmap``).

Point size scaling uses the formula :math:`s = 100 / \sqrt{N}`, giving
:math:`s \approx 0.5` for Slide-seq (N=41K) and :math:`s \approx 1.9` for
10x Visium (N~3K).

Chaining Stages with ``run``
-----------------------------

Stages can be chained using the ``run`` command. They are always executed in
pipeline order regardless of the order specified:

.. code-block:: bash

   # Stages are sorted: preprocess → train → analyze → figures
   spatial_factorization run figures train -c configs/slideseq/svgp.yaml
   # Executes: train → figures (preprocess not listed, so not run)

   # Full pipeline for one model
   spatial_factorization run all -c configs/slideseq/svgp.yaml

   # Only train and analyze
   spatial_factorization run train analyze -c configs/slideseq/svgp.yaml

When the config is a general YAML or directory, ``run`` automatically dispatches
to the multiplex runner for parallel execution. See :doc:`multiplex`.
