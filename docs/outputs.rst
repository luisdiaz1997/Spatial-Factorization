Output Files Reference
======================

This page documents every file produced by the pipeline, including shapes,
formats, and how to load them in Python.

Directory Layout
----------------

.. code-block:: text

   {output_dir}/
   ├── preprocessed/          ← Stage 1 output (shared by all models)
   ├── logs/                  ← Subprocess logs (one file per model)
   ├── run_status.json        ← Multiplex run summary
   ├── pnmf/                  ← Non-spatial model
   ├── svgp/                  ← SVGP model
   ├── mggp_svgp/             ← MGGP_SVGP model
   ├── lcgp/                  ← LCGP model
   └── mggp_lcgp/             ← MGGP_LCGP model

Preprocessed Data
-----------------

Written by ``preprocess``. Shared across all model variants for a given dataset.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - File
     - Shape / Type
     - Description
   * - ``X.npy``
     - ``(N, 2)`` float32
     - Spatial coordinates (divided by ``spatial_scale``).
   * - ``Y.npz``
     - ``(D, N)`` float32 sparse
     - Count matrix in sparse CSR format. Load with ``scipy.sparse.load_npz``.
   * - ``C.npy``
     - ``(N,)`` int32
     - Group codes, 0-indexed. ``-1`` entries are NaN (dropped by NaN filter).
   * - ``metadata.json``
     - JSON
     - ``gene_names`` (list of D), ``group_names`` (list of G), ``N``, ``D``, ``G``,
       filter statistics.

.. code-block:: python

   import numpy as np
   import scipy.sparse as sp
   import json

   X = np.load("outputs/slideseq/preprocessed/X.npy")          # (N, 2)
   Y = sp.load_npz("outputs/slideseq/preprocessed/Y.npz")      # (D, N) sparse
   C = np.load("outputs/slideseq/preprocessed/C.npy")          # (N,) int32

   with open("outputs/slideseq/preprocessed/metadata.json") as f:
       meta = json.load(f)
   gene_names  = meta["gene_names"]   # list of D gene names
   group_names = meta["group_names"]  # list of G group names

Training Outputs
----------------

Written by ``train``.

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - File
     - Type
     - Description
   * - ``model.pth``
     - PyTorch
     - State dict with ``model_state_dict``, ``prior_state_dict``, ``components``,
       and ``hyperparameters``. Always saved.
   * - ``model.pkl``
     - Pickle
     - Full PNMF model object. Saved for pnmf/svgp/lcgp. MGGP variants are not picklable.
   * - ``training.json``
     - JSON
     - ``n_components``, ``elbo``, ``training_time``, ``converged``,
       ``n_iterations``, ``timestamp``, ``model_config``, ``training_config``,
       ``data_info``, ``resumed`` (if ``--resume`` was used).
   * - ``elbo_history.csv``
     - CSV
     - Columns: ``iteration``, ``elbo``. One row per training iteration.
       Appended (not overwritten) when ``--resume`` is used.
   * - ``elbo_history.npy``
     - ``(T,)`` float
     - ELBO values as a numpy array (same data as CSV).
   * - ``config.yaml``
     - YAML
     - Snapshot of the config used for this training run.
   * - ``video_frames.npy``
     - ``(F, N, L)`` float
     - Factor snapshots captured during training (only if ``--video``).
   * - ``video_frame_iters.npy``
     - ``(F,)`` int
     - Training iteration for each video frame.

.. code-block:: python

   import torch
   import pandas as pd

   state = torch.load("outputs/slideseq/svgp/model.pth")
   hyperparams = state["hyperparameters"]   # dict with n_components, prior, K, etc.

   elbo_df = pd.read_csv("outputs/slideseq/svgp/elbo_history.csv")
   import matplotlib.pyplot as plt
   plt.plot(elbo_df["iteration"], elbo_df["elbo"])
   plt.xlabel("Iteration"); plt.ylabel("ELBO"); plt.show()

Analyze Outputs
---------------

Written by ``analyze``. All factor-indexed outputs are sorted by **descending Moran's I**
(Factor 0 = highest spatial autocorrelation).

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - File
     - Shape / Type
     - Description
   * - ``factors.npy``
     - ``(N, L)`` float32
     - :math:`\exp(F)` factor values. Non-negative, minimum > 0.
       Factor 0 has the highest Moran's I.
   * - ``scales.npy``
     - ``(N, L)`` float32
     - Factor uncertainty: standard deviation of :math:`q(F_{i\ell})`.
   * - ``loadings.npy``
     - ``(D, L)`` float32
     - Global gene loadings :math:`W`. Computed via ``transform_W`` over all cells.
   * - ``loadings_group_0.npy``
     - ``(D, L)`` float32
     - Per-group loadings for group 0 (first group in ``metadata.json``).
       One file per group, same Moran's I ordering as ``factors.npy``.
   * - ``Z.npy``
     - ``(M, 2)`` float32
     - Inducing point coordinates. SVGP: :math:`M \ll N`. LCGP: :math:`M = N` (all training points).
   * - ``groupsZ.npy``
     - ``(M,)`` int32
     - Inducing point group assignments. MGGP variants only.
   * - ``Lu.pt``
     - ``(L, M, M)`` float32
     - Cholesky variational covariance. SVGP/MGGP_SVGP only. Load with ``torch.load``.
   * - ``Lu.npy``
     - ``(L, N, K)`` float32
     - VNNGP-style sparse covariance. LCGP/MGGP_LCGP only.
   * - ``moran_i.csv``
     - CSV
     - Columns: ``factor_idx`` (0-indexed, post-reorder), ``moran_i``.
       Sorted descending by Moran's I.
   * - ``gene_enrichment.json``
     - JSON
     - Nested dict: ``{factor_idx: {group_name: lfc_array}}``.
       ``lfc_array`` is a list of :math:`D` log fold change values.
   * - ``metrics.json``
     - JSON
     - ``reconstruction_error`` (MSE), ``poisson_deviance``, ``moran_i_mean``
       (average Moran's I across factors).

.. code-block:: python

   import numpy as np
   import pandas as pd
   import json

   model_dir = "outputs/slideseq/mggp_svgp"

   factors  = np.load(f"{model_dir}/factors.npy")   # (N, L)
   scales   = np.load(f"{model_dir}/scales.npy")    # (N, L)
   loadings = np.load(f"{model_dir}/loadings.npy")  # (D, L)

   # Per-group loadings
   import os
   group_files = sorted(f for f in os.listdir(model_dir) if f.startswith("loadings_group_"))
   group_loadings = [np.load(f"{model_dir}/{f}") for f in group_files]  # list of (D, L)

   # Moran's I
   moran = pd.read_csv(f"{model_dir}/moran_i.csv")
   print(f"Top factor Moran's I: {moran['moran_i'].iloc[0]:.3f}")

   # Gene enrichment for factor 0, all groups
   with open(f"{model_dir}/gene_enrichment.json") as f:
       enrichment = json.load(f)
   factor_0_group_0 = enrichment["0"]["ClusterA"]  # list of D LFC values

   # Metrics
   with open(f"{model_dir}/metrics.json") as f:
       metrics = json.load(f)
   print(f"Poisson deviance: {metrics['poisson_deviance']:.2f}")

Figure Outputs
--------------

Written by ``figures`` to ``{model_dir}/figures/``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Description
   * - ``factors_spatial.png``
     - Grid of 2D scatter plots, one per factor, colored by factor value.
   * - ``scales_spatial.png``
     - Grid of 2D scatter plots, one per factor, colored by uncertainty.
   * - ``elbo_curve.png``
     - ELBO convergence curve across training iterations.
   * - ``points.png``
     - 2D scatter of cells colored by group. SVGP overlays inducing points (×).
   * - ``top_genes.png``
     - Top 10 loading genes per factor (horizontal bar charts).
   * - ``factors_with_genes.png``
     - Factor spatial maps annotated with top gene names.
   * - ``lu_scales_inducing.png``
     - Variational covariance scale at inducing locations (SVGP/MGGP_SVGP only).
   * - ``gene_enrichment.png``
     - Heatmap: factors (rows) × groups (columns) of log fold change.
   * - ``enrichment_factor_{n}.png``
     - Per-factor bar chart of LFC across groups.
   * - ``enrichment_by_group/{group}.png``
     - Per-group bar chart of LFC across factors.
   * - ``celltype_gene_loadings.png``
     - Heatmap of group-specific loadings. Skipped with ``--no-heatmap``.
   * - ``factor_gene_loadings.png``
     - Factor × top-gene loadings heatmap. Skipped with ``--no-heatmap``.

Multiplex Runner Outputs
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - Description
   * - ``{output_dir}/logs/{model}.log``
     - Full stdout/stderr from each training subprocess.
   * - ``run_status.json``
     - Per-job status summary. Written alongside the configs directory for
       directory-based runs; alongside ``output_dir`` for single-dataset runs.

Comparison Figures (``multianalyze``)
--------------------------------------

Written by ``multianalyze`` to ``{output_dir}/figures/``:

.. code-block:: text

   # Two-model comparison
   outputs/slideseq/figures/comparison_svgp_vs_mggp_svgp.png

   # Multi-model comparison
   outputs/slideseq/figures/comparison_svgp_vs_mggp_svgp_vs_pnmf_vs_lcgp_vs_mggp_lcgp.png
