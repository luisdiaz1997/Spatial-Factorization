Configuration Reference
=======================

All pipeline stages are configured via YAML files. This page documents every
available field.

Config File Structure
---------------------

.. code-block:: yaml

   name: slideseq               # Experiment name (used in logs and metadata)
   seed: 67                     # Random seed for reproducibility
   dataset: slideseq            # Dataset key (see Datasets page)
   output_dir: outputs/slideseq # Root directory for all outputs

   preprocessing:               # Dataset-specific preprocessing params
     ...

   model:                       # Model architecture params (passed to PNMF)
     ...

   training:                    # Optimizer and runtime params
     ...

Top-Level Fields
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Default
     - Description
   * - ``name``
     - required
     - Experiment name. Used in ``training.json`` metadata.
   * - ``seed``
     - ``42``
     - Random seed for PyTorch and NumPy.
   * - ``dataset``
     - ``slideseq``
     - Dataset key. See :doc:`datasets` for all valid values.
   * - ``output_dir``
     - ``outputs``
     - Root output directory. Model outputs go to ``{output_dir}/{model_name}/``.

Preprocessing Section
---------------------

.. code-block:: yaml

   preprocessing:
     spatial_scale: 50.0        # Divide raw coordinates by this factor
     filter_mt: true            # Drop mitochondrial genes (MT-prefix)
     min_counts: 100            # Drop cells with fewer total counts
     min_cells: 10              # Drop genes expressed in fewer cells
     min_group_fraction: 0.01   # Drop groups < this fraction of total cells
     min_group_size: 10         # Alternative: drop groups < N cells (absolute)
     subsample: null            # Subsample to this many cells (null = no subsampling)

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``spatial_scale``
     - ``50.0``
     - Raw coordinates are divided by this. With ``lengthscale=8.0``, a lengthscale
       of 8 corresponds to 400 raw coordinate units.
   * - ``filter_mt``
     - ``true``
     - Drop genes whose name starts with ``MT-`` (mitochondrial).
   * - ``min_counts``
     - ``100``
     - Minimum total UMI count per cell.
   * - ``min_cells``
     - ``10``
     - Minimum number of cells expressing a gene (used to filter rare genes).
   * - ``min_group_fraction``
     - not set
     - If set, takes precedence over ``min_group_size``. Groups with fewer than
       ``min_group_fraction * N`` cells are dropped.
   * - ``min_group_size``
     - ``10``
     - Minimum absolute cell count per group. Used only when ``min_group_fraction``
       is not set.
   * - ``subsample``
     - ``null``
     - If set to an integer, randomly subsample to that many cells before saving.

Dataset-specific fields (``path``, ``labels_path``, ``cell_type_column``) are also
placed in the ``preprocessing`` section. See :doc:`datasets` for details.

Model Section
-------------

.. code-block:: yaml

   model:
     # ── Universal ─────────────────────────────────────────────────
     n_components: 10           # L: number of latent factors
     mode: expanded             # ELBO mode: 'expanded' | 'simple' | 'lower-bound'
     loadings_mode: multiplicative  # W constraint mode
     training_mode: standard    # 'standard' | 'natural_gradient'
     E: 3                       # Monte Carlo samples per ELBO step
     scale_ll_D: true           # Scale log-likelihood by 1/D
     scale_kl_NM: true          # Scale KL by N/M

     # ── Spatial (all spatial models) ───────────────────────────────
     spatial: true              # Enable GP prior (false → PNMF baseline)
     groups: false              # Enable multi-group (MGGP) mode
     local: false               # Enable LCGP (locally conditioned) mode
     kernel: Matern32           # 'Matern32' | 'RBF'
     lengthscale: 8.0           # GP kernel lengthscale (in scaled coordinates)
     sigma: 1.0                 # GP signal variance (fixed)
     train_lengthscale: false   # Learn lengthscale during training
     group_diff_param: 1.0      # MGGP cross-group penalty (higher = more independent)

     # ── SVGP-specific ──────────────────────────────────────────────
     num_inducing: 3000         # M: inducing points (auto-clamped to N if N < M)
     cholesky_mode: exp         # Cholesky param: 'exp' | 'softplus'
     diagonal_only: false       # Diagonal-only covariance (faster, less expressive)
     inducing_allocation: derived  # Inducing split: 'proportional' | 'derived' | 'uniform'

     # ── LCGP-specific ──────────────────────────────────────────────
     K: 50                      # Local neighbors per point
     precompute_knn: true       # Precompute KNN at initialization
     neighbors: knn             # 'knn' (FAISS L2) | 'probabilistic' (kernel-weighted)

     # ── Naming override ────────────────────────────────────────────
     model_name_override: null  # Override output directory name

.. list-table:: Model fields reference
   :header-rows: 1
   :widths: 28 15 57

   * - Field
     - Default
     - Description
   * - ``n_components``
     - ``10``
     - Number of latent factors :math:`L`.
   * - ``mode``
     - ``expanded``
     - ELBO computation mode. ``expanded`` (default): analytic rate term + MC log-sum-exp.
       ``simple``: full MC. ``lower-bound``: fully analytic Jensen lower bound.
   * - ``loadings_mode``
     - ``multiplicative``
     - Non-negativity constraint on :math:`W`. ``projected``: gradient clamp at 0.
       ``softplus``: softplus transform. ``exp``: exponential. ``multiplicative``: multiplicative update.
   * - ``training_mode``
     - ``standard``
     - Optimization mode. ``standard``: Adam on all params. ``natural_gradient``:
       natural gradient for variational params, Adam for W.
   * - ``E``
     - ``3``
     - Monte Carlo samples per ELBO computation. Higher = lower variance, slower.
   * - ``scale_ll_D``
     - ``true``
     - Scale log-likelihood term by :math:`1/D`. Keeps ELBO magnitude dataset-invariant.
   * - ``scale_kl_NM``
     - ``true``
     - Scale KL term by :math:`N/M`. Correct ELBO scaling for sparse GPs.
   * - ``spatial``
     - ``false``
     - Enable spatial GP prior. Set ``true`` for SVGP/LCGP variants.
   * - ``groups``
     - ``false``
     - Enable multi-group kernel (MGGP). Requires group labels in preprocessed data.
   * - ``local``
     - ``false``
     - Enable LCGP mode (M=N, VNNGP-style covariance).
   * - ``kernel``
     - ``Matern32``
     - GP kernel. ``Matern32`` recommended for spatial data; ``RBF`` is smoother.
   * - ``lengthscale``
     - ``8.0``
     - GP kernel lengthscale in normalized coordinate units (after dividing by
       ``spatial_scale``). Tune this first if spatial structure looks wrong.
   * - ``sigma``
     - ``1.0``
     - GP signal variance. Fixed (not trained) unless ``train_lengthscale`` is also set.
   * - ``train_lengthscale``
     - ``false``
     - Whether to optimize the lengthscale during training.
   * - ``group_diff_param``
     - ``1.0``
     - MGGP cross-group correlation penalty :math:`\delta`. Higher = groups more
       independent. ``exp(-delta)`` is the between-group correlation.
   * - ``num_inducing``
     - ``3000``
     - Number of inducing points for SVGP. Auto-clamped to :math:`N` if the dataset
       has fewer spots. Larger = better approximation but slower.
   * - ``cholesky_mode``
     - ``exp``
     - Parameterization of the Cholesky diagonal. ``exp``: always positive.
       ``softplus``: smooth positive.
   * - ``diagonal_only``
     - ``false``
     - Use a diagonal (mean-field) variational covariance instead of full Cholesky.
       Faster and less memory but less expressive.
   * - ``inducing_allocation``
     - ``derived``
     - How to split inducing points across groups in MGGP_SVGP.
       ``proportional``: proportional to group size. ``derived``: computed from group structure.
   * - ``K``
     - ``50``
     - Number of local neighbors for LCGP. Controls the sparsity of :math:`L_u`.
   * - ``precompute_knn``
     - ``true``
     - Precompute KNN indices at the start of training. Recommended.
   * - ``neighbors``
     - ``knn``
     - KNN strategy for LCGP. ``knn``: deterministic FAISS L2.
       ``probabilistic``: kernel-weighted Gumbel-max sampling.
   * - ``model_name_override``
     - ``null``
     - If set, overrides the auto-computed ``model_name`` for the output directory.
       Useful for comparing two variants with the same spatial/groups/local flags.

Training Section
----------------

.. code-block:: yaml

   training:
     max_iter: 20000            # Maximum number of training iterations
     learning_rate: 2e-3        # Adam learning rate
     optimizer: Adam            # Optimizer (Adam only)
     tol: 1e-4                  # ELBO convergence tolerance
     verbose: false             # Print per-iteration ELBO (tqdm otherwise)
     device: gpu                # 'cpu' | 'gpu' | 'cuda:0' | 'cuda:1' ...
     batch_size: 7000           # Cells per mini-batch (null = full batch)
     y_batch_size: 2000         # Genes per mini-batch (null = full batch)
     shuffle: true              # Shuffle mini-batches each epoch
     analyze_batch_size: 10000  # Cells per batch in the analyze GP forward pass
     video_interval: 20         # Iterations between video frame captures

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Field
     - Default
     - Description
   * - ``max_iter``
     - ``10000``
     - Maximum training iterations. Training stops early if ELBO change < ``tol``.
   * - ``learning_rate``
     - ``0.01``
     - Adam learning rate. Start with ``2e-3`` for spatial models.
   * - ``tol``
     - ``1e-4``
     - Convergence tolerance on ELBO change between iterations.
   * - ``verbose``
     - ``false``
     - If ``true``, prints ``Iteration N: ELBO = X`` each step (useful for debugging).
       If ``false``, shows a tqdm progress bar.
   * - ``device``
     - ``cpu``
     - Compute device. ``gpu`` maps to PyTorch ``auto`` (first available GPU).
       Use ``cuda:0`` / ``cuda:1`` for explicit GPU assignment.
   * - ``batch_size``
     - ``null``
     - Number of cells per mini-batch. Set to a value smaller than :math:`N`
       for large datasets. Auto-clamped to :math:`N`.
   * - ``y_batch_size``
     - ``null``
     - Number of genes per mini-batch. Set smaller than :math:`D` for large gene sets.
       Auto-clamped to :math:`D`.
   * - ``shuffle``
     - ``true``
     - Whether to shuffle mini-batches each epoch.
   * - ``analyze_batch_size``
     - ``10000``
     - Cells per batch during the analyze stage GP forward pass. Reduce if OOM occurs.
   * - ``video_interval``
     - ``20``
     - Iterations between factor snapshot captures when ``--video`` is passed.

General vs Per-Model Configs
-----------------------------

A **general config** (e.g. ``general.yaml``) contains all hyperparameters for all model
variants but has **no** ``model.spatial`` key. Running ``spatial_factorization generate``
expands it into five per-model configs by injecting the appropriate ``spatial``, ``groups``,
and ``local`` flags.

A **per-model config** (e.g. ``svgp.yaml``) has ``model.spatial: true`` (or ``false``)
set explicitly and runs exactly one model.

The pipeline detects which type a config is by checking for the presence of ``model.spatial``.

.. note::
   You can pass ``--config-name mggp_lcgp.yaml`` to the ``run`` command to recursively
   find per-model configs by name across a directory tree, bypassing general config expansion.
   See :doc:`cli` for details.

``model_name`` Resolution
--------------------------

The output subdirectory for a model is determined by the config flags:

.. list-table::
   :header-rows: 1
   :widths: 15 15 15 15 40

   * - ``spatial``
     - ``groups``
     - ``local``
     - ``model_name``
     - Output directory
   * - ``false``
     - any
     - any
     - ``pnmf``
     - ``outputs/{dataset}/pnmf/``
   * - ``true``
     - ``false``
     - ``false``
     - ``svgp``
     - ``outputs/{dataset}/svgp/``
   * - ``true``
     - ``true``
     - ``false``
     - ``mggp_svgp``
     - ``outputs/{dataset}/mggp_svgp/``
   * - ``true``
     - ``false``
     - ``true``
     - ``lcgp``
     - ``outputs/{dataset}/lcgp/``
   * - ``true``
     - ``true``
     - ``true``
     - ``mggp_lcgp``
     - ``outputs/{dataset}/mggp_lcgp/``

If ``model_name_override`` is set in the config, that value is used directly instead.

Auto-Clamping
-------------

At training time, three config values are automatically clamped to the actual data dimensions:

- ``num_inducing`` → ``min(num_inducing, N)``
- ``batch_size`` → ``min(batch_size, N)``
- ``y_batch_size`` → ``min(y_batch_size, D)``

A note is printed when clamping occurs. This means configs written for large datasets
(``num_inducing=3000``) work correctly on small datasets (e.g. osmFISH N=4,839)
without manual adjustment.
