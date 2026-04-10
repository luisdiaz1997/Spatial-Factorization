Model Variants
==============

Spatial-Factorization provides five model variants that differ in the GP prior structure
and whether cell-type group information is incorporated.

Overview
--------

.. list-table::
   :header-rows: 1
   :widths: 18 28 14 10 30

   * - Model
     - PNMF flags
     - Groups
     - M
     - Complexity
   * - ``pnmf``
     - ``spatial=False``
     - No
     - —
     - :math:`\mathcal{O}(NL)`
   * - ``svgp``
     - ``spatial=True, local=False, multigroup=False``
     - No
     - :math:`M \ll N`
     - :math:`\mathcal{O}(NM + M^2)`
   * - ``mggp_svgp``
     - ``spatial=True, local=False, multigroup=True``
     - Yes
     - :math:`M \ll N`
     - :math:`\mathcal{O}(NM + M^2)`
   * - ``lcgp``
     - ``spatial=True, local=True, multigroup=False``
     - No
     - :math:`M = N`
     - :math:`\mathcal{O}(NK^2)`
   * - ``mggp_lcgp``
     - ``spatial=True, local=True, multigroup=True``
     - Yes
     - :math:`M = N`
     - :math:`\mathcal{O}(NK^2)`

The model name determines the output directory:
``outputs/{dataset}/{model_name}/``.

Non-Spatial Baseline: PNMF
--------------------------

**Config:** ``spatial: false``

The simplest model. Places an isotropic Gaussian prior on each factor independently.
No spatial coordinates are used.

**Use when:**

- You want a fast baseline to confirm the number of components is reasonable
- You suspect the biology is not spatially structured
- You need a non-spatial reference for ``multianalyze`` comparison

**Key parameters:** ``n_components``, ``E``, ``batch_size``, ``y_batch_size``

SVGP
----

**Config:** ``spatial: true, groups: false, local: false``

Sparse Variational Gaussian Process with :math:`M` inducing points
(:math:`M \ll N`, default 3000). Models spatial correlation via a Matérn-3/2 kernel
over 2D coordinates. Inducing points are initialized by K-means clustering of the
spatial coordinates.

**Use when:**

- Spatial structure is expected and :math:`N` is large (10K–1M cells)
- Cell-type group annotations are absent or not informative for the model
- ``mggp_svgp`` is too slow due to many groups

**Key parameters:** ``num_inducing``, ``lengthscale``, ``kernel``, ``train_lengthscale``

.. tip::
   Start with ``lengthscale=8.0`` (tuned for Slide-seq coordinates scaled by 50).
   If factors are too smooth, decrease it; if too noisy, increase it.

MGGP_SVGP
----------

**Config:** ``spatial: true, groups: true, local: false``

Multi-Group GP extension of SVGP. Uses a product kernel that captures within-group
and cross-group spatial correlations. Group labels (cell types, tissue layers) are
passed at training time, and inducing points are assigned group labels proportionally.

**Use when:**

- Cell-type or tissue region annotations are available
- You want the model to learn group-specific spatial programs
- The dataset has 2–20 well-defined groups

**Key parameters:** Same as SVGP plus ``group_diff_param`` (higher = groups more independent),
``inducing_allocation`` (how to split inducing points across groups)

.. note::
   MGGP models cannot be pickled due to an internal wrapper class. Only
   ``model.pth`` (PyTorch state dict) is saved; ``model.pkl`` is skipped.

LCGP
----

**Config:** ``spatial: true, groups: false, local: true``

Locally Conditioned GP with all :math:`N` training points as inducing points.
Uses a VNNGP-style sparse Cholesky covariance :math:`S = L_u L_u^\top` where
:math:`L_u \in \mathbb{R}^{N \times K}` is sparse (K non-zeros per row).

**Use when:**

- Full spatial coverage is needed (every cell is an inducing point)
- :math:`N` is moderate (up to ~300K) with GPU memory available
- You want to experiment with probabilistic neighborhood sampling

**Key parameters:** ``K`` (number of local neighbors, default 50),
``neighbors`` (``knn`` or ``probabilistic``)

.. list-table:: SVGP vs LCGP trade-offs
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - SVGP
     - LCGP
   * - Inducing points
     - :math:`M \ll N` (e.g. 3000)
     - :math:`M = N` (all training cells)
   * - Covariance
     - Full Cholesky :math:`(M \times M)`
     - Sparse :math:`(N \times K)`
   * - Saved ``Lu``
     - ``Lu.pt`` :math:`(L, M, M)` ~344 MB
     - ``Lu.npy`` :math:`(L, N, K)`
   * - Best for
     - Very large :math:`N`, approximate coverage
     - Full coverage, VNNGP expressivity

MGGP_LCGP
----------

**Config:** ``spatial: true, groups: true, local: true``

Most expressive model. Combines LCGP local conditioning with the MGGP multi-group
kernel. All :math:`N` training points are inducing points with group labels.

**Use when:**

- You need both group-aware covariance and full spatial coverage
- You have moderate :math:`N` (<300K) and informative group annotations

**Key parameters:** Same as LCGP plus MGGP group parameters

Hyperparameter Reference
------------------------

All hyperparameters are set in the ``model:`` section of the config YAML.
See :doc:`configuration` for the full reference.

.. list-table:: Most important hyperparameters per model
   :header-rows: 1
   :widths: 25 20 15 40

   * - Parameter
     - Models
     - Default
     - Notes
   * - ``n_components``
     - All
     - 10
     - Number of latent factors L
   * - ``lengthscale``
     - SVGP, MGGP_SVGP, LCGP, MGGP_LCGP
     - 8.0
     - GP kernel lengthscale in normalized coordinates
   * - ``num_inducing``
     - SVGP, MGGP_SVGP
     - 3000
     - Auto-clamped to N if N < 3000
   * - ``K``
     - LCGP, MGGP_LCGP
     - 50
     - Number of local neighbors
   * - ``group_diff_param``
     - MGGP_SVGP, MGGP_LCGP
     - 1.0
     - Higher = groups more independent
   * - ``neighbors``
     - LCGP, MGGP_LCGP
     - ``knn``
     - ``knn`` (FAISS) or ``probabilistic``
   * - ``train_lengthscale``
     - All spatial
     - ``false``
     - Enable to learn lengthscale from data

Picklability
------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Model
     - Picklable
     - Notes
   * - ``pnmf``
     - Yes
     - Both ``model.pkl`` and ``model.pth`` saved
   * - ``svgp``
     - Yes
     - Both ``model.pkl`` and ``model.pth`` saved
   * - ``lcgp``
     - Yes
     - Both ``model.pkl`` and ``model.pth`` saved
   * - ``mggp_svgp``
     - **No**
     - Only ``model.pth`` saved (MGGPWrapper local class)
   * - ``mggp_lcgp``
     - **No**
     - Only ``model.pth`` saved (MGGPWrapper local class)

All models can be fully loaded and re-analyzed from ``model.pth`` via the analyze stage.
