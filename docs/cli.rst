CLI Reference
=============

All functionality is accessed through the ``spatial_factorization`` command.
Run ``spatial_factorization --help`` or ``spatial_factorization COMMAND --help``
for inline help.

.. code-block:: bash

   spatial_factorization --version
   spatial_factorization --help

Commands
--------

``preprocess``
~~~~~~~~~~~~~~

.. code-block:: bash

   spatial_factorization preprocess -c CONFIG

Load and standardize the raw dataset. Run **once per dataset** — all model variants
share the same preprocessed data.

**Output:** ``{output_dir}/preprocessed/``

Options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c``, ``--config PATH``
     - Path to a per-model or general config YAML (required).

**Example:**

.. code-block:: bash

   spatial_factorization preprocess -c configs/slideseq/svgp.yaml

----

``train``
~~~~~~~~~

.. code-block:: bash

   spatial_factorization train -c CONFIG [--resume] [--video] [--probabilistic]

Train a PNMF model from a per-model config.

**Output:** ``{output_dir}/{model_name}/model.pth``, ``training.json``, ``elbo_history.csv``

Options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c``, ``--config PATH``
     - Path to a per-model config YAML (required).
   * - ``--resume``
     - Warm-start from an existing ``model.pth`` checkpoint. Appends to ELBO history.
       Falls back to training from scratch if no checkpoint exists.
   * - ``--video``
     - Capture factor snapshots every ``training.video_interval`` iterations.
       Saves ``video_frames.npy`` for later rendering.
   * - ``--probabilistic``
     - Override the KNN strategy to probabilistic for LCGP/MGGP_LCGP models.
       No-op for PNMF/SVGP/MGGP_SVGP. The saved checkpoint records
       ``neighbors: probabilistic``.

**Examples:**

.. code-block:: bash

   # Standard training
   spatial_factorization train -c configs/slideseq/svgp.yaml

   # Resume from checkpoint
   spatial_factorization train --resume -c configs/slideseq/mggp_lcgp.yaml

   # Resume with probabilistic KNN
   spatial_factorization train --resume --probabilistic -c configs/slideseq/lcgp.yaml

----

``analyze``
~~~~~~~~~~~

.. code-block:: bash

   spatial_factorization analyze -c CONFIG [--probabilistic]

Analyze a trained model: extract factors, compute Moran's I, gene loadings, enrichment,
and reconstruction metrics.

**Output:** ``{output_dir}/{model_name}/factors.npy``, ``moran_i.csv``, ``metrics.json``, etc.

Options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c``, ``--config PATH``
     - Path to a per-model config YAML (required).
   * - ``--probabilistic``
     - Load the LCGP/MGGP_LCGP model with probabilistic KNN for this analyze run.
       Outputs overwrite existing analyze artifacts in place. No-op for other models.

**Example:**

.. code-block:: bash

   spatial_factorization analyze --probabilistic -c configs/slideseq/mggp_lcgp.yaml

----

``figures``
~~~~~~~~~~~

.. code-block:: bash

   spatial_factorization figures -c CONFIG [--no-heatmap]

Generate publication figures from the analyze outputs.

**Output:** ``{output_dir}/{model_name}/figures/``

Options:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Description
   * - ``-c``, ``--config PATH``
     - Path to a per-model config YAML (required).
   * - ``--no-heatmap``
     - Skip ``celltype_gene_loadings.png`` and ``factor_gene_loadings.png``.
       Use for large :math:`D` (>10K genes) where the heatmaps are slow or unreadable.

----

``generate``
~~~~~~~~~~~~

.. code-block:: bash

   spatial_factorization generate -c GENERAL_CONFIG

Expand a general config into five per-model YAML files in the same directory.
Prints the generated file paths.

**Example:**

.. code-block:: bash

   spatial_factorization generate -c configs/slideseq/general.yaml
   # Generated 5 model configs:
   #   pnmf: configs/slideseq/pnmf.yaml
   #   svgp: configs/slideseq/svgp.yaml
   #   mggp_svgp: configs/slideseq/mggp_svgp.yaml
   #   lcgp: configs/slideseq/lcgp.yaml
   #   mggp_lcgp: configs/slideseq/mggp_lcgp.yaml

----

``run``
~~~~~~~

.. code-block:: bash

   spatial_factorization run STAGES... -c CONFIG [OPTIONS]

Run one or more pipeline stages. Stages are always executed in pipeline order
(``preprocess → train → analyze → figures``) regardless of the order listed.

**STAGES:** One or more of ``preprocess``, ``train``, ``analyze``, ``figures``, or ``all``.

Using ``all`` activates the parallel multiplex runner. See :doc:`multiplex` for details.

Config behavior:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Config type
     - Behavior
   * - Per-model YAML (has ``model.spatial``)
     - Runs that single model sequentially
   * - General YAML (no ``model.spatial``)
     - Expands to 5 per-model configs; runs all in parallel
   * - Directory
     - Recursively finds files matching ``--config-name``. General configs are expanded;
       per-model configs are used as-is. Falls back to every ``*.yaml`` if no match found.
   * - Directory + ``--skip-general``
     - Ignores general configs; every non-general ``*.yaml`` in the tree is a per-model config.

Options:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Description
   * - ``-c``, ``--config PATH``
     - Config YAML or directory (required).
   * - ``--config-name NAME``
     - Filename to ``rglob`` when config is a directory. Default: ``general.yaml``.
       Use ``general_test.yaml`` for quick tests or ``mggp_lcgp.yaml`` to target
       a specific per-model config across all datasets.
   * - ``--skip-general``
     - Ignore general configs; treat all non-general ``*.yaml`` files as per-model configs.
   * - ``--resume``
     - Warm-start training from checkpoint when available; train from scratch otherwise.
       Safe for batch runs (models without checkpoints just start fresh).
   * - ``--force``
     - Re-run preprocessing even if preprocessed data already exists.
   * - ``--dry-run``
     - Print the execution plan without running anything.
   * - ``--probabilistic``
     - Override LCGP KNN strategy to probabilistic for both ``train`` and ``analyze`` stages.
       No-op for non-LCGP models.
   * - ``--video``
     - Capture factor snapshots during training for later animation.
   * - ``--gpu-only``
     - Never fall back to CPU; only assign jobs to available GPUs.
   * - ``--failed``
     - Re-run only jobs that failed in the previous run (reads ``run_status.json``).
   * - ``--no-heatmap``
     - Skip heatmap figures in the figures stage.

**Examples:**

.. code-block:: bash

   # All stages, single model
   spatial_factorization run all -c configs/slideseq/svgp.yaml

   # Specific stages, single model
   spatial_factorization run train analyze figures -c configs/slideseq/svgp.yaml

   # All models, one dataset (parallel)
   spatial_factorization run all -c configs/slideseq/general.yaml

   # Quick test (10 iterations)
   spatial_factorization run all -c configs/slideseq/general_test.yaml

   # All datasets at once (recursive, parallel)
   spatial_factorization run all -c configs/ --config-name general_test.yaml

   # All LCGP models across datasets, resume with probabilistic KNN
   spatial_factorization run all -c configs/ --config-name mggp_lcgp.yaml \
       --resume --probabilistic

   # Re-run only failed jobs from last run
   spatial_factorization run all -c configs/ --failed

   # Dry run to inspect plan
   spatial_factorization run all -c configs/slideseq/general.yaml --dry-run

   # Force re-preprocessing
   spatial_factorization run all -c configs/slideseq/general.yaml --force

----

``multianalyze``
~~~~~~~~~~~~~~~~

.. code-block:: bash

   spatial_factorization multianalyze -c CONFIG MODEL1 MODEL2 [MODEL3...] [OPTIONS]

Compare matched factors across two or more trained models. Factors are matched
by greedy normalized L2 distance on the ``factors.npy`` arrays.

**Two-model mode** (exactly 2 models):
Finds the top ``--n-pairs`` matched factor pairs and renders each pair as
[2D spatial | 3D surface] side-by-side with model A on the top row and model B on the bottom.

**Three-or-more-model mode** (3+ models):
Finds the single best-matching reference factor between MODEL1 and MODEL2 (or ``--match-against``),
then matches that factor against all remaining models. Layout: top row = 2D spatial,
bottom row = 3D surface; one column per model.

Options:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Description
   * - ``-c``, ``--config PATH``
     - Any config for the dataset (used to resolve ``output_dir``).
   * - ``--n-pairs N``
     - Number of matched pairs to show (two-model mode only). Default: 2.
   * - ``--match-against MODEL``
     - Model to use as the reference in three-or-more-model mode.
       Default: MODEL2 (second positional argument).
   * - ``-o``, ``--output PATH``
     - Output file path. Default: ``{output_dir}/figures/comparison_*.png``.

**Examples:**

.. code-block:: bash

   # SVGP vs MGGP_SVGP — 2 pairs
   spatial_factorization multianalyze -c configs/slideseq/general.yaml svgp mggp_svgp

   # SVGP vs MGGP_SVGP — 4 pairs
   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp --n-pairs 4

   # All 5 models (svgp as reference)
   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp pnmf lcgp mggp_lcgp

   # Custom reference model
   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp pnmf mggp_svgp lcgp mggp_lcgp --match-against mggp_svgp

.. tip::
   Use ``svgp`` as the reference model (first argument) for 3+-model comparisons.
   ``pnmf`` should not be used as reference since it is non-spatial and its factors
   are not ordered spatially.
