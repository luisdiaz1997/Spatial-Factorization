Advanced Usage
==============

This page covers features for power users: checkpoint resume, probabilistic KNN,
training video capture, and cross-model factor comparison.

Checkpoint Resume
-----------------

Training spatial GP models takes hours on large datasets. ``--resume`` lets you
continue training from a saved checkpoint:

.. code-block:: bash

   spatial_factorization train --resume -c configs/slideseq/mggp_svgp.yaml

**Behavior:**

- If ``{model_dir}/model.pth`` exists: warm-starts from the saved prior and :math:`W` matrix
- If no checkpoint exists: trains from scratch (safe fallback — no error raised)
- ELBO history is **appended** to the existing ``elbo_history.csv`` with correct iteration counts
- ``training.json`` is updated with ``"resumed": true`` and cumulative iteration count

**Warm-start internals:**
``_create_warm_start_pnmf`` creates a PNMF subclass that overrides ``_create_spatial_prior``
(spatial models) and ``_initialize_W`` to inject the loaded parameters instead of random
initialization, then runs the normal ``fit()`` training loop. After training, the subclass
is reset to ``PNMF`` so the model can be pickled identically to a normal run.

**Requires_grad preservation:**
PyTorch state dicts save values only, not ``requires_grad`` flags. The resume path
explicitly re-applies the correct flags (``Z.requires_grad_(False)``,
``kernel.sigma.requires_grad_(False)``, etc.) before the optimizer is built,
preventing NaN gradients on the first backward step.

**Batch resume:**
Using ``--resume`` with ``run all`` is safe — models with a checkpoint warm-start,
models without a checkpoint train from scratch:

.. code-block:: bash

   # Resume all models across all datasets; train from scratch where no checkpoint exists
   spatial_factorization run all -c configs/ --config-name general.yaml --resume

Probabilistic KNN (LCGP/MGGP_LCGP)
-------------------------------------

By default, LCGP computes :math:`K` nearest neighbors via FAISS L2 (deterministic).
The ``--probabilistic`` flag switches to **kernel-weighted probabilistic sampling**:

For each cell :math:`x_i`, neighbors are sampled without replacement from all other cells,
with probability proportional to the kernel value :math:`k(x_i, z_j)`. Sampling uses
the **Gumbel-max trick**:

.. math::

   \tilde{w}_{ij} = \log k(x_i, z_j) + G_{ij}, \quad G_{ij} \sim \text{Gumbel}(0,1)

The top-:math:`K` cells by :math:`\tilde{w}` are selected as neighbors.

This produces stochastic, kernel-density-weighted neighborhoods that are not limited to
the strict nearest neighbors. Cells that are kernel-similar but not geometrically nearest
can be selected.

**Analyze only (overwrite in place):**

.. code-block:: bash

   spatial_factorization analyze --probabilistic -c configs/slideseq/lcgp.yaml
   spatial_factorization figures -c configs/slideseq/lcgp.yaml

Analyze outputs overwrite the existing files. The trained model is never touched.

**Training with probabilistic KNN:**

.. code-block:: bash

   spatial_factorization train --resume --probabilistic -c configs/slideseq/lcgp.yaml

The checkpoint is loaded with probabilistic neighbors (for the warm-start prior).
Training continues with probabilistic neighborhood sampling. The saved checkpoint
records ``neighbors: probabilistic`` so future analyze calls automatically use the
same strategy without needing ``--probabilistic``.

**Batch operation across datasets:**

.. code-block:: bash

   # Re-analyze all LCGP models with probabilistic KNN
   spatial_factorization run all -c configs/ --config-name lcgp.yaml --probabilistic

   # Resume all LCGP models with probabilistic KNN
   spatial_factorization run all -c configs/ --config-name mggp_lcgp.yaml \
       --resume --probabilistic

.. note::
   ``--probabilistic`` is a no-op for ``pnmf``, ``svgp``, and ``mggp_svgp``.
   The flag is silently ignored for those models.

Training Video
--------------

Capture factor snapshots during training for animation:

.. code-block:: bash

   spatial_factorization train --video -c configs/slideseq/svgp.yaml

Every ``video_interval`` iterations (default 20, configurable in ``training.video_interval``),
the current factor values are captured and appended to ``video_frames.npy``:

.. code-block:: text

   {model_dir}/
   ├── video_frames.npy      ← (n_frames, N, L) factor snapshots
   └── video_frame_iters.npy ← (n_frames,) iteration indices

For spatial models, each frame is computed via the GP predictive pass on all training
coordinates (batched to avoid OOM). For non-spatial PNMF, frames are ``exp(prior.mean)``.

The figures stage reads ``video_frames.npy`` and renders an animation.

ELBO Scaling Flags
------------------

Two flags control ELBO normalization (both default to ``true``):

``scale_ll_D``
   Scale the log-likelihood term by :math:`1/D` (number of genes). Keeps the ELBO
   magnitude comparable across datasets with different numbers of genes.

``scale_kl_NM``
   Scale the KL divergence term by :math:`N/M`. Correct ELBO scaling for sparse GP
   models where :math:`M \ll N` (SVGP). For LCGP where :math:`M = N`, this is 1.0.

Set these to ``false`` for video demos or direct comparison with checkpoints trained
before these flags were introduced:

.. code-block:: yaml

   model:
     scale_ll_D: false
     scale_kl_NM: false

Multi-Model Factor Comparison
-------------------------------

After training multiple models, compare their learned factors:

**Two-model mode:**

.. code-block:: bash

   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp --n-pairs 3

Shows the top 3 matched factor pairs as [2D spatial | 3D surface] side-by-side.

**All 5 models:**

.. code-block:: bash

   spatial_factorization multianalyze -c configs/slideseq/general.yaml \
       svgp mggp_svgp pnmf lcgp mggp_lcgp

Uses SVGP as reference. Finds the best SVGP ↔ MGGP_SVGP match, then matches that
factor against PNMF, LCGP, and MGGP_LCGP. Output: 2D panels (top row) and 3D surface
panels (bottom row), one column per model.

Factor matching uses **greedy normalized L2 distance** on the ``factors.npy`` arrays.
Factors are matched sequentially: the best pair is selected first, then removed from
the candidate pool, and so on.

Output is saved to ``{output_dir}/figures/comparison_{model1}_vs_{model2}_vs_...png``.

Running Tests Efficiently
--------------------------

For development and smoke testing, always use ``general_test.yaml`` (10 iterations):

.. code-block:: bash

   # Quick test of all models for one dataset
   spatial_factorization run all -c configs/slideseq/general_test.yaml

   # Quick test of all datasets at once
   spatial_factorization run all -c configs/ --config-name general_test.yaml

Clean up test outputs before re-running:

.. code-block:: bash

   # Remove model outputs but keep preprocessed data
   rm -rf outputs/slideseq/pnmf outputs/slideseq/svgp outputs/slideseq/mggp_svgp \
          outputs/slideseq/lcgp outputs/slideseq/mggp_lcgp \
          outputs/slideseq/logs outputs/slideseq/run_status.json

.. warning::
   Never delete files from outputs that represent real trained models.
   Only delete test outputs from ``general_test.yaml`` runs.

``model_name_override``
------------------------

Two config variants with the same spatial/groups/local flags would overwrite each
other's output directory. Use ``model_name_override`` to give them distinct names:

.. code-block:: yaml

   # config_a.yaml
   model:
     spatial: false
     model_name_override: pnmf_scaled
     scale_ll_D: true

   # config_b.yaml
   model:
     spatial: false
     model_name_override: pnmf_unscaled
     scale_ll_D: false

These write to ``outputs/slideseq/pnmf_scaled/`` and ``outputs/slideseq/pnmf_unscaled/``
respectively, allowing direct comparison.
