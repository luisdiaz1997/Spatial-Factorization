SDMBench Multi-Slide Example
=============================

The SDMBench DLPFC Visium dataset consists of 12 slides of human dorsolateral prefrontal
cortex (DLPFC), each with ~4,200 spots and ~33,000 genes. This example demonstrates
running all 12 slides in parallel and comparing results across slides.

Dataset
-------

- **Source:** SDMBench benchmark (h5ad files, one per slide)
- **Slides:** 12 (sample IDs 151507–151676)
- **N per slide:** ~4,200 spots
- **D per slide:** ~33,000 genes
- **Groups:** 5–7 tissue layers + white matter (``obs["Region"]``)
- **Technology:** 10x Visium
- **Tissue:** Human dorsolateral prefrontal cortex

The ``Region`` column contains NaN entries for spots outside annotated tissue layers.
These are automatically dropped by the NaN filter during preprocessing.

Configuration Structure
-----------------------

Each slide has its own config directory:

.. code-block:: text

   configs/sdmbench/
   ├── 151507/
   │   ├── general.yaml
   │   └── general_test.yaml
   ├── 151508/
   │   ├── general.yaml
   │   └── general_test.yaml
   └── ...  (12 slides total, through 151676)

Each ``general.yaml`` specifies the path to that slide's h5ad file:

.. code-block:: yaml

   name: sdmbench_151507
   dataset: sdmbench
   output_dir: outputs/sdmbench/151507

   preprocessing:
     path: /path/to/sdmbench/Data/151507.h5ad
     spatial_scale: 50.0
     min_group_fraction: 0.01

   model:
     n_components: 10
     num_inducing: 3000
     lengthscale: 8.0
     K: 50
     group_diff_param: 1.0

   training:
     max_iter: 20000
     batch_size: 4221       # ~N for this slide
     y_batch_size: 2000

Running All 12 Slides
---------------------

.. code-block:: bash

   # Quick test: all 12 slides × 5 models = 60 jobs (10 iterations each)
   spatial_factorization run all -c configs/sdmbench/ --config-name general_test.yaml

   # Full training: all 12 slides × 5 models in parallel
   spatial_factorization run all -c configs/sdmbench/ --config-name general.yaml

The runner collects all 12 ``general.yaml`` files, expands each into 5 per-model configs,
and dispatches up to ``len(GPUs) + 1`` training jobs simultaneously. With 2 GPUs:

- 2 training jobs run on GPUs
- 1 job runs on CPU
- Others queue

.. note::
   Job names across slides are unique: ``sdmbench_151507_mggp_svgp``,
   ``sdmbench_151508_mggp_svgp``, etc. Each slide gets its own ``logs/`` directory.

Single Slide
------------

.. code-block:: bash

   # One slide, quick test
   spatial_factorization run all -c configs/sdmbench/151507/general_test.yaml

   # One slide, specific model
   spatial_factorization run train analyze figures \
       -c configs/sdmbench/151507/mggp_svgp.yaml

Inspecting Run Status
---------------------

After a multi-slide run, check which jobs succeeded:

.. code-block:: python

   import json

   with open("configs/sdmbench/run_status.json") as f:
       status = json.load(f)

   failed = {k: v for k, v in status.items() if v["status"] != "success"}
   print(f"Failed jobs: {list(failed.keys())}")

Re-run failed jobs:

.. code-block:: bash

   spatial_factorization run all -c configs/sdmbench/ \
       --config-name general.yaml --failed

Large Gene Set Considerations
------------------------------

With ~33K genes per slide, heatmap figures are very slow. Use ``--no-heatmap``:

.. code-block:: bash

   spatial_factorization run all -c configs/sdmbench/ \
       --config-name general.yaml --no-heatmap

Also consider reducing ``y_batch_size`` if GPU memory is tight during training,
or increasing it toward 5000 if memory allows (fewer gene mini-batches = faster).

Cross-Slide Factor Comparison
------------------------------

After training all slides, you can compare factors across slides using ``multianalyze``
on one slide at a time:

.. code-block:: bash

   # Compare SVGP vs MGGP_SVGP for slide 151507
   spatial_factorization multianalyze \
       -c configs/sdmbench/151507/general.yaml svgp mggp_svgp

   # All 5 models for slide 151507
   spatial_factorization multianalyze \
       -c configs/sdmbench/151507/general.yaml \
       svgp mggp_svgp pnmf lcgp mggp_lcgp

To compare the same factor across slides programmatically:

.. code-block:: python

   import numpy as np

   slides = ["151507", "151508", "151509"]
   model  = "mggp_svgp"

   # Load Factor 0 (highest Moran's I) from each slide
   factor_0_per_slide = [
       np.load(f"outputs/sdmbench/{s}/{model}/factors.npy")[:, 0]
       for s in slides
   ]

   # Compare Moran's I across slides
   import pandas as pd
   for s in slides:
       moran = pd.read_csv(f"outputs/sdmbench/{s}/{model}/moran_i.csv")
       print(f"Slide {s} — Factor 0 Moran's I: {moran['moran_i'].iloc[0]:.3f}")
