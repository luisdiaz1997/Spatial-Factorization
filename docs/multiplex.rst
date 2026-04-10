Multiplex Pipeline
==================

The multiplex runner executes multiple models (and datasets) in parallel with
automatic GPU/CPU resource management and a live status display.

Overview
--------

Passing a general config or a directory to ``run all`` activates the multiplex runner:

.. code-block:: bash

   # All 5 models for Slide-seq, in parallel
   spatial_factorization run all -c configs/slideseq/general.yaml

   # All datasets, all models, in parallel
   spatial_factorization run all -c configs/ --config-name general_test.yaml

The runner:

1. Collects all per-model configs (expanding general configs via ``generate``)
2. Runs preprocessing sequentially once per unique ``output_dir``
3. Dispatches training jobs to GPUs + CPU
4. Dispatches analyze and figures jobs as training completes
5. Shows a live status table throughout

Config Collection
-----------------

When ``-c`` is a directory, the runner recursively searches for files matching
``--config-name`` (default: ``general.yaml``):

- **General configs** found: expanded via ``generate_configs`` into 5 per-model configs each
- **Per-model configs** found: used as-is without expansion
- **No match**: falls back to every ``*.yaml`` in the directory tree

This means you can mix general and per-model configs in the same directory tree and
the runner handles each file appropriately.

**Examples:**

.. code-block:: bash

   # Find and expand all general.yaml files
   spatial_factorization run all -c configs/

   # Find all general_test.yaml files (quick test of every dataset)
   spatial_factorization run all -c configs/ --config-name general_test.yaml

   # Find all mggp_lcgp.yaml files (per-model, no expansion)
   spatial_factorization run all -c configs/ --config-name mggp_lcgp.yaml

   # Every non-general *.yaml in the slideseq directory
   spatial_factorization run all --skip-general -c configs/slideseq/

Resource Scheduling
-------------------

The runner manages GPU and CPU resources to maximize throughput:

**GPU assignment**
   Each GPU runs at most one training job at a time (exclusive assignment).
   A job is assigned to the GPU with the most free memory at dispatch time.

**CPU fallback**
   When all GPUs are occupied, at least one CPU slot is always available.
   ``--gpu-only`` disables this fallback.

**Training priority**
   Training jobs always get GPU/CPU resources before analyze or figures jobs.
   Analyze only starts when no pending training jobs are waiting for resources.
   This prevents analyze from competing with training for GPU memory.

**Multi-dataset**
   When running across multiple datasets (e.g. ``-c configs/``), preprocessing
   runs once per unique ``output_dir``. Models from different datasets can train
   in parallel on the same GPU pool.

Live Status Table
-----------------

During parallel training a live-updating table shows the status of every job:

.. code-block:: text

                                           Training Progress
   ┏━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
   ┃ Job         ┃ Task      ┃ Device  ┃ Status    ┃ Epoch       ┃ ELBO             ┃ Time           ┃
   ┡━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
   │ lcgp        │ train     │ cuda:0  │ completed │ 200/200     │ -335521443.5     │ 0:00:15/-      │
   │ lcgp        │ analyze   │ cuda:0  │ analyzing │ 600/1000    │ -                │ 0:00:00/00:01  │
   │ mggp_lcgp   │ train     │ cuda:1  │ training  │ 50/200      │ -49833232.0      │ 0:00:02/0:01:30│
   │ mggp_lcgp   │ analyze   │ pending │ pending   │ -           │ -                │ 0:00:00/-      │
   │ mggp_svgp   │ train     │ cpu     │ completed │ 200/200     │ -45042568.0      │ 0:01:43/-      │
   │ mggp_svgp   │ analyze   │ cpu     │ analyzing │ 114/1000    │ -                │ 0:00:00/01:42  │
   │ pnmf        │ train     │ cuda:0  │ completed │ 200/200     │ -50454488.0      │ 0:00:08/-      │
   │ pnmf        │ analyze   │ cuda:0  │ completed │ 82/1000     │ -                │ 0:00:00/00:03  │
   │ svgp        │ train     │ cuda:1  │ training  │ 80/200      │ -47521234.5      │ 0:01:29/-      │
   └─────────────┴───────────┴─────────┴───────────┴─────────────┴──────────────────┴────────────────┘

Active jobs (training/analyzing) float to the top; completed/failed jobs sink to the bottom.

The **ELBO** column is parsed from both output formats:

- ``verbose=True``: ``Iteration 500: ELBO = -12345.67``
- ``verbose=False`` (tqdm): ``5000/10000 [... ELBO=-5.475e+05 ...]``

Logs
----

Each job writes its stdout/stderr to:

.. code-block:: text

   {output_dir}/logs/{model_name}.log

For multi-dataset runs, each dataset gets its own ``logs/`` directory under its
``output_dir``.

.. code-block:: bash

   # Follow a specific model's log in real-time
   tail -f outputs/slideseq/logs/mggp_svgp.log

Run Status
----------

After the run completes, a summary is written to ``run_status.json``:

.. code-block:: json

   {
     "slideseq_svgp": {
       "status": "success",
       "elbo": -47521234.5,
       "training_time": 89.4
     },
     "slideseq_mggp_svgp": {
       "status": "failed",
       "error": "CUDA out of memory"
     }
   }

For directory-based runs, ``run_status.json`` is written alongside the configs directory.

Use ``--failed`` to re-run only failed jobs from the previous run:

.. code-block:: bash

   spatial_factorization run all -c configs/ --failed

Dry Run
-------

Inspect the execution plan without running anything:

.. code-block:: bash

   spatial_factorization run all -c configs/slideseq/general.yaml --dry-run

Prints: configs found, models to be trained, GPU/CPU assignment plan.

Job Naming
----------

Job names are unique across nested directories and are constructed as:
``{output_dir_parts}_{model_name}``

For example, for ``configs/liver/healthy/mggp_svgp.yaml`` with
``output_dir: outputs/liver/healthy``, the job name is ``liver_healthy_mggp_svgp``.

This ensures the status table and log files are unambiguous when running
multiple datasets simultaneously.
