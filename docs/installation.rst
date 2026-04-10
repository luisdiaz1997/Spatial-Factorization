Installation
============

Prerequisites
-------------

- Python 3.14
- `conda <https://docs.conda.io/>`_ (recommended)
- CUDA-capable GPU (strongly recommended for SVGP/LCGP training on large datasets)

Spatial-Factorization depends on two sibling packages that must be installed from source:

- **PNMF** — sklearn-compatible Poisson factorization model
- **GPzoo** — Gaussian process backends (SVGP, MGGP, LCGP, kernels)

Install from Source
-------------------

.. code-block:: bash

   # 1. Clone all three repositories
   git clone https://github.com/luisdiaz1997/Spatial-Factorization.git
   git clone https://github.com/luisdiaz1997/Probabilistic-NMF.git
   git clone https://github.com/luisdiaz1997/GPzoo.git

   # 2. Create and activate a conda environment
   conda create -n factorization python=3.14
   conda activate factorization

   # 3. Install dependencies in order (PNMF → GPzoo → Spatial-Factorization)
   pip install -e Probabilistic-NMF/
   pip install -e GPzoo/
   pip install -e Spatial-Factorization/

Or use the helper script in the repo:

.. code-block:: bash

   cd Spatial-Factorization
   conda activate factorization
   ./scripts/install_deps.sh

Verify Installation
-------------------

.. code-block:: bash

   spatial_factorization --version
   spatial_factorization --help

You should see the version number and a list of available commands
(``preprocess``, ``train``, ``analyze``, ``figures``, ``generate``, ``run``, ``multianalyze``).

Optional Dependencies
---------------------

``faiss-gpu``
   GPU-accelerated FAISS for faster KNN computation in LCGP models.
   Install after activating the environment:

   .. code-block:: bash

      conda install -c conda-forge faiss-gpu

``rich``
   Required for the live status table during parallel training (``run all``).
   Installed automatically as a dependency.

``matplotlib``, ``seaborn``
   Required for figure generation (Stage 4). Installed automatically.

Data Storage
------------

Large output files (model checkpoints, factor arrays, figures) can grow to tens of GB.
We recommend storing outputs on a filesystem with ample space and symlinking into the repo:

.. code-block:: bash

   mkdir -p /large/storage/Spatial-Factorization/outputs
   ln -s /large/storage/Spatial-Factorization/outputs outputs/

.. warning::
   Never delete files in ``outputs/`` unless explicitly intending to. Trained models
   take hours to days to produce on large datasets.
