.. Spatial-Factorization documentation master file

Spatial-Factorization
=====================

**Spatial-Factorization** is a four-stage CLI pipeline for spatial transcriptomics analysis.
It applies probabilistic non-negative matrix factorization (PNMF) with Gaussian process (GP)
priors to decompose spatial gene expression data into spatially coherent latent factors.

.. code-block:: text

   preprocess  →  train  →  analyze  →  figures

Key features:

- **Five model variants**: non-spatial PNMF, SVGP, MGGP_SVGP, LCGP, MGGP_LCGP
- **Seven dataset loaders**: Slide-seq V2, MERFISH, 10x Visium, liver, colon, osmFISH, SDMBench (12 slides)
- **Parallel multi-model/multi-dataset runner** with live GPU/CPU scheduling
- **Checkpoint resume**, probabilistic KNN, and factor comparison across models
- **Spatially ordered factors**: all outputs sorted by descending Moran's I

Three-Package Ecosystem
-----------------------

.. code-block:: text

   Spatial-Factorization  ←  pipeline, CLI, datasets, analysis, figures
         ├── PNMF          ←  sklearn-compatible Poisson factorization model
         └── GPzoo         ←  GP backends (SVGP, MGGP, LCGP, kernels)

.. note::
   This project is under active development. All five model variants are implemented
   and tested on datasets ranging from 4K to 1.2M cells.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Background

   math
   models

.. toctree::
   :maxdepth: 2
   :caption: Reference

   configuration
   cli
   datasets
   outputs

.. toctree::
   :maxdepth: 2
   :caption: Pipeline

   pipeline
   multiplex
   advanced

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples/slideseq
   examples/merfish
   examples/sdmbench
   examples/colon


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
