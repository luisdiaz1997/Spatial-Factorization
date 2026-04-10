Datasets
========

Spatial-Factorization includes loaders for seven spatial transcriptomics datasets,
ranging from 4,839 to 1.2 million cells.

Supported Datasets
------------------

.. list-table::
   :header-rows: 1
   :widths: 15 25 10 12 30 8

   * - Key
     - Source
     - N
     - D
     - Groups
     - Auto-load
   * - ``slideseq``
     - squidpy ``slideseq_v2()``
     - ~41K
     - ~4K
     - ``obs["cluster"]``
     - Yes
   * - ``tenxvisium``
     - squidpy ``visium_hne_adata()``
     - ~3K
     - ~15K
     - ``obs["cluster"]``
     - Yes
   * - ``merfish``
     - squidpy ``merfish()``
     - 73K
     - 161
     - ``obs["Cell_class"]``
     - Yes
   * - ``sdmbench``
     - h5ad file
     - ~4.2K
     - ~33K
     - ``obs["Region"]``
     - No (path required)
   * - ``liver``
     - h5ad file
     - 90K (healthy) / 310K (diseased)
     - 317
     - ``obs["Cell_Type"]`` / ``obs["Cell_Type_final"]``
     - No (path required)
   * - ``osmfish``
     - h5ad file
     - 4.8K
     - 33
     - ``obs["ClusterName"]``
     - No (path required)
   * - ``colon``
     - h5ad + CSV
     - 1.2M
     - 492
     - CSV column ``cl46v1SubShort_ds``
     - No (paths required)

Auto-download datasets (``slideseq``, ``tenxvisium``, ``merfish``) are fetched via
squidpy on first use and cached in ``data/anndata/``.

Dataset Configuration
----------------------

The ``dataset`` key in the config YAML selects the loader. Dataset-specific parameters
go in the ``preprocessing`` section.

slideseq
~~~~~~~~

Mouse cerebellum, ~41K spots, ~4K genes. Automatically downloaded via squidpy.

.. code-block:: yaml

   dataset: slideseq
   preprocessing:
     spatial_scale: 50.0
     filter_mt: true
     min_counts: 100
     min_cells: 10
     min_group_fraction: 0.01

tenxvisium
~~~~~~~~~~

10x Visium H&E mouse brain, ~3K spots, ~15K genes. Auto-downloaded.

.. code-block:: yaml

   dataset: tenxvisium
   preprocessing:
     spatial_scale: 50.0
     min_group_fraction: 0.01

.. tip::
   With ~15K genes, use ``y_batch_size: 2000`` and ``--no-heatmap`` for figures
   to keep runtime reasonable.

merfish
~~~~~~~

squidpy MERFISH mouse brain, 73K cells, 161 genes. Auto-downloaded.

.. code-block:: yaml

   dataset: merfish
   preprocessing:
     spatial_scale: 50.0
     min_group_fraction: 0.01

sdmbench
~~~~~~~~

SDMBench Visium DLPFC dataset. 12 slides (sample IDs 151507–151676), each with
~4,200 spots and ~33K genes. Requires a path to the h5ad file.

.. code-block:: yaml

   dataset: sdmbench
   preprocessing:
     path: /path/to/151507.h5ad
     spatial_scale: 50.0
     min_group_fraction: 0.01

Each slide has its own config directory under ``configs/sdmbench/{slide_id}/``.

.. note::
   The ``Region`` column in SDMBench contains NaN entries for spots outside
   annotated tissue layers. These are automatically dropped by the NaN filter.

liver
~~~~~

Liver MERFISH dataset. Two conditions: healthy (90K cells) and diseased (310K cells).
Both use the same loader with different path and cell-type column.

**Healthy:**

.. code-block:: yaml

   dataset: liver
   preprocessing:
     path: /path/to/adata_healthy_merfish.h5ad
     cell_type_column: Cell_Type

**Diseased:**

.. code-block:: yaml

   dataset: liver
   preprocessing:
     path: /path/to/adata_healthy_diseased_merfish.h5ad
     cell_type_column: Cell_Type_final

.. note::
   Liver uses ``obsm["X_spatial"]`` for coordinates (not the standard ``obsm["spatial"]``).
   This is handled transparently by the loader.

osmfish
~~~~~~~

osmFISH SDMBench dataset. 4,839 cells, 33 genes. Small dataset useful for
quick end-to-end validation.

.. code-block:: yaml

   dataset: osmfish
   preprocessing:
     path: /path/to/osmfish.h5ad
     spatial_scale: 50.0

colon
~~~~~

Colon Cancer Vizgen MERFISH. 1.2 million cells, 492 genes. Group labels come
from a separate CSV file.

.. code-block:: yaml

   dataset: colon
   preprocessing:
     path: /path/to/colon.h5ad
     labels_path: /path/to/labels.csv
     subsample: null      # null = all 1.2M cells; set an integer to subsample

.. warning::
   At 1.2M cells, LCGP is not feasible (M=N=1.2M inducing points). Use SVGP
   or MGGP_SVGP. See :doc:`examples/colon` for configuration guidance.

Coordinate Systems
------------------

Each dataset stores spatial coordinates in a specific ``obsm`` key:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Dataset
     - Coordinate key
     - Notes
   * - ``slideseq``
     - ``obsm["spatial"]``
     - Standard squidpy convention
   * - ``tenxvisium``
     - ``obsm["spatial"]``
     - Standard squidpy convention
   * - ``merfish``
     - ``obsm["spatial"]``
     - Standard squidpy convention
   * - ``sdmbench``
     - ``obsm["spatial"]``
     - Standard squidpy convention
   * - ``liver``
     - ``obsm["X_spatial"]``
     - Non-standard; handled by loader
   * - ``osmfish``
     - ``obsm["spatial"]``
     - Standard
   * - ``colon``
     - ``obsm["spatial"]``
     - Stored as a pandas DataFrame; sorted by index before use

All coordinates are normalized by ``spatial_scale`` (default 50.0) before saving
to ``preprocessed/X.npy``.

Preprocessing Filters
----------------------

Two filters are applied after loading, before saving to ``preprocessed/``:

**NaN filter**
   Drops cells where any of the following is NaN or invalid:

   - Spatial coordinates (either X or Y is NaN)
   - Expression data (any gene count is NaN)
   - Group code (cell assigned to an undefined or NaN category)

**Small-group filter**
   Drops cells belonging to groups that are too small to be meaningful.
   Controlled by ``min_group_fraction`` (fraction of total cells) or
   ``min_group_size`` (absolute count). When ``min_group_fraction`` is set,
   it takes precedence.

   After filtering, surviving group codes are re-encoded contiguously (0..G'-1)
   preserving their original relative order.

Adding a Custom Dataset
-----------------------

To add a new dataset:

1. Create a loader class in ``spatial_factorization/datasets/`` that returns a
   ``SpatialData`` object with ``X`` (N×2 coordinates), ``Y`` (D×N counts),
   and ``groups`` (N group codes).
2. Register the loader in ``spatial_factorization/datasets/__init__.py``.
3. Add a config directory under ``configs/{dataset}/`` with ``general.yaml``
   and ``general_test.yaml``.

See ``spatial_factorization/datasets/slideseq.py`` as a reference implementation.
