Mathematical Background
=======================

This page describes the probabilistic models implemented in Spatial-Factorization.
All models share the same Poisson observation model and variational inference framework;
they differ in the prior placed on the latent factors.

Notation
--------

- :math:`N` — number of cells / spots
- :math:`D` — number of genes
- :math:`L` — number of latent components (``n_components``)
- :math:`G` — number of cell type groups
- :math:`M` — number of inducing points (SVGP: :math:`M \ll N`; LCGP: :math:`M = N`)
- :math:`K` — number of local neighbors (LCGP only)
- :math:`\mathbf{X} \in \mathbb{R}^{N \times 2}` — spatial coordinates
- :math:`\mathbf{Y} \in \mathbb{R}_+^{N \times D}` — count matrix (spots × genes)

Poisson Factorization Model
----------------------------

Each count entry is modeled as Poisson-distributed:

.. math::

   y_{id} \sim \text{Poisson}(\lambda_{id}), \qquad
   \lambda_{id} = \sum_{\ell=1}^{L} W_{d\ell} \exp(F_{i\ell})

where:

- :math:`\mathbf{W} \in \mathbb{R}_+^{D \times L}` are the **gene loadings** (non-negative)
- :math:`\mathbf{F} \in \mathbb{R}^{N \times L}` are the **latent factors** (log-space)

The factors :math:`\mathbf{F}` are random variables governed by a prior
:math:`p(\mathbf{F})` that depends on the model variant (see below).
The loadings :math:`\mathbf{W}` are optimized parameters.

Variational Inference
---------------------

We use **variational inference** to approximate the posterior
:math:`p(\mathbf{F} \mid \mathbf{Y})` with a tractable distribution
:math:`q(\mathbf{F})`.

For all model variants, the variational distribution is a fully-factorized Gaussian:

.. math::

   q(F_{i\ell}) = \mathcal{N}(\mu_{i\ell},\, \sigma_{i\ell}^2)

The **Evidence Lower BOund (ELBO)** is maximized jointly over :math:`\{\mu, \sigma, \mathbf{W}\}`:

.. math::

   \mathcal{L} = \mathbb{E}_{q(\mathbf{F})}[\log p(\mathbf{Y} \mid \mathbf{F})]
                 - \text{KL}[q(\mathbf{F}) \,\|\, p(\mathbf{F})]

Expected Log-Likelihood
~~~~~~~~~~~~~~~~~~~~~~~

For the Poisson likelihood, the expected log-likelihood per observation is:

.. math::

   \mathbb{E}_q\!\left[\log p(y_{id} \mid F_{i\cdot})\right] =
     y_{id}\, \mathbb{E}_q\!\left[\log \sum_\ell W_{d\ell} e^{F_{i\ell}}\right]
     - \sum_\ell W_{d\ell}\, \mathbb{E}_q\!\left[e^{F_{i\ell}}\right]
     - \log(y_{id}!)

Using the Gaussian moment-generating function, the second term is analytic:

.. math::

   \mathbb{E}_q\!\left[e^{F_{i\ell}}\right]
   = \exp\!\left(\mu_{i\ell} + \tfrac{1}{2}\sigma_{i\ell}^2\right)

The first term (log of a sum of exponentials) is estimated via Monte Carlo with
the reparameterization trick using :math:`E` samples (``E`` parameter in config,
default 3).

ELBO Modes
~~~~~~~~~~

Three ELBO computation modes are available (``model.mode`` in config):

``expanded`` (default)
   Monte Carlo for the log-sum-exp term; analytic MGF for the rate term. Best balance
   of speed and accuracy.

``simple``
   Full Monte Carlo estimation for all terms via
   ``torch.distributions.Poisson.log_prob()``. Slower but more general.

``lower-bound``
   Fully analytic lower bound via Jensen's inequality — no Monte Carlo sampling.
   Fastest mode, guarantees convergence.

Priors
------

Non-Spatial Prior (PNMF)
~~~~~~~~~~~~~~~~~~~~~~~~~

A factorized isotropic Gaussian prior:

.. math::

   F_{i\ell} \sim \mathcal{N}(0, 1)

All factors are treated as independent. No spatial information is used.

GP Kernel
~~~~~~~~~

All spatial priors use the **Matérn-3/2** kernel by default (``kernel: Matern32``):

.. math::

   k(x, x') = \sigma^2 \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
               \exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right), \qquad
   r = \|x - x'\|_2

Parameters:

- :math:`\sigma` — signal variance (fixed at 1.0 by default)
- :math:`\ell` — **lengthscale** (default 8.0, optionally trainable via ``train_lengthscale``)

The **RBF** kernel is also available (``kernel: RBF``):

.. math::

   k_{\text{RBF}}(x, x') = \sigma^2 \exp\!\left(-\frac{r^2}{2\ell^2}\right)

SVGP Prior
~~~~~~~~~~

The **Sparse Variational GP** places a GP prior over spatial coordinates using
:math:`M \ll N` inducing points :math:`\mathbf{Z} \in \mathbb{R}^{M \times 2}`:

.. math::

   f_\ell(\cdot) \sim \mathcal{GP}(0,\, k(\cdot, \cdot))

The variational distribution over the inducing outputs
:math:`\mathbf{u}_\ell = f_\ell(\mathbf{Z})` is:

.. math::

   q(\mathbf{u}_\ell) = \mathcal{N}(\mathbf{m}_\ell,\, \mathbf{S}_\ell)

with the Cholesky factor :math:`\mathbf{L}_u` such that
:math:`\mathbf{S}_\ell = \mathbf{L}_{u,\ell} \mathbf{L}_{u,\ell}^\top` (saved as ``Lu.pt``).

The predictive distribution at training points is:

.. math::

   q(f_\ell(\mathbf{X})) = \mathcal{N}(
       K_{NM} K_{MM}^{-1} \mathbf{m}_\ell,\;\;
       K_{NN} - K_{NM} K_{MM}^{-1}(K_{MM} - \mathbf{S}_\ell) K_{MM}^{-1} K_{MN}
   )

Compute: :math:`\mathcal{O}(NM + M^2)` per component.

MGGP_SVGP Prior
~~~~~~~~~~~~~~~

Extends SVGP with a **multi-group kernel** that models cross-group correlations.
Given group labels :math:`c_i \in \{0, \ldots, G-1\}`:

.. math::

   k_{\text{MGGP}}\!\left((x, c),\, (x', c')\right) =
   k_{\text{base}}(x, x') \cdot \rho(c, c')

where:

.. math::

   \rho(c, c') = \begin{cases}
     1 & \text{if } c = c' \\
     \exp(-\delta) & \text{if } c \neq c'
   \end{cases}

and :math:`\delta` is ``group_diff_param`` (default 1.0; higher = more independent groups).

Inducing points :math:`\mathbf{Z}` have associated group labels ``groupsZ``.
Saved: ``Lu.pt`` (:math:`L \times M \times M`), ``groupsZ.npy`` (:math:`M`).

LCGP Prior
~~~~~~~~~~

The **Locally Conditioned GP** uses all :math:`N` training points as inducing points
(:math:`M = N`). The variational covariance uses a VNNGP-style sparse factorization:

.. math::

   \mathbf{S}_\ell = \mathbf{L}_{u,\ell} \mathbf{L}_{u,\ell}^\top, \qquad
   \mathbf{L}_{u,\ell} \in \mathbb{R}^{N \times K}

where each row of :math:`\mathbf{L}_{u,\ell}` is non-zero only in the :math:`K`
nearest-neighbor columns of that point. This gives a sparse Cholesky factor
(saved as ``Lu.npy``, shape :math:`L \times N \times K`).

Compute: :math:`\mathcal{O}(NK^2)` per component.

**KNN strategies** for computing neighborhoods (``neighbors`` config field):

- ``knn`` (default): deterministic FAISS L2 nearest neighbors
- ``probabilistic``: kernel-weighted sampling via the Gumbel-max trick —
  each neighbor :math:`z_j` is sampled with probability proportional to
  :math:`k(x_i, z_j)`, without replacement

MGGP_LCGP Prior
~~~~~~~~~~~~~~~

Combines LCGP local conditioning with the MGGP multi-group kernel. All :math:`N`
training points serve as inducing points, with group labels ``groupsZ`` = ``C``
(the training group codes).

Factor Ordering
---------------

After training, all factor-related outputs are **reordered by descending Moran's I**
in the analyze stage. Factor 0 always has the highest spatial autocorrelation.

**Moran's I** for factor :math:`\ell` is:

.. math::

   I_\ell = \frac{N}{\sum_{ij} w_{ij}} \cdot
             \frac{\sum_{ij} w_{ij}(F_{i\ell} - \bar{F}_\ell)(F_{j\ell} - \bar{F}_\ell)}
                  {\sum_i (F_{i\ell} - \bar{F}_\ell)^2}

where :math:`w_{ij}` is a spatial weight matrix (inverse distance, KNN-based).

This reordering is applied consistently to: ``factors.npy``, ``scales.npy``,
``loadings.npy``, ``Lu.pt``/``Lu.npy``, per-group loadings, and gene enrichment.
