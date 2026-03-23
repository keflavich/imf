.. _make_cluster:

imf.make_cluster
================

This function enables sampling from the package's various mass functions.

.. currentmodule:: imf

.. autofunction:: imf.make_cluster

Sampling details
----------------

Random sampling
^^^^^^^^^^^^^^^

Random sampling in ``imf`` is done by repeatedly drawing samples from a uniform distribution and using the cumulative distribution function (CDF) associated with each IMF form to map them to masses until the total mass of the population exceeds the target mass ``mcluster``. Once the threshold is reached, ``imf`` determines what to do based on one of its stop criteria, set with the ``stop_criterion`` keyword. Here's what each criterion means for a sampled population in practice:

* ``'nearest'``: Include all stars drawn from an IMF (in drawing order) that bring the cumulative mass of the cluster closest to ``mcluster``. Sometimes exceeds ``mcluster``.

  * *Example:* Cluster with ``mcluster = 1000``, of which 950 :math:`M_\odot` are already included. The next three sampled stars have masses (0.2, 45, 10) :math:`M_\odot`. 995.2 :math:`M_\odot` (950 + 0.2 + 45) is closest to ``mcluster``, so the first two stars are included and none after.

* ``'before'``: Include all stars drawn from an IMF (in drawing order) with cumulative mass less than ``mcluster``. Never exceeds ``mcluster``.
* ``'after'``: Include all stars drawn from an IMF (in drawing order) with cumulative mass less than ``mcluster``, and also the next star. Always exceeds ``mcluster``.
* ``'sorted'``: Sort the stars by mass, then include or exclude massive stars based on the ``'nearest'`` criterion.


Optimal sampling
^^^^^^^^^^^^^^^^

Optimal sampling creates a population which perfectly reproduces the shape of its underlying mass function. ``imf`` implements this following the algorithm of `Schulz et al. (2015) <https://doi.org/10.1051/0004-6361/201425296>`__. In short, the total mass budget and IMF are used to find the mass of the most massive star in the cluster and the accompanying scale factor for the IMF; once these are found, successive members are added by locating the :math:`dm` along the scaled IMF that produce a :math:`dn` of 1 until the entire mass budget is consumed.
