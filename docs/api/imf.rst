.. _imf:

imf.imf
=======

Contains all implemented IMF forms, as well as the basic class for mass functions. Running ``import imf`` will provide direct access to all of these functions, i.e. calling functions (e.g. ``make_cluster``) and classes (e.g. ``Salpeter``) can then be done with ``imf.{function/class}``.

``imf`` also pre-defines some particularly common IMF forms as global objects which can be imported: these are as follows:

* ``salpeter`` returns the power law IMF of [S55]_.
* ``kroupa`` returns the broken power law IMF of [K01]_.
* ``chabrier`` or ``chabrierpowerlaw`` returns the log-normal + power law IMF of [C03]_.
* ``lognormal`` or ``chabrierlognormal`` returns an IMF which is just the log-normal component of [C03]_.
* ``chabrier2005`` returns the slightly modified IMF of [C05]_.

.. note::
   Many IMF forms involve power laws in some way; however, notation may
   vary from usage to usage. This package adopts the convention that when
   working with a ``MassFunction``, the power law shape is determined by
   the parameter :math:`\alpha`, which corresponds to :math:`dn/dm`, while
   the shape parameter for :math:`dn/d\,{\rm log}\, m` is
   :math:`\Gamma=\alpha-1`.

   For a mass function with a decreasing power law slope, a positive
   value should be provided for ``alpha`` or ``powers`` (keyword
   depending on the function), i.e. this package adopts a negative sign
   convention such that the more commonly occurring decreasing power laws
   are more "natural" to construct. Anything notated as a power law
   "exponent" in this document assumes this convention.

.. automodule:: imf.imf
   :members:

.. [S55] `Salpeter (1955) <https://doi.org/10.1086/145971>`_
.. [K01] `Kroupa (2001) <https://doi.org/10.1046/j.1365-8711.2001.04022.x>`_
.. [C03] `Chabrier (2003) <https://doi.org/10.1086/376392>`_
.. [C05] `Chabrier (2005) <https://doi.org/10.48550/arXiv.astro-ph/0409465>`_
