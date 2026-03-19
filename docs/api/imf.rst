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

.. automodule:: imf.imf
   :members:


.. [S55] `Salpeter (1955) <https://scixplorer.org/abs/1955ApJ...121..161S/abstract>`_
.. [K01] `Kroupa (2001) <https://scixplorer.org/abs/2001MNRAS.322..231K/abstract>`_
.. [C03] `Chabrier (2003) <https://scixplorer.org/abs/2003PASP..115..763C/abstract>`_
.. [C05] `Chabrier (2005) <https://scixplorer.org/abs/2005ASSL..327...41C/abstract>`_
