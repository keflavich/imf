.. _distributions:

imf.distributions
=================

The base statistical distributions underpinning IMF forms. All distributions are defined within an interval [m1, m2], which are arguments for all distributions unless otherwise noted.

.. warning::
   Due to the construction of the package, the power law slope sign convention
   for ``Distributions`` is the opposite of the convention for ``MassFunctions``,
   i.e. negative slopes correspond to decreasing power laws.
   ``KoenConvolvedPowerLaw`` and ``PadoanTF`` are exceptions, as they inherit
   the conventions of their companion papers.

   Power laws with :math:`dn/dm \\propto m^{-1}` will not have defined CDFs
   due to the lack of an appropriate analytical form. 

.. automodule::	imf.distributions
   :members:
