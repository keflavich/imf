Documentation
=============

This package is very shy on documentation at the moment.  There's a nice set of examples here:
https://github.com/keflavich/imf/tree/master/examples

but almost nothing else has been documented.


Simple Example
--------------

Plot the Kroupa and Chabrier mass functions:

.. python::

   import numpy as np
   import imf
   import pylab as pl

   k_masses = np.linspace(imf.kroupa.mmin, imf.kroupa.mmax, 1000)

   # Chabrier2005 has no maximum mass
   c_masses = np.linspace(imf.chabrier2005.mmin, imf.kroupa.mmax, 1000)

   pl.loglog(c_masses, imf.chabrier2005(c_masses), label='Chabrier 2005 IMF')
   pl.loglog(c_masses, imf.chabrier2005(c_masses), label='Chabrier 2005 IMF')
   pl.loglog(k_masses, imf.kroupa(k_masses), label='Kroupa IMF')
   pl.legend(loc='best')
   pl.xlabel("Stellar Mass (M$_\odot$)")
   pl.ylabel("P(M)")


Model a 10,000 solar mass cluster with two IMFs:

.. python::

   import numpy as np
   import imf
   import pylab as pl

   cluster1 = imf.make_cluster(1e4, massfunc='kroupa')
   cluster2 = imf.make_cluster(1e4, massfunc=imf.Chabrier2005(mmax=imf.kroupa.mmax))

   pl.hist(cluster1, bins=np.geomspace(0.03, 120), label='Kroupa', alpha=0.5)
   pl.hist(cluster2, bins=np.geomspace(0.03, 120), label='Chabrier', alpha=0.5)
   pl.xscale('log')
   pl.yscale('log')
   pl.legend(loc='best')
   pl.xlabel("Stellar Mass (M$_\odot$)")
   pl.ylabel("N(M)")
   


Advanced
^^^^^^^^

.. toctree::
   :maxdepth: 1
