"""
Protostellar luminosity functions as described by Offner and McKee, 2011


Alternatively, perhaps try to construct a probabilistic P(L; m, m_f) given a
series of stellar evolution codes?
"""

import numpy as np
import scipy.integrate
import warnings

from .imf import MassFunction, ChabrierPowerLaw

chabrierpowerlaw = ChabrierPowerLaw()

class McKeeOffner_PLF(MassFunction):
    def __init__(self, j=1, n=1, jf=3/4., mmin=0.033, mmax=3.0, imf=chabrierpowerlaw, **kwargs):
        """
        Incomplete.  The PLF requires a protostellar evolution code as part of its input.
        """
        raise NotImplementedError
        self.j = j
        self.jf = jf
        self.n = n
        self.mmin = mmin
        self.mmax = mmax
        self.imf = imf

        def den_func(x):
            return self.imf(x)*x**(-self.jf)
        self.denominator = scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, luminosity, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""
        if taper:

            def num_func(x, luminosity_):
                tf = (1-(luminosity_/x)**(1-self.j))**0.5
                return self.imf(x)*x**(self.j-self.jf-1) * tf

            def integrate(lolim, luminosity_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax,
                                                args=(luminosity_,),
                                                **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin <
                                                         luminosity,
                                                         luminosity,
                                                         self.mmin),
                                                luminosity)

        else:
            def num_func(x):
                return self.imf(x)*x**(self.j-self.jf-1)

            def integrate(lolim):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin <
                                                         luminosity,
                                                         luminosity,
                                                         self.mmin))

        result = (1-self.j) * luminosity**(1-self.j) * numerator / self.denominator
        if integral_form:
            warnings.warn("The 'integral form' of the Chabrier PMF is not correctly normalized; "
                          "it is just PMF(m) * m")
            return result * self.normfactor * luminosity
            raise ValueError("Integral version not yet computed")
        else:
            return result * self.normfactor
