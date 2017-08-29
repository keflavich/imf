"""
Protostellar mass functions as described by McKee and Offner, 2010
"""

import numpy as np
import scipy.integrate

from .imf import MassFunction, chabrier, Chabrier2005

chabrier2005 = Chabrier2005()

class ChabrierPMF(MassFunction):
    def __init__(self, j=1, n=1, jf=3/4., mlow=0.033, mhigh=3.0, **kwargs):
        """
        """
        self.j = j
        self.jf = jf
        self.n = n
        self.mlow = mlow
        self.mhigh = mhigh

        def den_func(x):
            return chabrier2005(x)*x**(-self.jf)
        self.denominator = scipy.integrate.quad(den_func, self.mlow, self.mhigh, **kwargs)[0]

    def __call__(self, mass, **kwargs):
        def num_func(x):
            return chabrier2005(x)*x**(self.j-self.jf-1)

        def integrate(lolim):
            integral = scipy.integrate.quad(num_func, lolim, self.mhigh, **kwargs)[0]
            return integral

        numerator = np.vectorize(integrate)(np.where(self.mlow < mass, mass, self.mlow))

        result = (1-self.j) * mass**(1-self.j) * numerator / self.denominator
        return result

ChabrierPMF_IS = ChabrierPMF(j=0, jf=0, )
ChabrierPMF_TC = ChabrierPMF(j=0.5, jf=0.75, )
ChabrierPMF_CA = ChabrierPMF(j=2/3., jf=1.0, )

class SalpeterPMF(MassFunction):
    def __init__(self, j=1, jf=3/4., alpha=2.35, mhigh=3.0):
        self.alpha = alpha
        self.mhigh = mhigh
        self.j = j
        self.jf = jf

    def __call__(self, mass, **kwargs):
        alpha = (self.alpha-1+self.jf-self.j)
        fm = 1 - (mass/self.mhigh)**(alpha)
        result = fm * mass**(-((self.alpha-2)+self.jf))
        return result

SalpeterPMF_IS = ChabrierPMF(j=0, jf=0, )
SalpeterPMF_TC = ChabrierPMF(j=0.5, jf=0.75, )
SalpeterPMF_CA = ChabrierPMF(j=2/3., jf=1.0, )
