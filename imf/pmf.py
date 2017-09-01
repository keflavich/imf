"""
Protostellar mass functions as described by McKee and Offner, 2010
"""

import numpy as np
import scipy.integrate
import warnings

from .imf import MassFunction, Chabrier2005, Kroupa

chabrier2005 = Chabrier2005()

class McKeeOffner_PMF(MassFunction):
    def __init__(self, j=1, n=1, jf=3/4., mmin=0.033, mmax=3.0, massfunc=chabrier2005, **kwargs):
        """
        """
        self.j = j
        self.jf = jf
        self.n = n
        self.mmin = mmin
        self.mmax = mmax
        self.massfunc = massfunc

        def den_func(x):
            return self.massfunc(x)*x**(-self.jf)
        self.denominator = scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, mass, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""
        if taper:

            def num_func(x, mass_):
                tf = (1-(mass_/x)**(1-self.j))**0.5
                return self.massfunc(x)*x**(self.j-self.jf-1) * tf

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x):
                return self.massfunc(x)*x**(self.j-self.jf-1)

            def integrate(lolim):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin))

        result = (1-self.j) * mass**(1-self.j) * numerator / self.denominator
        if integral_form:
            warnings.warn("The 'integral form' of the Chabrier PMF is not correctly normalized; "
                          "it is just PMF(m) * m")
            return result * self.normfactor * mass
            raise ValueError("Integral version not yet computed")
        else:
            return result * self.normfactor

class McKeeOffner_2CTC(MassFunction):
    """ 2-component Turbulent Core variant """
    def __init__(self, Rmdot=3.6, j=0.5, jf=3/4., mmin=0.033, mmax=3.0,
                 massfunc=chabrier2005, **kwargs):
        """
        """
        self.j = j
        self.jf = jf
        self.mmin = mmin
        self.mmax = mmax
        self.Rmdot = Rmdot
        self.massfunc = massfunc

        def den_func(x):
            return self.massfunc(x) * (2/((1+Rmdot**2*x**1.5)**0.5+1))
        self.denominator = scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, mass, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""
        if taper:

            def num_func(x, mass_):
                tf = (1-(mass_/x)**(1-self.j))**0.5
                return self.massfunc(x)*(1./x)**(1-self.j) * (2/((1+self.Rmdot**2*x**1.5)**0.5+1)) * tf

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x):
                return chabrier2005(x)*(1./x)**(1-self.j) * (2/((1+self.Rmdot**2*x**1.5)**0.5+1))

            def integrate(lolim):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin))

        result = (1-self.j) * mass**(1-self.j) * numerator / self.denominator
        if integral_form:
            warnings.warn("The 'integral form' of the Chabrier PMF is not correctly normalized; "
                          "it is just PMF(m) * m")
            return result * self.normfactor * mass
            raise ValueError("Integral version not yet computed")
        else:
            return result * self.normfactor

ChabrierPMF_IS = McKeeOffner_PMF(j=0, jf=0, )
ChabrierPMF_TC = McKeeOffner_PMF(j=0.5, jf=0.75, )
ChabrierPMF_CA = McKeeOffner_PMF(j=2/3., jf=1.0, )
ChabrierPMF_2CTC = McKeeOffner_2CTC()

class McKeeOffner_SalpeterPMF(MassFunction):
    " special case; above is now generalized to obsolete this "
    def __init__(self, j=1, jf=3/4., alpha=2.35, mmax=3.0):
        self.alpha = alpha
        self.mmax = mmax
        self.j = j
        self.jf = jf

    def __call__(self, mass, **kwargs):
        alpha = (self.alpha-1+self.jf-self.j)
        fm = 1 - (mass/self.mmax)**(alpha)
        result = fm * mass**(-((self.alpha-2)+self.jf))
        return result

SalpeterPMF_IS = McKeeOffner_SalpeterPMF(j=0, jf=0, )
SalpeterPMF_TC = McKeeOffner_SalpeterPMF(j=0.5, jf=0.75, )
SalpeterPMF_CA = McKeeOffner_SalpeterPMF(j=2/3., jf=1.0, )

kroupa = Kroupa()
KroupaPMF_IS = McKeeOffner_PMF(j=0, jf=0, massfunc=kroupa)
KroupaPMF_TC = McKeeOffner_PMF(j=0.5, jf=0.75, massfunc=kroupa)
KroupaPMF_CA = McKeeOffner_PMF(j=2/3., jf=1.0, massfunc=kroupa)
KroupaPMF_2CTC = McKeeOffner_2CTC(massfunc=kroupa)
