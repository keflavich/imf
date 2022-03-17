"""
Protostellar mass functions as described by McKee and Offner, 2010
"""

import numpy as np
import scipy.integrate
import warnings

from .imf import MassFunction, ChabrierPowerLaw, Kroupa

chabrierpowerlaw = ChabrierPowerLaw()

class McKeeOffner_PMF(MassFunction):
    default_mmin = 0.033
    default_mmax = 3.0

    def __init__(self, j=1, n=1, jf=3/4., mmin=default_mmin, mmax=default_mmax,
                 imf=chabrierpowerlaw, **kwargs):
        """
        """
        super().__init__(mmin=mmin, mmax=mmax)
        self.j = j
        self.jf = jf
        self.n = n
        self.imf = imf

        def den_func(x):
            return self.imf(x)*x**(-self.jf)
        self.denominator = scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, mass, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""
        if taper:

            def num_func(x, mass_):
                tf = (1-(mass_/x)**(1-self.j))**0.5
                return self.imf(x)*x**(self.j-self.jf-1) * tf

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x):
                return self.imf(x)*x**(self.j-self.jf-1)

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
    default_mmin = 0.033
    default_mmax = 3.0

    def __init__(self, Rmdot=3.6, j=0.5, jf=3/4., mmin=default_mmin, mmax=default_mmax,
                 imf=chabrierpowerlaw, **kwargs):
        """
        """
        super().__init__(mmin=mmin, mmax=mmax)
        self.j = j
        self.jf = jf
        self.Rmdot = Rmdot
        self.imf = imf

        def den_func(x):
            return self.imf(x) * (2/((1+Rmdot**2*x**1.5)**0.5+1))
        self.denominator = scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, mass, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""
        if taper:

            def num_func(x, mass_):
                tf = (1-(mass_/x)**(1-self.j))**0.5
                return self.imf(x)*(1./x)**(1-self.j) * (2/((1+self.Rmdot**2*x**1.5)**0.5+1)) * tf

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x):
                return self.imf(x)*(1./x)**(1-self.j) * (2/((1+self.Rmdot**2*x**1.5)**0.5+1))

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
    default_mmin = 0.033
    default_mmax = 3.0

    def __init__(self, j=1, jf=3/4., alpha=2.35, mmin=default_mmin, mmax=default_mmax):
        super().__init__(mmin=mmin, mmax=mmax)
        self.alpha = alpha
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
KroupaPMF_IS = McKeeOffner_PMF(j=0, jf=0, imf=kroupa)
KroupaPMF_TC = McKeeOffner_PMF(j=0.5, jf=0.75, imf=kroupa)
KroupaPMF_CA = McKeeOffner_PMF(j=2/3., jf=1.0, imf=kroupa)
KroupaPMF_2CTC = McKeeOffner_2CTC(imf=kroupa)

class McKeeOffner_AcceleratingSF_PMF(MassFunction):
    default_mmin = 0.033
    default_mmax = 3.0

    def __init__(self, j=1, n=1, jf=3/4., mmin=default_mmin, mmax=default_mmax,
                 tau=1, # current time, Myr
                 tm=0.54, # SF timescale, Myr
                 tf1=0.50, # accretion  timescale for a 1-msun star
                 imf=chabrierpowerlaw, **kwargs):
        """
        McKee & Offner 2010 Protostellar Mass Function with an accelerating star formation rate
        """
        super().__init__(mmin=mmin, mmax=mmax)
        self.j = j
        self.jf = jf
        self.n = n
        self.tau = tau
        self.tm = tm
        self.imf = imf
        self.tf1 = tf1

        def den_func(x):
            return self.imf(x)*(1-np.exp(-self.tf1*x**(1-self.jf)/self.tau))/x
        self.denominator = self.tau * scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]
        assert self.denominator > 0

        self.normfactor = 1

    def __call__(self, mass, tau=None, taper=False, integral_form=False, **kwargs):
        """

        Parameters
        ----------
        tm : float
            Star formation timescale in Myrs
        """
        if tau is None:
            tau = self.tau

        if taper:

            raise NotImplementedError()
        #    def num_func(x, mass_):
        #        tf = (1-(mass_/x)**(1-self.j))**0.5
        #        return self.massfunc(x)*x**(self.j-self.jf-1) * tf

        #    def integrate(lolim, mass_):
        #        integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
        #        return integral

        #    numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x, mass_):
                tm = self.tf1 * mass_**(1-self.j) / (x**(self.jf-self.j))
                return self.imf(x)*x**(self.j-self.jf-1) * np.exp(-tm / tau)

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

            assert np.all(numerator >= 0)

        result = self.tf1 * (1-self.j) * mass**(1-self.j) * numerator / self.denominator

        # it is possible with the time-evolution case to get negative values for high masses;
        # these should (probably?) just be zero'd
        result = np.where(mass > self.mmax, np.nan, result)

        if integral_form:
            warnings.warn("The 'integral form' of the Chabrier PMF is not correctly normalized; "
                          "it is just PMF(m) * m")
            return result * self.normfactor * mass
            raise ValueError("Integral version not yet computed")
        else:
            return result * self.normfactor

ChabrierPMF_AcceleratingSF_IS = McKeeOffner_AcceleratingSF_PMF(j=0, jf=0, )
ChabrierPMF_AcceleratingSF_TC = McKeeOffner_AcceleratingSF_PMF(j=0.5, jf=0.75, )
ChabrierPMF_AcceleratingSF_CA = McKeeOffner_AcceleratingSF_PMF(j=2/3., jf=1.0, )
#ChabrierPMF_AcceleratingSF_2CTC = McKeeOffner_AcceleratingSF_2CTC()
