"""
Protostellar mass functions as described by McKee and Offner, 2010
"""

import numpy as np
import scipy.integrate
from scipy.optimize import root_scalar
import warnings

from .imf import MassFunction, ChabrierPowerLaw, Kroupa

chabrierpowerlaw = ChabrierPowerLaw()

###this is new###

hist_values = {'is' : (0, 0, 1.54e-6, 10, 1.5),
               'tc' : (0.5, 0.75, 4.9e-6, 0.1, 0.75),
               'ca' : (2/3, 1., 6.9e-6, 1e4, 0.5)}

def scaling(history,value=None):
    params = hist_values[history]
    if value is None:
        value = params[3]
    return params[2] * (value / params[3])**params[4]

class ProtoMassFunction:
    def __init__(self,imf,
                 history=None,
                 j=None,jf=None,scale_value=None,
                 n=1,mmin=None,mmax=None):
        self.imf = imf

        self.history = history
        
        if self.history is None:
            self.j = j
            self.jf = jf

        self.scale_value = scale_value

        self.n = n

        self.mmin = self.imf.mmin if mmin is None else mmin
        self.mmax = self.imf.mmax if mmax is None else mmax

    def __call__(self,mass,
                 taper=False,integral_form=False,
                 **kwargs):
        avg_time = self.weight_average(self.tf,taper)

        def integrand(mf,mass_):
            if taper:
                tf = self.tf(mf,taper=taper)
                def root_t(t,mf,mass_):
                    term1 = t * (1 - (t / tf)**self.n / (self.n + 1))
                    term2 = mass_**(1 - self.j) / self.scale_value / (1 - self.j) / mf**(self.jf - self.j)
                    prime_term1 = 1 - (t / tf)**self.n / (self.n + 1)
                    prime_term2 = self.n / (self.n + 1) * (t / tf)**self.n
                    return term1 - term2, prime_term1 - prime_term2
                    
                def taper_factor(mf,mass_):
                    sol = root_scalar(root_t,args=(mf,mass_),x0=0,fprime=True)
                    return 1 - (sol.root / tf)**self.n
                
                return self.imf(mf) * mass_**(1 - self.j) * mf**(self.j - self.jf) / self.scale_value / taper_factor(mf,mass_)

            else:
                return self.imf(mf) * mass_**(1 - self.j) * mf**(self.j - self.jf) / self.scale_value

        def integral(lolim,mass_,**kwargs):
            return scipy.integrate.quad(integrand,lolim,self.mmax,args=(mass_),**kwargs)[0]
        
        ret = np.vectorize(integral)(np.where(self.mmin < mass, mass, self.mmin),mass)
        return ret / avg_time
        
    def weight_average(self,func,*args):
        """
        Integrates a function of stellar mass f(m) over the
        base IMF of a PMF.
        """
        def weighted_func(x):
            return self.imf(x) * func(x,*args)
        
        num = scipy.integrate.quad(weighted_func, self.mmin, self.mmax)[0]
        return num * self.imf.normfactor

    def tf(self,x,taper=False):
        factor = (self.n + 1) / self.n if taper else 1
        tf1 = factor / (1 - self.j) / self.scale_value
        return tf1 * x**(1 - self.jf)

    @property
    def imf(self):
        return self._imf

    @imf.setter
    def imf(self,val):
        self._imf = val

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self,x):
        if x is None:
            self._history = x
        else:
            if not x in hist_values.keys():
                raise ValueError("history must be one of 'is'/'tc'/'ca'")
        
            self._history = x
            self._j = hist_values[x][0]
            self._jf = hist_values[x][1]
            self._scale_value = hist_values[x][2]

    @property
    def j(self):
        return self._j

    @j.setter
    def j(self,x):
        if self.history in hist_values.keys():
            raise ValueError('j cannot take on alternate values for a defined history')
        else:
            self._j = x

    @property
    def jf(self):
        return self._jf

    @jf.setter
    def jf(self,x):
        if self.history	in hist_values.keys():
            raise ValueError('jf cannot take on alternate values for a defined history')
        else:
            self._jf = x

    @property
    def scale_value(self):
        return self._scale_value

    @scale_value.setter
    def scale_value(self,x):
        if self.history in hist_values.keys():
            self._scale_value = scaling(self.history,x)
        else:
            self._scale_value = x
        
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self,x):
        if x <= 0:
            raise ValueError('n must be > 0')
        self._n = x

    @property
    def mmin(self):
        return self._mmin

    @mmin.setter
    def mmin(self,x):
        self._mmin = x
        self.imf._mmin = x
        self.imf.normalize()

    @property
    def mmax(self):
        return self._mmax

    @mmax.setter
    def mmax(self,x):
        self._mmax = x
        self.imf._mmax = x
        self.imf.normalize()

#class PMF_2C(ProtoMassFunction):
#    """
#    description
#    """
#    def __init__(self,etc.):
#        return 0
        
###end new###
        
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

        self.denominator = scipy.integrate.quad(self.den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, mass, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""       
        if taper:

            def num_func(x, mass_):
                tf = (1 - (mass_ / x)**(1 - self.j))**0.5
                return self.imf(x) * x**(self.j - self.jf - 1) * tf

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x):
                return self.imf(x) * x**(self.j - self.jf - 1)

            def integrate(lolim):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin))

        result = (1 - self.j) * mass**(1 - self.j) * numerator / self.denominator
        if integral_form:
            warnings.warn("The 'integral form' of the Chabrier PMF is not correctly normalized; "
                          "it is just PMF(m) * m")
            return result * self.normfactor * mass
            raise ValueError("Integral version not yet computed")
        else:
            return result * self.normfactor
    
    def den_func(self,x):
        return self.imf(x) * x**(-self.jf)
    
    @property
    def mmin(self):
        return self._mmin
    
    @mmin.setter
    def mmin(self,mass,**kwargs):
        self._mmin = mass
        self.denominator = scipy.integrate.quad(self.den_func, self.mmin, self.mmax, **kwargs)[0]

    @property
    def mmax(self):
        return self._mmax
        
    @mmax.setter
    def mmax(self,mass,**kwargs):
        self._mmax = mass
        self.denominator = scipy.integrate.quad(self.den_func, self.mmin, self.mmax, **kwargs)[0]

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

        self.denominator = scipy.integrate.quad(self.den_func, self.mmin, self.mmax, **kwargs)[0]

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

    def den_func(self,x):
        return self.imf(x) * (2 / ((1 + self.Rmdot**2 * x**1.5)**0.5 + 1))


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
