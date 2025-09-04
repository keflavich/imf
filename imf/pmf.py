"""
Protostellar mass functions as described by McKee and Offner, 2010
"""

import numpy as np
import scipy.integrate
import warnings

from .imf import MassFunction, ChabrierPowerLaw, Kroupa
from . import distributions

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

class PMF(MassFunction):
    """
    documentation
    """
    def __init__(self,imf,
                 mmin=None,mmax=None,
                 history='is',
                 j=None,jf=None,scale_value=None,
                 n=1,tau=1):
        self.distr = None
        
        self._imf = imf
        self.imf.normalize()

        self._mmin = self.imf.mmin if mmin is None else mmin
        self._mmax = self.imf.mmax if mmax is None else mmax

        self.history = history

        if self.history is None:
            self._j = j
            self._jf = jf
        
        self.scale_value = scale_value

        self._n = n
        self._tau = tau
        
        self.distr = distributions.PMF(self.imf,self.mmin,self.mmax,
                                       self.j,self.jf,self.scale_value,
                                       self.n,self.tau)
        self.normfactor = 1
        
    def __call__(self,mass,
                 integral_form=False,
                 taper=False,
                 accelerating=False,
                 **kwargs):

        self.distr.taper = taper
        self.distr.accelerating = accelerating
        
        if integral_form:
            return self.distr.cdf(mass) * self.normfactor
        else:
            return self.distr.pdf(mass) * self.normfactor

    def mass_weighted(self,x,
                      taper=False,
                      accelerating=False):
        return self(x,taper=taper,accelerating=accelerating) * x

    def tf(self,mf,taper=False):
        """
        Returns the expected formation time of a star with
        final mass mf following the accretion history
        underlying the PMF.
        """
        return self.distr._tf(mf,taper)

    def average_time(self,taper=False,accelerating=False):
        """
        Returns the IMF-averaged star formation time of the
        PMF.
        """
        return self.distr._average_time(taper,accelerating)
        
    @property
    def imf(self):
        return self._imf

    @imf.setter
    def imf(self,x):
        self._imf = x
        self._imf.normalize()
        self.distr.imf = self._imf
        self.distr.calculate('all')

    @property
    def mmin(self):
        return self._mmin

    @mmin.setter
    def mmin(self,x):
        self._mmin = x
        self.distr.mmin = x
        self.distr._calculate('all')
            
    @property
    def mmax(self):
        return self._mmax

    @mmax.setter
    def mmax(self,x):
        self._mmax = x
        self.distr.mmax = x
        self.distr._calculate('all')

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

            if self.distr is not None:
                self.distr.j = self.j
                self.distr.jf = self.jf
                self.distr.scale_value = self.scale_value
                self.distr._calculate('all')
            
    @property
    def j(self):
        return self._j

    @j.setter
    def j(self,x):
        if self.history in hist_values.keys():
            raise ValueError('j cannot take on alternate values for a defined history')
        else:
            self._j = x
            self.distr.j = x
            self.distr._calculate('all')

    @property
    def jf(self):
        return self._jf

    @jf.setter
    def jf(self,x):
        if self.history	in hist_values.keys():
            raise ValueError('jf cannot take on alternate values for a defined history')
        else:
            self._jf = x
            self.distr.jf = x
            self.distr._calculate('all')

    @property
    def scale_value(self):
        return self._scale_value

    @scale_value.setter
    def scale_value(self,x):
        if self.history in hist_values.keys():
            self._scale_value = scaling(self.history,x)
        else:
            self._scale_value = x

        if self.distr is not None:
            self.distr.scale_value = self._scale_value
            self.distr._calculate('all')
        
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self,x):
        if x <= 0:
            raise ValueError('n must be > 0')
        self._n = x
        self.distr.n = x
        self.distr._calculate('taper')

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self,x):
        self._tau = x
        self.distr.tau = x
        self.distr._calculate('accelerating')


hist_values_2C = {'tc' : (0.5, 0.75, 3.6),
                  'ca' : (2/3, 1., 3.2)}

class PMF_2C(MassFunction):
    """
    documentation
    """
    def __init__(self,imf,
                 mmin=None,mmax=None,
                 history='is',
                 j=None,jf=None,
                 R_mdot=None,T=10,
                 n=1,tau=1):
        self.distr = None
        
        self._imf = imf
        self.imf.normalize()

        self._mmin = self.imf.mmin if mmin is None else mmin
        self._mmax = self.imf.mmax if mmax is None else mmax

        self.history = history

        if self.history is None:
            self._j = j
            self._jf = jf
            self._R_mdot = R_mdot

        self._n = n
        self._tau = tau

        self._T = T
        self.m_is = scaling('is',self.T)
        
        self.distr = distributions.PMF_2C(self.imf,self.mmin,self.mmax,
                                          self.j,self.jf,
                                          self.R_mdot,self.m_is,
                                          self.n,self.tau)
        self.normfactor = 1

    def __call__(self,mass,
                 integral_form=False,
                 taper=False,
                 accelerating=False,
                 **kwargs):

        self.distr.taper = taper
        self.distr.accelerating = accelerating
        
        if integral_form:
            return self.distr.cdf(mass) * self.normfactor
        else:
            return self.distr.pdf(mass) * self.normfactor

    def mass_weighted(self,x,
                      taper=False,
                      accelerating=False):
        return self(x,taper=taper,accelerating=accelerating) * x

    def tf(self,mf,taper=False):
        """
        Returns the expected formation time of a star with
        final mass mf following the accretion history
        underlying the PMF.
        """
        return self.distr._tf(mf,taper)

    def average_time(self,taper=False,accelerating=False):
        """
        Returns the IMF-averaged star formation time of the
        PMF.
        """
        return self.distr._average_time(taper,accelerating)
        
    @property
    def imf(self):
        return self._imf

    @imf.setter
    def imf(self,x):
        self._imf = x
        self._imf.normalize()
        self.distr.imf = self._imf
        self.distr.calculate('all')

    @property
    def mmin(self):
        return self._mmin

    @mmin.setter
    def mmin(self,x):
        self._mmin = x
        self.distr.mmin = x
        self.distr._calculate('all')
            
    @property
    def mmax(self):
        return self._mmax

    @mmax.setter
    def mmax(self,x):
        self._mmax = x
        self.distr.mmax = x
        self.distr._calculate('all')

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self,x):
        if x is None:
            self._history = x
        else:
            if not x in hist_values_2C.keys():
                raise ValueError("history must be one of 'tc'/'ca'")
        
            self._history = x
            self._j = hist_values_2C[x][0]
            self._jf = hist_values_2C[x][1]
            self._R_mdot = hist_values_2C[x][2]

            if self.distr is not None:
                self.distr.j = self.j
                self.distr.jf = self.jf
                self.distr.R_mdot = self.R_mdot
                self.distr._calculate('all')
            
    @property
    def j(self):
        return self._j

    @j.setter
    def j(self,x):
        if self.history in hist_values_2C.keys():
            raise ValueError('j cannot take on alternate values for a defined history')
        else:
            self._j = x
            self.distr.j = x
            self.distr._calculate('all')

    @property
    def jf(self):
        return self._jf

    @jf.setter
    def jf(self,x):
        if self.history	in hist_values_2C.keys():
            raise ValueError('jf cannot take on alternate values for a defined history')
        else:
            self._jf = x
            self.distr.jf = x
            self.distr._calculate('all')

    @property
    def R_mdot(self):
        return self._R_mdot

    @R_mdot.setter
    def R_mdot(self,x):
        self._R_mdot = x
        self.distr.R_mdot = x
        self.distr._calculate('all')

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self,x):
        self._T = x
        self.m_is = scaling('is',x)

        self.distr.m_is = self.m_is
        self.distr._calculate('all')
        
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self,x):
        if x <= 0:
            raise ValueError('n must be > 0')
        self._n = x
        self.distr.n = x
        self.distr._calculate('taper')

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self,x):
        self._tau = x
        self.distr.tau = x
        self.distr._calculate('accelerating')
       
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
                return self.imf(x) * x**(self.j - self.jf) / tf * self.n / (self.n + 1)

            def integrate(lolim, mass_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, args=(mass_,), **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin < mass, mass, self.mmin), mass)

        else:
            def num_func(x):
                return self.imf(x) * x**(self.j - self.jf)

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
        return self.imf(x) * x**(1 - self.jf)
    
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
