import numpy as np
from scipy.integrate import quad,cumulative_trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root_scalar
from scipy.special import hyp2f1

from .imf import MassFunction
from .distributions import Distribution

hist_values = {'is' : (0, 0, 1.54e-6, 10, 1.5),
               'tc' : (0.5, 0.75, 4.9e-6, 0.1, 0.75),
               'ca' : (2/3, 1., 6.9e-6, 1e4, 0.5)}

def scaling(history,value=None):
    r"""
    Calculates the final untapered accretion rate for a star 
    of unit mass (in $M_\odot$ / yr) for the accretion histories
    implemented in McKee/Offner (2010).

    Parameters
    ----------
    history: str
        Accretion history of stars; accepts 'is' (isothermal sphere),
        'tc' (turbulent core), and 'ca' (competitive accretion).
    value: float
        Value of the scaling parameter relevant for the accretion history.
        If None, defaults to the fiducial value (10 K for IS, 0.1 g cm^-2
        for TC, 10^4 cm^-3 for CA)
    """
    if history not in hist_values.keys():
        raise ValueError(f'history must be one of {hist_values.keys()}')
    
    params = hist_values[history]
    if value is None:
        value = params[3]
    return params[2] * (value / params[3])**params[4]

class PMF(MassFunction):
    r"""
    Calculates the Protostellar Luminosity Function (PMF)
    corresponding to a supplied IMF and accretion history
    using the formalism of McKee/Offner (2010). 
    
    Parameters
    ----------
    imf: MassFunction
        The IMF of the emerging stellar population
    mmin: float
        Minimum final mass for stars (default = IMF min)
    mmax: float
        Maximum final mass for stars (default = IMF max)
    history: str
        Accretion history of stars; accepts 'is' (isothermal sphere),
        'tc' (turbulent core), and 'ca' (competitive accretion). If
        None, custom values can be input for j, jf, and scale_value
        (default = 'is')
    j_exp: float
        Value setting the scaling of accretion rate with current mass
        (default = None)
    jf_exp: float
        Value setting the scaling of accretion rate with final mass
        (default = None)
    scale_value: float
        Final untapered accretion rate for a star of unit mass,
        in $M_\odot$ / yr (default = None)
    n: float
        Exponent governing the tapering factor (default = 1)
    tau: float
        Time constant for accelerating star formation, in Myr (default = 1)
    npts: int
        Number of points at which to evaluate the PMF for interpolation
        (default = 200)
    """
    def __init__(self,imf,
                 mmin=None,mmax=None,
                 history='is',
                 j_exp=None,jf_exp=None,scale_value=None,
                 n=1,tau=1,npts=200):
        self.distr = None
        
        self._imf = imf
        self.imf.normalize()

        self._mmin = self.imf.mmin if mmin is None else mmin
        self._mmax = self.imf.mmax if mmax is None else mmax

        self.history = history

        if self.history is None:
            self._j_exp = j_exp
            self._jf_exp = jf_exp
        
        self.scale_value = scale_value

        self._n = n
        self._tau = tau
        
        self.distr = dist_pmf(self.imf,self.mmin,self.mmax,
                              self.j,self.jf,self.scale_value,
                              self.n,self.tau,npts)
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

    def integrate(self,mlow,mhigh,
                  taper=False,
                  accelerating=False,
                  **kwargs):
        def func(x):
            return self(x,taper=taper,accelerating=accelerating)

        return quad(func,mlow,mhigh,**kwargs)

    def m_integrate(self,mlow,mhigh,
                    taper=False,
                    accelerating=False,
                    **kwargs):
        def func(x):
            return self.mass_weighted(x,taper=taper,accelerating=accelerating)

        return quad(func,mlow,mhigh,**kwargs)

    def log_integrate(self,mlow,mhigh,
                      taper=False,
                      accelerating=False,
                      **kwargs):
        def logform(x):
            return self(x,taper=taper,accelerating=accelerating) / x

        return quad(logform,mlow,mhigh,**kwargs)

    #PMFs are normalized by construction if the underlying IMF is normalized
    #(which it is)
    def normalize(self):
        pass

    def weight_average(self,func,
                       taper=False,
                       accelerating=False,
                       *args,**kwargs):
        def weighted_func(x):
            return self(x,taper=taper,accelerating=accelerating) * func(x,*args)
        
        return quad(weighted_func,self.mmin,self.mmax,**kwargs)
        
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
            self._j_exp = hist_values[x][0]
            self._jf_exp = hist_values[x][1]
            self._scale_value = hist_values[x][2]

            if self.distr is not None:
                self.distr.j_exp = self.j_exp
                self.distr.jf_exp = self.jf_exp
                self.distr.scale_value = self.scale_value
                self.distr._calculate('all')
            
    @property
    def j_exp(self):
        return self._j_exp

    @j_exp.setter
    def j_exp(self,x):
        if self.history in hist_values.keys():
            raise ValueError('j_exp cannot take on alternate values for a defined history')
        self._j_exp = x
        self.distr.j_exp = x
        self.distr._calculate('all')

    @property
    def jf_exp(self):
        return self._jf_exp

    @jf_exp.setter
    def jf_exp(self,x):
        if self.history	in hist_values.keys():
            raise ValueError('jf_exp cannot take on alternate values for a defined history')
        self._jf_exp = x
        self.distr.jf_exp = x
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

class dist_pmf(Distribution):
    """
    Manages the PDF/CDF for a PMF.
    """
    def __init__(self,imf,m1,m2,
                 j_exp,jf_exp,scale_value,
                 n,tau,npts):
        self.imf = imf
        self.m1 = m1
        self.m2 = m2
        self.j_exp = j_exp
        self.jf_exp = jf_exp
        self.scale_value = scale_value
        self.n = n
        self.tau = tau

        self._points = np.geomspace(min(self.m1,1e-3),self.m2,npts)
        self._func_dict = None
        self._calculate('all')

        self._taper = False
        self._accelerating = False

    def _make_bases(self,taper,accelerating):
        """
        Construct interpolators for the PDF/CDF/PPF underlying
        a PMF for a particular combination of tapered accretion
        and accelerating star formation.
        """
        def pmf(mass,taper,accelerating):
            avg_time = self._average_time(taper,accelerating)

            def m_dot(mf,mass_):
                return self.scale_value * (mass_ / mf)**self.j_exp * mf**self.jf_exp

            def integrand(mf,mass_):
                if taper:
                    tf = self._tf(mf,taper)
                    def root_t(t,mf,mass_):
                        term1 = t * (1 - (t / tf)**self.n / (self.n + 1))
                        term2 = mass_**(1 - self.j_exp) / self.scale_value / (1 - self.j_exp) / mf**(self.jf_exp - self.j_exp)
                        prime_term1 = 1 - (t / tf)**self.n / (self.n + 1)
                        prime_term2 = self.n / (self.n + 1) * (t / tf)**self.n
                        return term1 - term2, prime_term1 - prime_term2

                    def taper_factor(mf,mass_):
                        sol = root_scalar(root_t,args=(mf,mass_),x0=0,fprime=True)
                        return 1 - (sol.root / tf)**self.n

                    t_factor = taper_factor(mf,mass_)
                    if accelerating:
                        tm = (1 - t_factor)**(1 / self.n) * tf

                else:
                    t_factor = 1
                    if accelerating:
                        tm = mass_**(1 - self.j_exp) / mf**(self.jf_exp - self.j_exp) / self.scale_value / (1 - self.j_exp)
                a_factor = np.exp(-tm / self.tau / 1e6) if accelerating else 1

                return self.imf(mf) * mass_ / m_dot(mf,mass_) / t_factor * a_factor

            def integral(lolim,mass_,**kwargs):
                return quad(integrand,lolim,self.m2,args=(mass_),**kwargs)[0]

            ret = np.vectorize(integral)(np.where(self.m1 < mass, mass, self.m1),mass)
            return np.where(ret / avg_time > 0, ret / avg_time, 0) #ensure the PMF is always >= 0

        base = pmf(self._points,taper,accelerating)
        pdf = base / self._points
        cdf = cumulative_trapezoid(pdf,self._points,initial=0)
        zero_arg = np.max(np.nonzero(np.diff(cdf))) + 1
        return (PchipInterpolator(self._points,pdf),
                PchipInterpolator(self._points,cdf),
                PchipInterpolator(cdf[:zero_arg+1],self._points[:zero_arg+1]))

    def _calculate(self,mode):
        """
        Calculate the PDF/CDF/PPF as needed. "mode" determines
        which versions are recalculated.
        """
        not_ok = (self.j_exp is None) | (self.jf_exp is None) | (self.scale_value is None)
        if not_ok:
            raise ValueError('Cannot calculate a PMF without a history or all of (j_exp, jf_exp, scale_value)')

        keys = ['pdf','cdf','ppf']
        if mode == 'all':
            func_dict = {key: [] for key in keys}
            modes = [(0,0),(1,0),(0,1),(1,1)]
            for mm in modes:
                bases = self._make_bases(*mm)
                for ii,key in enumerate(keys):
                    func_dict[key].append(bases[ii])
            self._func_dict = func_dict

        elif mode == 'taper':
            modes = [(1,0),(1,1)]
            for ii,mm in enumerate(modes):
                bases = self._make_bases(*mm)
                for jj,key in enumerate(keys):
                    self._func_dict[key][2*ii+1] = bases[jj]

        elif mode == 'accelerating':
            modes = [(0,1),(1,1)]
            for ii,mm in enumerate(modes):
                bases = self._make_bases(*mm)
                for j,key in enumerate(keys):
                    self._func_dict[key][ii+2] = bases[jj]

    def _pick_function(self,functype,taper,accelerating):
        return self._func_dict[functype][int(taper+2*accelerating)]

    def _update_functions(self):
        self._pdf = self._pick_function('pdf',self.taper,self.accelerating)
        self._cdf = self._pick_function('cdf',self.taper,self.accelerating)
        self._ppf = self._pick_function('ppf',self.taper,self.accelerating)

    def _tf(self,mf,taper):
        factor = (self.n + 1) / self.n if taper else 1
        tf1 = factor / (1 - self.j_exp) / self.scale_value
        return tf1 * mf**(1 - self.jf_exp)

    def _average_time(self,taper,accelerating):
        if accelerating:
            def accel_weight(mf,taper=False):
                return 1e6 * self.tau * (1 - np.exp(-self._tf(mf,taper=taper) / self.tau / 1e6))
            ret = self.imf.weight_average(accel_weight,taper)
        else:
            ret = self.imf.weight_average(self._tf,taper)
        return ret

    def pdf(self,x):
        return self._pdf(x,extrapolate=False)

    def cdf(self,x):
        return self._cdf(x,extrapolate=False)

    def ppf(self,x):
        return self._ppf(x,extrapolate=False)

    def rvs(self,N):
        samp = np.random.uniform(self.cdf(min(self._points)),self.cdf(self.m2),size=N)
        return self.ppf(samp)

    @property
    def taper(self):
        return self._taper

    @taper.setter
    def taper(self,x):
        self._taper = x
        self._update_functions()

    @property
    def accelerating(self):
        return self._accelerating

    @accelerating.setter
    def accelerating(self,x):
        self._accelerating = x
        self._update_functions()
        
hist_values_2C = {'tc' : (0.5, 0.75, 3.6),
                  'ca' : (2/3, 1., 3.2)}

class PMF_2C(PMF):
    r"""
    Calculates a two-component (i.e. blended accretion) PMF.

    Parameters
    ----------
    imf: MassFunction
        The IMF of the emerging stellar population
    mmin: float
        Minimum final mass for stars (default = IMF min)
    mmax: float
        Maximum final mass for stars (default = IMF max)
    history: str
        Accretion history of stars; accepts 'tc' (turbulent core),
        and 'ca' (competitive accretion). If None, custom values
        can be input for j, jf, and R_mdot (default = 'tc')
    j_exp: float
        Value setting the scaling of the non-IS accretion rate
        with current mass (default = None)
    jf_exp: float
        Value setting the scaling of the non-IS accretion rate 
        with final mass (default = None)  
    R_mdot: float
        Ratio of the characteristic accretion rate of the blended history
        and IS accretion (default = None)
    T: float
        Average gas temperature, in K. Sets the scaling for 
        IS accretion (default = 10)
    n: float
        Exponent governing the tapering factor (default = 1)
    tau: float
        Time constant for accelerating star formation, in Myr (default = 1)
    npts: int
        Number of points at which to evaluate the PMF for interpolation
        (default = 200)
    """
    def __init__(self,imf,
                 mmin=None,mmax=None,
                 history='tc',
                 j_exp=None,jf_exp=None,
                 R_mdot=None,T=10,
                 n=1,tau=1,npts=200):
        self.distr = None
        
        self._imf = imf
        self.imf.normalize()

        self._mmin = self.imf.mmin if mmin is None else mmin
        self._mmax = self.imf.mmax if mmax is None else mmax

        self.history = history

        if self.history is None:
            self._j_exp = j_exp
            self._jf_exp = jf_exp
            self._R_mdot = R_mdot

        self._n = n
        self._tau = tau

        self._T = T
        self.m_is = scaling('is',self.T)
        
        self.distr = dist_pmf_2c(self.imf,self.mmin,self.mmax,
                                 self.j_exp,self.jf_exp,
                                 self.R_mdot,self.m_is,
                                 self.n,self.tau,npts)
        self.normfactor = 1

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
            self._j_exp = hist_values_2C[x][0]
            self._jf_exp = hist_values_2C[x][1]
            self._R_mdot = hist_values_2C[x][2]

            if self.distr is not None:
                self.distr.j_exp = self.j_exp
                self.distr.jf_exp = self.jf_exp
                self.distr.R_mdot = self.R_mdot
                self.distr._calculate('all')
            
    @property
    def j_exp(self):
        return self._j_exp

    @j_exp.setter
    def j_exp(self,x):
        if self.history in hist_values_2C.keys():
            raise ValueError('j_exp cannot take on alternate values for a defined history')
        self._j_exp = x
        self.distr.j_exp = x
        self.distr._calculate('all')

    @property
    def jf_exp(self):
        return self._jf_exp

    @jf_exp.setter
    def jf_exp(self,x):
        if self.history	in hist_values_2C.keys():
            raise ValueError('jf_exp cannot take on alternate values for a defined history')
        self._jf_exp = x
        self.distr.jf_exp = x
        self.distr._calculate('all')

    @property
    def R_mdot(self):
        return self._R_mdot

    @R_mdot.setter
    def R_mdot(self,x):
        if self.history is hist_values_2C.keys():
            raise ValueError('R_mdot cannot take on alternate values for a defined history')
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

class dist_pmf_2c(dist_pmf):
    """
    Manages the PDF/CDF for two-component PMFs.
    """
    def __init__(self,imf,m1,m2,
                 j_exp,jf_exp,
                 R_mdot,m_is,
                 n,tau,npts):
        self.imf = imf
        self.m1 = m1
        self.m2 = m2
        self.j_exp = j_exp
        self.jf_exp = jf_exp
        self.R_mdot = R_mdot
        self.m_is = m_is
        self.n = n
        self.tau = tau

        self._points = np.geomspace(min(self.m1,1e-3),self.m2,npts)
        self._func_dict = None
        self._calculate('all')
        self._taper = False
        self._accelerating = False

    def _make_bases(self,taper,accelerating):
        """
        Construct interpolators for the PDF/CDF/PPF underlying
        a PMF for a particular combination of tapered accretion
        and accelerating star formation.
        """
        def pmf(mass,taper,accelerating):
            avg_time = self._average_time(taper,accelerating)

            def m_dot(mf,mass_):
                return self.m_is * np.sqrt(1 + self.R_mdot**2 *
                                           (mass_ / mf)**(2 * self.j_exp) *
                                           mf**(2 * self.jf_exp))

            def integrand(mf,mass_):
                def base_tm(mf,mass_):
                    return mass_ / self.m_is * hyp2f1(0.5,0.5/self.j_exp,1+0.5/self.j_exp,
                                                      -(self.R_mdot*(mass_/mf)**self.j_exp*mf**self.jf_exp)**2)

                if taper:
                    tf = self._tf(mf,taper)
                    def root_t(t,mf,mass_):
                        term1 = t * (1 - (t / tf)**self.n / (self.n + 1))
                        term2 = base_tm(mf,mass_)
                        prime_term1 = 1 - (t / tf)**self.n / (self.n + 1)
                        prime_term2 = self.n / (self.n + 1) * (t / tf)**self.n
                        return term1 - term2, prime_term1 - prime_term2

                    def taper_factor(mf,mass_):
                        sol = root_scalar(root_t,args=(mf,mass_),x0=0,fprime=True)
                        return 1 - (sol.root / tf)**self.n

                    t_factor = taper_factor(mf,mass_)
                    if accelerating:
                        tm = (1 - t_factor)**(1 / self.n) * tf
                else:
                    t_factor = 1
                    if accelerating:
                        tm = base_tm(mf,mass_)
                a_factor = np.exp(-tm / self.tau / 1e6) if accelerating else 1

                return self.imf(mf) * mass_ / m_dot(mf,mass_) / t_factor * a_factor

            def integral(lolim,mass_,**kwargs):
                return quad(integrand,lolim,self.m2,args=(mass_),**kwargs)[0]

            ret = np.vectorize(integral)(np.where(self.m1 < mass, mass, self.m1),mass)
            return np.where(ret / avg_time > 0, ret / avg_time, 0) #ensure the PMF is always >= 0
        
        base = pmf(self._points,taper,accelerating)
        pdf = base / self._points
        cdf = cumulative_trapezoid(pdf,self._points,initial=0)
        return (PchipInterpolator(self._points,pdf),
                PchipInterpolator(self._points,cdf),
                PchipInterpolator(cdf[np.nonzero(pdf)],self._points[np.nonzero(pdf)]))

    def _calculate(self,mode):
        """
        Calculate the PDF/CDF/PPF as needed. "mode" determines
        which versions are recalculated.
        """
        not_ok = (self.j_exp is None) | (self.jf_exp is None) | (self.R_mdot is None)
        if not_ok:
            raise ValueError('Cannot calculate a PMF without a history or all of (j_exp, jf_exp, R_mdot)')

        keys = ['pdf','cdf','ppf']
        if mode == 'all':
            func_dict = {key: [] for key in keys}
            modes = [(0,0),(1,0),(0,1),(1,1)]
            for mm in modes:
                bases = self._make_bases(*mm)
                for ii,key in enumerate(keys):
                    func_dict[key].append(bases[ii])
            self._func_dict = func_dict

        elif mode == 'taper':
            modes = [(1,0),(1,1)]
            for ii,mm in enumerate(modes):
                bases = self._make_bases(*mm)
                for jj,key in enumerate(keys):
                    self._func_dict[key][2*ii+1] = bases[jj]

        elif mode == 'accelerating':
            modes = [(0,1),(1,1)]
            for ii,mm in enumerate(modes):
                bases = self._make_bases(*mm)
                for j,key in enumerate(keys):
                    self._func_dict[key][ii+2] = bases[jj]

    def _tf(self,mf,taper):
        factor = (self.n + 1) / self.n if taper else 1
        body = mf / self.m_is * hyp2f1(0.5,0.5/self.j_exp,
                                       1+0.5/self.j_exp,
                                       -(self.R_mdot * mf**self.jf_exp)**2)
        return factor * body
