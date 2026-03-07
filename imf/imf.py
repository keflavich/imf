"""
Various codes to work with the initial mass function
"""

from __future__ import print_function
import numpy as np
import types
import scipy.integrate
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.stats import norm
from astropy import units as u
from . import distributions

import warnings


class MassFunction(object):
    """
    Generic class establishing basic operations for mass functions.
    Intended for subclassing.
    """

    def __init__(self, mmin=None, mmax=None):
        """
        Parameters
        ----------
        mmin: float or None
            Minimum stellar mass
        mmax: float or None
            Maximum stellar mass
        """
        self._mmin = self.default_mmin if mmin is None else mmin
        self._mmax = self.default_mmax if mmax is None else mmax

    def dndm(self, m, **kwargs):
        """
        The differential form of the mass function, d N(M) / dM
        """
        return self(m, integral_form=False, **kwargs)

    def n_of_m(self, m, **kwargs):
        """
        The integral form of the mass function, N(M)
        """
        return self(m, integral_form=True, **kwargs)

    def mass_weighted(self, m, **kwargs):
        return self(m, integral_form=False, **kwargs) * m

    def integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass function over some range
        """
        return scipy.integrate.quad(self, mlow, mhigh)

    def m_integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass-weighted mass function over some range (this tells
        you the fraction of mass in the specified range)
        """
        return scipy.integrate.quad(self.mass_weighted, mlow, mhigh, **kwargs)

    def log_integrate(self, mlow, mhigh, **kwargs):
        def logform(x):
            return self(x) / x

        return scipy.integrate.quad(logform, mlow, mhigh, **kwargs)

    def normalize(self, mmin=None, mmax=None, log=False, **kwargs):
        """
        Set self.normfactor such that the integral of the function over the
        range (mmin, mmax) = 1
        """
        if mmin is None:
            mmin = self.mmin
        if mmax is None:
            mmax = self.mmax

        self.normfactor = 1

        if log:
            integral = self.log_integrate(mmin, mmax, **kwargs)
        else:
            integral = self.integrate(mmin, mmax, **kwargs)
        self.normfactor = 1. / integral[0]

        assert self.normfactor > 0

    def weight_average(self, func, *args, **kwargs):
        """
        Integrate a function of stellar mass f(m) over the IMF
        """
        def weighted_func(x):
            return self(x) * func(x, *args)

        return scipy.integrate.quad(weighted_func, self.mmin, self.mmax, **kwargs)[0]

    @property
    def mmin(self):
        return self._mmin

    @mmin.setter
    def mmin(self, x):
        self._mmin = x

    @property
    def mmax(self):
        return self._mmax

    @mmax.setter
    def mmax(self, x):
        self._mmax = x


class Salpeter(MassFunction):
    default_mmin = 0.3
    default_mmax = 120

    def __init__(self, alpha=2.35, mmin=default_mmin, mmax=default_mmax):
        """
        Create a default Salpeter mass function, i.e. a power-law mass function
        the Salpeter 1955 IMF: dn/dm ~ m^-2.35

        Parameters
        ----------
        alpha: float
            The exponent of the power law (default = 2.35)
        """
        super().__init__(mmin=mmin, mmax=mmax)

        self.alpha = alpha
        self.normfactor = 1
        self.distr = distributions.PowerLaw(-self.alpha, self.mmin, self.mmax)

    def __call__(self, m, integral_form=False):
        if not integral_form:
            return self.distr.pdf(m) * self.normfactor
        else:
            return self.distr.cdf(m) * self.normfactor


class BrokenPowerLaw(MassFunction):
    # kroupa = BrokenPowerLaw(breaks={0.08: -0.3, 0.5: 1.3, 'last': 2.3}, mmin=0.03, mmax=120)
    default_mmin = 0.03
    default_mmax = 120

    def __init__(self,
                 mmin=default_mmin,
                 mmax=default_mmax,
                 powers=[0.3, 1.3, 2.3],
                 breaks=[0.08, 0.5],
                 ):
        """
        Powers should be positive values for decreasing slopes
        """
        super().__init__(mmin=mmin, mmax=mmax)

        self.powers = list(powers)
        self.breaks = list(breaks)
        self.distr = distributions.BrokenPowerLaw([-x for x in self.powers],
                                                  [self.mmin, *self.breaks, self.mmax])
        self.normfactor = 1

    def __call__(self, m, integral_form=False):
        """

        Parameters
        ----------
        m: float array
            The mass at which to evaluate the function (Msun)
        p1, p2, p3: floats
            The power-law slopes of the different segments of the IMF
        break1, break2: floats
            The mass breakpoints at which to use the different power laws
        """

        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def integrate(self, mlow, mhigh, numerical=False):
        """
        Integrate the mass function over some range
        """
        if mhigh < mlow:
            raise ValueError("Must have mlow <= mhigh in integral")
        if numerical:
            return super().integrate(mlow, mhigh)

        return (self.distr.cdf(mhigh) -
                self.distr.cdf(mlow)) * self.normfactor, 0

    def m_integrate(self, mlow, mhigh, numerical=False, **kwargs):
        """
        Integrate the mass function over some range
        """
        if mhigh < mlow:
            raise ValueError("Must have mlow <= mhigh in integral")

        if numerical:
            return super(Kroupa, self).m_integrate(mlow, mhigh, **kwargs)
        else:
            distr1 = distributions.BrokenPowerLaw(
                [-x + 1 for x in self.powers],
                [self.mmin, *self.breaks, self.mmax])
            ratio = distr1.pdf(self.break1) / self.distr.pdf(
                self.break1) / self.break1
            return ((distr1.cdf(mhigh) - distr1.cdf(mlow)) / ratio, 0)


class Kroupa(BrokenPowerLaw):
    # kroupa = BrokenPowerLaw(breaks={0.08: -0.3, 0.5: 1.3, 'last': 2.3}, mmin=0.03, mmax=120)
    default_mmin = 0.03
    default_mmax = 120

    def __init__(self,
                 mmin=default_mmin,
                 mmax=default_mmax,
                 p1=0.3,
                 p2=1.3,
                 p3=2.3,
                 break1=0.08,
                 break2=0.5,
                 powers=None,
                 breaks=None
                 ):
        """
        The Kroupa IMF with two power-law breaks, p1 and p2. See __call__ for
        details.
        """

        if powers is not None:
            p1, p2, p3 = powers
        else:
            powers = [p1, p2, p3]
        if breaks is not None:
            break1, break2 = breaks
        else:
            breaks = [break1, break2]

        super().__init__(mmin=mmin, mmax=mmax, powers=powers, breaks=breaks)

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.break1 = break1
        self.break2 = break2

    def __call__(self, m, integral_form=False):
        """
        Kroupa 2001 IMF (http://arxiv.org/abs/astro-ph/0009005,
        http://adsabs.harvard.edu/abs/2001MNRAS.322..231K) eqn 2

        Parameters
        ----------
        m: float array
            The mass at which to evaluate the function (Msun)
        p1, p2, p3: floats
            The power-law slopes of the different segments of the IMF
        break1, break2: floats
            The mass breakpoints at which to use the different power laws
        """
        return super().__call__(m, integral_form)


class Kirkpatrick2024(BrokenPowerLaw):
    """
    https://ui.adsabs.harvard.edu/abs/2024ApJS..271...55K/abstract
    """
    default_mmin = 0.03
    default_mmax = 120

    def __init__(self,
                 mmin=default_mmin,
                 mmax=default_mmax,
                 powers=[0.6, 0.25, 1.3, 2.3],
                 breaks=[0.05, 0.22, 0.55],
                 ):
        super().__init__(mmin=mmin, mmax=mmax, powers=powers, breaks=breaks)


class ChabrierLogNormal(MassFunction):
    """
    Eqn 18 of https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract
    is eqn3 of https://ui.adsabs.harvard.edu/abs/2003ApJ...586L.133C/abstract

    \\xi = 0.086 exp (-(log m - log 0.22)^2 / (2 * 0.57**2)) 

    This function is a pure lognormal; see ChabrierPowerLaw for the version
    with a power-law extension to high mass

    Parameters
    ----------
    lognormal_center : float
    lognormal_width : float
    mmin : float
    mmax : float
    leading_constant : float
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 lognormal_center=0.22,
                 lognormal_width=0.57*np.log(10),
                 leading_constant=0.086):
        super().__init__(mmin=mmin, mmax=mmax)

        self.multiplier = leading_constant
        self.lognormal_width = lognormal_width
        self.normfactor = 1
        self.distr = distributions.TruncatedLogNormal(mu=lognormal_center,
                                                      sig=self.lognormal_width,
                                                      m1=self.mmin,
                                                      m2=self.mmax)

    def __call__(self, mass, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(mass) * self.multiplier * self.normfactor
        else:
            return self.distr.pdf(mass) * self.multiplier * self.normfactor


class ChabrierPowerLaw(MassFunction):
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self,
                 lognormal_center=0.22,
                 lognormal_width=0.57*np.log(10),
                 mmin=default_mmin,
                 mmax=default_mmax,
                 alpha=2.3,
                 mmid=1):
        """
        From Equation 18 of Chabrier 2003
        https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C/abstract

        Parameters
        ----------
        lognormal_center : float
        lognormal_width : float
            The lognormal width.  Scipy.stats.lognorm uses log_n,
            so we need to scale this b/c Chabrier uses log_10
        mmin : float
        mmax : float
        alpha : float
            The high-mass power-law slope
        mmid : float
            The mass to transition from lognormal to power-law


        Notes
        -----
        A previous version of this function used sigma=0.55,
        center=0.2, and alpha=2.35, which come from McKee & Offner 2010 
        (https://ui.adsabs.harvard.edu/abs/2010ApJ...716..167M/abstract)
        but those exact numbers don't appear in Chabrier 2005
        """
        # The numbers are from Eqn 3 of
        # https://ui.adsabs.harvard.edu/abs/2005ASSL..327...41C/abstract
        # There is no equation 3 in that paper, though?
        # importantly the lognormal center is the exp(M) where M is the mean of ln(mass)
        # normal distribution
        super().__init__(mmin=mmin, mmax=mmax)
        self._mmid = mmid
        if self.mmax <= self._mmid:
            raise ValueError("The Chabrier Mass Function does not support "
                             "mmax <= mmid")
        self._alpha = alpha
        self._lognormal_width = lognormal_width
        self._lognormal_center = lognormal_center
        self.normfactor = 1
        self.distr = distributions.CompositeDistribution([
            distributions.TruncatedLogNormal(self._lognormal_center,
                                             self._lognormal_width,
                                             self.mmin,
                                             self._mmid),
            distributions.PowerLaw(-self._alpha, self._mmid, self.mmax)
        ])

    def __call__(self, x, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(x) * self.normfactor
        else:
            return self.distr.pdf(x) * self.normfactor

class PadoanTF(MassFunction):
    """
    IMF implementing the form derived in Padoan & Nordlund (2002)
    emerging from turbulent fragmentation theory.
    """
    default_mmin = 0.01
    default_mmax = 200

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 b=1.8, T0=10, n0=5e2,
                 sigma=None, mach=10, npts=None):
        """        
        Parameters
        ----------
        b: float
            Spectral index of the turbulence power spectrum (default = 1.8)
        T0: float
            Average gas temperature in K (default = 10)
        n0: float
            Average gas number density in cm^-3 (default = 5e2)
        sigma: float or None
            Standard deviation of the log of gas density
        mach: float
            Mach number of the turbulent flow. Used to calculate sigma 
            if sigma is None
        npts: int
            Number of points at which to evaluate the function for
            interpolation (default = 200)
        """
        if ~np.logical_and(np.isfinite(mmin), np.isfinite(mmax)):
            raise ValueError("PN IMF uses interpolation; mmin and mmax must be finite")
        if sigma is None and mach is None:
            raise ValueError('PN IMF requires either stdev of density distribution (sigma) or rms Mach number (mach)')
        init_sigma = np.sqrt(np.log(1 + (mach / 2)**2)) if sigma is None else sigma
        self._mach = 2 * np.sqrt(np.exp(sigma**2) - 1) if mach is None else mach

        self.distr = distributions.PadoanTF(mmin, mmax,
                                            b, T0, n0, init_sigma,
                                            npts=npts)
        self.normfactor = 1

    def __call__(self, m, integral_form=False):
        if integral_form:
            return self.distr.cdf(m)
        else:
            return self.distr.pdf(m)

    @property
    def mmin(self):
        return self.distr.m1

    @mmin.setter
    def mmin(self, x):
        self.distr.m1 = x
        self.distr._calculate()

    @property
    def mmax(self):
        return self.distr.m2

    @mmax.setter
    def mmax(self, x):
        self.distr.m2 = x
        self.distr._calculate()

    @property
    def b(self):
        return self.distr.b

    @b.setter
    def b(self, x):
        self.distr.b = x
        self.distr._calculate()

    @property
    def T0(self):
        return self.distr.T0

    @T0.setter
    def T0(self, x):
        self.distr.T0 = x
        self.distr._calculate()

    @property
    def n0(self):
        return self.distr.n0

    @n0.setter
    def n0(self, x):
        self.distr.n0 = x
        self.distr._calculate()

    @property
    def sigma(self):
        return self.distr.sigma

    def set_sigma(self, x, update_mach=True):
        self.distr.sigma = x
        self.distr._calculate()
        if update_mach:
            self._mach = 2 * np.sqrt(np.exp(x**2) - 1)

    @property
    def mach(self):
        return self._mach

    def set_mach(self, x, update_sigma=True):
        self._mach = x
        if update_sigma:
            self.sigma = np.sqrt(np.log(1 + (self._mach / 2)**2))
            self.distr._calculate()


class Schechter(MassFunction):
    default_mmin = 0.03
    default_mmax = 200

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 alpha=2.35, m0=100, npts=None):
        """
        Create a Schechter-like mass function; a power law
        with a high-mass exponential cutoff.

        Parameters
        ----------
        alpha: float
            Power law slope (default = -2.35)
        m0: float
            Characteristic mass for exponential decay (default = 100)
        npts: int
            Number of points to use for interpolation (default = 200)
        """
        super().__init__(mmin=mmin, mmax=mmax)
        if ~np.logical_and(np.isfinite(mmin), np.isfinite(self.mmax)):
            warnings.warn('function uses interpolation; non-finite mass bounds prevent random sampling')
        self.alpha = alpha
        self.m0 = m0

        self.distr = distributions.CutoffPowerLaw(-self.alpha, self.mmin, self.mmax, self.m0,
                                                  npts=npts)
        self.normalize()

    def __call__(self, mass, integral_form=False):
        if integral_form:
            return self.normfactor * self.distr.cdf(mass)
        else:
            return self.normfactor * self.distr.pdf(mass)


class ModifiedSchechter(Schechter):
    default_mmin = 0.03
    default_mmax = 200

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 alpha=2.35, ml=0.5, mu=100, npts=None):
        """
        A Schechter-like mass function with an additional
        low-lever exponential cutoff.

        Parameters
        ----------
        alpha: float
            Power law slope (default = -2.35)
        ml: float
            Characteristic mass for the low-level cutoff
            (default = 0.5)
        mu: float
            Characteristic mass for the high-level cutoff
            (default = 100)
        npts: int
            Number of points to use for interpolation (default = 200)
        """
        super().__init__(mmin=mmin, mmax=mmax)
        self.alpha = alpha
        self.ml = ml
        self.mu = mu

        self.distr = distributions.ModifiedCutoffPowerLaw(-self.alpha,
                                                          self.mmin, self.mmax,
                                                          self.ml, self.mu,
                                                          npts=npts)
        self.normalize()

    def __call__(self, mass, integral_form=False):
        if integral_form:
            return self.normfactor * self.distr.cdf(mass)
        else:
            return self.normfactor * self.distr.pdf(mass)


# these are global objects
salpeter = Salpeter()
kroupa = Kroupa()
lognormal = chabrierlognormal = ChabrierLogNormal()
chabrier = chabrierpowerlaw = ChabrierPowerLaw()
chabrier2005 = ChabrierPowerLaw(lognormal_width=0.55*np.log(10),
                                lognormal_center=0.2, alpha=2.35)

massfunctions = {'kroupa': Kroupa, 'salpeter': Salpeter,
                 'chabrierlognormal': ChabrierLogNormal,
                 'chabrierpowerlaw': ChabrierPowerLaw,
                 'chabrier': ChabrierPowerLaw,
                 }
reverse_mf_dict = {v: k for k, v in massfunctions.items()}
expectedmass_cache = {}


def get_massfunc(massfunc, mmin=None, mmax=None, **kwargs):
    if isinstance(massfunc, MassFunction):
        if mmax is not None and massfunc.mmax != mmax:
            raise ValueError("mmax was specified, but a massfunction instance"
                             " was specified with a different mmax")
        if mmin is not None and massfunc.mmin != mmin:
            raise ValueError("mmin was specified, but a massfunction instance"
                             " was specified with a different mmin")
        return massfunc
    elif massfunc in massfunctions.values():
        # if the massfunction is a known MassFunc class
        return massfunc(mmin=mmin, mmax=mmax, **kwargs)
    elif massfunc in massfunctions:
        # if the massfunction is the _name_ of a massfunc class
        return massfunctions[massfunc](mmin=mmin, mmax=mmax, **kwargs)
    else:
        raise ValueError("massfunc must either be a string in the set %s or a MassFunction instance"
                         % (", ".join(massfunctions.keys())))


def get_massfunc_name(massfunc):
    if massfunc in reverse_mf_dict:
        return reverse_mf_dict[massfunc]
    elif type(massfunc) is str:
        return massfunc
    elif hasattr(massfunc, '__name__'):
        return massfunc.__name__
    else:
        raise ValueError("invalid mass function")


def m_integrate(fn=kroupa, bins=np.logspace(-2, 2, 500)):
    xax = (bins[:-1] + bins[1:]) / 2.
    integral = xax * (bins[1:] - bins[:-1]) * (fn(bins[:-1]) +
                                               fn(bins[1:])) / 2.

    return xax, integral


def cumint(fn=kroupa, bins=np.logspace(-2, 2, 500)):
    xax, integral = integrate(fn, bins)
    return integral.cumsum() / integral.sum()


def m_cumint(fn=kroupa, bins=np.logspace(-2, 2, 500)):
    xax, integral = m_integrate(fn, bins)
    return integral.cumsum() / integral.sum()


def inverse_imf(p,
                mmin=None,
                mmax=None,
                massfunc='kroupa',
                **kwargs):
    """
    Inverse mass function.  Given a likelihood value in the range [0, 1),
    return the appropriate mass.  This just calls the mass function's ppdf
under the hood.


    Parameters
    ----------
    p: np.array
        An array of floats in the range [0, 1).  These should be uniformly random
        numbers.
    mmin: float
    mmax: float
        Minimum and maximum stellar mass in the distribution
    massfunc: string or function
        massfunc can be 'kroupa', 'chabrier', 'salpeter', 'schechter', or a
        function
    """

    mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax)

    # this should be the entirety of "inverse-imf".  The rest is a hack
    if hasattr(mfc, 'distr'):
        return mfc.distr.ppf(p)
    else:
        raise NotImplementedError

#### This section contains the functions required to optimally sample a cluster####


def _prefactor(max_star, massfunc):
    """
    Returns the multiplier required for an IMF to have at most one star above m_max.
    """
    return 1 / massfunc.integrate(max_star, massfunc.mmax)[0]


def _M_cluster(m, massfunc):
    """
    Returns the mass of a cluster distributed according to some IMF where the 
    largest star has mass m.
    """
    k = _prefactor(m, massfunc)
    return k * massfunc.m_integrate(massfunc.mmin, m)[0] + m


def _max_star(m, M_res, massfunc):
    """
    Returns the most massive star capable of forming in a cluster of mass M_res
    according to the m_max/M_cluster relation. Formatted for use with root finding.
    """
    return M_res - _M_cluster(m, massfunc)


def _max_star_prime(m, M_res, massfunc):
    """
    Returns the derivative of _max_star at mass m. Used for Newton's method in
    the case of an infinite upper bound on the provided mass function.
    """
    term1 = _prefactor(m, massfunc)**2 * massfunc(m) * massfunc.m_integrate(massfunc.mmin, m)[0]
    term2 = m * massfunc(m) * _prefactor(m, massfunc)
    return -term1 - term2 - 1


def _get_next_m(m, last_m, k, massfunc):
    """
    Returns the next smallest star in an optimally sampled cluster given the 
    previous star and overall IMF. Formatted for use with root finding.
    """
    return k*massfunc.m_integrate(m, last_m)[0]-m


def _opt_sample(M_res, massfunc, tolerance):
    """
    Returns a numpy array containing stellar masses that optimally sample 
    from a provided MassFunction to make a cluster with mass M_res.
    """
    # retrieve mass bounds from provided massfunc
    mmin = massfunc.mmin
    mmax = massfunc.mmax
    finMax = np.isfinite(mmax)

    # finding all the component stars requires a cutoff--ensure there is one
    if not np.logical_or(np.isfinite(np.log(mmin)), np.isfinite(np.log(tolerance))):
        raise ValueError('Optimal sampling requires either mmin or tolerance to be finite and greater than zero.')

    if finMax:
        # bracket from min to ALMOST max (max gives an undefined prefactor)
        sol = root_scalar(_max_star, args=(M_res, massfunc), bracket=[mmin, 0.9999*mmax])
    else:
        # use Newton's method
        sol = root_scalar(_max_star, args=(M_res, massfunc), x0=10*mmin, fprime=_max_star_prime)
    k = _prefactor(sol.root, massfunc)
    M_tot = sol.root
    star_masses = [sol.root]
    m_i = sol.root

    while np.abs(M_res-M_tot) > np.maximum(mmin, tolerance):
        try:
            m_i_plus = root_scalar(lambda x: k * massfunc.integrate(x, m_i)[0]-1,
                                   bracket=[mmin, m_i]).root
        except(ValueError):
            print(f'Reached provided lower mass bound; stopping')
            break
        m = k * massfunc.m_integrate(m_i_plus, m_i)[0]
        star_masses.append(m)
        M_tot += m
        m_i = m_i_plus

    return np.array(star_masses), M_tot

##############################################################################


def make_cluster(mcluster,
                 massfunc='kroupa',
                 tolerance=0.0,
                 sampling='random',
                 stop_criterion='nearest',
                 mmin=None,
                 mmax=None,
                 verbose=False,
                 silent=False,
                 **kwargs):
    """
    Sample from an IMF to make a cluster.  Returns the masses of all stars in the cluster

    Parameters
    ----------
    mcluster : float
        The target cluster mass.
    massfunc : string or MassFunction
        A mass function to use.
    tolerance : float
        tolerance is how close the cluster mass must be to the requested mass.
        It can be zero, but this does not guarantee that the final cluster mass will be
        exactly `mcluster`
    sampling: 'random' or 'optimal'
        Optimal sampling is implemented by solving Equations 9-11 in Section 2.2
        of https://ui.adsabs.harvard.edu/abs/2013pss5.book..115K/abstract.
        Optimal sampling is only to be used in the context of a variable M_max
        that is a function of the cluster mass, e.g., eqn 24 of Schulz+ 2015.
    stop_criterion : 'nearest', 'before', 'after', 'sorted'
        The criterion to stop random sampling when the total cluster mass is reached.
        See, e.g., Krumholz et al 2015: https://ui.adsabs.harvard.edu/abs/2015MNRAS.452.1447K/abstract.
        Does not factor into optimal sampling.        
    """
    # use most common mass to guess needed number of samples
    # nsamp = mcluster / mostcommonmass[get_massfunc_name(massfunc)]
    # masses = inverse_imf(np.random.random(int(nsamp)), massfunc=massfunc, **kwargs)

    # mtot = masses.sum()
    # if verbose:
    #    print(("%i samples yielded a cluster mass of %g (%g requested)" %
    #          (nsamp, mtot, mcluster)))

    # catch wrong keywords early
    ok_samplings = ['random', 'optimal']
    ok_criteria = ['nearest', 'before', 'after', 'sorted']
    if not sampling in ok_samplings:
        raise ValueError("Sampling should be either 'random' or 'optimal' (see documentation)")
    if (sampling == 'random') and not stop_criterion in ok_criteria:
        raise ValueError("Stop criterion for random sampling should be 'nearest', 'before', 'after', or 'sorted' (see documentation)")

    if sampling == 'optimal':
        mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax, **kwargs)
        masses, mtot = _opt_sample(mcluster, mfc, tolerance=tolerance)
        if verbose:
            print(f'Sampled {len(masses)} new stars.')
        if not silent:
            print(f'Total cluster mass is {np.round(mtot, 3)} (limit was {int(mcluster)})')

    else:
        mcluster = u.Quantity(mcluster, u.M_sun).value

        mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax, **kwargs)

        if (massfunc, mfc.mmin, mfc.mmax) in expectedmass_cache:
            expected_mass = expectedmass_cache[(massfunc, mfc.mmin, mfc.mmax)]
            assert expected_mass > 0
        else:
            expected_mass = mfc.m_integrate(mfc.mmin, mfc.mmax)[0]
            assert expected_mass > 0
            expectedmass_cache[(massfunc, mfc.mmin, mfc.mmax)] = expected_mass

        if verbose:
            print("Expected mass is {0:0.3f}".format(expected_mass))

        mtot = 0
        masses = []

        while mtot < mcluster + tolerance:
            # at least 1 sample, but potentially many more
            nsamp = int(np.ceil((mcluster + tolerance - mtot) / expected_mass))
            assert nsamp > 0
            newmasses = mfc.distr.rvs(nsamp)
            masses = np.concatenate([masses, newmasses])
            mtot = masses.sum()
            if verbose:
                print("Sampled %i new stars.  Total is now %g" %
                      (int(nsamp), mtot))

            if mtot >= mcluster + tolerance:  # don't force exact equality; that would yield infinite loop
                mcum = masses.cumsum()
                if stop_criterion == 'sorted':
                    masses = np.sort(masses)
                    if np.abs(masses[:-1].sum() - mcluster) < np.abs(masses.sum() -
                                                                     mcluster):
                        # if the most massive star makes the cluster a worse fit, reject it
                        # (this follows Krumholz+ 2015 appendix A1)
                        last_ind = len(masses) - 1
                    else:
                        last_ind = len(masses)
                else:
                    if stop_criterion == 'nearest':
                        # find the closest one, and use +1 to include it
                        last_ind = np.argmin(np.abs(mcum - mcluster)) + 1
                    elif stop_criterion == 'before':
                        last_ind = np.argmax(mcum > mcluster)
                    elif stop_criterion == 'after':
                        last_ind = np.argmax(mcum > mcluster) + 1
                masses = masses[:last_ind]
                mtot = masses.sum()
                if verbose:
                    print(
                        "Selected the first %i out of %i masses to get %g total" %
                        (last_ind, len(mcum), mtot))
                # force the break, because some stopping criteria can push mtot < mcluster
                break

        if not silent:
            print("Total cluster mass is %g (limit was %g)" % (mtot, mcluster))

    return masses


class KoenConvolvedPowerLaw(MassFunction):
    """
    Implementaton of error-convolved power-law described in the 2009 Koen/Kondlo
    paper, "Fitting power-law distributions to data with measurement errors."
    When instantiated, the error convolutions (equations (3) and (5) from KK09)
    are performed for a fixed set of points, and calls to the function interpolate
    between these values. This implementation is preferred for those looking to work 
    extensively with a single mass function, including using it to create clusters.

    Parameters
    ----------
    mmin, mmax: floats
        The upper and lower bounds for the power law distribution.
    gamma: float
        The specified gamma for the distribution. Slope = -gamma - 1.
    sigma: float
        Specified spread of error. Assumes normal distribution with mean 0 and variance sigma.
    npts: int
        Number of points at which to evaluate the function for interpolation
        (default = 200)
    quad_sub_limit: int
        Limit of the number of subdivisions allowed for scipy.integrate.quad,
        which handles integration (default = 50)
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin, mmax, gamma, sigma, npts=200, quad_sub_limit=50):
        if mmax < mmin:
            raise ValueError("mmax must be greater than mmin")
        if not np.all(np.isfinite(np.log([mmin, mmax]))):
            raise ValueError('KoenConvolvedPowerLaw requires finite, positive mass bounds')

        super().__init__(mmin, mmax)
        self._gamma = gamma
        self._sigma = sigma
        self._quad_sub_limit = quad_sub_limit
        self.distr = distributions.KoenConvolvedPowerLaw(self.mmin, self.mmax,
                                                         self.gamma, self.sigma, npts)
        self.normfactor = 1. / self.distr.cdf(self.mmax)

    def __call__(self, m, integral_form=False):
        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass function over some range
        """
        if 'limit' not in kwargs.keys():
            return scipy.integrate.quad(self, mlow, mhigh,
                                        limit=self._quad_sub_limit, **kwargs)
        else:
            return scipy.integrate.quad(self, mlow, mhigh, **kwargs)

    def m_integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass-weighted mass function over some range (this 
        tells you the fraction of mass in the specified range)
        """
        if 'limit' not in kwargs.keys():
            return scipy.integrate.quad(self.mass_weighted, mlow, mhigh,
                                        limit=self._quad_sub_limit, **kwargs)
        else:
            return scipy.integrate.quad(self.mass_weighted,
                                        mlow, mhigh, **kwargs)

    @property
    def gamma(self):
        return self._gamma

    @property
    def sigma(self):
        return self._sigma

    @property
    def quad_sub_limit(self):
        return self._quad_sub_limit

    @quad_sub_limit.setter
    def quad_sub_limit(self, x):
        self._quad_sub_limit = x


class SpotKoenConvolvedPowerLaw(MassFunction):
    """
    Implementation of Koen/Kondlo 2009 error-convolved powerlaw,
    but evaluation is done on the spot in contrast to KoenConvolvedPowerLaw,
    which evaluates at a series of points beforehand and then interpolates 
    when called. This implementation is good for those looking for 
    improved accuracy or wanting to work with multiple mass functions.
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin, mmax, gamma, sigma):
        if mmax < mmin:
            raise ValueError("mmax must be greater than mmin")
        if not np.all(np.isfinite(np.log([mmin, mmax]))):
            raise ValueError('KoenConvolvedPowerLaw requires finite, positive mass bounds')

        super().__init__(mmin, mmax)
        self.sigma = sigma
        self.gamma = gamma
        self.normfactor = 1 / self._integrate(self.mmax, integral_form=True)

    def _coef(self, integral_form):
        if integral_form:
            return (1 / (self.sigma * np.sqrt(2 * np.pi) *
                         (self.mmin**-self.gamma - self.mmax**-self.gamma)))
        else:
            return (self.gamma / ((self.sigma * np.sqrt(2 * np.pi)) *
                                  ((self.mmin**-self.gamma) -
                                   (self.mmax**-self.gamma))))

    def _integrand(self, x, y, integral_form):
        if integral_form:
            return ((self.mmin**-self.gamma - x**-self.gamma) * np.exp(
                (-1 / 2) * ((y - x) / self.sigma)**2))
        else:
            return (x**-(self.gamma + 1)) * np.exp(-.5 * (
                (y - x) / self.sigma)**2)

    def _mirror_steps(self):
        # Sub-intervals for the integration to capture small changes at both ends
        x = np.geomspace(self.mmin, self.mmax, 100)
        mir_x = self.mmax-(x[::-1]-self.mmin)
        dx = x[1:]-x[:-1]
        cutoff = min(self.sigma, 1)
        break1 = np.searchsorted(dx, cutoff)
        break2 = np.searchsorted(-dx[::-1], -cutoff)
        xpt = x[break1]
        mirxpt = mir_x[break2]
        x1, x2 = min(xpt, mirxpt), max(xpt, mirxpt)
        x = np.append(x[x < x1], np.linspace(x1, x2,
                                            int((x2-x1)/cutoff)))
        x = np.append(x, mir_x[mir_x > x2])
        return x

    def _integrate(self, y, integral_form):
        steps = self._mirror_steps()
        chunks = []
        for i in range(len(steps)-1):
            l, u = steps[i], steps[i+1]
            area = quad(self._integrand, l, u, args=(y, integral_form))[0]
            chunks.append(area)
        if integral_form:
            ret = self._coef(integral_form) * np.sum(chunks) + norm.cdf(
                (y - self.mmax) / self.sigma)
        else:
            ret = self._coef(integral_form)*np.sum(chunks)
        return ret

    def __call__(self, m, integral_form=False):
        vector_int = np.vectorize(self._integrate)
        m = np.asarray(m)
        return self.normfactor * vector_int(m, integral_form)


class KoenTruePowerLaw(MassFunction):
    """
    Implementaton of error free power-law described in 2009 Koen Kondlo paper,
    Fitting power-law distributions to data with measurement errors

    This is a power law with truncations on the low and high end.

    Equations (2) and (4)

    Parameters
    ----------
    m: float
        The mass at which to evaluate the function
    mmin, mmax: floats
        The upper and lower bounds for the power law distribution
    gamma: floats
        The specified gamma for the distribution, related to the slope, alpha = -gamma + 1
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin, mmax, gamma):
        super().__init__(mmin, mmax)
        self.gamma = gamma

    def __call__(self, m, integral_form=False):
        m = np.asarray(m)
        if self.mmax < self.mmin:
            raise ValueError('mmax must be greater than mmin')
        if integral_form:
            # Returns
            # -------
            # Probability that m < x for the given CDF with specified mmin, mmax, sigma, and gamma
            # True for L<=x
            pdf = ((self.mmin**-self.gamma - np.power(m, -self.gamma)) /
                   self.mmin**-self.gamma - self.mmax**-self.gamma)
            return_value = (pdf * ((m > self.mmin) & (m < self.mmax)) + 1.0 *
                            (m >= self.mmax) + 0 * (m < self.mmin))
            return return_value

        else:
            # Returns
            # ------
            # Probability of getting x given the PDF with specified mmin, mmax, and gamma
            # Answers it gives are true from mmin<=x<=mmax
            cdf = (self.gamma * np.power(m, -(self.gamma + 1)) /
                   (self.mmin**-self.gamma - self.mmax**-self.gamma))
            return_value = (cdf * ((m > self.mmin) & (m < self.mmax)) + 0 *
                            (m > self.mmax) + 0 * (m < self.mmin))
            return return_value
