from __future__ import print_function
import numpy as np
import types
import scipy.integrate
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.stats import norm
from astropy import units as u

from . import distributions

import warnings


class MassFunction(object):
    """
    Generic class establishing basic operations for mass functions.
    Intended for subclassing.

    Parameters
    ----------
    mmin: float or None
        Minimum stellar mass
    mmax: float or None
        Maximum stellar mass
    """

    def __init__(self, mmin=None, mmax=None):
        self._mmin = self.default_mmin if mmin is None else mmin
        self._mmax = self.default_mmax if mmax is None else mmax

    def dn_dm(self, m, **kwargs):
        """
        The differential form of the mass function, dN/dM.
        Identical to ``MassFunction.__call__``.
        """
        return self(m, integral_form=False, **kwargs)

    def n_of_m(self, m, **kwargs):
        """
        The integral form of the mass function, N(M). Identical to
        ``MassFunction.__call__(integral_form=True)``.
        """
        return self(m, integral_form=True, **kwargs)

    def mass_weighted(self, m, **kwargs):
        """
        The log form of the mass function, dN/d log M.
        """
        return self(m, integral_form=False, **kwargs) * m

    def integrate(self, mlow, mhigh, **kwargs):
        """
        Integrates the mass function over the range [mlow, mhigh].

        Returns
        -------
        result: float
            Result of integration
        error: float
            Error from integration method
        """
        return scipy.integrate.quad(self, mlow, mhigh)

    def m_integrate(self, mlow, mhigh, **kwargs):
        """
        Integrates the mass-weighted mass function over the range
        [mlow, mhigh].

        Returns
        -------
        result: float
            Result of integration
        error: float
            Error from integration method
        """
        return scipy.integrate.quad(self.mass_weighted, mlow, mhigh, **kwargs)

    def log_integrate(self, mlow, mhigh, **kwargs):
        """
        Integrates the mass function in log space (i.e. with
        respect to :math:`dm/m`).

        Returns
        -------
        result: float
            Result of integration
        error: float
            Error from integration method
        """
        def logform(x):
            return self(x) / x

        return scipy.integrate.quad(logform, mlow, mhigh, **kwargs)

    def normalize(self, mmin=None, mmax=None, log=False, **kwargs):
        """
        Normalizes the integral of the function over the range 
        [mmin, mmax]. If ``log`` is ``True``, use ``log_integrate``
        instead of ``integrate``.
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
        Integrates some function dependent on stellar mass :math:`f(m)`
        over a mass function.

        Returns
        -------
        result: float
            Result of integration
        error: float
            Error from integration method
        """
        def weighted_func(x):
            return self(x) * func(x, *args)

        num = scipy.integrate.quad(weighted_func, self.mmin, self.mmax, **kwargs)[0]
        den = self.integrate(self.mmin, self.mmax)[0]

        return num / den

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
    """
    The `Salpeter (1955) <https://doi.org/10.1086/145971>`__ mass 
    function, i.e. a power-law  with :math:`dn/dm \\propto m^{-\\alpha}`.
    Default mass range is [0.3, 120] :math:`M_\\odot`.

    Parameters
    ----------
    alpha: float
        Power law exponent (default = 2.35)
    """
    default_mmin = 0.3
    default_mmax = 120

    def __init__(self, alpha=2.35, mmin=default_mmin, mmax=default_mmax):
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
    """
    A generic broken power law mass function. Powers should
    be positive values for decreasing slopes. Break points are
    specifically transitions between segments; mmin and mmax
    (i.e. the edges) are set separately. Default mass range is 
    [0.03, 120] :math:`M_\\odot`.

    Parameters
    ----------
    powers: list/array
        The power law exponents of the different segments of the IMF
    breaks: list/array
        The break points between power laws
    """
    default_mmin = 0.03
    default_mmax = 120

    def __init__(self,
                 mmin=default_mmin,
                 mmax=default_mmax,
                 powers=[0.3, 1.3, 2.3],
                 breaks=[0.08, 0.5],
                 ):
        super().__init__(mmin=mmin, mmax=mmax)

        self.powers = list(powers)
        self.breaks = list(breaks)
        self.distr = distributions.BrokenPowerLaw([-x for x in self.powers],
                                                  [self.mmin, *self.breaks, self.mmax])
        self.normfactor = 1

    def __call__(self, m, integral_form=False):
        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def integrate(self, mlow, mhigh, numerical=False):
        """
        The ``numerical`` keyword toggles between analytic integration
        (``False``) and numeric integration (``True``); the default
        is ``False``.
        """
        if mhigh < mlow:
            raise ValueError("Must have mlow <= mhigh in integral")
        if numerical:
            return super().integrate(mlow, mhigh)

        return (self.distr.cdf(mhigh) -
                self.distr.cdf(mlow)) * self.normfactor, 0

    def m_integrate(self, mlow, mhigh, numerical=False, **kwargs):
        """
        Also has a ``numerical`` keyword (default = ``False``).
        """
        if mhigh < mlow:
            raise ValueError("Must have mlow <= mhigh in integral")

        if numerical:
            return super().m_integrate(mlow, mhigh, **kwargs)
        else:
            #raise NotImplementedError("Analytic m_integrate not implemented for BrokenPowerLaw; use numerical=True to use the default numerical integration")
            # marking as not implemented because there's a variable definition error that requires some thinking to fix - this _might_ be fixed, but we need to check
            distr1 = distributions.BrokenPowerLaw(
                [-x + 1 for x in self.powers],
                [self.mmin, *self.breaks, self.mmax])
            ratio = distr1.pdf(self.breaks[0]) / self.distr.pdf(
                self.breaks[0]) / self.breaks[0]
            return ((distr1.cdf(mhigh) - distr1.cdf(mlow)) / ratio, 0)


class Kroupa(BrokenPowerLaw):
    """
    The `Kroupa (2001) <https://doi.org/10.1046/j.1365-8711.2001.04022.x>`__
    parameterization of the IMF. Default mass range is [0.03, 120] 
    :math:`M_\\odot`.
    """
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
        return super().__call__(m, integral_form)


class Kirkpatrick2024(BrokenPowerLaw):
    """
    The local IMF derived in the census of `Kirkpatrick et al. (2024) 
    <https://doi.org/10.3847/1538-4365/ad24e2>`__. Default mass range 
    is [0.03, 120] :math:`M_\\odot`.
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


class ChabrierPowerLaw(MassFunction):
    """
    The log-normal + power law IMF from `Chabrier 2003
    <https://doi.org/10.1086/376392>`__. Default mass range 
    is [0, :math:`\\infty`] :math:`M_\\odot`. 

    Parameters
    ----------
    lognormal_center: float
        Midpoint of the log-normal distribution's CDF, i.e. the mean of
        the log-normal with respect to :math:`d \\, \\log \\, m`
    lognormal_width: float
        The lognormal width parameter in base 10 (default = 0.57)
    alpha: float
        Power law exponent (default = 2.3)
    mmid: float
        Transition point between log-normal and power law
        (default = 1)
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self,
                 lognormal_center=0.22,
                 lognormal_width=0.57,
                 mmin=default_mmin,
                 mmax=default_mmax,
                 alpha=2.3,
                 mmid=1):
        super().__init__(mmin=mmin, mmax=mmax)
        self._mmid = mmid
        if self.mmax <= self._mmid:
            raise ValueError("The Chabrier Mass Function does not support "
                             "mmax <= mmid")
        self._alpha = alpha
        self._lognormal_width = lognormal_width * np.log(10)  # convert to base e for scipy
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


class ChabrierLogNormal(MassFunction):
    """
    A purely log-normal version of the `Chabrier (2003) 
    <https://doi.org/10.1086/376392>`__ IMF. Accepts the same 
    log-normal shape parameters as ``ChabrierPowerLaw``.
    Default mass range is [0, :math:`\\infty`] :math:`M_\\odot`. 

    Parameters
    ----------
    leading_constant: float
        Leading constant for the log-normal (default = 0.086)
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 lognormal_center=0.22,
                 lognormal_width=0.57,
                 leading_constant=0.086):
        super().__init__(mmin=mmin, mmax=mmax)

        self.multiplier = leading_constant
        self.lognormal_width = lognormal_width * np.log(10)
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


class Schechter(MassFunction):
    """
    A `Schechter <https://en.wikipedia.org/wiki/Press%E2%80%93Schechter_formalism>`__-like 
    mass function; a power law with a high-mass exponential 
    cutoff. Default mass range is [0.03, 200] :math:`M_\\odot`.
    Uses interpolation for sampling.

    Parameters
    ----------
    alpha: float
        Power law exponent (default = 2.35)
    m0: float
        Characteristic mass for exponential decay (default = 100)
    npts: int
        Number of points to use for interpolation (default = 200)
    """
    default_mmin = 0.03
    default_mmax = 200

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 alpha=2.35, m0=100, npts=None):
        super().__init__(mmin=mmin, mmax=mmax)
        if ~np.logical_and(np.isfinite(mmin), np.isfinite(self.mmax)):
            warnings.warn('function uses interpolation; non-finite mass bounds prevent random sampling')
        self.alpha = alpha
        self.m0 = m0

        self.distr = distributions.CutoffPowerLaw(-self.alpha, self.mmin,
                                                  self.mmax, self.m0,
                                                  npts=npts)
        self.normalize()

    def __call__(self, mass, integral_form=False):
        if integral_form:
            return self.normfactor * self.distr.cdf(mass)
        else:
            return self.normfactor * self.distr.pdf(mass)


class ModifiedSchechter(Schechter):
    """
    A `Schechter <https://en.wikipedia.org/wiki/Press%E2%80%93Schechter_formalism>`__-like 
    mass function with an additional low-level exponential cutoff.
    Uses interpolation for evaluation and sampling.

    Parameters
    ----------
    alpha: float
        Power law exponent (default = 2.35)
    ml: float
        Characteristic mass for the low-level cutoff
        (default = 0.5)
    mu: float
        Characteristic mass for the high-level cutoff
        (default = 100)
    npts: int
        Number of points to use for interpolation (default = 200)
    """
    default_mmin = 0.03
    default_mmax = 200

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 alpha=2.35, ml=0.5, mu=100, npts=None):
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


class KoenConvolvedPowerLaw(MassFunction):
    """
    An IMF based on an error-convolved power law as described in 
    `Koen/Kondlo (2009) <https://doi.org/10.1111/j.1365-2966.2009.14956.x>`__.
    This implementation is preferred for those looking to work extensively 
    with a single mass function, including using it to create clusters.
    Uses interpolation for evaluation and sampling.

    Parameters
    ----------
    mmin, mmax: floats
        The upper and lower bounds for the power law distribution
    alpha: float
        Power law exponent
    sigma: float
        Specified spread of error. Assumes normal distribution with 
        mean 0 and variance sigma
    npts: int
        Number of points at which to evaluate the function for
        interpolation (default = 200)
    quad_sub_limit: int
        Limit of the number of subdivisions allowed for 
        ``scipy.integrate.quad``, which handles integration 
        (default = 50)
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin, mmax, alpha, sigma, npts=200, quad_sub_limit=50):
        if mmax < mmin:
            raise ValueError("mmax must be greater than mmin")
        if not np.all(np.isfinite(np.log([mmin, mmax]))):
            raise ValueError('KoenConvolvedPowerLaw requires finite, positive mass bounds')

        super().__init__(mmin, mmax)
        self._gamma = alpha - 1
        self._sigma = sigma
        self._quad_sub_limit = quad_sub_limit
        self.distr = distributions.KoenConvolvedPowerLaw(self.mmin, self.mmax,
                                                         self.gamma, self.sigma,
                                                         npts)
        self.normfactor = 1. / self.distr.cdf(self.mmax)

    def __call__(self, m, integral_form=False):
        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def integrate(self, mlow, mhigh, **kwargs):
        if 'limit' not in kwargs.keys():
            return scipy.integrate.quad(self, mlow, mhigh,
                                        limit=self._quad_sub_limit, **kwargs)
        else:
            return scipy.integrate.quad(self, mlow, mhigh, **kwargs)

    def m_integrate(self, mlow, mhigh, **kwargs):
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
    def alpha(self):
        return self._gamma + 1

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
    The same formalism as ``KoenConvolvedPowerLaw``, but evaluation is 
    done on the spot instead of interpolating between precomputed
    values. This implementation is good for those looking for 
    improved accuracy or wanting to work with multiple mass functions at
    a time (e.g. for comparison).
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin, mmax, alpha, sigma):
        if mmax < mmin:
            raise ValueError("mmax must be greater than mmin")
        if not np.all(np.isfinite(np.log([mmin, mmax]))):
            raise ValueError('KoenConvolvedPowerLaw requires finite, positive mass bounds')

        super().__init__(mmin, mmax)
        self.sigma = sigma
        self.gamma = alpha - 1
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

    @property
    def alpha(self):
        return self.gamma + 1


class PadoanTF(MassFunction):
    """
    An IMF implementing the form derived in `Padoan & 
    Nordlund (2002) <https://doi.org/10.1086/341790>`_
    emerging from turbulent fragmentation theory. Default
    mass range is [0.01, 200] :math:`M_\\odot`. Uses
    interpolation for evaluation and sampling.

    Parameters
    ----------
    b: float
        Spectral index of the turbulence power spectrum 
        (default = 1.8)
    T0: float
        Average gas temperature in K (default = 10)
    n0: float
        Average gas number density in 1 / cm3 (default = 5e2)
    sigma: float
        Standard deviation of the log of gas density (default = 
        ``None``)
    mach: float
        Mach number of the turbulent flow. Used to calculate sigma 
        if sigma is ``None`` (default = 10)
    npts: int
        Number of points at which to evaluate the function for
        interpolation (default = 200)
    """
    default_mmin = 0.01
    default_mmax = 200

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 b=1.8, T0=10, n0=5e2,
                 sigma=None, mach=10, npts=None):
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
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

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
        """
        Set the width of the gas density distribution.
        Accepts floats. If ``update_mach`` is ``True``,
        update the Mach number to match ``sigma`` (default =
        ``True``).
        """
        self.distr.sigma = x
        self.distr._calculate()
        if update_mach:
            self._mach = 2 * np.sqrt(np.exp(x**2) - 1)

    @property
    def mach(self):
        return self._mach

    def set_mach(self, x, update_sigma=True):
        """
        Set the Mach number of the turbulent flow.
        Accepts floats. If ``update_sigma`` is ``True``,
        update the width of the density distribution to
        match ``mach`` (default = ``True``).
        """
        self._mach = x
        if update_sigma:
            self.sigma = np.sqrt(np.log(1 + (self._mach / 2)**2))
            self.distr._calculate()


# these are global objects
salpeter = Salpeter()
kroupa = Kroupa()
lognormal = chabrierlognormal = ChabrierLogNormal()
chabrier = chabrierpowerlaw = ChabrierPowerLaw()
chabrier2005 = ChabrierPowerLaw(lognormal_width=0.55,
                                lognormal_center=0.2, alpha=2.35)

massfunctions = {'kroupa': Kroupa, 'salpeter': Salpeter,
                 'chabrier': ChabrierPowerLaw,
                 'chabrierpowerlaw': ChabrierPowerLaw,
                 'chabrierlognormal': ChabrierLogNormal
                 }
reverse_mf_dict = {v: k for k, v in massfunctions.items()}


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
    Inverse mass function.  Given a likelihood value in the 
    range [0, 1), return the appropriate mass.  This calls the 
    mass function's PPF under the hood.

    Parameters
    ----------
    p: np.array
        An array of floats in the range [0, 1).  These should be 
        uniformly random numbers.
    mmin: float
        Minimum stellar mass for the mass function if none
        exists already (default = ``None``)
    mmax: float
        Maximum stellar mass for the mass function if none
        exists already (default = ``None``)
    massfunc: string or MassFunction
        ``massfunc`` can be ``'salpeter'``, ``'kroupa'``, 
        ``'chabrier'``, or an existing function
    """

    mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax)

    # this should be the entirety of "inverse-imf".  The rest is a hack
    if hasattr(mfc, 'distr'):
        return mfc.distr.ppf(p)
    else:
        raise NotImplementedError
