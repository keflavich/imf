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

class MassFunction(object):
    """
    Generic Mass Function class

    (this is mostly meant to be subclassed by other functions, not used itself)
    """
    def __init__(self, mmin=None, mmax=None):
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

    @property
    def mmin(self):
        return self._mmin

    @property
    def mmax(self):
        return self._mmax



class Salpeter(MassFunction):
    default_mmin = 0.3
    default_mmax = 120

    def __init__(self, alpha=2.35, mmin=default_mmin, mmax=default_mmax):
        """
        Create a default Salpeter mass function, i.e. a power-law mass function
        the Salpeter 1955 IMF: dn/dm ~ m^-2.35
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


class Kroupa(MassFunction):
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
                 break2=0.5):
        """
        The Kroupa IMF with two power-law breaks, p1 and p2. See __call__ for
        details.
        """
        super().__init__(mmin=mmin, mmax=mmax)

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.break1 = break1
        self.break2 = break2
        self.distr = distributions.BrokenPowerLaw([-p1, -p2, -p3],
                                                  [self.mmin, break1, break2, self.mmax])
        self.normfactor = 1

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
            return super(Kroupa, self).integrate(mlow, mhigh)

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
                [-self.p1 + 1, -self.p2 + 1, -self.p3 + 1],
                [self.mmin, self.break1, self.break2, self.mmax])
            ratio = distr1.pdf(self.break1) / self.distr.pdf(
                self.break1) / self.break1
            return ((distr1.cdf(mhigh) - distr1.cdf(mlow)) / ratio, 0)


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

        self.distr = distributions.TruncatedLogNormal(mu=lognormal_center,
                                                      sig=self.lognormal_width,
                                                      m1=self.mmin,
                                                      m2=self.mmax)

    def __call__(self, mass, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(mass) * self.multiplier
        else:
            return self.distr.pdf(mass) * self.multiplier


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
        self.distr = distributions.CompositeDistribution([
            distributions.TruncatedLogNormal(self._lognormal_center,
                                             self._lognormal_width,
                                             self.mmin,
                                             self._mmid),
            distributions.PowerLaw(-self._alpha, self._mmid, self.mmax)
        ])

    def __call__(self, x, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(x)
        else:
            return self.distr.pdf(x)



class Schechter(MassFunction):
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin=default_mmin, mmax=default_mmax):
        raise NotImplementedError("Schechter function needs to be refactored")
        super().__init__(mmin=mmin, mmax=mmax)

    def __call__(self, m, A=1, beta=2, m0=100, integral_form=False):
        """
        A Schechter function with arbitrary defaults
        (integral may not be correct - exponent hasn't been dealt with at all)
        
        (TODO: this should be replaced with a Truncated Power Law Distribution)

        $$ A m^{-\\beta} e^{-m/m_0} $$

        Parameters
        ----------
            m: np.ndarray
                List of masses for which to compute the Schechter function
            A: float
                Arbitrary amplitude of the Schechter function
            beta: float
                Power law exponent
            m0: float
                Characteristic mass (mass at which exponential decay takes over)

        Returns
        -------
            p(m) - the (unnormalized) probability of an object of a given mass
            as a function of that object's mass
            (though you could interpret mass as anything, it's just a number)
        """
        if integral_form:
            beta -= 1
        return A * m**-beta * np.exp(-m / m0) * (m > self.mmin) * (m < self.mmax)

class ModifiedSchechter(Schechter):
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin=default_mmin, mmax=default_mmax):
        self.schechter = super().__init__(mmin=mmin, mmax=mmax)

    def __call__(self, m, m1, **kwargs):
        """
        A Schechter function with a low-level exponential cutoff

        (TODO: this should be replaced with a Truncated Power Law Distribution)

        Parameters
        ----------
            m: np.ndarray
                List of masses for which to compute the Schechter function
            m1: float
                Characteristic minimum mass (exponential decay below this mass)
            ** See schecter for other parameters **

        Returns
        -------
            p(m) - the (unnormalized) probability of an object of a given mass
            as a function of that object's mass
            (though you could interpret mass as anything, it's just a number)
        """
        return self.schechter(m, **kwargs) * np.exp(-m1 / m) * (m > self.mmin) * (m < self.mmax)

try:
    import scipy

    def schechter_cdf(m, A=1, beta=2, m0=100, mmin=10, mmax=None, npts=1e4):
        """
        Return the CDF value of a given mass for a set mmin, mmax
        mmax will default to 10 m0 if not specified

        Analytic integral of the Schechter function:
        http://www.wolframalpha.com/input/?i=integral%28x^-a+exp%28-x%2Fm%29+dx%29
        """
        if mmax is None:
            mmax = 10 * m0

        # integrate the CDF from the minimum to maximum
        posint = -mmax**(1 - beta) * scipy.special.expn(beta, mmax / m0)
        negint = -mmin**(1 - beta) * scipy.special.expn(beta, mmin / m0)
        tot = posint - negint

        # normalize by the integral
        ret = (-m**(1 - beta) * scipy.special.expn(beta, m / m0) -
               negint) / tot

        return ret

    def sh_cdf_func(**kwargs):
        return lambda x: schechter_cdf(x, **kwargs)
except ImportError:
    pass

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
#                 'schechter': Schechter, 'modified_schechter': ModifiedSchecter}
reverse_mf_dict = {v: k for k, v in massfunctions.items()}
# salpeter and schechter selections are arbitrary
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

####This section contains the functions required to optimally sample a cluster####

def _prefactor(max_star,massfunc):
    """
    Returns the multiplier required for an IMF to have at most one star above m_max.
    """
    return 1/massfunc.integrate(max_star,massfunc.mmax)[0]

def _M_cluster(m,massfunc):
    """
    Returns the mass of a cluster distributed according to some IMF where the 
    largest star has mass m.
    """
    k = _prefactor(m,massfunc)
    return k*massfunc.m_integrate(massfunc.mmin,m)[0]+m

def _max_star(m,M_res,massfunc):
    """
    Returns the most massive star capable of forming in a cluster of mass M_res
    according to the m_max/M_cluster relation. Formatted for use with root finding.
    """
    return M_res-_M_cluster(m,massfunc)

def _max_star_prime(m,M_res,massfunc):
    """
    Returns the derivative of _max_star at mass m. Used for Newton's method in
    the case of an infinite upper bound on the provided mass function.
    """
    term1 = _prefactor(m,massfunc)**2*massfunc(m)*massfunc.m_integrate(massfunc.mmin,m)[0]
    term2 = m*massfunc(m)*_prefactor(m,massfunc)
    return -term1-term2-1

def _get_next_m(m,last_m,k,massfunc):
    """
    Returns the next smallest star in an optimally sampled cluster given the 
    previous star and overall IMF. Formatted for use with root finding.
    """
    return k*massfunc.m_integrate(m,last_m)[0]-m

def _opt_sample(M_res,massfunc,tolerance):
    """
    Returns a numpy array containing stellar masses that optimally sample 
    from a provided MassFunction to make a cluster with mass M_res.
    """
    #retrieve mass bounds from provided massfunc
    mmin = massfunc.mmin
    mmax = massfunc.mmax
    finMax = np.isfinite(mmax)

    #finding all the component stars requires a cutoff--ensure there is one
    if not np.isfinite(np.log(mmin)):
        if not np.isfinite(np.log(tolerance)):
            raise ValueError('Optimal sampling requires either mmin or tolerance to be finite and greater than zero.')

    if finMax:
        #bracket from min to ALMOST max (max gives an undefined prefactor)
        sol = root_scalar(_max_star,args=(M_res,massfunc),bracket=[mmin,0.9999*mmax])
    else:
        #use Newton's method
        sol = root_scalar(_max_star,args=(M_res,massfunc),x0=10*mmin,fprime=_max_star_prime)
    k = _prefactor(sol.root,massfunc)
    M_tot = sol.root
    star_masses = [sol.root]

    while np.abs(M_res-M_tot) > np.maximum(mmin,tolerance):
        try:
            sol = root_scalar(_get_next_m,args=(star_masses[-1],k,massfunc),
                              bracket=[mmin,star_masses[-1]])
        except(ValueError):
            print(f'Reached provided lower mass bound; stopping')
            break
        m = sol.root    
        star_masses.append(m)
        M_tot += m
    
    return np.array(star_masses),M_tot

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
    ==========
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
    
    #catch wrong keywords early
    ok_samplings = ['random','optimal']
    ok_criteria = ['nearest','before','after','sorted']
    if not sampling in ok_samplings:
        raise ValueError("Sampling should be either 'random' or 'optimal' (see documentation)")
    if (sampling == 'random') and not stop_criterion in ok_criteria:
        raise ValueError("Stop criterion for random sampling should be 'nearest', 'before', 'after', or 'sorted' (see documentation)")

    if sampling == 'optimal':
        mfc = get_massfunc(massfunc, mmin=mmin, mmax=mmax, **kwargs)
        masses,mtot = _opt_sample(mcluster,mfc,tolerance=tolerance)
        if verbose:
            print(f'Sampled {len(masses)} new stars.')
        if not silent:
            print(f'Total cluster mass is {np.round(mtot,3)} (limit was {int(mcluster)})')

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


mass_luminosity_interpolator_cache = {}


def mass_luminosity_interpolator(name):
    if name in mass_luminosity_interpolator_cache:
        return mass_luminosity_interpolator_cache[name]
    elif name == 'VGS':

        # non-extrapolated
        vgsMass = [
            51.3, 44.2, 41.0, 38.1, 35.5, 33.1, 30.8, 28.8, 26.9, 25.1, 23.6,
            22.1, 20.8, 19.5, 18.4
        ]
        vgslogL = [
            6.154, 6.046, 5.991, 5.934, 5.876, 5.817, 5.756, 5.695, 5.631,
            5.566, 5.499, 5.431, 5.360, 5.287, 5.211
        ]
        vgslogQ = [
            49.18, 48.99, 48.90, 48.81, 48.72, 48.61, 48.49, 48.34, 48.16,
            47.92, 47.63, 47.25, 46.77, 46.23, 45.69
        ]
        # mass extrapolated
        vgsMe = np.concatenate([
            np.linspace(0.03, 0.43, 100),
            np.linspace(0.43, 2, 100),
            np.linspace(2, 20, 100), vgsMass[::-1],
            np.linspace(50, 150, 100)
        ])
        # log luminosity extrapolated
        vgslogLe = np.concatenate([
            np.log10(0.23 * np.linspace(0.03, 0.43, 100)**2.3),
            np.log10(np.linspace(0.43, 2, 100)**4),
            np.log10(1.5 * np.linspace(2, 20, 100)**3.5), vgslogL[::-1],
            np.polyval(np.polyfit(np.log10(vgsMass[:3]), vgslogL[:3], 1),
                       np.log10(np.linspace(50, 150, 100)))
        ])
        # log Q (lyman continuum) extrapolated
        vgslogQe = np.concatenate([
            np.zeros(100),  # 0.03-0.43 solar mass stars produce 0 LyC photons
            np.zeros(100),  # 0.43-2.0 solar mass stars produce 0 LyC photons
            np.polyval(np.polyfit(np.log10(vgsMass[-3:]), vgslogQ[-3:], 1),
                       np.log10(np.linspace(8, 18.4, 100))),
            vgslogQ[::-1],
            np.polyval(np.polyfit(np.log10(vgsMass[:3]), vgslogQ[:3], 1),
                       np.log10(np.linspace(50, 150, 100)))
        ])

        mass_luminosity_interpolator_cache[name] = vgsMe, vgslogLe, vgslogQe

        return mass_luminosity_interpolator_cache[name]
    elif name == 'Ekstrom':
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = 1e7  # effectively infinite

        # this query should cache
        tbl = Vizier.get_catalogs('J/A+A/537/A146/iso')[0]

        match = tbl['logAge'] == 6.5
        masses = tbl['Mass'][match]
        lums = tbl['logL'][match]
        mass_0 = 0.033
        lum_0 = np.log10((mass_0 / masses[0])**3.5 * 10**lums[0])
        mass_f = 200  # extrapolate to 200 Msun...
        lum_f = np.log10(10**lums[-1] * (mass_f / masses[-1])**1.35)

        masses = np.array([mass_0] + masses.tolist() + [mass_f])
        lums = np.array([lum_0] + lums.tolist() + [lum_f])

        # TODO: come up with a half-decent approximation here?  based on logTe?
        logQ = lums - 0.5

        mass_luminosity_interpolator_cache[name] = masses, lums, logQ

        return mass_luminosity_interpolator_cache[name]
    else:
        raise ValueError("Bad grid name {0}".format(name))


def lum_of_star(mass, grid='Ekstrom'):
    """
    Determine total luminosity of a star given its mass

    Two grids:
        (1) VGS:
    Uses the Vacca, Garmany, Shull 1996 Table 5 Log Q and Mspec parameters

    returns LogL in solar luminosities
    **WARNING** Extrapolates for M not in [18.4, 50] msun

    http://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation

    (2) Ekstrom 2012:
    Covers 0.8 - 64 Msun, extrapolated out of that
    """
    masses, lums, _ = mass_luminosity_interpolator(grid)
    return np.interp(mass, masses, lums)


def lum_of_cluster(masses, grid='Ekstrom'):
    """
    Determine the log of the integrated luminosity of a cluster
    Only M>=8msun count

    masses is a list or array of masses.
    """
    #if max(masses) < 8: return 0
    logL = lum_of_star(masses, grid=grid)  #[masses >= 8])
    logLtot = np.log10((10**logL).sum())
    return logLtot

def lyc_of_star(mass, grid='VGS'):
    """
    Determine lyman continuum luminosity of a star given its mass
    Uses the Vacca, Garmany, Shull 1996 Table 5 Log Q and Mspec parameters

    returns LogQ
    """
    masses, _, logQ = mass_luminosity_interpolator(grid)

    return np.interp(mass, masses, logQ)

def lyc_of_cluster(masses, grid='VGS'):
    """
    Determine the log of the integrated lyman continuum luminosity of a cluster
    Only M>=8msun count

    masses is a list or array of masses.
    """
    if max(masses) < 8:
        return 0
    logq = lyc_of_star(masses[masses >= 8], grid=grid)
    logqtot = np.log10((10**logq).sum())
    return logqtot


def color_from_mass(mass, outtype=float):
    """
    Use vendian.org colors:
   100 O2(V)        150 175 255   #9db4ff
    50 O5(V)        157 180 255   #9db4ff
    20 B1(V)        162 185 255   #a2b9ff
    10 B3(V)        167 188 255   #a7bcff
     8 B5(V)        170 191 255   #aabfff
     6 B8(V)        175 195 255   #afc3ff
   2.2 A1(V)        186 204 255   #baccff
   2.0 A3(V)        192 209 255   #c0d1ff
  1.86 A5(V)        202 216 255   #cad8ff
   1.6 F0(V)        228 232 255   #e4e8ff
   1.5 F2(V)        237 238 255   #edeeff
   1.3 F5(V)        251 248 255   #fbf8ff
   1.2 F8(V)        255 249 249   #fff9f9
     1 G2(V)        255 245 236   #fff5ec
  0.95 G5(V)        255 244 232   #fff4e8
  0.90 G8(V)        255 241 223   #fff1df
  0.85 K0(V)        255 235 209   #ffebd1
  0.70 K4(V)        255 215 174   #ffd7ae
  0.60 K7(V)        255 198 144   #ffc690
  0.50 M2(V)        255 190 127   #ffbe7f
  0.40 M4(V)        255 187 123   #ffbb7b
  0.35 M6(V)        255 187 123   #ffbb7b
  0.30 M8(V)        255 167 123   #ffbb7b  # my addition
    """

    mcolor = { # noqa: E131
             100: (150, 175, 255),
              50: (157, 180, 255),
              20: (162, 185, 255),
              10: (167, 188, 255),
               8: (170, 191, 255),
               6: (175, 195, 255),
             2.2: (186, 204, 255),
             2.0: (192, 209, 255),
            1.86: (202, 216, 255),
             1.6: (228, 232, 255),
             1.5: (237, 238, 255),
             1.3: (251, 248, 255),
             1.2: (255, 249, 249),
               1: (255, 245, 236),
            0.95: (255, 244, 232),
            0.90: (255, 241, 223),
            0.85: (255, 235, 209),
            0.70: (255, 215, 174),
            0.60: (255, 198, 144),
            0.50: (255, 190, 127),
            0.40: (255, 187, 123),
            0.35: (255, 187, 123),
            0.30: (255, 177, 113),
            0.20: (255, 107, 63),
            0.10: (155, 57, 33),
            0.10: (155, 57, 33),
           0.003: (105, 27, 0),
            }

    keys = sorted(mcolor.keys())

    reds, greens, blues = zip(*[mcolor[k] for k in keys])

    r = np.interp(mass, keys, reds)
    g = np.interp(mass, keys, greens)
    b = np.interp(mass, keys, blues)

    if outtype == int:
        return (r, g, b)
    elif outtype == float:
        return (r / 255., g / 255., b / 255.)
    else:
        raise NotImplementedError


def color_of_cluster(cluster, colorfunc=color_from_mass):
    colors = np.array([colorfunc(m) for m in cluster])
    luminosities = 10**np.array([lum_of_star(m) for m in cluster])
    mean_color = (colors *
                  luminosities[:, None]).sum(axis=0) / luminosities.sum()
    return mean_color

def coolplot(clustermass, massfunc=kroupa, log=True, **kwargs):
    """
    "cool plot" is just because the plot is kinda neat.

    This function creates a cluster using `make_cluster`, assigns each star a
    color based on the vendian.org colors using `color_from_mass`, and assigns
    each star a random Y-value distributed underneath the specified mass
    function's curve.

    Parameters
    ----------
    clustermass: float
        The mass of the cluster in solar masses
    massfunc: str
        A MassFunction instance
    log: bool
        Is the Y-axis log-scaled?

    Returns
    -------
    cluster: array
        The array of stellar masses that makes up the cluster
    yax: array
        The array of Y-values associated with the stellar masses
    colors: list
        A list of color tuples associated with each star
    """
    cluster = make_cluster(clustermass,
                           massfunc=massfunc,
                           mmax=massfunc.mmax,
                           **kwargs)
    colors = [color_from_mass(m) for m in cluster]

    maxmass = cluster.max()
    pmin = massfunc(maxmass)
    if log:
        yax = [
            np.random.rand() * (np.log10(massfunc(m)) - np.log10(pmin)) +
            np.log10(pmin) for m in cluster
        ]
    else:
        yax = [
            np.random.rand() * ((massfunc(m)) / (pmin)) + (pmin)
            for m in cluster
        ]

    assert all(np.isfinite(yax))

    return cluster, yax, colors

    # import pylab as pl
    # pl.scatter(cluster, yax, c=colors, s=np.log10(cluster)*5)


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
        Number of evenly log-spaced points at which to evaluate the function
        (function calls interpolate between these). Defaults to 200.
    quad_sub_limit: int
        Limit of the number of subdivisions allowed for scipy.integrate.quad,
        which handles integration. Defaults to scipy's default of 50.
    """
    default_mmin = 0
    default_mmax = np.inf

    def __init__(self, mmin, mmax, gamma, sigma, npts=200, quad_sub_limit=50):
        if mmax < mmin:
            raise ValueError("mmax must be greater than mmin")
        if not np.all(np.isfinite(np.log([mmin,mmax]))):
            raise ValueError('KoenConvolvedPowerLaw requires finite, positive mass bounds')
        
        super().__init__(mmin, mmax)
        self._gamma = gamma
        self._sigma = sigma
        self._quad_sub_limit = quad_sub_limit
        self.distr = distributions.KoenConvolvedPowerLaw(self.mmin,self.mmax,
                                                         self.gamma,self.sigma,npts)
        self._normfactor = 1. / self.distr.cdf(self.mmax)

    def __call__(self, m, integral_form=False):
        if integral_form:
            return self._normfactor*self.distr.cdf(m)
        else:
            return self._normfactor*self.distr.pdf(m)
    
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
    def quad_sub_limit(self,x):
        self._quad_sub_limit = x

    @property
    def normfactor(self):
        return self._normfactor

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
        if not np.all(np.isfinite(np.log([mmin,mmax]))):
            raise ValueError('KoenConvolvedPowerLaw requires finite, positive mass bounds')

        super().__init__(mmin, mmax)
        self.sigma = sigma
        self.gamma = gamma
        self.normfactor = 1/self._integrate(self.mmax,integral_form=True)

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
        #Sub-intervals for the integration to capture small changes at both ends
        x = np.geomspace(self.mmin,self.mmax,100)
        mir_x = self.mmax-(x[::-1]-self.mmin)
        dx = x[1:]-x[:-1]
        cutoff = min(self.sigma,1)
        break1 = np.searchsorted(dx,cutoff)
        break2 = np.searchsorted(-dx[::-1],-cutoff)
        xpt = x[break1]
        mirxpt = mir_x[break2]
        x1, x2 = min(xpt,mirxpt), max(xpt,mirxpt)
        x = np.append(x[x < x1],np.linspace(x1,x2,
                                            int((x2-x1)/cutoff)))
        x = np.append(x,mir_x[mir_x > x2])
        return x

    def _integrate(self, y, integral_form):
        steps = self._mirror_steps()
        chunks = []
        for i in range(len(steps)-1):
            l,u = steps[i],steps[i+1]
            area = quad(self._integrand,l,u,args=(y,integral_form))[0]
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
