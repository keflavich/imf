"""
Various codes to work with the initial mass function
"""

from __future__ import print_function
import numpy as np
import types
import scipy.integrate
from scipy.special import erf
from six import iteritems
import scipy.integrate as integrate
from scipy.integrate import quad
from . import distributions


class MassFunction(object):
    """
    Generic Mass Function class

    (this is mostly meant to be subclassed by other functions, not used itself)
    """

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
        self.normfactor = 1./integral[0]

        assert self.normfactor > 0


class Salpeter(MassFunction):

    def __init__(self, alpha=2.35, mmin=0.3, mmax=120):
        """
        Create a default Salpeter mass function, i.e. a power-law mass function
        the Salpeter 1955 IMF: dn/dm ~ m^-2.35
        """
        self.mmin = mmin
        self.mmax = mmax
        self.alpha = alpha
        self.normfactor = 1

    @property
    def distr(self):
        return distributions.PowerLaw(-self.alpha, self.mmin, self.mmax)

    def __call__(self, m, integral_form=False):
        if not integral_form:
            return self.distr.pdf(m) * self.normfactor
        else:
            return self.distr.cdf(m) * self.normfactor


# three codes for dn/dlog(m)
salpeter = Salpeter()
# kroupa
# chabrier


class Kroupa(MassFunction):
    # kroupa = BrokenPowerLaw(breaks={0.08: -0.3, 0.5: 1.3, 'last': 2.3}, mmin=0.03, mmax=120)
    def __init__(self, mmin=0.03, mmax=120, p1=0.3, p2=1.3, p3=2.3,
                 break1=0.08, break2=0.5):
        """
        The Kroupa IMF with two power-law breaks, p1 and p2. See __call__ for
        details.
        """
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.break1 = break1
        self.break2 = break2
        self.distr = distributions.BrokenPowerLaw([-p1, -p2, -p3],
                                                  [mmin, break1, break2, mmax])
        self.mmin = mmin
        self.mmax = mmax
        self.normfactor = 1

    @property
    def mmin(self):
        return self.distr.m1

    @mmin.setter
    def mmin(self, value):
        self.distr.m1 = value

    @property
    def mmax(self):
        return self.distr.m2

    @mmax.setter
    def mmax(self, value):
        self.distr.m2 = value

    def __call__(self, m, integral_form=False):
        """
        Kroupa 2001 IMF (http://arxiv.org/abs/astro-ph/0009005,
        http://adsabs.harvard.edu/abs/2001MNRAS.322..231K)

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
        if mhigh <= mlow:
            raise ValueError("Must have mlow < mhigh in integral")
        if numerical:
            return super(Kroupa, self).integrate(mlow, mhigh)

        return (self.distr.cdf(mhigh) - self.distr.cdf(mlow)) * self.normfactor, 0


    def m_integrate(self, mlow, mhigh, numerical=False, **kwargs):
        """
        Integrate the mass function over some range
        """
        if mhigh <= mlow:
            raise ValueError("Must have mlow < mhigh in integral")

        if numerical:
            return super(Kroupa, self).m_integrate(mlow, mhigh, **kwargs)
        else:
            distr1 = distributions.BrokenPowerLaw([-self.p1+1, -self.p2+1,
                                                   -self.p3+1],
                                                  [self.mmin, self.break1,
                                                   self.break2, self.mmax])
            ratio = distr1.pdf(self.break1)/self.distr.pdf(self.break1)/self.break1
            return ((distr1.cdf(mhigh)-distr1.cdf(mlow))/ratio, 0)




kroupa = Kroupa()


class Chabrier(MassFunction):
    def __init__(self):
        self.mmin = 0.57*np.log(10)
        self.multiplier = 0.86

    @property
    def distr(self):
        return distributions.LogNormal(0.22, self.mmin)

    def __call__(self, mass, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(mass)*self.multiplier
        else:
            return self.distr.pdf(mass)*self.multiplier


chabrier = Chabrier()

class Chabrier2005(MassFunction):
    def __init__(self, lognormal_center=0.2, lognormal_width=0.55*np.log(10),
                 mmin=0.033, mmax=np.inf, alpha=2.35, mmid=1):
        # The numbers are from Eqn 3 of
        # https://ui.adsabs.harvard.edu/abs/2005ASSL..327...41C/abstract
        # importantly the lognormal center is the exp(M) where M is the mean of ln(mass)
        # normal distribution
        self.mmin = mmin
        self.mmid = mmid
        self.mmax = mmax
        self.alpha = alpha
        self.lognormal_width = lognormal_width
        self.lognormal_center = lognormal_center

    @property
    def distr(self):
        return distributions.CompositeDistribution(
            [distributions.TruncatedLogNormal(self.lognormal_center,
                                              self.lognormal_width,
                                              self.mmin,
                                              self.mmid),
             distributions.PowerLaw(-self.alpha, self.mmid, self.mmax)])

    def __call__(self, x, integral_form=False, **kw):
        if integral_form:
            return self.distr.cdf(x)
        else:
            return self.distr.pdf(x)


chabrier2005 = Chabrier2005()


def schechter(m, A=1, beta=2, m0=100, integral=False):
    """
    A Schechter function with arbitrary defaults
    (integral may not be correct - exponent hasn't been dealt with at all)

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
    if integral:
        beta -= 1
    return A*m**-beta * np.exp(-m/m0)

def modified_schechter(m, m1, **kwargs):
    """
    A Schechter function with a low-level exponential cutoff
    "
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
    return schechter(m, **kwargs) * np.exp(-m1/m)

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
            mmax = 10*m0

        # integrate the CDF from the minimum to maximum
        posint = -mmax**(1-beta) * scipy.special.expn(beta, mmax/m0)
        negint = -mmin**(1-beta) * scipy.special.expn(beta, mmin/m0)
        tot = posint-negint

        # normalize by the integral
        ret = (-m**(1-beta) * scipy.special.expn(beta, m/m0) - negint) / tot

        return ret

    def sh_cdf_func(**kwargs):
        return lambda x: schechter_cdf(x, **kwargs)
except ImportError:
    pass

def m_integrate(fn=kroupa, bins=np.logspace(-2, 2, 500)):
    xax = (bins[:-1]+bins[1:])/2.
    integral = xax*(bins[1:]-bins[:-1]) * (fn(bins[:-1])+fn(bins[1:])) / 2.

    return xax, integral

def cumint(fn=kroupa, bins=np.logspace(-2, 2, 500)):
    xax, integral = integrate(fn, bins)
    return integral.cumsum() / integral.sum()

def m_cumint(fn=kroupa, bins=np.logspace(-2, 2, 500)):
    xax, integral = m_integrate(fn, bins)
    return integral.cumsum() / integral.sum()

massfunctions = {'kroupa': kroupa, 'salpeter': salpeter, 'chabrier': chabrier,
                 'schechter': schechter, 'modified_schechter': modified_schechter}
reverse_mf_dict = {v: k for k, v in iteritems(massfunctions)}
# salpeter and schechter selections are arbitrary
mostcommonmass = {'kroupa': 0.08, 'salpeter': 0.01, 'chabrier': 0.23,
                  'schecter': 0.01, 'modified_schechter': 0.01}
expectedmass_cache = {}

def get_massfunc(massfunc):
    if isinstance(massfunc, types.FunctionType) or hasattr(massfunc, '__call__'):
        return massfunc
    elif type(massfunc) is str:
        return massfunctions[massfunc]
    else:
        raise ValueError("massfunc must either be a string in the set %s or a function"
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

def inverse_imf(p, nbins=1000, mmin=None, mmax=None, massfunc='kroupa',
                **kwargs):
    """
    Inverse mass function.  Creates a cumulative distribution function from the
    mass function and samples it using the given randomly distributed values
    ``p``.


    Parameters
    ----------
    p: np.array
        An array of floats in the range [0, 1).  These should be uniformly random
        numbers.
    nbins: int
        The number of bins in the cumulative distribution function to sample
        over.  More bins results in (marginally) higher precision.
    mmin: float
    mmax: float
        Minimum and maximum stellar mass in the distribution
    massfunc: string or function
        massfunc can be 'kroupa', 'chabrier', 'salpeter', 'schechter', or a
        function
    """

    mfc = get_massfunc(massfunc)

    if mmin is not None and hasattr(mfc, 'mmin') and mmin != mfc.mmin:
        orig_mmin = mfc.mmin
        mfc.mmin = mmin
    if mmax is not None and hasattr(mfc, 'mmax') and mmax != mfc.mmax:
        orig_mmax = mfc.mmax
        mfc.mmax = mmax

    mmin = mfc.mmin
    mmax = mfc.mmax

    ends = np.logspace(np.log10(mmin), np.log10(mmax), nbins)
    masses = (ends[1:] + ends[:-1])/2.
    dm = np.diff(ends)


    # the full probability distribution function N(M) dm
    mf = mfc(masses, **kwargs)

    # integrate by taking the cumulative sum of x dx
    mfcum = (mf*dm).cumsum()

    # normalize to sum (this turns into a cdf)
    mfcum /= mfcum.max()

    result = np.interp(p, mfcum, masses)

    if 'orig_mmin' in locals():
        mfc.mmin = orig_mmin
    if 'orig_mmax' in locals():
        mfc.mmax = orig_mmax

    return result


def make_cluster(mcluster, massfunc='kroupa', verbose=False, silent=False,
                 tolerance=0.0, stop_criterion='nearest', mmax=120, mmin=None):
    """
    Sample from an IMF to make a cluster.  Returns the masses of all stars in the cluster

    massfunc must be a string
    tolerance is how close the cluster mass must be to the requested mass.
    If the last star is greater than this tolerance, the total mass will not be within
    tolerance of the requested

    stop criteria can be: 'nearest', 'before', 'after', 'sorted'

    """

    # use most common mass to guess needed number of samples
    # nsamp = mcluster / mostcommonmass[get_massfunc_name(massfunc)]
    # masses = inverse_imf(np.random.random(int(nsamp)), massfunc=massfunc, **kwargs)

    # mtot = masses.sum()
    # if verbose:
    #    print(("%i samples yielded a cluster mass of %g (%g requested)" %
    #          (nsamp, mtot, mcluster)))

    mfc = get_massfunc(massfunc)
    if mmin is not None and hasattr(mfc, 'mmin') and mfc.mmin != mmin:
        orig_mmin = mfc.mmin
        mfc.mmin = mmin
    if mmax is not None and hasattr(mfc, 'mmax') and mfc.mmax != mmax:
        orig_mmax = mfc.mmax
        mfc.mmax = mmax

    if (massfunc, mfc.mmin, mmax) in expectedmass_cache:
        expected_mass = expectedmass_cache[(massfunc, mfc.mmin, mmax)]
        assert expected_mass > 0
    else:
        expected_mass = mfc.m_integrate(mfc.mmin, mmax)[0]
        assert expected_mass > 0
        expectedmass_cache[(massfunc, mfc.mmin, mmax)] = expected_mass

    if verbose:
        print("Expected mass is {0:0.3f}".format(expected_mass))


    mtot = 0
    masses = []

    while mtot < mcluster + tolerance:
        # at least 1 sample, but potentially many more
        nsamp = int(np.ceil((mcluster+tolerance-mtot) / expected_mass))
        assert nsamp > 0
        newmasses = mfc.distr.rvs(nsamp)
        masses = np.concatenate([masses, newmasses])
        mtot = masses.sum()
        if verbose:
            print("Sampled %i new stars.  Total is now %g" % (int(nsamp), mtot))

        if mtot > mcluster+tolerance: # don't force exact equality; that would yield infinite loop
            mcum = masses.cumsum()
            if stop_criterion == 'sorted':
                masses = np.sort(masses)
                if np.abs(masses[:-1].sum()-mcluster) < np.abs(masses.sum() - mcluster):
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
                print("Selected the first %i out of %i masses to get %g total"
                      % (last_ind, len(mcum), mtot))
            # force the break, because some stopping criteria can push mtot < mcluster
            break

    if not silent:
        print("Total cluster mass is %g (limit was %g)" % (mtot, mcluster))

    if 'orig_mmin' in locals():
        mfc.mmin = orig_mmin
    if 'orig_mmax' in locals():
        mfc.mmax = orig_mmax

    return masses

mass_luminosity_interpolator_cache = {}

def mass_luminosity_interpolator(name):
    if name in mass_luminosity_interpolator_cache:
        return mass_luminosity_interpolator_cache[name]
    elif name == 'VGS':

        # non-extrapolated
        vgsMass = [51.3, 44.2, 41.0, 38.1, 35.5, 33.1, 30.8, 28.8, 26.9, 25.1,
                   23.6, 22.1, 20.8, 19.5, 18.4]
        vgslogL = [6.154, 6.046, 5.991, 5.934, 5.876, 5.817, 5.756, 5.695,
                   5.631, 5.566, 5.499, 5.431, 5.360, 5.287, 5.211]
        vgslogQ = [49.18, 48.99, 48.90, 48.81, 48.72, 48.61, 48.49, 48.34,
                   48.16, 47.92, 47.63, 47.25, 46.77, 46.23, 45.69]
        # mass extrapolated
        vgsMe = np.concatenate([
            np.linspace(0.03, 0.43, 100),
            np.linspace(0.43, 2, 100),
            np.linspace(2, 20, 100),
            vgsMass[::-1],
            np.linspace(50, 150, 100)])
        # log luminosity extrapolated
        vgslogLe = np.concatenate([
            np.log10(0.23*np.linspace(0.03, 0.43, 100)**2.3),
            np.log10(np.linspace(0.43, 2, 100)**4),
            np.log10(1.5*np.linspace(2, 20, 100)**3.5),
            vgslogL[::-1],
            np.polyval(np.polyfit(np.log10(vgsMass[:3], vgslogL[:3], 1),
                                  np.log10(np.linspace(50, 150, 100))))])
        # log Q (lyman continuum) extrapolated
        vgslogQe = np.concatenate([
            np.zeros(100), # 0.03-0.43 solar mass stars produce 0 LyC photons
            np.zeros(100), # 0.43-2.0 solar mass stars produce 0 LyC photons
            np.polyval(np.polyfit(np.log10(vgsMass[-3:], vgslogQ[-3:], 1),
                                  np.log10(np.linspace(8, 18.4, 100)))),
            vgslogQ[::-1],
            np.polyval(np.polyfit(np.log10(vgsMass[:3], vgslogQ[:3], 1),
                                  np.log10(np.linspace(50, 150, 100))))
        ])

        mass_luminosity_interpolator_cache[name] = vgsMe, vgslogLe, vgslogQ

        return mass_luminosity_interpolator_cache[name]
    elif name == 'Ekstrom':
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = 1e7 # effectively infinite

        # this query should cache
        tbl = Vizier.get_catalogs('J/A+A/537/A146/iso')[0]

        match = tbl['logAge'] == 6.5
        masses = tbl['Mass'][match]
        lums = tbl['logL'][match]
        mass_0 = 0.033
        lum_0 = np.log10((mass_0/masses[0])**3.5 * 10**lums[0])
        mass_f = 200 # extrapolate to 200 Msun...
        lum_f = np.log10(10**lums[-1] * (mass_f/masses[-1])**1.35)

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
    logL = lum_of_star(masses, grid=grid) #[masses >= 8])
    logLtot = np.log10( (10**logL).sum() )
    return logLtot

def lyc_of_star(mass):
    """
    Determine lyman continuum luminosity of a star given its mass
    Uses the Vacca, Garmany, Shull 1996 Table 5 Log Q and Mspec parameters

    returns LogQ
    """
    masses, _, logQ =  mass_luminosity_interpolator(grid)

    return np.interp(mass, masses, logQ)

def lyc_of_cluster(masses):
    """
    Determine the log of the integrated lyman continuum luminosity of a cluster
    Only M>=8msun count

    masses is a list or array of masses.
    """
    if max(masses) < 8: return 0
    logq = lyc_of_star(masses[masses >= 8])
    logqtot = np.log10( (10**logq).sum() )
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

    mcolor = {
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
        return (r/255., g/255., b/255.)
    else:
        raise NotImplementedError

def color_of_cluster(cluster, colorfunc=color_from_mass):
    colors       = np.array([colorfunc(m) for m in cluster])
    luminosities = 10**np.array([lum_of_star(m) for m in cluster])
    mean_color = (colors*luminosities[:, None]).sum(axis=0)/luminosities.sum()
    return mean_color

def coolplot(clustermass, massfunc='kroupa', log=True, **kwargs):
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
        The name of the mass function to use, determined using the
        `get_massfunc` function.
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
    cluster = make_cluster(clustermass, massfunc=massfunc, mmax=massfunc.mmax,
                           **kwargs)
    colors = [color_from_mass(m) for m in cluster]
    massfunc = get_massfunc(massfunc)
    maxmass = cluster.max()
    pmin = massfunc(maxmass)
    if log:
        yax = [np.random.rand()*(np.log10(massfunc(m))-np.log10(pmin))
               + np.log10(pmin) for m in cluster]
    else:
        yax = [np.random.rand()*((massfunc(m))/(pmin)) + (pmin) for m in cluster]

    assert all(np.isfinite(yax))

    return cluster, yax, colors

    # import pylab as pl
    # pl.scatter(cluster, yax, c=colors, s=np.log10(cluster)*5)



class KoenConvolvedPowerLaw(MassFunction):
    """
    Implementaton of convolved errror power-law described in 2009 Koen, Kondlo
    paper, Fitting power-law distributions to data with measurement errors.
    Equations (3) and (5)

    Parameters
    ----------
    m: float
        The mass at which to evaluate the function
    mmin, mmax: floats
        The upper and lower bounds for the power law distribution
    gamma: floats
        The specified gamma for the distribution, slope = -gamma - 1
    sigma: float or None
        specified spread of error, assumes Normal distribution with mean 0 and variance sigma.
    """


    def __init__(self, mmin, mmax, gamma, sigma):
        self.mmin = mmin
        self.mmax = mmax
        self.sigma = sigma
        self.gamma = gamma

    def __call__(self, m, integral_form=False):
        m = np.asarray(m)
        if self.mmax<self.mmin:
            raise ValueError("mmax must be greater than mmin")

        if integral_form:
            #       Returns
            #       -------
            #       Probability that m < x for the given CDF with specified
            #       mmin, mmax, sigma, and gamma

            def error(t):
                return np.exp(-(t**2)/2)

            error_coeffecient = 1/np.sqrt(2*np.pi)

            def error_integral(y):
                error_integral = quad(error, -np.inf, (y-self.mmax)/self.sigma)[0]
                return error_integral

            vector_errorintegral = np.vectorize(error_integral)
            phi = vector_errorintegral(m) * error_coeffecient

            def integrand(x, y):
                return ((self.mmin**-self.gamma - x**-self.gamma) *
                        np.exp((-1/2)*((y-x)/self.sigma)**2))

            coef = (1 / (self.sigma*np.sqrt(2*np.pi) *
                         (self.mmin**-self.gamma -
                          self.mmax**-self.gamma)))

            def eval_integral(y):
                integral = quad(integrand, self.mmin, self.mmax, args=(y))[0]
                return integral

            vector_integral = np.vectorize(eval_integral)
            probability = phi + coef * vector_integral(m)
            return probability


        else:
            # Returns
            # ------
            # Probability of getting x given the PDF with specified mmin, mmax, sigma, and gamma
            def integrand(x, y):
                return (x**-(self.gamma+1)) * np.exp(-.5*((y-x)/self.sigma)**2)

            coef = (self.gamma/((self.sigma*np.sqrt(2*np.pi)) *
                                ((self.mmin**-self.gamma) -
                                 (self.mmax**-self.gamma))))

            def Integral(y):
                I = quad(integrand, self.mmin, self.mmax, args=(y))[0]
                return I

            vector_I = np.vectorize(Integral)
            return  coef * vector_I(m)


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

    def __init__(self, mmin, mmax, gamma):
        self.mmin = mmin
        self.mmax = mmax
        self.gamma = gamma

    def __call__(self, m, integral_form=False):
        m = np.asarray(m)
        if self.mmax<self.mmin:
            raise ValueError('mmax must be greater than mmin')
        if integral_form:
            # Returns
            # -------
            # Probability that m < x for the given CDF with specified mmin, mmax, sigma, and gamma
            # True for L<=x
            pdf = ((self.mmin**-self.gamma -
                    np.power(m, -self.gamma))/self.mmin**-self.gamma -
                   self.mmax**-self.gamma)
            return_value = (pdf * ((m > self.mmin) & (m < self.mmax)) + 1.0 *
                            (m >= self.mmax) + 0 * (m <self.mmin))
            return return_value

        else:
            # Returns
            # ------
            # Probability of getting x given the PDF with specified mmin, mmax, and gamma
            # Answers it gives are true from mmin<=x<=mmax
            cdf = (self.gamma*np.power(m,
                                       -(self.gamma+1))/(self.mmin**-self.gamma
                                                         -
                                                         self.mmax**-self.gamma))
            return_value = (cdf * ((m > self.mmin) & (m < self.mmax)) + 0 *
                            (m > self.mmax) + 0 * (m < self.mmin))
            return return_value
