"""
Various codes to work with the initial mass function
"""
from __future__ import print_function
import numpy as np
import types # I use typechecking.  Is there a better way to do this?  (see inverse_imf below)
import scipy.integrate
import warnings
from astropy.extern.six import iteritems

class MassFunction(object):
    """
    Generic Mass Function class
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

    def integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass function over some range
        """
        return scipy.integrate.quad(self, mlow, mhigh, **kwargs)

    def m_integrate(self, mlow, mhigh, **kwargs):
        """
        Integrate the mass-weighted mass function over some range (this tells
        you the fraction of mass in the specified range)
        """
        def mform(x):
            return self(x, integral_form=True)
        return scipy.integrate.quad(mform, mlow, mhigh, **kwargs)

    def log_integrate(self, mlow, mhigh, **kwargs):
        def logform(x):
            return self(x) / x
        return scipy.integrate.quad(logform, mlow, mhigh, **kwargs)

    def normalize(self, log=False, **kwargs):
        """
        Set self.normfactor such that the integral of the function over the
        range (mlow, mhigh) = 1
        """
        if log:
            integral = self.log_integrate(self.mlow, self.mhigh, **kwargs)
        else:
            integral = self.integrate(self.mlow, self.mhigh, **kwargs)
        self.normfactor = 1./integral[0]


class Salpeter(MassFunction):

    def __init__(self, alpha=2.35):
        """
        Create a default Salpeter mass function, i.e. a power-law mass function
        the Salpeter 1955 IMF: dn/dm ~ m^-2.35
        """
        self.alpha = alpha

    def __call__(self, m, integral_form=False):
        if integral_form:
            return m**(-(self.alpha - 1))
        else:
            return m**(-self.alpha)


# three codes for dn/dlog(m)
salpeter = Salpeter()

class BrokenPowerLaw(MassFunction):
    def __init__(self, breaks, mmin, mmax):
        self.breaks = breaks
        self.normalization = self.integrate(mmin, mmax)[0]

    def __call__(self, m, integral_form=False):
        zeta = 0
        b_low = 0
        alp_low = 0
        for ii,b in enumerate(self.breaks):
            if integral_form:
                alp = self.breaks[b] - 1
            else:
                alp = self.breaks[b]
            if b == 'last':
                zeta += m**(-alp) * (b_low**(-alp+alp_low)) * (m>b_low)
            else:
                mask = ((m<b)*(m>b_low))
                zeta += m**(-alp) * (b**(-alp+alp_low)) *mask
                alp_low = alp
                b_low = b

        if hasattr(self,'normalization'):
            return zeta/self.normalization
        else:
            return zeta

#kroupa = BrokenPowerLaw(breaks={0.08:-0.3, 0.5:1.3, 'last':2.3},mmin=0.03,mmax=120)

class Kroupa(MassFunction):
    def __init__(self, mmin=0.03):
        """
        """
        self.mmin = mmin

    def __call__(self, m, p1=0.3, p2=1.3, p3=2.3, break1=0.08, break2=0.5,
                 mmin=None, mmax=120, integral_form=False):
        """
        Kroupa 2001 IMF (http://arxiv.org/abs/astro-ph/0009005, http://adsabs.harvard.edu/abs/2001MNRAS.322..231K)

        Parameters
        ----------
        m : float array
            The mass at which to evaluate the function (Msun)
        p1,p2,p3 : floats
            The power-law slopes of the different segments of the IMF
        break1,break2 : floats
            The mass breakpoints at which to use the different power laws
        mmin : float or None
            The minimum mass of the MF.  Defaults to using the class' mmin, but
            can be overridden
        """

        m = np.array(m)

        mmin = mmin if mmin is not None else self.mmin

        binv = ((break1**(-(p1-1)) - mmin**(-(p1-1)))/(1-p1) +
                (break2**(-(p2-1)) - break1**(-(p2-1))) * (break1**(p2-p1))/(1-p2) +
                (- break2**(-(p3-1))) * (break1**(p2-p1)) * (break2**(p3-p2))/(1-p3))
        b = 1./binv
        c = b * break1**(p2-p1)
        d = c * break2**(p3-p2)

        zeta = (b*(m**(-(p1))) * (m<break1) +
                c*(m**(-(p2))) * (m>=break1) * (m<break2) +
                d*(m**(-(p3))) * (m>=break2))

        if integral_form:
            return zeta * m
        else:
            return zeta

kroupa = Kroupa()

def lognormal(m, offset=0.22, width=0.57, scale=0.86):
    """
    A lognormal IMF.  The default parameters correspond to the Chabrier IMF
    """
    return scale * np.exp(-1*(np.log10(m)-np.log10(offset))**2/(2*width**2))

def chabrier(m, integral_form=False):
    """
    Chabrier 2003 IMF
    http://adsabs.harvard.edu/abs/2003PASP..115..763C
    (only valid for m < 1 msun)

    not sure which of these to use...

    integral is NOT IMPLEMENTED
    """
    if integral_form:
        # ...same as kroupa?  This might not be right...
        warnings.warn("I don't know if this integral is correct.  It's implemented very naively.")
        raise NotImplementedError("Chabrier integral NOT IMPLEMENTED")
        return lognormal(m)
        #http://stats.stackexchange.com/questions/9501/is-it-possible-to-analytically-integrate-x-multiplied-by-the-lognormal-probabi
        #alpha = 

    # This system MF can be parameterized by the same type of lognormal form as
    # the single MF (eq. [17]), with the same normalization at 1 Msun, with the
    # coefficients (Chabrier 2003)
    return lognormal(m)
    #return 0.86 * np.exp(-1*(np.log10(m)-np.log10(0.22))**2/(2*0.57**2))
    # This analytic form for the disk MF for single objects below 1 Msun, within these uncertainties, is given by the following lognormal form (Chabrier 2003):
    #return 0.158 * np.exp(-1*(np.log10(m)-np.log10(0.08))**2/(2*0.69**2))

class Chabrier(MassFunction):
    def __call__(self, mass, integral_form=False):
        return chabrier(mass, integral_form=integral_form)

class Chabrier2005(MassFunction):
    """
    Chabrier 2005 IMF as expressed by McKee & Offner 2010

    The logarithmic integral is normalized

    >>> scipy.integrate.quad(lambda x: imf.imf.Chabrier2005()(x) / x, 0.033, 3.0)
        (1.0034751070852832, 1.237415792054719e-08)
    """
    def __init__(self, mlow=0.033, mmid=1.0, mhigh=3.0, psi1=0.35, psi2=0.16,
                 width=0.55, center=0.2, salpeterslope=1.35):
        """
        """
        self.mlow = mlow
        self.mmid = mmid
        self.mhigh = mhigh
        # psi1 and psi2 are technically derived as normalizations...
        self.psi1 = psi1
        self.psi2 = psi2
        self.width = width
        self.center = center
        self.salpeterslope = salpeterslope

        self.normfactor = 1

    def __call__(self, mass, integral_form=False):
        lower = np.array(mass < self.mmid).astype('bool')
        result = self.psi1 * np.exp(-(np.log10(mass)-np.log10(self.center))**2/(2*self.width**2)) * lower
        result += self.psi2 * mass**-self.salpeterslope * (~lower)
        if integral_form:
            return result * mass * self.normfactor
        else:
            return result * self.normfactor


def schechter(m,A=1,beta=2,m0=100, integral=False):
    """
    A Schechter function with arbitrary defaults
    (integral may not be correct - exponent hasn't been dealt with at all)

    $$ A m^{-\\beta} e^{-m/m_0} $$

    Parameters
    ----------
        m : np.ndarray
            List of masses for which to compute the Schechter function
        A : float
            Arbitrary amplitude of the Schechter function
        beta : float
            Power law exponent
        m0 : float
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
        m : np.ndarray
            List of masses for which to compute the Schechter function
        m1 : float
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
    def schechter_cdf(m,A=1,beta=2,m0=100,mmin=10,mmax=None,npts=1e4):
        """
        Return the CDF value of a given mass for a set mmin,mmax
        mmax will default to 10 m0 if not specified

        Analytic integral of the Schechter function:
        http://www.wolframalpha.com/input/?i=integral%28x^-a+exp%28-x%2Fm%29+dx%29
        """
        if mmax is None:
            mmax = 10*m0

        # integrate the CDF from the minimum to maximum
        # undefined posint = -m0 * mmax**-beta * (mmax/m0)**beta * scipy.special.gammainc(1-beta, mmax/m0)
        # undefined negint = -m0 * mmin**-beta * (mmin/m0)**beta * scipy.special.gammainc(1-beta, mmin/m0)
        posint = -mmax**(1-beta) * scipy.special.expn(beta, mmax/m0)
        negint = -mmin**(1-beta) * scipy.special.expn(beta, mmin/m0)
        tot = posint-negint

        # normalize by the integral
        # undefined ret = (-m0 * m**-beta * (m/m0)**beta * scipy.special.gammainc(1-beta, m/m0)) / tot
        ret = (-m**(1-beta) * scipy.special.expn(beta, m/m0) - negint)/ tot

        return ret

    def sh_cdf_func(**kwargs):
        return lambda x: schechter_cdf(x, **kwargs)
except ImportError:
    pass




#def schechter_inv(m):
#    """
#    Return p(m)
#    """
#    return scipy.interpolate.interp1d(shfun,arange(.1,20,.01),bounds_error=False,fill_value=20.)

def integrate(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax = (bins[:-1]+bins[1:])/2.
    integral = (bins[1:]-bins[:-1]) * (fn(bins[:-1])+fn(bins[1:])) / 2.

    return xax,integral

def m_integrate(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax = (bins[:-1]+bins[1:])/2.
    integral = xax*(bins[1:]-bins[:-1]) * (fn(bins[:-1])+fn(bins[1:])) / 2.

    return xax,integral

def cumint(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax,integral = integrate(fn,bins)
    return integral.cumsum() / integral.sum()

def m_cumint(fn=kroupa, bins=np.logspace(-2,2,500)):
    xax,integral = m_integrate(fn,bins)
    return integral.cumsum() / integral.sum()

massfunctions = {'kroupa':kroupa, 'salpeter':salpeter, 'chabrier':chabrier, 'schechter':schechter,'modified_schechter':modified_schechter}
reverse_mf_dict = {v:k for k,v in iteritems(massfunctions)}
# salpeter and schechter selections are arbitrary
mostcommonmass = {'kroupa':0.08, 'salpeter':0.01, 'chabrier':0.23, 'schecter':0.01,'modified_schechter':0.01}
expectedmass_cache = {}

def get_massfunc(massfunc):
    if isinstance(massfunc, types.FunctionType) or hasattr(massfunc,'__call__'):
        return massfunc
    elif type(massfunc) is str:
        return massfunctions[massfunc]
    else:
        raise ValueError("massfunc must either be a string in the set %s or a function" % (",".join(massfunctions.keys())))

def get_massfunc_name(massfunc):
    if massfunc in reverse_mf_dict:
        return reverse_mf_dict[massfunc]
    elif type(massfunc) is str:
        return massfunc
    elif hasattr(massfunc,'__name__'):
        return massfunc.__name__
    else:
        raise ValueError("invalid mass function")

def inverse_imf(p, nbins=1000, mmin=0.03, mmax=120, massfunc='kroupa', **kwargs):
    """
    Inverse mass function

    massfunc can be 'kroupa', 'chabrier', 'salpeter', 'schechter', or a function
    """

    masses = np.logspace(np.log10(mmin),np.log10(mmax),nbins)
    mf = get_massfunc(massfunc)(masses, integral_form=True, **kwargs)
    mfcum = mf.cumsum()
    mfcum /= mfcum.max() # normalize to sum (cdf)

    return np.interp(p, mfcum, masses)

def make_cluster(mcluster, massfunc='kroupa', verbose=False, silent=False,
                 tolerance=0.0, stop_criterion='nearest', mmax=120, **kwargs):
    """
    Sample from an IMF to make a cluster.  Returns the masses of all stars in the cluster

    massfunc must be a string
    tolerance is how close the cluster mass must be to the requested mass.
    If the last star is greater than this tolerance, the total mass will not be within
    tolerance of the requested

    stop criteria can be: 'nearest', 'before', 'after', 'sorted'

    kwargs are passed to `inverse_imf`
    """

    # use most common mass to guess needed number of samples
    #nsamp = mcluster / mostcommonmass[get_massfunc_name(massfunc)]
    #masses = inverse_imf(np.random.random(int(nsamp)), massfunc=massfunc, **kwargs)

    #mtot = masses.sum()
    #if verbose:
    #    print(("%i samples yielded a cluster mass of %g (%g requested)" %
    #          (nsamp,mtot,mcluster)))

    if (massfunc, get_massfunc(massfunc).mmin, mmax) in expectedmass_cache:
        expected_mass = expectedmass_cache[(massfunc,
                                            get_massfunc(massfunc).mmin, mmax)]
    else:
        expected_mass = get_massfunc(massfunc).m_integrate(get_massfunc(massfunc).mmin,
                                                           mmax)[0]
        expectedmass_cache[(massfunc, get_massfunc(massfunc).mmin, mmax)] = expected_mass

    if verbose:
        print("Expected mass is {0:0.3f}".format(expected_mass))

    mtot = 0
    masses = []

    while mtot < mcluster + tolerance:
        # at least 1 sample, but potentially many more
        nsamp = np.ceil((mcluster-mtot) / expected_mass)
        newmasses = inverse_imf(np.random.random(int(nsamp)), massfunc=massfunc, **kwargs)
        masses = np.concatenate([masses,newmasses])
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
                print("Selected the first %i out of %i masses to get %g total" % (last_ind,len(mcum),mtot))
            # force the break, because some stopping criteria can push mtot < mcluster
            break

    if not silent:
        print("Total cluster mass is %g (limit was %g)" % (mtot,mcluster))

    return masses

# Vacca Garmany Shull log(lyman continuum) parameters
# Power-law extrapolated from 18 to 8 and from 50 to 150
# (using, e.g., ",".join(["%0.2f" % p for p in polyval(polyfit(log10(vgsmass[:5]),vgslogq[:5],1),log10(linspace(50,150,6)))[::-1]])
# where vgsmass does *not* include the extrapolated values)
vgsmass = [150.,  130.,  110.,   90.,   70.,  51.3,44.2,41.0,38.1,35.5,33.1,30.8,28.8,26.9,25.1,23.6,22.1,20.8,19.5,18.4,18.,  16.,  14.,  12.,  10.,   8.][::-1]
vgslogq = [50.51,50.34,50.13,49.88,49.57,49.18,48.99,48.90,48.81,48.72,48.61,48.49,48.34,48.16,47.92,47.63,47.25,46.77,46.23,45.69,45.58,44.65,43.60,42.39,40.96,39.21][::-1]

# non-extrapolated
vgsM    = [51.3,44.2,41.0,38.1,35.5,33.1,30.8,28.8,26.9,25.1,23.6,22.1,20.8,19.5,18.4]
vgslogL = [6.154,6.046,5.991,5.934,5.876,5.817,5.756,5.695,5.631,5.566,5.499,5.431,5.360,5.287,5.211]
vgslogQ = [49.18,48.99,48.90,48.81,48.72,48.61,48.49,48.34,48.16,47.92,47.63,47.25,46.77,46.23,45.69]
# mass extrapolated
vgsMe = np.concatenate([
    np.linspace(0.03,0.43,100),
    np.linspace(0.43,2,100),
    np.linspace(2,20,100),
    vgsM[::-1],
    np.linspace(50,150,100)])
# log luminosity extrapolated
vgslogLe = np.concatenate([
    np.log10(0.23*np.linspace(0.03,0.43,100)**2.3),
    np.log10(np.linspace(0.43,2,100)**4),
    np.log10(1.5*np.linspace(2,20,100)**3.5),
    vgslogL[::-1],
    np.polyval(np.polyfit(np.log10(vgsM)[:3],vgslogL[:3],1),np.log10(np.linspace(50,150,100)))])
# log Q (lyman continuum) extrapolated
vgslogQe = np.concatenate([
    np.zeros(100), # 0.03-0.43 solar mass stars produce 0 LyC photons
    np.zeros(100), # 0.43-2.0 solar mass stars produce 0 LyC photons
    np.polyval(np.polyfit(np.log10(vgsM)[-3:],vgslogQ[-3:],1),np.log10(np.linspace(8,18.4,100))),
    vgslogQ[::-1],
    np.polyval(np.polyfit(np.log10(vgsM)[:3],vgslogQ[:3],1),np.log10(np.linspace(50,150,100)))])

def lum_of_star(mass):
    """
    Determine total luminosity of a star given its mass
    Uses the Vacca, Garmany, Shull 1996 Table 5 Log Q and Mspec parameters

    returns LogL in solar luminosities
    **WARNING** Extrapolates for M not in [18.4,50] msun

    http://en.wikipedia.org/wiki/Mass%E2%80%93luminosity_relation
    """
    return np.interp(mass, vgsMe, vgslogLe)

def lum_of_cluster(masses):
    """
    Determine the log of the integrated luminosity of a cluster
    Only M>=8msun count

    masses is a list or array of masses.
    """
    #if max(masses) < 8: return 0
    logL = lum_of_star(masses) #[masses >= 8])
    logLtot = np.log10( (10**logL).sum() )
    return logLtot

def lyc_of_star(mass):
    """
    Determine lyman continuum luminosity of a star given its mass
    Uses the Vacca, Garmany, Shull 1996 Table 5 Log Q and Mspec parameters

    returns LogQ
    """

    return np.interp(mass, vgsMe, vgslogQe)

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
             100 :(150,175,255),
              50 :(157,180,255),
              20 :(162,185,255),
              10 :(167,188,255),
               8 :(170,191,255),
               6 :(175,195,255),
             2.2 :(186,204,255),
             2.0 :(192,209,255),
            1.86 :(202,216,255),
             1.6 :(228,232,255),
             1.5 :(237,238,255),
             1.3 :(251,248,255),
             1.2 :(255,249,249),
               1 :(255,245,236),
            0.95 :(255,244,232),
            0.90 :(255,241,223),
            0.85 :(255,235,209),
            0.70 :(255,215,174),
            0.60 :(255,198,144),
            0.50 :(255,190,127),
            0.40 :(255,187,123),
            0.35 :(255,187,123),
            0.30 :(255,177,113),
            0.20 :(255,107,63),
            0.10 :(155,57,33),
            0.10 :(155,57,33),
            0.003 :(105,27,0),
            }

    keys = sorted(mcolor.keys())

    reds,greens,blues = zip(*[mcolor[k] for k in keys])

    r = np.interp(mass,keys,reds)
    g = np.interp(mass,keys,greens)
    b = np.interp(mass,keys,blues)

    if outtype == int:
        return (r,g,b)
    elif outtype == float:
        return (r/255.,g/255.,b/255.)
    else:
        raise NotImplementedError

def color_of_cluster(cluster, colorfunc=color_from_mass):
    colors       = np.array([colorfunc(m) for m in cluster])
    luminosities = 10**np.array([lum_of_star(m) for m in cluster])
    mean_color = (colors*luminosities[:,None]).sum(axis=0)/luminosities.sum()
    return mean_color

def coolplot(clustermass, massfunc='kroupa', **kwargs):
    cluster = make_cluster(clustermass, massfunc=massfunc, **kwargs)
    colors = [color_from_mass(m) for m in cluster]
    massfunc = get_massfunc(massfunc)
    maxmass = cluster.max()
    pmin = massfunc(maxmass)
    yax = [np.random.rand()*(np.log10(massfunc(m))-np.log10(pmin)) + np.log10(pmin) for m in cluster]

    return cluster,yax,colors

    # import pylab as pl
    # pl.scatter(cluster, yax, c=colors, s=np.log10(cluster)*5)
