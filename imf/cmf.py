from __future__ import print_function
import numpy as np
from astropy import units as u
from astropy import constants
import scipy
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid, quad
from scipy.interpolate import PchipInterpolator

from . import imf
from .distributions import Distribution
from .imf import MassFunction

import warnings


class PN_CMF(MassFunction):
    """
    Core mass function derived from a population generated according to
    `Padoan/Nordlund (2011) <https://doi.org/10.1088/2041-8205/741/1/L22>`__.
    The base CMF consists of cores which have not or will not collapse
    and are visible at the crossing time of the parent cloud. Uses
    interpolation for evaluation and sampling.

    Parameters
    ----------
    mmin: float
        Minimum permissible core mass (default = 0.01)
    mmax: float
        Maximum permissible core mass (default = 120)
    T0: :math:`{\\rm K}` or equivalent
        Mean temperature of the parent cloud (default = 10 K)
    L0: :math:`{\\rm pc}` or equivalent
        Cloud size (default = 10 pc)
    v0: :math:`{\\rm km \\, s}^{-1}` or equivalent
        RMS velocity of the parent cloud at R = 1 pc
        (default = 0.8 km / s)
    rho0: :math:`{\\rm g \\, cm}^{-3}` or equivalent
        Mean mass density of gas in the parent cloud
        (default = 2e-21 g / cm3)
    massfunc: MassFunction
        Mass function determining the final masses of cores.
        Defaults to a ``Salpeter`` instance with the provided mass range.
    sampling: str
        Method to use for sampling the provided mass function.
        Accepts ``"random"`` or ``"optimal"``. If ``None``, 
        defaults to random sampling.
    stop_criterion: str
        Stop criterion for random sampling; accepts the same arguments
        as ``imf.make_cluster``. If ``None``, defaults to ``"nearest"``.
    eff: float
        Efficiency of core formation. The core mass budget is
        eff * cloud mass (default = 0.26)
    beta: float
        Ratio of gas to magnetic pressure in postshock gas (default = 0.4)
    b: float
        Spectral index of the turbulence power spectrum (default = 1.8)
    T_mean: :math:`{\\rm K}` or equivalent
        Mean core temperature (default = 7 K)
    mu: float
        Mean molecular weight of gas (default = 2.33)
    bins: int or str
        Number of histogram bins (in log space) or type of estimator to
        use for bin width; accepts the same arguments as `NumPy histograms
        <https://numpy.org/doc/stable/reference/generated/numpy.histogram.html>`__ 
        (default = ``'auto'``)
    """

    default_mmin = 0.01
    default_mmax = 120

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 T0=10*u.K, L0=10*u.pc,
                 v0=4.9*u.km/u.s, rho0=2e-21*u.g/u.cm**3,
                 massfunc=None, sampling=None, stop_criterion=None,
                 eff=0.26, beta=0.4, b=1.8,
                 T_mean=7*u.K, mu=2.33, bins='auto'):

        if mmin is None:
            mmin = default_mmin
        if mmax is None:
            mmax = default_mmax
        if sampling is None:
            sampling = 'random'
        if stop_criterion is None:
            stop_criterion = 'nearest'

        self._tcross = (L0 / v0).to(u.Myr)
        m0 = (4 / 3 * np.pi * L0**3 * rho0).to(u.M_sun)  # total cloud mass

        # Mach number defined on largest scale (assumed)
        MS0 = (v0 / ((constants.k_B * T0 / (mu * constants.m_p))**0.5).to(u.km/u.s)).value

        # implementing eqn 1
        sigma_rho = MS0 / np.sqrt((1 + beta**-1)) / 2.  # stdev of density [eqn 3]
        s = np.sqrt(np.log(1 + sigma_rho**2))  # lognormal shape

        n0 = (rho0 / constants.m_p / mu).to(u.cm**-3).value
        self._massfunc = imf.Salpeter(mmin=mmin,mmax=mmax) if massfunc is None else massfunc
        self._maccr = imf.make_cluster(mcluster=(m0*eff).to(u.M_sun).value,
                                       massfunc=self.massfunc,
                                       sampling=sampling,
                                       stop_criterion=stop_criterion,
                                       silent=True) * u.M_sun

        # loc is s**2/2 instead of -s**2/2 because this is the _mass-weighted_ log-normal density PDF
        # (this is only weakly hinted at in PN11 by the words "converted to mass fraction" just before eqn 1)
        # see e.g. Hopkins 2013 eqn 4 for the mass-weighted definitions
        ln_rho = scipy.stats.norm.rvs(loc=s**2/2,
                                      scale=s,
                                      size=len(self.maccr))
        x = np.exp(ln_rho)
        self._rho = x * rho0  # densities around each core

        self._birthdays = np.random.random(len(self.maccr)) * self.tcross  # core birthdays (assuming flat formation over crossing time)

        a = (b - 1) / 2
        self._taccr = (self.tcross * sigma_rho**((4 - 4 * a) / (3 - 2 * a)) *
                       (self.maccr / m0)**((1 - a) / (3 - 2 * a))).to(u.Myr)

        c_s = ((constants.k_B * T_mean / (mu * constants.m_p))**0.5).to(u.km/u.s)
        self._mbe = (1.182 * c_s**3 / (constants.G**1.5 * self.rho**0.5)).to(u.M_sun)

        self.distr = dist_pn(mmin, mmax,
                             self.maccr, self.taccr, self.mbe,
                             self.rho, self.tcross, self.birthdays,
                             bins)

        self.distr._calculate()
        self.distr._update_functions()

        self.normfactor = 1

    def __call__(self, m,
                 integral_form=False):

        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def get_masses(self, tnow=1, cores='prestellar', visible_only=True):
        """
        Returns the masses of cores meeting the specifications (in 
        :math:`M_\odot`). ``'tnow'``, ``'visible_only'``, and ``'cores'``
        accept the same arguments as the setter methods for time, 
        visibility, and core type respectively.
        """
        return self.distr._core_masses(tnow, visible_only, cores)

    def set_time(self, x):
        """
        Sets the time at which the CMF is sampled. Accepts ints 
        and floats;  units are in terms of cloud crossing time.
        """
        self.distr._time = x
        self.distr._calculate()
        self.distr._update_functions()

    def set_visible(self, x):
        """
        Sets whether the CMF includes only cores likely to be visible
        or all sampled cores. Accepts ``True`` or ``False``.
        """
        self.distr._visible = bool(x)
        self.distr._calculate()
        self.distr._update_functions()

    def set_cores(self, x):
        """
        Sets the type of cores included in the CMF. Accepts ``'prestellar'``,
        ``'stellar'``, ``'transient'``, ``'nonstellar'``, or ``'all'``.
        """
        if x not in ('stellar', 'prestellar', 'transient', 'nonstellar', 'all'):
            raise ValueError("Allowed values are 'prestellar', 'stellar', 'transient', 'nonstellar', or 'all'")

        self.distr._cores = x
        self.distr._update_functions()

    @property
    def mtot(self):
        """
        Total mass of all cores (in :math:`M_\odot`)
        """
        return sum(self.maccr)

    @property
    def mmin(self):
        return self.distr.m1

    @property
    def mmax(self):
        return self.distr.m2

    @property
    def tcross(self):
        return self._tcross

    @property
    def maccr(self):
        return self._maccr

    @property
    def rho(self):
        return self._rho

    @property
    def birthdays(self):
        return self._birthdays

    @property
    def taccr(self):
        return self._taccr

    @property
    def mbe(self):
        return self._mbe

    @property
    def massfunc(self):
        return self._massfunc

    @property
    def time(self):
        return self.distr.time

    @property
    def visible(self):
        return self.distr.visible

    @property
    def cores(self):
        return self.distr.cores


class dist_pn(Distribution):
    """
    Manages the PDF/CDF for a population of cores generated
    through `Padoan/Nordlund (2011) <https://doi.org/10.1088/2041-8205/741/1/L22>`__
    turbulent fragmentation.
    """

    def __init__(self, cmin, cmax,
                 maccr, taccr, mbe, rho,
                 tcross, birthdays, bins):

        self.cmin = cmin
        self.cmax = cmax
        self.maccr = maccr
        self.taccr = taccr
        self.tbe = (self.taccr * (self.maccr / mbe)**(-1/3.)).to(u.s)
        self.tff = ((3 * np.pi / (32 * constants.G * rho))**0.5).to(u.s)
        self.mmax = (self.maccr * ((self.tbe + self.tff) / self.taccr)**3).to(u.M_sun)
        self.belowBE = self.maccr < mbe
        self.bins = bins

        self.tcross = tcross
        self.birthdays = birthdays
        self._time = 1
        self._visible = True
        self._cores = 'nonstellar'

        keys = ['prestellar', 'stellar', 'transient', 'nonstellar', 'all']
        self._func_dict = {key: None for key in keys}

    def _core_masses(self, tnow, visible, cores):
        """
        Returns the current masses of cores meeting particular
        conditions
        """
        age = tnow * self.tcross - self.birthdays
        isBorn = age > 0
        isTransient = self.belowBE #cores which will never collapse
        isPrestellar = np.logical_and(age < self.tbe + self.tff, ~isTransient) #cores which will collapse but haven't
        isStellar = np.logical_and(age >= self.tbe + self.tff, ~isTransient) #cores which have collapsed
        isForming = age < self.taccr
        
        mnow = ((age / self.taccr)**3 * self.maccr).to(u.M_sun)
        mnow[mnow > self.maccr] = self.maccr[mnow > self.maccr]  # cap current mass to final sampled mass

        cut = np.copy(isBorn)
        
        if cores == 'prestellar':
            cut = np.logical_and(cut, isPrestellar)
        elif cores == 'stellar':
            cut = np.logical_and(cut, isStellar)
        elif cores == 'transient':
            cut = np.logical_and(cut, isTransient)
            if visible:
                cut = np.logical_and(cut, isForming) # transient cores are modeled as only visible during formation
        elif cores == 'nonstellar':
            trans_cut = np.logical_and(isTransient, isForming) if visible else isTransient
            cut = np.logical_and(cut, np.logical_or(isPrestellar, trans_cut))
        elif cores == 'all':
            collapse_cut = np.logical_or(isPrestellar, isStellar)
            trans_cut = np.logical_and(isTransient, isForming) if visible else isTransient
            cut = np.logical_and(cut, np.logical_or(collapse_cut, trans_cut))

        core_masses = mnow[cut]
        core_masses = core_masses[core_masses.value > self.cmin]  # only consider cores above minimum provided core mass
        return core_masses

    def _calculate(self):
        """
        Constructs the PDF/CDF/PPF from the generated core
        population
        """
        keys = ['prestellar', 'stellar', 'transient', 'nonstellar', 'all']

        for key in keys:
            core_masses = self._core_masses(self.time, self.visible, key)

            edges = 10**np.histogram_bin_edges(np.log10(core_masses.value), bins=self.bins) * u.M_sun
            N, edges = np.histogram(core_masses, bins=edges)
            centers = ((edges[:-1] + edges[1:]) / 2).value

            # construct function dictionary
            try:
                pdf = N / centers
                norm = np.trapezoid(pdf, x=centers)
                pdf /= norm
                cdf = cumulative_trapezoid(pdf, centers, initial=0)
                cdf_unq, indices = np.unique(cdf, return_index=True)

                functions = [PchipInterpolator(centers, pdf),
                             PchipInterpolator(centers, cdf),
                             PchipInterpolator(cdf_unq, centers[indices])]
                self._func_dict[key] = functions
            except(ValueError):
                warnings.warn(f"Insufficient cores of type '{key}' to construct a function")
                self._func_dict[key] = None

    def pdf(self, x):
        return self._pdf(x, extrapolate=False)

    def cdf(self, x):
        return self._cdf(x, extrapolate=False)

    def ppf(self, x):
        return self._ppf(x, extrapolate=False)

    def rvs(self, N):
        samp = np.random.uniform(min(self._ppf.x), max(self._ppf.x), size=N)
        return self.ppf(samp)

    def _pick_functions(self, cores):
        functions = self._func_dict[cores]
        if functions is not None:
            return functions
        else:
            return None

    def _update_functions(self):
        functions = self._pick_functions(self.cores)
        if functions is not None:
            self._pdf, self._cdf, self._ppf = functions[0], functions[1], functions[2]
            self.m1 = self._pdf.x[0]
            self.m2 = self._pdf.x[-1]
        else:
            self._pdf = None
            self._cdf = None
            self._ppf = None
            self.m1 = None
            self.m2 = None

    @property
    def time(self):
        return self._time

    @property
    def visible(self):
        return self._visible

    @property
    def cores(self):
        return self._cores


class HC_CMF(MassFunction):
    """
    Generalized core mass function following the formalism of
    Hennebelle/Chabrier (`2008 <https://doi.org/10.1086/589916>`_,
    `2009 <https://doi.org/10.1088/0004-637X/702/2/1428>`_,
    `2013 <https://doi.org/10.1088/0004-637X/770/2/150>`_). The 
    base CMF is the time-dependent version from the 2013 paper.
    Uses interpolation for evaluation and sampling.

    Parameters
    ----------
    mmin: float
        Minimum permissible core mass (default = 0.01)
    mmax: float
        Maximum permissible core mass (default = 300)
    clump_size: :math:`{\\rm pc}` or equivalent
        Radius of the parent clump (default = 1 pc)
    n_cl: float
        Clump density normalization at 1 pc. Number density is
        n_cl * 1e3 (default = 5)
    mu: float
        Mean molecular weight of gas (default = 2.33)
    Cs0: :math:`{\\rm km \\, s}^{-1}` or equivalent
        Average isothermal sound speed for a cloud with
        number density 1e4 / cm3 (default = 0.2 km / s)
    T0: :math:`{\\rm K}` or equivalent
        Mean temperature of the parent clump. Used to calculate
        sound speed if none is provided (default = 10 K)
    v0: :math:`{\\rm km \\, s}^{-1}` or equivalent
        RMS velocity of the parent clump at R = 1 pc
        (default = 0.8 km / s)
    eta: float
        Exponent governing the behavior of dispersion velocity with scale
        (default = ``None``)
    n_pow: float
        Index of 3D velocity power spectrum. Used to derive eta
        if no eta is provided (default = 3.8)
    b_forcing: float
        Forcing parameter of turbulence (default = 0.4)
    eos: str
        String specifying which equation of state to use for gas
        in the parent clump. Accepts ``'isothermal'``, ``'polytropic'``,
        and ``'barotropic'``; see papers for implementation details
        (default = ``'isothermal'``)
    gamma1: float
        Exponent in a non-isothermal EOS. The sole exponent in a
        polytropic case and the lower-density exponent in a
        barotropic case. Only used if eos != ``'isothermal'``
        (default = 0.7)
    gamma2: float
        High-density exponent in a barotropic EOS. Only used if
        eos == ``'barotropic'`` (default = 1.1)
    rho_crit: :math:`{\\rm g \\, cm}^{-3}` or equivalent
        Critical density in a barotropic EOS (i.e. where the piecewise
        halves meet). Only used if eos == ``'barotropic'``
        (default = 1e-18 g / cm3)
    m: float
        Exponent governing the combination of the piecewise components
        of a barotropic EOS; higher = less blending. Only used if
        eos == ``'barotropic'`` (default = 3)
    include_B: bool
        Whether or not to include support from a magnetic field in
        CMF calculation. (default = ``False``)
    B0: :math:`{\\rm gauss}` or equivalent
        Mean magnetic field strength. Only used if include_B is ``True``
        (default = 10 microgauss)
    gammab: float
        Exponent governing the relationship between magnetic field
        strength and gas density. Only used if include_B is ``True``
        (default = 0.1)
    npts: int
        Number of points at which to evaluate the CMF for interpolation
        (default = 200)
    """

    default_mmin = 0.01
    default_mmax = 300

    def __init__(self, mmin=default_mmin, mmax=default_mmax,
                 clump_size=1*u.pc, n_cl=5, mu=2.33,
                 Cs0=0.2*u.km/u.s, T0=10*u.K,
                 v0=0.8*u.km/u.s, eta=None,
                 n_pow=3.8, b_forcing=0.5,
                 eos='isothermal', gamma1=0.7,
                 gamma2=1.1, rho_crit=1e-18*u.g/u.cm**3, m=3,
                 include_B=False, B0=10*u.uG, gammab=0.1,
                 npts=200):

        if eta is None:
            eta = (n_pow - 3) / 2

        eos_types = ['isothermal', 'polytropic', 'barotropic']
        if eos not in eos_types:
            raise ValueError(f'EOS must be one of the following: {eos_types}')
        cs_mod = gamma1 if eos != 'isothermal' else 1

        # scale density according to Larson laws
        n0 = (n_cl * 1e3 * u.cm**-3) / (clump_size.to(u.pc).value)**0.7
        rho0 = (n0 * mu * constants.m_p).to(u.g/u.cm**3)
        self._mcloud = (4 * np.pi / 3 * clump_size**3 * rho0).to(u.M_sun).value

        # scale sound speed according to EOS
        if Cs0 is None:
            Cs0 = np.sqrt(constants.k_B * T0 / mu / constants.m_p)
        Cs = Cs0.to(u.km/u.s) * np.sqrt((n0.value / 1e4)**(cs_mod-1))

        self.distr = dist_hc(mmin, mmax,
                             clump_size, rho0, Cs,
                             v0, eta, b_forcing,
                             eos, gamma1, gamma2, rho_crit, m,
                             include_B, B0, gammab, npts)

        self.set_timedep(True)

        self.normfactor = 1

    def __call__(self, m,
                 integral_form=False):

        if integral_form:
            return self.normfactor * self.distr.cdf(m)
        else:
            return self.normfactor * self.distr.pdf(m)

    def set_timedep(self, x):
        """
        Sets the type of CMF to use. If ``True``, use the time-dependent 
        CMF of HC13; if ``False``, use the time-independent form of HC08/09.
        """
        self.distr.time_dep = bool(x)

    @property
    def mcloud(self):
        r"""
        Total cloud mass (in :math:`M_\odot`)
        """
        return self._mcloud

    @property
    def mmin(self):
        return self.distr.m1

    @property
    def mmax(self):
        return self.distr.m2

    @property
    def clump_size(self):
        return self.distr.clump_size

    @property
    def rho0(self):
        return self.distr.rho0

    @property
    def Cs(self):
        return self.distr.Cs

    @property
    def v0(self):
        return self.distr.v0

    @property
    def eta(self):
        return self.distr.eta

    @property
    def b_forcing(self):
        return self.distr.b_forcing

    @property
    def eos(self):
        return self.distr.eos

    @property
    def gamma1(self):
        return self.distr.gamma1

    @property
    def gamma2(self):
        return self.distr.gamma2

    @property
    def rho_crit(self):
        return self.distr.rho_crit

    @property
    def rho_crit(self):
        return self.distr.rho_crit

    @property
    def B0(self):
        return self.distr.B0

    @property
    def gammab(self):
        return self.distr.gammab

    @property
    def time_dep(self):
        return self.distr.time_dep


class dist_hc(Distribution):
    """
    Manages the PDF/CDF for a CMF generated according to the
    Hennebelle/Chabrier turbulent fragmentation formalism.
    """

    def __init__(self, m1, m2,
                 clump_size, rho0, Cs,
                 v0, eta, b_forcing,
                 eos, gamma1, gamma2, rho_crit, m,
                 include_B, B0, gammab, npts):

        self.m1 = m1
        self.m2 = m2

        self.clump_size = clump_size
        self.rho0 = rho0
        self.Cs = Cs

        self.v0 = v0
        self.eta = eta
        self.b_forcing = b_forcing

        self.eos = eos
        if self.eos != 'isothermal':
            self.gamma1 = gamma1
        if self.eos == 'barotropic':
            self.gamma2 = gamma2
            self.rho_crit = rho_crit
            self.m = m

        self.include_B = include_B
        self.B0 = B0
        self.gammab = gammab

        self.npts = npts
        keys = ['pdf', 'cdf', 'ppf']
        self._func_dict = {key: [] for key in keys}

        self._calculate()

    def _calculate(self):
        """
        Calculates the PDF/CDF/PPF
        """
        # use EOS to set thermal Cs (for Mj/Lj/Mstar)
        cs_mod = 1 if self.eos == 'isothermal' else self.gamma1
        Cs = self.Cs  # * cs_mod #papers suggest gamma should go here, but not in the HC IDL code

        # add support from magnetic field
        mag_coef = 1 if self.include_B else 0
        gauss = u.g**0.5 / u.cm**0.5 / u.s  # define a custom gauss unit to work in cgs
        B0 = self.B0.to(u.G).value * gauss
        Va_sq = (B0**2 / (24 * np.pi * self.rho0) / Cs**2).to(u.dimensionless_unscaled)

        # use EOS/support to define M and dM/dR
        # including D, the thermal and magnetic terms of M
        if self.eos == 'barotropic':
            Kcrit = ((self.rho_crit / self.rho0).decompose())**(self.gamma1-self.gamma2)

            def R_M(R_, M_):
                A = (M_ / R_**3)**((self.gamma1-1)*self.m) + Kcrit**self.m * (M_ / R_**3)**((self.gamma2-1)*self.m)
                return R_ * (A**(1/self.m) + Mstar**2 * R_**(2*self.eta) + mag_coef * Va_sq * (M_ / R_**3)**(2*self.gammab-1)) - M_

            def D_funcs(rho_):
                A = rho_**((self.gamma1-1)*self.m) + Kcrit**self.m * rho_**((self.gamma2-1)*self.m)
                D = A**(1/self.m) + mag_coef * Va_sq * rho_**(2*self.gammab-1)
                dD = (A**(1/self.m-1) *
                      ((self.gamma1 - 1) * rho_**((self.gamma1-1)*self.m-1) +
                       Kcrit**self.m * (self.gamma2 - 1) * rho_**((self.gamma2-1)*self.m-1)) +
                      mag_coef * (2 * self.gammab - 1) * Va_sq * rho_**(2*self.gammab-2))
                return D, dD

        else:
            def R_M(R_, M_):
                return R_ * ((M_ / R_**3)**(cs_mod-1) + Mstar**2 * R_**(2 * self.eta) + mag_coef * Va_sq * (M_ / R_**3)**(2*self.gammab-1)) - M_

            def D_funcs(rho_):
                D = rho_**(cs_mod-1) + mag_coef * Va_sq * rho_**(2*self.gammab-1)
                dD = (cs_mod-1) * rho_**(cs_mod-2) + mag_coef * Va_sq * (2 * self.gammab - 1) * rho_**(2*self.gammab-2)
                return D, dD

        # equations 33/34 (specifying dM/dR) of HC13 hold for all relevant definitions of D
        def dM_dR(M_, R_):
            rho = M_ / R_**3
            D, dD = D_funcs(rho)
            B = D - 3 * rho * dD + (2 * self.eta + 1) * Mstar**2 * R_**(2*self.eta)
            C = 1 - R_**-2 * dD
            return B / C

        # root find for R
        def get_root(M_):
            return root_scalar(R_M, x0=1, args=(M_)).root

        # Jeans mass/length; leading coefficients are as in the IDL code
        aj = np.pi**2.5 / 6
        Mj = (aj * Cs**3 / np.sqrt(constants.G**3 * self.rho0)).to(u.M_sun) * cs_mod**1.5  # in the IDL code, gamma is explicit for Mj/Lj
        Lj = ((np.pi**0.5 / 2)**(1/3) * Cs / np.sqrt(constants.G * self.rho0)).to(u.pc) * cs_mod**0.5
        Li = self.clump_size.to(u.pc) / Lj
        # Mach number
        Mstar = self.v0 / Cs * (Lj.value)**self.eta / np.sqrt(3)
        Mach = np.sqrt(3) * Mstar * Li**self.eta

        # determine maximum possible "core" mass given provided sizescale
        mmax = root_scalar(lambda md, rd: R_M(rd, md), x0=1, args=(Li)).root * Mj
        self.m2 = min(mmax.value, self.m2)

        # set points for interpolation based on allowable mass range
        self._points = np.geomspace(self.m1, self.m2, self.npts)

        Mt = self._points / Mj.value
        Rt = np.vectorize(get_root)(Mt)
        delta = np.log(Mt / Rt**3)

        # calculate variance and CMF dP/dR term
        sig_0 = np.log(1 + self.b_forcing**2 * Mach**2)
        sig_sq = sig_0 * (1 - (Rt / Li)**(2*self.eta))
        dsigma_dR = -self.eta / np.sqrt(sig_sq) * (sig_0 - sig_sq) / Rt
        corr = dsigma_dR / np.sqrt(sig_sq) * (delta + sig_sq / 2)

        # determine maximum possible "core" mass given provided sizescale
        mmax = root_scalar(lambda md, rd: R_M(rd, md), x0=1, args=(Li)).root * Mj

        # calculate unscaled, dimensionless CMF
        dM = dM_dR(Mt, Rt)
        N = (1 / Mt / dM * ((3 / Rt - dM / Mt) + corr) /
             np.sqrt(2 * np.pi * sig_sq) *
             np.exp(-(delta - sig_sq / 2)**2 / 2 / sig_sq))

        # get rid of impossible entries
        N[~np.isfinite(N)] = 0

        # store time-independent PDF
        norm = np.trapezoid(N, x=self._points)
        cdf = cumulative_trapezoid(N/norm, self._points, initial=0)
        cdf_unq, indices = np.unique(cdf, return_index=True)

        self._func_dict['pdf'].append(PchipInterpolator(self._points, N/norm))
        self._func_dict['cdf'].append(PchipInterpolator(self._points, cdf))
        self._func_dict['ppf'].append(PchipInterpolator(cdf_unq, self._points[indices]))

        # store time-dependent PDF
        N *= np.sqrt(np.exp(delta))
        norm = np.trapezoid(N, x=self._points)
        cdf = cumulative_trapezoid(N/norm, self._points, initial=0)
        cdf_unq, indices = np.unique(cdf, return_index=True)

        self._func_dict['pdf'].append(PchipInterpolator(self._points, N/norm))
        self._func_dict['cdf'].append(PchipInterpolator(self._points, cdf))
        self._func_dict['ppf'].append(PchipInterpolator(cdf_unq, self._points[indices]))

    def _pick_function(self, functype, time_dep):
        return self._func_dict[functype][int(time_dep)]

    def _update_functions(self):
        self._pdf = self._pick_function('pdf', self.time_dep)
        self._cdf = self._pick_function('cdf', self.time_dep)
        self._ppf = self._pick_function('ppf', self.time_dep)

    def pdf(self, x):
        return self._pdf(x, extrapolate=False)

    def cdf(self, x):
        return self._cdf(x, extrapolate=False)

    def ppf(self, x):
        return self._ppf(x, extrapolate=False)

    def rvs(self, N):
        samp = np.random.uniform(min(self._ppf.x), max(self._ppf.x), size=N)
        return self.ppf(samp)

    @property
    def time_dep(self):
        return self._time_dep

    @time_dep.setter
    def time_dep(self, x):
        self._time_dep = x
        self._update_functions()
