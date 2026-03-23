import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator
from astropy.table import Table

import os
import warnings
from glob import glob
loc = os.path.dirname(__file__)

from .imf import MassFunction
from .distributions import Distribution

hist_values = {'is': (0, 0, 1.54e-6, 10, 1.5),
               'tc': (0.5, 0.75, 4.9e-6, 0.1, 0.75),
               'ca': (2/3, 1., 6.9e-6, 1e4, 0.5)}


def scaling(history, value=None):
    params = hist_values[history]
    if value is None:
        value = params[3]
    return params[2] * (value / params[3])**params[4]


def get_fname(history, n_comp=1, taper=False):
    prefix = 'taper_' if taper else ''
    if n_comp == 1:
        return prefix + history
    else:
        return prefix + f'2c{history}'


class PLF(MassFunction):
    r"""
    Calculates the Protostellar Luminosity Function (PLF)
    corresponding to a supplied IMF and accretion history
    using the formalism of Offner/McKee (2011).

    Currently not implemented due to algorithmic issues with
    interpolation (see Appendix A of the companion paper,
    Richardson+ 2026).

    Parameters
    ----------
    imf: MassFunction
        The IMF of the emerging stellar population
    lmin: float
        Lower limit to luminosity in $L_\odot$ (default = 0.01)
    lmax: float
        Upper limit to luminosity in $L_\odot$ (default = 100)
    history: str
        Accretion history of stars; accepts 'is' (isothermal sphere),
        'tc' (turbulent core), and 'ca' (competitive accretion)
        (default = 'is')
    f_epi: float
        The fraction of mass accreted in episodic bursts (default = 0.25)
    n: float
        Exponent governing the tapering factor (default = 1)
    tau: float
        Time constant for accelerating star formation, in Myr (default = 1)
    n_lpts: int
        Number of points at which to evaluate the function for interpolation
        (default = 200)
    n_mpts: int
        Number of points at which to evaluate the luminosity-to-mass
        mapping for interpolation (default = 50)
    proto_trackdir: str
        The location of the files containing the protostellar evolutionary
        track information. Points to files in imf generated using a
        modified Klassen+ (2012) code by default, but will accept files
        with the same format
    """

    def __init__(self, imf,
                 lmin=0.01, lmax=100,
                 history='is',
                 f_epi=0.25,
                 n=1, tau=1,
                 n_lpts=None, n_mpts=None,
                 proto_trackdir=f'{loc}/data/K12_protoev_tables',
                 **kwargs
                 ):

        raise NotImplementedError('PLFs are currently disabled due to interpolation issues')

        self._imf = imf
        self.imf.normalize()

        self._lmin = lmin
        self._lmax = lmax

        self.history = history

        self._f_epi = f_epi

        self._n = n
        self._tau = tau

        self._trackdir = proto_trackdir
        self._interps = self._make_interps(self._trackdir, **kwargs)

        self.distr = dist_plf(self.imf, self.lmin, self.lmax,
                              self.j_exp, self.jf_exp, self.scale_value,
                              self.n, self.tau,
                              self.interps,
                              n_lpts, n_mpts)
        self.distr._taper = False
        self.distr._accelerating = False
        self.distr._update_functions()

        self.normfactor = 1

    def __call__(self, lum,
                 integral_form=False
                 ):

        if integral_form:
            return self.normfactor * self.distr.cdf(lum)
        else:
            return self.normfactor * self.distr.pdf(lum)

    def _make_interps(self, trackdir, **kwargs):
        """
        Set up interpolators for L and dL/dm by reading
        evolutionary track data
        """
        interps = []
        for taper in (False, True):
            table = Table.read(f'{trackdir}/{get_fname(self.history, taper=taper)}_val_table.fits')
            l_tot = table['lint'] + (1 - self.f_epi) * table['lacc']
            ok = np.isfinite(l_tot)
            mf_unq = np.unique(table['mf'])
            mf_min, mf_max = np.min(mf_unq), np.max(mf_unq)
            m_unq = np.unique(table['m'])
            l_data = np.copy(l_tot)
            l_data[~ok] = 0

            l_data = l_data.reshape(len(mf_unq), len(m_unq))
            l_interp = RegularGridInterpolator((mf_unq, m_unq), l_data, method='pchip')

            grad = np.gradient(l_data, m_unq, axis=1)
            grad_interp = RegularGridInterpolator((mf_unq, m_unq), grad, method='pchip')

            interps.extend([l_interp, grad_interp])

        if np.logical_or(mf_min < self.imf.mmin, mf_max > self.imf.mmax):
            warnings.warn('IMF mmin/mmax is outside the range of values covered by evolutionary tracks; '
                          'stars outside this range will not contribute to the PLF')
        return interps

    def tf(self, mf, taper=False):
        """
        Returns the expected formation time of a star with
        final mass mf following the accretion history
        underlying the PMF
        """
        return self.distr._tf(mf, taper)

    def average_time(self, taper=False, accelerating=False):
        """
        Returns the IMF-averaged star formation time of the
        PMF
        """
        return self.distr._average_time(taper, accelerating)

    def set_taper(self, x):
        """
        Sets whether or not the accretion history is tapered.
        Accepts True or False
        """
        self.distr.taper = x

    def set_accel(self, x):
        """
        Sets whether or not the assumed star formation rate
        is accelerating. Accepts True or False
        """
        self.distr.accelerating = x

    @property
    def imf(self):
        return self._imf

    @imf.setter
    def imf(self, x):
        self._imf = x
        self._imf.normalize()
        self.distr.imf = self._imf
        self.distr.calculate('all')

    @property
    def lmin(self):
        return self._lmin

    @lmin.setter
    def lmin(self, x):
        self._lmin = x
        self.distr.lmin = x
        self.distr._calculate('all')

    @property
    def lmax(self):
        return self._lmax

    @lmax.setter
    def lmax(self, x):
        self._lmax = x
        self.distr.lmax = x
        self.distr._calculate('all')

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self, x, fmt=None):
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
            self.distr.interps = self._make_interps(self._trackdir)
            self.distr._calculate('all')

    @property
    def f_epi(self):
        return self._f_epi

    @f_epi.setter
    def f_epi(self, x):
        self._f_epi = x
        self.distr.interps = self._make_interps(self._trackdir)
        self.distr.calculate('all')

    @property
    def mmin(self):
        return self.lmin

    @property
    def mmax(self):
        return self.lmax

    @property
    def j_exp(self):
        return self._j_exp

    @property
    def jf_exp(self):
        return self._jf_exp

    @property
    def scale_value(self):
        return self._scale_value

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, x):
        if x <= 0:
            raise ValueError('n must be > 0')
        self._n = x
        self.distr.n = x
        self.distr._calculate('taper')

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, x):
        self._tau = x
        self.distr.tau = x
        self.distr._calculate('accelerating')

    @property
    def interps(self):
        return self._interps

    @property
    def taper(self):
        return self.distr.taper

    @property
    def accelerating(self):
        return self.distr.accelerating


class dist_plf(Distribution):
    """
    Manages the PDF/CDF for a PLF.

    Currently not implemented due to algorithmic issues with
    interpolation
    """

    def __init__(self, imf, l1, l2,
                 j_exp, jf_exp, scale_value,
                 n, tau,
                 n_lpts, n_mpts,
                 interps):

        raise NotImplementedError('PLFs are currently disabled due to interpolation issues')

        self.imf = imf
        self.l1 = l1
        self.l2 = l2
        self.j_exp = j_exp
        self.jf_exp = jf_exp
        self.scale_value = scale_value
        self.n = n
        self.tau = tau
        self.n_mpts = n_mpts
        self.interps = interps

        self._points = np.geomspace(self.l1, self.l2, n_lpts)
        self._func_dict = None
        self._taper = False
        self._accelerating = False
        self.interp_idx = 0

        self._calculate('all')

    def _make_bases(self, taper, accelerating):
        """
        Construct interpolators for the PDF/CDF/PPF underlying
        a PLF for a particular combination of tapered accretion
        and accelerating star formation.
        """
        def plf(lum, taper, accelerating):
            avg_time = self._average_time(taper, accelerating)

            def get_unknowns(mf, lum_):
                interp_l = self.interps[self.interp_idx]

                masses = np.geomspace(0.1, mf, self.n_mpts)  # in-bounds mass points for a 1D interpolation, beginning at the point where the code initializes a protostar
                m_of_l = PchipInterpolator(interp_l((np.ones(len(masses))*mf, masses)), masses)
                ret_m = m_of_l(lum_)

                interp_grad = self.interps[self.interp_idx+1]
                ret_l = abs(ret_m * interp_grad((mf, ret_m)) / lum_)

                return ret_m, ret_l

            def m_dot(mf, mass_):
                return self.scale_value * (mass_ / mf)**self.j_exp * mf**self.jf_exp

            def integrand(mf, lum_):

                mass_, l_factor = get_unknowns(mf, lum_)

                if taper:
                    tf = self._tf(mf, taper)

                    def root_t(t, mf, mass_):
                        term1 = t * (1 - (t / tf)**self.n / (self.n + 1))
                        term2 = mass_**(1 - self.j_exp) / self.scale_value / (1 - self.j_exp) / mf**(self.jf_exp - self.j_exp)
                        prime_term1 = 1 - (t / tf)**self.n / (self.n + 1)
                        prime_term2 = self.n / (self.n + 1) * (t / tf)**self.n
                        return term1 - term2, prime_term1 - prime_term2

                    def taper_factor(mf, mass_):
                        sol = root_scalar(root_t, args=(mf, mass_), x0=0, fprime=True)
                        return 1 - (sol.root / tf)**self.n

                    t_factor = taper_factor(mf, mass_)
                    if accelerating:
                        tm = (1 - t_factor)**(1 / self.n) * tf

                else:
                    t_factor = 1
                    if accelerating:
                        tm = mass_**(1 - self.j_exp) / mf**(self.jf_exp - self.j_exp) / self.scale_value / (1 - self.j_exp)
                a_factor = np.exp(-tm / self.tau / 1e6) if accelerating else 1

                ret = self.imf(mf) * mass_ / m_dot(mf, mass_) * a_factor / t_factor / l_factor
                if np.isfinite(ret):
                    return ret
                else:
                    return 0

            def integral(lolim, mass_, **kwargs):
                return quad(integrand, mmin, mmax, args=(lum_), **kwargs)[0]

            ret = np.vectorize(integral)(lum, self.imf.mmin, self.imf.mmax)
            return np.where(ret / avg_time > 0, ret / avg_time, 0)  # ensure the PLF is always >= 0

        base = plf(self._points, taper, accelerating)
        pdf = base / self._points
        cdf = cumulative_trapezoid(pdf, self._points, initial=0)
        nonzero_args = np.nonzero(np.diff(cdf))[0]
        start = np.min(nonzero_args)
        end = np.max(nonzero_args) + 1
        return (PchipInterpolator(self._points, pdf),
                PchipInterpolator(self._points, cdf),
                PchipInterpolator(cdf[start:end+1], self._points[start:end+1]))

    def _calculate(self, mode):
        """
        Calculate the PDF/CDF/PPF as needed. "mode" determines
        which versions are recalculated.
        """
        not_ok = (self.j_exp is None) | (self.jf_exp is None) | (self.scale_value is None)

        keys = ['pdf', 'cdf', 'ppf']
        if mode == 'all':
            func_dict = {key: [] for key in keys}
            modes = [(0, 0), (1, 0), (0, 1), (1, 1)]
            for mm in modes:
                bases = self._make_bases(*mm)
                for ii, key in enumerate(keys):
                    func_dict[key].append(bases[ii])
            self._func_dict = func_dict

        elif mode == 'taper':
            modes = [(1, 0), (1, 1)]
            for ii, mm in enumerate(modes):
                bases = self._make_bases(*mm)
                for jj, key in enumerate(keys):
                    self._func_dict[key][2*ii+1] = bases[jj]

        elif mode == 'accelerating':
            modes = [(0, 1), (1, 1)]
            for ii, mm in enumerate(modes):
                bases = self._make_bases(*mm)
                for jj, key in enumerate(keys):
                    self._func_dict[key][ii+2] = bases[jj]

    def _pick_function(self, functype, taper, accelerating):
        return self._func_dict[functype][int(taper+2*accelerating)]

    def _update_functions(self):
        self._pdf = self._pick_function('pdf', self.taper, self.accelerating)
        self._cdf = self._pick_function('cdf', self.taper, self.accelerating)
        self._ppf = self._pick_function('ppf', self.taper, self.accelerating)

    def _tf(self, mf, taper):
        factor = (self.n + 1) / self.n if taper else 1
        tf1 = factor / (1 - self.j_exp) / self.scale_value
        return tf1 * mf**(1 - self.jf_exp)

    def _average_time(self, taper, accelerating):
        if accelerating:
            def accel_weight(mf, taper=False):
                return 1e6 * self.tau * (1 - np.exp(-self._tf(mf, taper=taper) / self.tau / 1e6))
            ret = self.imf.weight_average(accel_weight, taper)
        else:
            ret = self.imf.weight_average(self._tf, taper)
        return ret

    def pdf(self, x):
        return self._pdf(x, extrapolate=False)

    def cdf(self, x):
        return self._cdf(x, extrapolate=False)

    def ppf(self, x):
        return self._ppf(x, extrapolate=False)

    def rvs(self, N):
        samp = np.random.uniform(self.cdf(self.l1), self.cdf(self.l2), size=N)
        return self.ppf(samp)

    @property
    def taper(self):
        return self._taper

    @taper.setter
    def taper(self, x):
        self._taper = bool(x)
        self._update_functions()
        self.interp_idx = int(3 * self._taper)

    @property
    def accelerating(self):
        return self._accelerating

    @accelerating.setter
    def accelerating(self, x):
        self._accelerating = bool(x)
        self._update_functions()
