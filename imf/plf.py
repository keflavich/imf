"""
Protostellar luminosity functions as described by Offner and McKee, 2011


Alternatively, perhaps try to construct a probabilistic P(L; m, m_f) given a
series of stellar evolution codes?
"""

import numpy as np
import scipy.integrate
from scipy.interpolate import CloughTocher2DInterpolator,RegularGridInterpolator
from astropy.table import Table

import os
import warnings
from glob import glob
loc = os.path.dirname(__file__)

from .imf import MassFunction, ChabrierPowerLaw
from . import distributions

chabrierpowerlaw = ChabrierPowerLaw()

hist_values = {'is' : (0, 0, 1.54e-6, 10, 1.5),
               'tc' : (0.5, 0.75, 4.9e-6, 0.1, 0.75),
               'ca' : (2/3, 1., 6.9e-6, 1e4, 0.5)}

def scaling(history,value=None):
    params = hist_values[history]
    if value is None:
        value = params[3]
    return params[2] * (value / params[3])**params[4]

def get_fname(history,n_comp=1,taper=False):
    prefix = 'taper_' if taper else ''
    if n_comp == 1:
        return prefix + history
    else:
        return prefix + f'2c{history}'
    
class PLF(MassFunction):
    """
    documentation
    """
    def __init__(self,imf,
                 lmin=0.01,lmax=100,
                 history='is',
                 f_epi=0.75,
                 n=1,tau=1,
                 proto_trackdir=f'{loc}/data/K12_protoev_tables',
                 **kwargs):
        self.distr = None
        
        self._imf = imf
        self.imf.normalize()

        self._lmin = lmin
        self._lmax = lmax

        self.history = history

        self._f_epi = f_epi
        
        self._n = n
        self._tau = tau

        self._trackdir = proto_trackdir
        self._interps = self._make_interps(self._trackdir,**kwargs)
        
        self.distr = distributions.PLF(self.imf,self.lmin,self.lmax,
                                       self.j,self.jf,self.scale_value,
                                       self.n,self.tau,
                                       self.interps)
        self.normfactor = 1
        
    def __call__(self,lum,
                 integral_form=False,
                 taper=False,
                 accelerating=False):

        self.distr.taper = taper
        self.distr.accelerating = accelerating
        
        if integral_form:
            return self.distr.cdf(lum) * self.normfactor
        else:
            return self.distr.pdf(lum) * self.normfactor

    def _make_interps(self,trackdir,**kwargs):
        interps = []
        for taper in (False,True):
            table = Table.read(f'{trackdir}/{get_fname(self.history,taper=taper)}_val_table.fits')
            l_tot = table['lint'] + self.f_epi * table['lacc']
            ok = np.isfinite(l_tot)
            mf_unq = np.unique(table['mf'])
            mf_min, mf_max = np.min(mf_unq), np.max(mf_unq)
            m_unq = np.unique(table['m'])
            l_data = np.copy(l_tot)
            l_data[~ok] = 0

            l_data = l_data.reshape(len(mf_unq),len(m_unq))
            l_interp = RegularGridInterpolator((mf_unq,m_unq),l_data,method='pchip')

            grad = np.gradient(l_data,m_unq,axis=1)
            #may need to deal with discontinuities?
            grad_interp = RegularGridInterpolator((mf_unq,m_unq),grad,method='pchip')
            
            interps.extend([l_interp,grad_interp])

        if np.logical_or(mf_min < self.imf.mmin,mf_max > self.imf.mmax):
            warnings.warn('IMF mmin/mmax is outside the range of values covered by evolutionary tracks; '
                          'stars outside this range cannot contribute to the PLF')
        return interps
        
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
    def lmin(self):
        return self._lmin

    @lmin.setter
    def lmin(self,x):
        self._lmin = x
        self.distr.lmin = x
        self.distr._calculate('all')
            
    @property
    def lmax(self):
        return self._lmax

    @lmax.setter
    def lmax(self,x):
        self._lmax = x
        self.distr.lmax = x
        self.distr._calculate('all')

    @property
    def history(self):
        return self._history

    @history.setter
    def history(self,x,fmt=None):
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
            self.distr.interps = self._make_interps(self._trackdir)
            self.distr._calculate('all')
            
    @property
    def f_epi(self):
        return self._f_epi

    @f_epi.setter
    def f_epi(self,x):
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
    def j(self):
        return self._j

    @property
    def jf(self):
        return self._jf

    @property
    def scale_value(self):
        return self._scale_value
        
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

    @property
    def interps(self):
        return self._interps

class McKeeOffner_PLF(MassFunction):
    def __init__(self, j=1, n=1, jf=3/4., mmin=0.033, mmax=3.0, imf=chabrierpowerlaw, **kwargs):
        """
        Incomplete.  The PLF requires a protostellar evolution code as part of its input.
        """
        raise NotImplementedError
        self.j = j
        self.jf = jf
        self.n = n
        self.mmin = mmin
        self.mmax = mmax
        self.imf = imf

        def den_func(x):
            return self.imf(x)*x**(-self.jf)
        self.denominator = scipy.integrate.quad(den_func, self.mmin, self.mmax, **kwargs)[0]

        self.normfactor = 1

    def __call__(self, luminosity, taper=False, integral_form=False, **kwargs):
        """ Unclear if integral_form is right..."""
        if taper:

            def num_func(x, luminosity_):
                tf = (1-(luminosity_/x)**(1-self.j))**0.5
                return self.imf(x)*x**(self.j-self.jf-1) * tf

            def integrate(lolim, luminosity_):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax,
                                                args=(luminosity_,),
                                                **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin <
                                                         luminosity,
                                                         luminosity,
                                                         self.mmin),
                                                luminosity)

        else:
            def num_func(x):
                return self.imf(x)*x**(self.j-self.jf-1)

            def integrate(lolim):
                integral = scipy.integrate.quad(num_func, lolim, self.mmax, **kwargs)[0]
                return integral

            numerator = np.vectorize(integrate)(np.where(self.mmin <
                                                         luminosity,
                                                         luminosity,
                                                         self.mmin))

        result = (1-self.j) * luminosity**(1-self.j) * numerator / self.denominator
        if integral_form:
            warnings.warn("The 'integral form' of the Chabrier PMF is not correctly normalized; "
                          "it is just PMF(m) * m")
            return result * self.normfactor * luminosity
            raise ValueError("Integral version not yet computed")
        else:
            return result * self.normfactor
