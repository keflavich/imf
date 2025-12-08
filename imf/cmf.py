from __future__ import print_function
import numpy as np
from astropy import units as u
from astropy import constants
import scipy.stats

from . import imf

class PN_CMF:
    """
    Padoan & Nordlund IMF - from http://adsabs.harvard.edu/abs/2011ApJ...741L..22P

    Parameters
    ----------

    tnow : float
        Time at which to evaluate the mass function in units of the
        crossing time


    Does not match their figures yet!
    """
    def __init__(self,mmin,mmax,
                 T0=10*u.K, L0=10*u.pc,
                 v0=4.9*u.km/u.s, rho0=2e-21*u.g/u.cm**3,
                 alpham1=1.35, eff=0.26, beta=0.4,
                 T_mean=7*u.K, mean_mol_wt=2.33):

        self._tcross = (L0 / v0).to(u.Myr)
        m0 = (4 / 3 * np.pi * L0**3 * rho0).to(u.M_sun) #total cloud mass

        self._massfunc = imf.Salpeter(mmin=mmin,mmax=mmax,
                                alpha=alpham1+1)
        self._maccr = imf.make_cluster(mcluster=(m0*eff).to(u.M_sun).value,
                                 massfunc=self.massfunc,
                                 silent=True) * u.M_sun

        MS0 = (v0 / ((constants.k_B * T0 / (mean_mol_wt * constants.m_p))**0.5).to(u.km/u.s)).value

        sigma_rho = MS0 / np.sqrt((1 + beta**-1)) / 2. #stdev of density
        s = np.sqrt(np.log(1 + sigma_rho**2)) #lognormal shape
        #pdf_func = scipy.stats.lognorm(s)
        #x = pdf_func.rvs(len(self.maccr))
        ln_rho = scipy.stats.norm.rvs(loc=-s**2/2,scale=s,
                                      size=len(self.maccr))
        x = np.exp(ln_rho)
        self._rho = x * rho0 #densities around each core

        self._birthdays = np.random.random(len(self.maccr)) * self.tcross #core birthdays (assuming flat formation over crossing time)

        a = (3 - (3 / alpham1)) / 2
        self._taccr = (self.tcross * sigma_rho**((4 - 4 * a) / (3 - 2 * a)) *
                       (self.maccr / m0)**((1 - a) / (3 - 2 * a))).to(u.Myr)

        c_s = ((constants.k_B * T_mean / (mean_mol_wt * constants.m_p))**0.5).to(u.km/u.s)
        self._mbe = (1.182 * c_s**3 / (constants.G**1.5 * self.rho**0.5)).to(u.M_sun)

    def __call__(self,
                 tnow=1, #tnow is in # of crossing times
                 visible_only=True,
                 cores='prestellar',
                 return_masses=False):

        #core_types = ['prestellar','stellar','all']

        tbe = (self.taccr * (self.maccr / self.mbe)**(-1/3.)).to(u.s)
        tff = ((3 * np.pi / (32 * constants.G * self.rho))**0.5).to(u.s)
        mmax = (self.maccr * ((tbe + tff) / self.taccr)**3).to(u.M_sun)

        age = tnow * self.tcross - self.birthdays
        isBorn = age > 0
        isPrestellar = age < tbe + tff
        belowBE = self.maccr < self.mbe
        isStellar = np.logical_and(~isPrestellar,~belowBE)
        isForming = age < self.taccr

        m_f = np.vstack([mmax.value, self.maccr.value]).min(axis=0)*u.M_sun
        mnow = ((age / self.taccr)**3 * self.maccr).to(u.M_sun)
        mnow[mnow > m_f] = m_f[mnow > m_f]

	#We assume that cores that do not reach their BE mass are seen only during
        #their formation time, taccr

        if ~visible_only:
            if cores == 'prestellar':
                cut = np.logical_and(isBorn,~isStellar)
            elif cores == 'stellar':
                cut = np.logical_and(isBorn,isStellar)
            else:
                cut = np.ones(len(mnow))
        else:
            cut = np.logical_and(isBorn,isForming)
            if cores == 'prestellar':
                cut = np.logical_and(cut,isPrestellar)
            elif cores == 'stellar':
                cut = np.logical_and(cut,isStellar)

        core_masses = mnow[cut]

        edges = 10**np.histogram_bin_edges(np.log10(core_masses.value)) * u.M_sun
        hist,edges = np.histogram(core_masses,bins=edges)
        if return_masses:
            return hist,edges,core_masses
        else:
            return hist,edges

    def will_collapse(self):
        return self.maccr > self.mbe / 2

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

def hc13_mf(mass, sizescale, n17=3.8, alpha_ct=0.75, mean_mol_wt=2.33,
            V0=0.8*u.km/u.s, meandens=5000*u.cm**-3, temperature=10*u.K,
            eta=0.45, b_forcing=0.4, Mach=6):
    """ Equation 21 of Hennebelle & Chabrier 2013

    Parameters
    ----------
    mass : np.array
        Masses at which to evaluate the PDF
    sizescale : pc equivalent
        The size of the clump (I think - extremely difficult to find this)
    n17 : float
        The "n" value in Equation 17, quoted to be 3.8 shortly afterward in the
        text
    alpha_ct : float
        "a dimensionless coefficient of the order of a few"
        I derived 0.75 from equation 9
    V0 : float
        u0 * 0.8 km/s according to the bottom of page 6, under eqn 36,
        which references eqn 16
    eta : None or float
        derived from Equation 17, but can be specified directly
    b_forcing : float
        The Forcing Parameter `b` from equation 4
    Mach : float
        Mach number
    """

    rho_bar = meandens * mean_mol_wt * constants.m_p

    c_s = ((constants.k_B * temperature /
            (mean_mol_wt*constants.m_p))**0.5).to(u.km/u.s)

    sigma = (np.log(1+b_forcing**2 * Mach**2))**0.5

    if eta is None:
        # eqn 17
        eta = (n17 - 3.)/2.

    alpha_g = 3/5. # for a uniform density fluctuation
    # eqn 9
    phit = 2 * alpha_ct * (24 / np.pi**2 / alpha_g)

    # dimensionless geometrical factor of the order of unity
    # For a sphere, becoMes:
    aJ = np.pi**2.5 / 6
    # a geometrical factor, typically of the order of 4pi/3
    Cm = 4 * np.pi / 3

    # eqn 13
    MJ0 = (aJ / Cm * c_s**3 * constants.G**-1.5 * rho_bar**-0.5).to(u.M_sun)
    
    # eqn 14
    lambdaJ0 = ((np.pi**1.5 / Cm)**(1./3) * c_s * (constants.G*rho_bar)**-0.5).to(u.pc)
    
    # Eqn 7 of Paper I
    # delta = np.log(rho/rho_bar
    # R = (mass/rho_bar)**(1/3.) * np.exp(-delta/3.) / lambdaJ0

    Rtwiddle = (sizescale / lambdaJ0).to(u.dimensionless_unscaled)

    Mtwiddle = (mass / MJ0).to(u.dimensionless_unscaled)

    # eqn 20
    Mstar = (3**-0.5 * V0 / c_s * (lambdaJ0 / u.pc)**eta).to(u.dimensionless_unscaled)

    # after eqn 21
    N0 = rho_bar / MJ0
    # PROBLEM: N0 is defined to be rho_bar / MJ0, but that is a dimensional
    # quantity with units cm^-3. This is a contradiction that means some
    # definition here is wrong.

    # eqn 21
    N = (2./phit * N0 * Rtwiddle**-6 *
         (1 + (1 - eta) * Mstar**2 * Rtwiddle**(2*eta)) /
         (1+(2*eta+1)*Mstar**2*Rtwiddle**(2*eta)) *
         (Mtwiddle / Rtwiddle**3)**(-1 - np.log(Mtwiddle/Rtwiddle**3) / 2 / sigma**2) *
         np.exp(-sigma**2/8.) / ((2 * np.pi)**0.5 / sigma))

    return N
