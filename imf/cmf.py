import numpy as np
from astropy import units as u
from astropy import constants
import scipy.stats

from . import imf

def padoan_mf(mmin=0.01*u.M_sun, mmax=120*u.M_sun, T0=10*u.K,
              T_mean=7*u.K, L0=10*u.pc, rho0=2e-21*u.g/u.cm**3, MS0=25,
              beta=0.4, alpham1=1.35, v0=4.9*u.km/u.s, eff=0.26):
    """
    Padoan & Nordlund IMF - from http://adsabs.harvard.edu/abs/2011ApJ...741L..22P

    Does not match their figures yet!
    """


    tcross = L0/v0
    # total molecular cloud mass
    m0 = 4/3.*np.pi*L0**3*rho0

    massfunc = imf.Salpeter(alpha=alpham1+1)
    massfunc.__name__ = 'salpeter'
    m_accr = imf.make_cluster(mcluster=(m0*eff).to(u.M_sun).value,
                              massfunc=massfunc,
                              mmin=mmin.to(u.M_sun).value,
                              mmax=mmax.to(u.M_sun).value,
                              silent=True)*u.M_sun

    sigma_squared = np.log(1+(MS0/2.)**2 * (1+1./beta)**-1)

    # two positional parameters: x, s such that pdf=lognorm(x,s)
    # x = (ln(rho/rho0) + (s/2))^2
    # s = s
    s = sigma_squared
    pdf_func = scipy.stats.lognorm(s)
    x = pdf_func.rvs(len(m_accr))
    rho = rho0 * np.exp(x**0.5 - s/2)

    sigma = (1+beta**-1)**-0.5 * MS0/2.

    b=4.-(3./alpham1)
    a=(b-1)/2.
    taccr = tcross * sigma**((4-4*a)/(3-2*a)) * (m_accr/m0)**((1-a)/(3-2*a))

    c_s = ((constants.k_B * T_mean / (2.34*constants.m_p))**0.5).to(u.km/u.s)
    mbe = (1.182 * c_s**3 / (constants.G**1.5 * rho**0.5)).to(u.M_sun)
    tbe = (taccr * (m_accr/mbe)**(-1/3.)).to(u.s)
    tff = ((3*np.pi/(32*constants.G*rho))**0.5).to(u.s)
    mmax = (m_accr * ((tbe+tff)/taccr)**3).to(u.M_sun)

    m_f = np.vstack([mmax.value, m_accr.value]).min(axis=0)*u.M_sun
    will_collapse = m_accr > mbe

    return m_f, will_collapse
