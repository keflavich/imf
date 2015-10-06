import numpy as np
from astropy import units as u
from astropy import constants
import scipy.stats

from . import imf

def pn11_mf(tnow=1, mmin=0.01*u.M_sun, mmax=120*u.M_sun, T0=10*u.K,
            T_mean=7*u.K, L0=10*u.pc, rho0=2e-21*u.g/u.cm**3, MS0=25,
            beta=0.4, alpham1=1.35, v0=4.9*u.km/u.s, eff=0.26):
    """
    Padoan & Nordlund IMF - from http://adsabs.harvard.edu/abs/2011ApJ...741L..22P

    Parameters
    ----------
    tnow : float
        The time at which to evaluate the mass function in units of the
        crossing time

    Does not match their figures yet!
    """


    tcross = (L0/v0).to(u.Myr)
    # total molecular cloud mass
    m0 = 4/3.*np.pi*L0**3*rho0

    massfunc = imf.Salpeter(alpha=alpham1+1)
    massfunc.__name__ = 'salpeter'
    maccr = imf.make_cluster(mcluster=(m0*eff).to(u.M_sun).value,
                              massfunc=massfunc,
                              mmin=mmin.to(u.M_sun).value,
                              mmax=mmax.to(u.M_sun).value,
                              silent=True)*u.M_sun

    sigma_squared = np.log(1+(MS0/2.)**2 * (1+1./beta)**-1)

    # two positional parameters: x, s such that pdf=lognorm(x,s)
    # p(x) = const * exp(-(ln(x)/s)^2 / 2)
    # p(rho/rho0) = const * exp(-(ln(rho/rho0) + sigma_squared/2)^2 / (2*sigma_squared))
    # (ln(rho/rho0)+sigma_squared/2)^2/(2*sigma_squared) = ln(x)^2/s^2 /2
    # s^2 = sigma_squared
    # ln(x) = ln(rho/rho0)+sigma_squared/2
    # x = np.exp(ln(rho/rho0)+sigma_squared/2)
    # ln(rho/rho0) = ln(x) - sigma_squared/2
    # rho = rho0 * exp(ln(x) - sigma_squared/2)
    #### wrong x = (ln(rho/rho0) + (s/2))^2
    #### wrong s = s
    s = sigma_squared**0.5
    pdf_func = scipy.stats.lognorm(s)
    x = pdf_func.rvs(len(maccr))
    rho = rho0 * np.exp(np.log(x) - sigma_squared/2)

    sigma = (1+beta**-1)**-0.5 * MS0/2.

    # core birthdays: "Finally, we associate a random age to each core,
    # assuming for simplicity that the SFR is uniform over time and independent
    # of core mass."
    age = np.random.random(len(maccr)) * tcross
    born = age < tnow*tcross

    b=4.-(3./alpham1)
    a=(b-1)/2.
    taccr = tcross * sigma**((4.-4.*a)/(3.-2.*a)) * (maccr/m0)**((1-a)/(3-2*a))
    print("taccr=[{0} - {1}]".format(taccr.to(u.Myr).min(), taccr.to(u.Myr).max()))

    c_s = ((constants.k_B * T_mean / (2.34*constants.m_p))**0.5).to(u.km/u.s)
    mbe = (1.182 * c_s**3 / (constants.G**1.5 * rho**0.5)).to(u.M_sun)
    tbe = (taccr * (maccr/mbe)**(-1/3.)).to(u.s)
    tff = ((3*np.pi/(32*constants.G*rho))**0.5).to(u.s)
    mmax = (maccr * ((tbe+tff)/taccr)**3).to(u.M_sun)

    # tnow = number of crossing times
    mnow = (((tnow*tcross-age)/taccr)**3 * maccr).to(u.M_sun)
    m_f = np.vstack([mmax.value, maccr.value]).min(axis=0)*u.M_sun
    print("{0} of {1} have mass greater than final at t={2}."
          " {3} are unborn.".format((mnow>m_f).sum(), len(mnow), tnow*tcross,
                                    np.sum(~born)))
    mnow[mnow > m_f] = m_f[mnow>m_f]
    will_collapse = maccr > mbe/2.

    return mnow[born], m_f[born], will_collapse[born], maccr[born], mbe[born], mmax[born]

def test_pn11(nreal=10, nbins=50):
    mnow, mf, wc, maccr, mbe, mmax = pn11_mf()
    import pylab as pl
    pl.figure(1).clf()
    toplot = mnow > 0.05*u.M_sun
    gtmax = maccr > mmax
    ltbe = maccr < mbe
    btw = (~gtmax) & (~ltbe)
    pl.loglog(mnow[toplot&gtmax], (maccr/mnow)[toplot&gtmax], 'kd', markerfacecolor='none')
    pl.loglog(mnow[toplot&ltbe],  (maccr/mnow)[toplot&ltbe], 'r.', markerfacecolor='none')
    pl.loglog(mnow[toplot&btw],   (maccr/mnow)[toplot&btw], 'b+', markerfacecolor='none')

    pl.figure(2).clf()
    pl.hist(maccr/mnow, bins=np.logspace(0,2,12), histtype='step', color='k')
    pl.hist((maccr/mnow)[mnow>(mbe/2.)], bins=np.logspace(0,2,12),
            histtype='step', linestyle='dashed', color='k')

    pl.figure(4)
    pl.clf()
    pl.hist(mnow, bins=np.logspace(-2,2,nbins), histtype='step', log=True,
            edgecolor='k')
    pl.hist(mnow[wc], bins=np.logspace(-2.02,1.98,nbins), histtype='step', log=True,
            edgecolor='b', facecolor=(0,0,1,0.25))
    pl.hist(mbe, bins=np.logspace(-1.98,2.02, nbins), histtype='step',
            log=True, linestyle='dashed', color='g')
    pl.hist(mmax, bins=np.logspace(-1.96,2.04, nbins), histtype='step',
            log=True, linestyle='dashed', color='m')
    pl.gca().set_xscale('log')

    many_realizations = [pn11_mf() for ii in range(nreal)]
    mnow = np.hstack([x.value for x,y,z,w,v,t in many_realizations]).ravel()
    mfwc = np.hstack([x[z].value for x,y,z,w,v,t in many_realizations]).ravel()

    counts,bins = np.histogram(mf.ravel(), bins=np.logspace(-2,2,nbins*nreal))
    bbins = (bins[:-1]+bins[1:])/2.
    ok = np.log(counts) > 0
    ppars = np.polyfit(np.log(bbins)[ok], np.log(counts)[ok], 1)
    pl.plot(bbins, np.exp(ppars[1]) * bbins**ppars[0], 'r')
    return counts,bbins
    #pl.hist(mf.ravel(), bins=np.logspace(-2,2,nbins*nreal), histtype='step', log=True,
    #        edgecolor='r')
    #pl.hist(mfwc.ravel(), bins=np.logspace(-2,2,nbins*nreal), histtype='step',
    #        edgecolor='g',
    #        log=True)
