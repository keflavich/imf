from __future__ import print_function
import numpy as np
from astropy import units as u
from astropy import constants
import scipy.stats

from . import imf

def pn11_mf(tnow=1, mmin=0.01*u.M_sun, mmax=120*u.M_sun, T0=10*u.K,
            T_mean=7*u.K, L0=10*u.pc, rho0=2e-21*u.g/u.cm**3, MS0=25,
            beta=0.4, alpham1=1.35, v0=4.9*u.km/u.s, eff=0.26,
            mean_mol_wt=2.33):
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
    m0 = (4/3.*np.pi*L0**3*rho0).to(u.M_sun)

    massfunc = imf.Salpeter(alpha=alpham1+1)
    massfunc.__name__ = 'salpeter'
    maccr = imf.make_cluster(mcluster=(m0*eff).to(u.M_sun).value,
                             massfunc=massfunc,
                             mmin=mmin.to(u.M_sun).value,
                             mmax=mmax.to(u.M_sun).value,
                             silent=True)*u.M_sun

    # sigma_squared of log(rho/rho0)
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

    s = sigma_squared**0.5
    pdf_func = scipy.stats.lognorm(s)
    x = pdf_func.rvs(len(maccr))
    rho = rho0 * np.exp(np.log(x) - sigma_squared/2)

    sigma = (1+beta**-1)**-0.5 * MS0/2.

    # core birthdays: "Finally, we associate a random age to each core,
    # assuming for simplicity that the SFR is uniform over time and independent
    # of core mass."
    # Confirmed: "what I did was to assume constant star formation rate during
    # the whole period t=t_0, where t_0 is the crossing time, and also the time
    # when the mass functions are computed. So cores are formed with a uniform
    # distribution (of birth times) between 0 and t_0."
    birthday = np.random.random(len(maccr)) * tcross
    born = birthday < tnow*tcross

    b = 4.-(3./alpham1)
    a = (b-1)/2.
    taccr = (tcross * sigma**((4.-4.*a)/(3.-2.*a)) *
             (maccr/m0)**((1-a)/(3-2*a))).to(u.Myr)
    print(("taccr=[{0} - {1}]".format(taccr.to(u.Myr).min(), taccr.to(u.Myr).max())))

    c_s = ((constants.k_B * T_mean / (mean_mol_wt*constants.m_p))**0.5).to(u.km/u.s)
    print(("c_s = {0}".format(c_s)))
    mbe = (1.182 * c_s**3 / (constants.G**1.5 * rho**0.5)).to(u.M_sun)
    tbe = (taccr * (maccr/mbe)**(-1/3.)).to(u.s)
    tff = ((3*np.pi/(32*constants.G*rho))**0.5).to(u.s)
    mmax = (maccr * ((tbe+tff)/taccr)**3).to(u.M_sun)

    # tnow = number of crossing times
    age = (tnow*tcross-birthday)
    mnow = ((age/taccr)**3 * maccr).to(u.M_sun)
    prestellar = age < tbe+tff
    ltbe = maccr < mbe
    stellar = (~prestellar) & (~ltbe)
    forming = age < taccr
    m_f = np.vstack([mmax.value, maccr.value]).min(axis=0)*u.M_sun

    mnow[mnow > m_f] = m_f[mnow > m_f]
    will_collapse = maccr > mbe/2.

    # We assume that cores that do not reach their BE mass are seen only during
    # their formation time, taccr,
    notseen = ltbe & ~forming

    core_mass = mnow[born & (~notseen) & (~stellar)].sum()
    stellar_mass = mnow[stellar].sum()
    print(("{0} of {1} have mass greater than final at t={2}."
           " {3} are unborn.  {4} are stellar.  "
           "{7} are not seen ({8:0.02f}%) because they are older than "
           "one accretion time and have M<M_BE. "
           "The cloud mass is {9}. "
           "The CFE={5}"
           " and SFE={6}".format((mnow > m_f).sum(), len(mnow), tnow*tcross,
                                 np.sum(~born), np.sum(stellar),
                                 (core_mass/m0).decompose().value,
                                 (stellar_mass/m0).decompose().value,
                                 notseen.sum(),
                                 (notseen.sum()/float(notseen.size))*100,
                                 m0
         )))

    return (mnow[born], m_f[born], will_collapse[born], maccr[born], mbe[born],
            mmax[born], forming[born])

def test_pn11(nreal=5, nbins=50, **kwargs):
    mnow, mf, wc, maccr, mbe, mmax, forming = pn11_mf(**kwargs)
    import pylab as pl
    pl.figure(1).clf()
    ltbe = maccr < mbe
    # We assume that cores that do not reach their BE mass are seen only during
    # their formation time, taccr,
    notseen = ltbe & ~forming
    toplot = (mnow > 0.05*u.M_sun) & (~notseen)
    gtmax = maccr > mmax
    btw = (~gtmax) & (~ltbe)
    pl.loglog(mnow[toplot & gtmax], (maccr/mnow)[toplot & gtmax], 'kd',
              markerfacecolor='none')
    print(("{0} have maccr<mbe and m>0.05".format((toplot & ltbe).sum())))
    pl.loglog(mnow[toplot & ltbe], (maccr/mnow)[toplot & ltbe], 'r.',
              markersize=2, alpha=0.5,
              markerfacecolor='none')
    pl.loglog(mnow[toplot & btw], (maccr/mnow)[toplot & btw], 'b+',
              markerfacecolor='none')
    pl.gca().set_ylim(0.5, 500)
    pl.gca().set_xlim(0.05, 50)
    pl.ylabel("$m_{accr}/m$")
    pl.xlabel("$m$, i.e. $m_{now}$")

    pl.figure(2).clf()
    pl.hist((maccr/mnow)[mnow > 0.1*u.M_sun], bins=np.logspace(0, 2, 12),
            histtype='step', color='k', log=True)
    pl.hist((maccr/mnow)[mnow > (mbe/2.)], bins=np.logspace(0, 2, 12),
            histtype='step', linestyle='dashed', color='k', log=True)
    pl.gca().set_xscale('log')
    pl.ylabel("$N(m_{accr}/m)$")
    pl.xlabel("$m_{accr}/m$")

    pl.figure(3).clf()
    pl.loglog(mnow[toplot & gtmax], (mnow/mbe)[toplot & gtmax], 'kd',
              markerfacecolor='none')
    pl.loglog(mnow[toplot & ltbe], (mnow/mbe)[toplot & ltbe], 'r.',
              markersize=2,
              markerfacecolor='none', alpha=0.5)
    pl.loglog(mnow[toplot & btw], (mnow/mbe)[toplot & btw], 'b+',
              markerfacecolor='none')
    ct, bn = np.histogram(mnow[(mnow > mbe/2.) & toplot & (maccr < mbe)],
                          bins=np.logspace(np.log10(0.05), np.log10(20)))
    ctall, bn = np.histogram(mnow[(mnow > mbe/2.) & toplot],
                             bins=np.logspace(np.log10(0.05), np.log10(20)))
    bbn = (bn[1:]+bn[:-1])/2.
    pl.loglog(bbn, ct/ctall.astype('float'), 'k-')
    pl.ylabel("$m_{now}/m_{BE}$")
    pl.xlabel("$m$, i.e. $m_{now}$")

    pl.figure(4)
    pl.clf()
    pl.hist(mnow, bins=np.logspace(-2, 2, nbins), histtype='step', log=True,
            edgecolor='k', label='$m$')
    pl.hist(mnow[wc], bins=np.logspace(-2.02, 1.98, nbins), histtype='step', log=True,
            edgecolor='b', facecolor=(0, 0, 1, 0.25), label='$m>m_{BE}/2$')
    pl.hist(mbe, bins=np.logspace(-1.98, 2.02, nbins), histtype='step',
            log=True, linestyle='dashed', color='g', label='$m_{BE}$')
    pl.hist(mmax, bins=np.logspace(-1.96, 2.04, nbins), histtype='step',
            log=True, linestyle='dashed', color='m', label='$m_{max}$')
    pl.gca().set_xscale('log')
    pl.legend(loc='best')
    pl.ylabel("$N(M)$")
    pl.xlabel("$m$, i.e. $m_{now}$")

    many_realizations = [pn11_mf(**kwargs) for ii in range(nreal)]
    mnow_many = np.hstack([x.value for x, y, z, w, v, s, t in many_realizations]).ravel()

    counts, bins = np.histogram(mnow_many.ravel(), bins=np.logspace(-2, 2, nbins*nreal))
    bbins = (bins[:-1]+bins[1:])/2.
    ok = np.log(counts) > 0
    ppars = np.polyfit(np.log(bbins)[ok], np.log(counts)[ok], 1)
    pl.plot(bbins, np.exp(ppars[1]) * bbins**ppars[0], 'r')

    pl.figure(5).clf()
    pl.plot(maccr[toplot], mbe[toplot], 'k,')

    return mnow, mf, wc, maccr, mbe, mmax

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
    phit = 2 * alpha_ct * (24/np.pi**2/alpha_g)

    # dimensionless geometrical factor of the order of unity
    # For a sphere, becoMes:
    aJ = np.pi**2.5/6.
    # a geometrical factor, typically of the order of 4pi/3
    Cm = 4*np.pi/3

    # eqn 13
    MJ0 = (aJ / Cm * c_s**3 * constants.G**-1.5 * rho_bar**-0.5).to(u.M_sun)

    # eqn 14
    lambdaJ0 = (np.pi**0.5 * c_s / Cm * (constants.G*rho_bar)**-0.5).to(u.pc)

    # Eqn 7 of Paper I
    # delta = np.log(rho/rho_bar
    # R = (mass/rho_bar)**(1/3.) * np.exp(-delta/3.) / lambdaJ0

    Rtwiddle = (sizescale / lambdaJ0).to(u.dimensionless_unscaled)

    Mtwiddle = (mass / MJ0).to(u.dimensionless_unscaled)

    # eqn 20
    Mstar = (3**-0.5 * V0/c_s * (lambdaJ0/(u.pc))**eta).to(u.dimensionless_unscaled)

    # after eqn 21
    N0 = rho_bar / MJ0
    # PROBLEM: N0 is defined to be rho_bar / MJ0, but that is a dimensional
    # quantity with units cm^-3. This is a contradiction that means some
    # definition here is wrong.

    # eqn 21
    N = (2./phit * N0 * Rtwiddle**-6 * (1 + (1-eta)*Mstar**2*Rtwiddle**(2*eta)) /
         (1+(2*eta+1)*Mstar**2*Rtwiddle**(2*eta)) *
         (Mtwiddle/Rtwiddle**3)**(-1-1/(2*sigma**2)*np.log(Mtwiddle/Rtwiddle**3)) *
         np.exp(sigma**2/8.) * ((2*np.pi)**0.5 * sigma)
        ).to(u.dimensionless_unscaled)

    return N


def test_hc13():
    masses = np.logspace(-2, 2, 100)*u.M_sun
    sizescale = 10*u.pc
    return hc13_mf(mass=masses, sizescale=sizescale)
