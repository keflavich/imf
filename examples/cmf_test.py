import numpy as np

from imf.cmf import *

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

def test_hc13():
    masses = np.logspace(-2, 2, 100)*u.M_sun
    sizescale = 10*u.pc
    return hc13_mf(mass=masses, sizescale=sizescale)
