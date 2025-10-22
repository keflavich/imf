import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from os import system

from imf.cmf import *

def test_pn11(nbins=50,nreal=5):

    system('mkdir -p cmf_test')
    
    mmin = 0.01
    mmax = 120
    cmf = PN_CMF(0.01,120)

    tcross = 1
    
    tbe = (cmf.taccr * (cmf.maccr / cmf.mbe)**(-1/3.)).to(u.s)
    tff = ((3 * np.pi / (32 * constants.G * cmf.rho))**0.5).to(u.s)
    mmax = (cmf.maccr * ((tbe + tff) / cmf.taccr)**3).to(u.M_sun)

    age = tnow * tcross - cmf.birthdays
    isBorn = age > 0
    isPrestellar = age < tbe + tff
    belowBE = cmf.maccr < cmf.mbe
    isStellar = np.logical_and(~isPrestellar,~belowBE)
    isForming = age < cmf.taccr

    m_f = np.vstack([mmax.value, cmf.maccr.value]).min(axis=0)*u.M_sun
    mnow = ((age / cmf.taccr)**3 * cmf.maccr).to(u.M_sun)
    mnow[mnow > m_f] = m_f[mnow > m_f]

    toplot = isBorn & isPrestellar & isForming
    gtmax = maccr > mmax

    plt.figure()
    plt.scatter(mnow[toplot & belowBE],cmf.maccr[toplot & belowBE],
                color='orange',marker='.',
                alpha=0.5,
                label=r'$m_{\rm accr} < m_{\rm BE}$')
    plt.scatter(mnow[toplot & ~gtmax & ~belowBE],
                cmf.maccr[toplot & ~gtmax & ~belowBE],
		color='b',marker='+',
                label=r'$m_{\rm accr} < m_{\rm max}$')
    plt.scatter(mnow[toplot & gtmax & ~belowBE],
                cmf.maccr[toplot & gtmax & ~belowBE],
		color='k',marker='d',
                markerfacecolor='none',
                label=r'$m_{\rm accr} > m_{\rm max}$')
    
    plt.xlim(0.03, 20)
    plt.ylim(0.6,700)
    plt.xlabel('$m$ ($M_\odot$)',fontsize='large')
    plt.ylabel(r'$m_{\rm accr} / m$',fontsize='large')
    plt.legend()
    plt.savefig('cmf_test/fig1.pdf',dpi=300,bbox_inches='tight')
    plt.close()
    
    plt.figure()
    edges = np.geomspace(1,60,13)
    plt.hist((cmf.maccr / mnow)[mnow > 0.1*u.M_sun & toplot],
             bins=edges,histtype='step',color='k',
             label=r'$m > 0.1 M_\odot$')
    plt.hist((cmf.maccr / mnow)[mnow > (cmf.mbe / 2)],
             bins=edges,histtype='step',linestyle='dashed',
             color='k',label=r'$m > m_{\rm BE} / 2$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$m_{\rm accr} / m$',fontsize='large')
    plt.legend()
    plt.savefig('cmf_test/fig2.pdf',dpi=300,bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.scatter(mnow[toplot & belowBE],
                (mnow / cmf.mbe)[toplot & belowBE],
                color='orange',marker='.',
                alpha=0.5,
                label=r'$m_{\rm accr} < m_{\rm BE}$')
    plt.scatter(mnow[toplot & ~gtmax & ~belowBE],
                (mnow / cmf.mbe)[toplot & ~gtmax & ~belowBE],
                color='b',marker='+',
                label=r'$m_{\rm accr} < m_{\rm max}$')
    plt.scatter(mnow[toplot & gtmax & ~belowBE],
                (mnow / cmf.mbe)[toplot & gtmax & ~belowBE],
                color='k',marker='d',
                markerfacecolor='none',
                label=r'$m_{\rm accr} > m_{\rm max}$')

    '''
    ct, bn = np.histogram(mnow[(mnow > mbe/2.) & toplot & (maccr < mbe)],
                          bins=np.logspace(np.log10(0.05), np.log10(20)))
    ctall, bn = np.histogram(mnow[(mnow > mbe/2.) & toplot],
                             bins=np.logspace(np.log10(0.05), np.log10(20)))
    bbn = (bn[1:]+bn[:-1])/2.
    pl.loglog(bbn, ct/ctall.astype('float'), 'k-')
    '''

    plt.xlim(0.03,20)
    plt.ylim(0.01,20)
    plt.xlabel('$m$ ($M_\odot$)',fontsize='large')
    plt.ylabel(r'$m / m_{\rm BE}$',fontsize='large')
    plt.legend()
    plt.savefig('cmf_test/fig3.pdf',dpi=300,bbox_inches='tight')
    plt.close()

    plt.figure()
    edges = np.geomspace(mmin,mmax,nbins+1)
    plt.hist(mnow[toplot], bins=edges, histtype='step',
             edgecolor='k', label=r'$m$ ($t=t_0$)')
    plt.hist(mnow[toplot & mnow > cmf.mbe / 2], bins=edges,
             histtype='step',color='b',
             label=r'$m > m_{\rm BE} / 2$ ($t=t_0$)')
    plt.hist(cmf.maccr[~belowBE], bins=edges,
             histtype='step',color='g',
             label=r'$m_{\rm accr}$ (stellar)$')

    for i in range(nreal):
        all_ms = []
        mf = PN_CMF()
        N, edges, masses = mf(1,visible_only=False,return_masses=True)
        all_ms.extend(masses)

    counts, edges = np.histogram(all_ms,bins=np.geomspace(mmin,mmax,nbins * nreal+1))
    av_edges = (edges[:-1] + edges[1:]) / 2.
    plt.plot(av_edges,counts,'r--',label='f{nreal} realizations')
    #ok = np.log(counts) > 0
    #ppars = np.polyfit(np.log(av_edges)[ok], np.log(counts)[ok], 1)
    #pl.plot(bbins, np.exp(ppars[1]) * bbins**ppars[0], 'r')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1)
    plt.xlabel(r'$m$ ($M_\odot$)',fontsize='large')
    plt.ylabel(r'$N(m)$')
    plt.legend()

def test_hc13():
    masses = np.logspace(-2, 2, 100)*u.M_sun
    sizescale = 10*u.pc
    return hc13_mf(mass=masses, sizescale=sizescale)
