import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from os import system

from imf.cmf import *

def test_pn11(tnow=1,nbins=50,nreal=5):

    system('mkdir -p plots/cmf_test')
    
    ml = 0.01
    mu = 120
    cmf = PN_CMF(ml,mu)
    
    tbe = (cmf.taccr * (cmf.maccr / cmf.mbe)**(-1/3.)).to(u.s)
    tff = ((3 * np.pi / (32 * constants.G * cmf.rho))**0.5).to(u.s)
    mmax = (cmf.maccr * ((tbe + tff) / cmf.taccr)**3).to(u.M_sun)

    age = tnow * cmf.tcross - cmf.birthdays
    isBorn = age > 0
    isPrestellar = age < tbe + tff
    belowBE = cmf.maccr < cmf.mbe
    isStellar = np.logical_and(~isPrestellar,~belowBE)
    isForming = age < cmf.taccr

    m_f = np.vstack([mmax.value, cmf.maccr.value]).min(axis=0)*u.M_sun
    mnow = ((age / cmf.taccr)**3 * cmf.maccr).to(u.M_sun)
    mnow[mnow > m_f] = m_f[mnow > m_f]

    toplot = isBorn & isPrestellar & isForming
    gtmax = cmf.maccr > mmax

    #figure 1
    plt.figure()
    plt.scatter(mnow[toplot & belowBE],
                cmf.maccr[toplot & belowBE]/mnow[toplot & belowBE],
                color='r',marker='.',
                alpha=0.1,
                label=r'$m_{\rm accr} < m_{\rm BE}$')
    plt.scatter(mnow[toplot & ~gtmax & ~belowBE],
                cmf.maccr[toplot & ~gtmax & ~belowBE]/mnow[toplot & ~gtmax & ~belowBE],
		color='b',marker='+',
                label=r'$m_{\rm accr} < m_{\rm max}$')
    plt.plot(mnow[toplot & gtmax & ~belowBE],
             cmf.maccr[toplot & gtmax & ~belowBE]/mnow[toplot & gtmax & ~belowBE],
	     'kD',markerfacecolor='none',
             label=r'$m_{\rm accr} > m_{\rm max}$')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.03, 20)
    plt.ylim(0.6,700)
    plt.xlabel(r'$m$ ($M_\odot$)',fontsize='large')
    plt.ylabel(r'$m_{\rm accr} / m$',fontsize='large')
    plt.legend()
    plt.savefig('plots/cmf_test/fig1.pdf',dpi=300,bbox_inches='tight')
    plt.close()
    
    #figure 2
    plt.figure()
    edges = np.geomspace(1,60,13)
    plt.hist((cmf.maccr / mnow)[(mnow > 0.1*u.M_sun) & toplot],
             bins=edges,histtype='step',color='k',
             label=r'$m > 0.1 M_\odot$')
    plt.hist((cmf.maccr / mnow)[(mnow > (cmf.mbe / 2)) & toplot],
             bins=edges,histtype='step',linestyle='dashed',
             color='k',label=r'$m > m_{\rm BE} / 2$')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$m_{\rm accr} / m$',fontsize='large')
    plt.legend()
    plt.savefig('plots/cmf_test/fig2.pdf',dpi=300,bbox_inches='tight')
    plt.close()

    #figure 3
    plt.figure()
    plt.scatter(mnow[toplot & belowBE],
                (mnow / cmf.mbe)[toplot & belowBE],
                color='r',marker='.',
                alpha=0.1,
                label=r'$m_{\rm accr} < m_{\rm BE}$')
    plt.scatter(mnow[toplot & ~gtmax & ~belowBE],
                (mnow / cmf.mbe)[toplot & ~gtmax & ~belowBE],
                color='b',marker='+',
                label=r'$m_{\rm accr} < m_{\rm max}$')
    plt.plot(mnow[toplot & gtmax & ~belowBE],
             (mnow / cmf.mbe)[toplot & gtmax & ~belowBE],
             'kD',markerfacecolor='none',
             label=r'$m_{\rm accr} > m_{\rm max}$')

    '''
    ct, bn = np.histogram(mnow[(mnow > mbe/2.) & toplot & (maccr < mbe)],
                          bins=np.logspace(np.log10(0.05), np.log10(20)))
    ctall, bn = np.histogram(mnow[(mnow > mbe/2.) & toplot],
                             bins=np.logspace(np.log10(0.05), np.log10(20)))
    bbn = (bn[1:]+bn[:-1])/2.
    pl.loglog(bbn, ct/ctall.astype('float'), 'k-')
    '''

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.03,20)
    plt.ylim(0.01,20)
    plt.xlabel('$m$ ($M_\odot$)',fontsize='large')
    plt.ylabel(r'$m / m_{\rm BE}$',fontsize='large')
    plt.legend()
    plt.savefig('plots/cmf_test/fig3.pdf',dpi=300,bbox_inches='tight')
    plt.close()

    #figure 4
    plt.figure()
    edges = np.geomspace(ml,mu,nbins+1)
    plt.hist(mnow[toplot].value, bins=edges, histtype='step',
             edgecolor='k', label=r'$m$ ($t=t_0$)')
    plt.hist(mnow[toplot & (mnow > cmf.mbe / 2)].value, bins=edges,
             histtype='step',color='b',
             label=r'$m > m_{\rm BE} / 2$ ($t=t_0$)')

    _, _, masses = cmf(tnow=2,cores='stellar',visible_only=False,return_masses=True)
    plt.hist(masses.value, bins=edges,
             histtype='step',color='g',
             label=r'$N(m)$, stellar ($t=2t_0$)')

    all_ms = []
    for i in range(nreal):
        mf = PN_CMF(ml,mu)
        _, _, masses_ = mf(tnow=tnow,visible_only=True,return_masses=True)
        all_ms.extend(masses_)

    all_ms = np.array([v.value for v in all_ms])
    counts, edges = np.histogram(all_ms,bins=edges)
    av_edges = (edges[:-1] + edges[1:]) / 2.
    plt.plot(av_edges,counts/nreal,'r--',label=f'{nreal} realizations')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.03,300)
    plt.ylim(1,1e3)
    plt.xlabel(r'$m$ ($M_\odot$)',fontsize='large')
    plt.ylabel(r'$N(m)$')
    plt.legend()
    plt.savefig('plots/cmf_test/fig4.pdf',dpi=300,bbox_inches='tight')

def test_hc13():
    masses = np.logspace(-2, 2, 100)*u.M_sun
    sizescale = 10*u.pc
    return hc13_mf(mass=masses, sizescale=sizescale)

def main():
    test_pn11(nbins=50)
    #test_hc13()

if __name__ == '__main__':
    main()
