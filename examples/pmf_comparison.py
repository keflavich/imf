#make plots illustrating the shape of various PMFs in comparison
#to the underlying IMF in both linear/log forms

import numpy as np
import pylab as pl

import imf.imf, imf.pmf
from imf.pmf import PMF, PMF_2C

import os
plotdir = 'plots/pmf'
os.makedirs(plotdir, exist_ok=True)

pl.rc('font', size=16)
pl.rc('lines', linewidth=2)

mmin = 0.033
chabrier = imf.chabrier2005
chabrier.mmin = mmin
kroupa = imf.Kroupa(mmin=mmin)

for mmax in (3, 120):
    print(f'Beginning mmax = {mmax}.')

    kroupa.mmax = mmax
    kroupa.normalize()
    chabrier.mmax = mmax
    chabrier.normalize()
    
    c_is = PMF(chabrier,history='is')
    c_tc = PMF(chabrier,history='tc')
    c_ca = PMF(chabrier,history='ca')
    c_2c = PMF_2C(chabrier,history='tc')

    k_is = PMF(kroupa,history='is')
    k_tc = PMF(kroupa,history='tc')
    k_ca = PMF(kroupa,history='ca')
    k_2c = PMF_2C(kroupa,history='tc')

    print("Now plotting.")
    masses = np.logspace(np.log10(mmin), np.log10(mmax), 100)

    for mass_weighted in (True,False):
        fname = 'mass_weighted' if mass_weighted else '__call__'
        y_label = r'$m\times P(m)$' if mass_weighted else r'Normalized $P(m)$'
        
        fig1 = pl.figure(1)
        fig1.clf()
        ax = fig1.gca()
        ax.set_title("Steady State McKee/Offner + Chabrier PMF",y=1.03)
        ax.loglog(masses, chabrier.__getattribute__(fname)(masses), label="IMF", color='k')
        ax.loglog(masses, c_is.__getattribute__(fname)(masses), label="IS", color='r', linestyle=':')
        ax.loglog(masses, c_tc.__getattribute__(fname)(masses), label="TC", color='g', linestyle='-.')
        ax.loglog(masses, c_ca.__getattribute__(fname)(masses), label="CA", color='orange', linestyle=(0,(5,2,1,2,1,2,1,2)))
        ax.loglog(masses, c_2c.__getattribute__(fname)(masses), label="2CTC", color='b', linestyle='--')
        ax.set_xlabel(r"(Proto)Stellar Mass (M$_\odot$)")
        ax.set_ylabel(y_label)
        if mass_weighted:
            ax.set_yscale('linear')
            ax.set_ylim(0,0.4)
        else:
            ax.set_ylim(np.min(chabrier(masses))/2)

        pl.legend(loc='best')
        pl.savefig(f'{plotdir}/'+'steadystate_pmf_chabrier{0}_mmax{1}.pdf'.format("_integral" if mass_weighted else "",
                                                                                  int(mmax)), bbox_inches='tight')

        fig2 = pl.figure(2)
        fig2.clf()
        ax2 = fig2.gca()
        ax2.set_title("Tapered McKee/Offner + Chabrier PMF",y=1.03)
        ax2.loglog(masses, chabrier.__getattribute__(fname)(masses), label="IMF", color='k')
        ax2.loglog(masses, c_is.__getattribute__(fname)(masses, taper=True), label="IS", color='r', linestyle=':')
        ax2.loglog(masses, c_tc.__getattribute__(fname)(masses, taper=True), label="TC", color='g', linestyle='-.')
        ax2.loglog(masses, c_ca.__getattribute__(fname)(masses, taper=True), label="CA", color='orange', linestyle=(0,(5,2,1,2,1,2,1,2)))
        ax2.loglog(masses, c_2c.__getattribute__(fname)(masses, taper=True), label="2CTC", color='b', linestyle='--')
        ax2.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax2.set_ylabel(y_label)
        if mass_weighted:
            ax2.set_yscale('linear')
            ax2.set_ylim(0,0.4)
        else:
            ax2.set_ylim(np.min(chabrier(masses))/2)

        pl.legend(loc='best')
        pl.savefig(f'{plotdir}/'+'taperedaccretion_pmf_chabrier{0}_mmax{1}.pdf'.format("_integral" if mass_weighted else "",
                                                                                       int(mmax)), bbox_inches='tight')

        fig3 = pl.figure(3)
        fig3.clf()
        ax3 = fig3.gca()
        ax3.set_title("Steady State McKee/Offner + Kroupa PMF", pad=15)
        ax3.loglog(masses, kroupa.__getattribute__(fname)(masses), label="IMF", color='k')
        ax3.loglog(masses, k_is.__getattribute__(fname)(masses), label="IS", color='r', linestyle=':')
        ax3.loglog(masses, k_tc.__getattribute__(fname)(masses), label="TC", color='g', linestyle='-.')
        ax3.loglog(masses, k_ca.__getattribute__(fname)(masses), label="CA", color='orange', linestyle=(0,(5,2,1,2,1,2,1,2)))
        ax3.loglog(masses, k_2c.__getattribute__(fname)(masses), label="2CTC", color='b', linestyle='--')
        ax3.set_xlabel("(Proto)Stellar Mass $\\left(\\mathrm{M}_\odot\\right)$")
        ax3.set_ylabel(y_label)
        if mass_weighted:
            ax3.set_yscale('linear')
            ax3.set_ylim(0,0.4)
        else:
            ax3.set_ylim(np.min(kroupa(masses))/2)

        pl.legend(loc='best')
        pl.savefig(f'{plotdir}/'+'steadystate_pmf_kroupa{0}_mmax{1}.pdf'.format("_integral" if mass_weighted else "",
                                                                                int(mmax)), bbox_inches='tight')

        fig4 = pl.figure(4)
        fig4.clf()
        ax4 = fig4.gca()
        ax4.set_title("Tapered McKee/Offner + Kroupa PMF")
        ax4.loglog(masses, kroupa.__getattribute__(fname)(masses), label="IMF", color='k')
        ax4.loglog(masses, k_is.__getattribute__(fname)(masses, taper=True), label="IS", color='r', linestyle=':')
        ax4.loglog(masses, k_tc.__getattribute__(fname)(masses, taper=True), label="TC", color='g', linestyle='-.')
        ax4.loglog(masses, k_ca.__getattribute__(fname)(masses, taper=True), label="CA", color='orange', linestyle=(0,(5,2,1,2,1,2,1,2)))
        ax4.loglog(masses, k_2c.__getattribute__(fname)(masses, taper=True), label="2CTC", color='b', linestyle='--')
        ax4.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax4.set_ylabel("m P(m)" if mass_weighted else "Normalized P(M)")
        if mass_weighted:
            ax4.set_yscale('linear')
            ax4.set_ylim(0,0.4)
        else:
            ax4.set_ylim(np.min(kroupa(masses))/2)

        pl.legend(loc='best')
        pl.savefig(f'{plotdir}/'+'taperedaccretion_pmf_kroupa{0}_mmax{1}.pdf'.format("_integral" if mass_weighted else "",
                                                                                     int(mmax)), bbox_inches='tight')
