# plot the PMF for accelerating star formation over various timescales
# in comparison to the underlying IMF in both linear/log forms

import pylab as pl
import numpy as np
import os
plotdir = 'plots/pmf'
os.makedirs(plotdir, exist_ok=True)

import imf.imf, imf.pmf
from imf.pmf import PMF, PMF_2C

mmin = 0.033
chabrier = imf.chabrier2005
chabrier.mmin = mmin

for mmax in (3, 120):

    print(f'Beginning mmax = {mmax}.')

    chabrier.mmax = mmax
    chabrier.normalize()

    c_is = PMF(chabrier, history='is')
    c_tc = PMF(chabrier, history='tc')
    c_ca = PMF(chabrier, history='ca')
    c_2c = PMF_2C(chabrier, history='tc')

    print("Now plotting.")
    masses = np.logspace(np.log10(mmin), np.log10(mmax), 100)

    for mass_weighted in (True, False):
        fname = 'mass_weighted' if mass_weighted else '__call__'
        fig1 = pl.figure(1)
        fig1.clf()
        ax = fig1.gca()
        ax.set_title("Accelerating SF McKee/Offner + Chabrier PMF")
        ax.loglog(masses, chabrier.__getattribute__(fname)(masses), label="IMF", color='k')
        for tau, lw in zip((0.1, 1.0, 10.0), (1, 2, 3,)):
            c_is.tau = tau
            ax.loglog(masses, c_is.__getattribute__(fname)(masses, accelerating=True), label="IS", color='r', linewidth=lw, linestyle=':')
            c_tc.tau = tau
            ax.loglog(masses, c_tc.__getattribute__(fname)(masses, accelerating=True), label="TC", color='g', linewidth=lw, linestyle='-.')
            c_ca.tau = tau
            ax.loglog(masses, c_ca.__getattribute__(fname)(masses, accelerating=True), label="CA", color='y', linewidth=lw, linestyle='-.')
            c_2c.tau = tau
            ax.loglog(masses, c_2c.__getattribute__(fname)(masses, accelerating=True), label="2CTC", color='b', linestyle='--')
        ax.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax.set_ylabel("m P(m)" if mass_weighted else "Normalized P(M)")
        if mass_weighted:
            ax.set_yscale('linear')
            ax.set_ylim(0, 0.4)
        else:
            ax.axis([mmin, mmax, 1e-4, 1])

        pl.legend(loc='best')
        pl.savefig(f'{plotdir}/' +
                   'acceleratingSF_pmf_chabrier{0}_mmax{1}.pdf'.format("_integral" if mass_weighted else "", int(mmax)),
                   bbox_inches='tight')
