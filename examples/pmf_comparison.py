import imf.imf, imf.pmf, imp
imp.reload(imf.imf)
imp.reload(imf.pmf)
imp.reload(imf.imf)
imp.reload(imf.pmf)
from imf.pmf import ChabrierPMF_IS, ChabrierPMF_TC, ChabrierPMF_CA, ChabrierPMF_2CTC
from imf.pmf import KroupaPMF_IS, KroupaPMF_TC, KroupaPMF_CA, KroupaPMF_2CTC
import pylab as pl
import numpy as np

mmin = 0.033
mmax = 3.0

for mmax in (3, 120):

    print("Normalizing.")
    ChabrierPMF_IS.mmax = mmax
    ChabrierPMF_TC.mmax = mmax
    ChabrierPMF_CA.mmax = mmax
    ChabrierPMF_2CTC.mmax = mmax

    ChabrierPMF_IS.normalize(log=True, mmin=mmin, mmax=mmax)
    ChabrierPMF_TC.normalize(log=True, mmin=mmin, mmax=mmax)
    ChabrierPMF_CA.normalize(log=True, mmin=mmin, mmax=mmax)
    ChabrierPMF_2CTC.normalize(log=True, mmin=mmin, mmax=mmax)

    KroupaPMF_IS.mmax = mmax
    KroupaPMF_TC.mmax = mmax
    KroupaPMF_CA.mmax = mmax
    KroupaPMF_2CTC.mmax = mmax

    KroupaPMF_IS.normalize(log=True, mmin=mmin, mmax=mmax)
    KroupaPMF_TC.normalize(log=True, mmin=mmin, mmax=mmax)
    KroupaPMF_CA.normalize(log=True, mmin=mmin, mmax=mmax)
    KroupaPMF_2CTC.normalize(log=True, mmin=mmin, mmax=mmax)

    chabrier2005 = imf.Chabrier2005()
    chabrier2005.normalize(log=True, mmin=mmin, mmax=mmax)
    kroupa = imf.Kroupa()
    kroupa.normalize(log=True, mmin=mmin, mmax=mmax)

    print("Now plotting.")
    masses = np.logspace(np.log10(mmin), np.log10(mmax), 100)

    for integral_form in (False,):
        fig1 = pl.figure(1)
        fig1.clf()
        ax = fig1.gca()
        ax.set_title("Steady State McKee/Offner + Chabrier PMF")
        ax.loglog(masses, chabrier2005(masses, integral_form=integral_form), label="IMF", color='k')
        ax.loglog(masses, ChabrierPMF_IS(masses, integral_form=integral_form), label="IS", color='r', linestyle=':')
        ax.loglog(masses, ChabrierPMF_TC(masses, integral_form=integral_form), label="TC", color='g', linestyle='-.')
        ax.loglog(masses, ChabrierPMF_CA(masses, integral_form=integral_form), label="CA", color='y', linestyle='-.')
        ax.loglog(masses, ChabrierPMF_2CTC(masses, integral_form=integral_form), label="2CTC", color='b', linestyle='--')
        ax.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax.set_ylabel("N(M)dm" if integral_form else "Normalized P(M)")
        ax.axis([mmin, mmax, 1e-4, 1])

        pl.legend(loc='best')
        pl.savefig('steadystate_pmf_chabrier{0}_mmax{1}.png'.format("_integral" if integral_form else "",
                                                                    int(mmax)), bbox_inches='tight')


        fig2 = pl.figure(2)
        fig2.clf()
        ax = fig2.gca()
        ax.set_title("Tapered McKee/Offner + Chabrier PMF")
        ax.loglog(masses, chabrier2005(masses, integral_form=integral_form), label="IMF", color='k')
        ax.loglog(masses, ChabrierPMF_IS(masses, integral_form=integral_form, taper=True), label="IS", color='r', linestyle=':')
        ax.loglog(masses, ChabrierPMF_TC(masses, integral_form=integral_form, taper=True), label="TC", color='g', linestyle='-.')
        ax.loglog(masses, ChabrierPMF_CA(masses, integral_form=integral_form, taper=True), label="CA", color='y', linestyle='-.')
        ax.loglog(masses, ChabrierPMF_2CTC(masses, integral_form=integral_form, taper=True), label="2CTC", color='b', linestyle='--')
        ax.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax.set_ylabel("N(M)dm" if integral_form else "Normalized P(M)")
        ax.axis([mmin, mmax, 1e-4, 1])

        pl.legend(loc='best')
        pl.savefig('taperedaccretion_pmf_chabrier{0}_mmax{1}.png'.format("_integral" if integral_form else "",
                                                                         int(mmax)), bbox_inches='tight')


        fig3 = pl.figure(3)
        fig3.clf()
        ax3 = fig3.gca()
        ax3.set_title("Steady State McKee/Offner + Kroupa PMF")
        ax3.loglog(masses, kroupa(masses, integral_form=integral_form), label="IMF", color='k')
        ax3.loglog(masses, KroupaPMF_IS(masses, integral_form=integral_form), label="IS", color='r', linestyle=':')
        ax3.loglog(masses, KroupaPMF_TC(masses, integral_form=integral_form), label="TC", color='g', linestyle='-.')
        ax3.loglog(masses, KroupaPMF_CA(masses, integral_form=integral_form), label="CA", color='y', linestyle='-.')
        ax3.loglog(masses, KroupaPMF_2CTC(masses, integral_form=integral_form), label="2CTC", color='b', linestyle='--')
        ax3.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax3.set_ylabel("N(M)dm" if integral_form else "Normalized P(M)")
        ax3.axis([mmin, mmax, 1e-4, 1])

        pl.legend(loc='best')
        pl.savefig('steadystate_pmf_kroupa{0}_mmax{1}.png'.format("_integral" if integral_form else "",
                                                                  int(mmax)), bbox_inches='tight')


        fig4 = pl.figure(4)
        fig4.clf()
        ax4 = fig4.gca()
        ax4.set_title("Tapered McKee/Offner + Kroupa PMF")
        ax4.loglog(masses, kroupa(masses, integral_form=integral_form), label="IMF", color='k')
        ax4.loglog(masses, KroupaPMF_IS(masses, integral_form=integral_form, taper=True), label="IS", color='r', linestyle=':')
        ax4.loglog(masses, KroupaPMF_TC(masses, integral_form=integral_form, taper=True), label="TC", color='g', linestyle='-.')
        ax4.loglog(masses, KroupaPMF_CA(masses, integral_form=integral_form, taper=True), label="CA", color='y', linestyle='-.')
        ax4.loglog(masses, KroupaPMF_2CTC(masses, integral_form=integral_form, taper=True), label="2CTC", color='b', linestyle='--')
        ax4.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax4.set_ylabel("N(M)dm" if integral_form else "Normalized P(M)")
        ax4.axis([mmin, mmax, 1e-4, 1])

        pl.legend(loc='best')
        pl.savefig('taperedaccretion_pmf_kroupa{0}_mmax{1}.png'.format("_integral" if integral_form else "",
                                                                       int(mmax)), bbox_inches='tight')
