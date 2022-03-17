import imf.imf, imf.pmf, imp
from imf.pmf import ChabrierPMF_AcceleratingSF_IS, ChabrierPMF_AcceleratingSF_TC, ChabrierPMF_AcceleratingSF_CA#, ChabrierPMF_AcceleratingSF_2CTC
import pylab as pl
import numpy as np
imp.reload(imf.imf)
imp.reload(imf.pmf)
imp.reload(imf.imf)
imp.reload(imf.pmf)

mmin = 0.033

for mmax in (3, 120):

    print("Normalizing.")
    ChabrierPMF_AcceleratingSF_IS.mmax = mmax
    ChabrierPMF_AcceleratingSF_TC.mmax = mmax
    ChabrierPMF_AcceleratingSF_CA.mmax = mmax
    #ChabrierPMF_AcceleratingSF_2CTC.mmax = mmax

    ChabrierPMF_AcceleratingSF_IS.normalize(log=True, mmin=mmin, mmax=mmax)
    ChabrierPMF_AcceleratingSF_TC.normalize(log=True, mmin=mmin, mmax=mmax)
    ChabrierPMF_AcceleratingSF_CA.normalize(log=True, mmin=mmin, mmax=mmax)
    #ChabrierPMF_AcceleratingSF_2CTC.normalize(log=True, mmin=mmin, mmax=mmax)


    chabrierpowerlaw = imf.ChabrierPowerLaw()
    chabrierpowerlaw.normalize(log=True, mmin=mmin, mmax=mmax)

    print("Now plotting.")
    masses = np.logspace(np.log10(mmin), np.log10(mmax), 100)

    for mass_weighted in (True,False):
        fname = 'mass_weighted' if mass_weighted else '__call__'
        fig1 = pl.figure(1)
        fig1.clf()
        ax = fig1.gca()
        ax.set_title("Accelerating SF McKee/Offner + Chabrier PMF")
        ax.loglog(masses, chabrierpowerlaw.__getattribute__(fname)(masses), label="IMF", color='k')
        for tau, lw in zip((0.1, 1.0, 10.0), (1,2,3,)):
            ax.loglog(masses, ChabrierPMF_AcceleratingSF_IS.__getattribute__(fname)(masses, tau=tau), label="IS", color='r', linewidth=lw, linestyle=':')
            ax.loglog(masses, ChabrierPMF_AcceleratingSF_TC.__getattribute__(fname)(masses, tau=tau), label="TC", color='g', linewidth=lw, linestyle='-.')
            ax.loglog(masses, ChabrierPMF_AcceleratingSF_CA.__getattribute__(fname)(masses, tau=tau), label="CA", color='y', linewidth=lw, linestyle='-.')
        #ax.loglog(masses, ChabrierPMF_AcceleratingSF_2CTC.__getattribute__(fname)(masses), label="2CTC", color='b', linestyle='--')
        ax.set_xlabel("(Proto)Stellar Mass (M$_\odot$)")
        ax.set_ylabel("m P(m)" if mass_weighted else "Normalized P(M)")
        ax.axis([mmin, mmax, 1e-4, 1])

        pl.legend(loc='best')
        pl.savefig('acceleratingSF_pmf_chabrier{0}_mmax{1}.png'
                   .format("_integral" if mass_weighted else "", int(mmax)),
                   bbox_inches='tight')
