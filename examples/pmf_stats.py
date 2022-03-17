import imf.imf, imf.pmf, imp
imp.reload(imf.imf)
imp.reload(imf.pmf)
imp.reload(imf.imf)
imp.reload(imf.pmf)
from imf.pmf import ChabrierPMF_IS, ChabrierPMF_TC, ChabrierPMF_CA, ChabrierPMF_2CTC
from imf.pmf import KroupaPMF_IS, KroupaPMF_TC, KroupaPMF_CA, KroupaPMF_2CTC
from imf.pmf import McKeeOffner_AcceleratingSF_PMF, ChabrierPMF_AcceleratingSF_IS, ChabrierPMF_AcceleratingSF_TC, ChabrierPMF_AcceleratingSF_CA#, ChabrierPMF_AcceleratingSF_2CTC
import pylab as pl
import numpy as np

mmin = 0.033
mmax = 120

print("Normalizing.")
ChabrierPMF_IS.mmax = mmax
ChabrierPMF_TC.mmax = mmax
ChabrierPMF_CA.mmax = mmax
ChabrierPMF_2CTC.mmax = mmax

ChabrierPMF_IS.normalize(log=False, mmin=mmin, mmax=mmax)
ChabrierPMF_TC.normalize(log=False, mmin=mmin, mmax=mmax)
ChabrierPMF_CA.normalize(log=False, mmin=mmin, mmax=mmax)
ChabrierPMF_2CTC.normalize(log=False, mmin=mmin, mmax=mmax)

ChabrierPMF_AcceleratingSF_IS.mmax = mmax
ChabrierPMF_AcceleratingSF_TC.mmax = mmax
ChabrierPMF_AcceleratingSF_CA.mmax = mmax
#ChabrierPMF_AcceleratingSF_2CTC.mmax = mmax

ChabrierPMF_AcceleratingSF_IS.normalize(log=True, mmin=mmin, mmax=mmax)
ChabrierPMF_AcceleratingSF_TC.normalize(log=True, mmin=mmin, mmax=mmax)
ChabrierPMF_AcceleratingSF_CA.normalize(log=True, mmin=mmin, mmax=mmax)
#ChabrierPMF_AcceleratingSF_2CTC.normalize(log=True, mmin=mmin, mmax=mmax)



KroupaPMF_IS.mmax = mmax
KroupaPMF_TC.mmax = mmax
KroupaPMF_CA.mmax = mmax
KroupaPMF_2CTC.mmax = mmax

KroupaPMF_IS.normalize(log=False, mmin=mmin, mmax=mmax)
KroupaPMF_TC.normalize(log=False, mmin=mmin, mmax=mmax)
KroupaPMF_CA.normalize(log=False, mmin=mmin, mmax=mmax)
KroupaPMF_2CTC.normalize(log=False, mmin=mmin, mmax=mmax)

chabrierpowerlaw = imf.ChabrierPowerLaw()
chabrierpowerlaw.normalize(log=False, mmin=mmin, mmax=mmax)
kroupa = imf.Kroupa()
kroupa.normalize(log=False, mmin=mmin, mmax=mmax)

print("Done normalizing")

mfs = {'ChabrierPMF_IS': ChabrierPMF_IS,
       'ChabrierPMF_TC': ChabrierPMF_TC,
       'ChabrierPMF_CA': ChabrierPMF_CA,
       'ChabrierPMF_2CTC': ChabrierPMF_2CTC,
       'ChabrierIMF': chabrierpowerlaw,
       'KroupaPMF_IS': KroupaPMF_IS,
       'KroupaPMF_TC': KroupaPMF_TC,
       'KroupaPMF_CA': KroupaPMF_CA,
       'KroupaPMF_2CTC': KroupaPMF_2CTC,
       'KroupaIMF': kroupa,
      }

for tau in (0.1, 0.5, 1.0, 2.0):
    mfs['ChabrierPMF_AcceleratingSF_IS_tau{0}'.format(tau)] = McKeeOffner_AcceleratingSF_PMF(j=0, jf=0, tau=tau, mmax=mmax)
    mfs['ChabrierPMF_AcceleratingSF_TC_tau{0}'.format(tau)] = McKeeOffner_AcceleratingSF_PMF(j=0.5, jf=0.75, tau=tau, mmax=mmax)
    mfs['ChabrierPMF_AcceleratingSF_CA_tau{0}'.format(tau)] = McKeeOffner_AcceleratingSF_PMF(j=2/3., jf=1.0, tau=tau, mmax=mmax)

for mf in sorted(mfs):
    total = mfs[mf].m_integrate(mmin, mmax)[0]
    gt10 = mfs[mf].m_integrate(10, mmax)[0]
    print("Mass fraction for {1} M>10 = {0:0.3f}".format(gt10/total, mf))
