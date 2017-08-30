import imf.imf, imf.pmf, imp
imp.reload(imf.imf)
imp.reload(imf.pmf)
imp.reload(imf.imf)
imp.reload(imf.pmf)
from imf.pmf import ChabrierPMF_IS, ChabrierPMF_TC, ChabrierPMF_CA, ChabrierPMF_2CTC
from imf.pmf import KroupaPMF_IS, KroupaPMF_TC, KroupaPMF_CA, KroupaPMF_2CTC
import pylab as pl
import numpy as np

ChabrierPMF_IS.normalize(log=True)
ChabrierPMF_TC.normalize(log=True)
ChabrierPMF_CA.normalize(log=True)
ChabrierPMF_2CTC.normalize(log=True)

KroupaPMF_IS.normalize(log=True)
KroupaPMF_TC.normalize(log=True)
KroupaPMF_CA.normalize(log=True)
KroupaPMF_2CTC.normalize(log=True)

chabrier2005 = imf.Chabrier2005()
chabrier2005.normalize(log=True)
kroupa = imf.Kroupa()
kroupa.normalize(log=True)

masses = np.logspace(np.log10(0.033), np.log10(3.0), 100)

fig1 = pl.figure(1)
fig1.clf()
ax = fig1.gca()
ax.set_title("Steady State McKee/Offner + Chabrier PMF")
ax.semilogx(masses, chabrier2005(masses), label="IMF", color='k')
ax.semilogx(masses, ChabrierPMF_IS(masses), label="IS", color='r', linestyle=':')
ax.semilogx(masses, ChabrierPMF_TC(masses), label="TC", color='g', linestyle='-.')
ax.semilogx(masses, ChabrierPMF_CA(masses), label="CA", color='y', linestyle='-.')
ax.semilogx(masses, ChabrierPMF_2CTC(masses), label="2CTC", color='b', linestyle='--')

pl.legend(loc='best')


fig2 = pl.figure(2)
fig2.clf()
ax = fig2.gca()
ax.set_title("Tapered McKee/Offner + Chabrier PMF")
ax.semilogx(masses, chabrier2005(masses), label="IMF", color='k')
ax.semilogx(masses, ChabrierPMF_IS(masses, taper=True), label="IS", color='r', linestyle=':')
ax.semilogx(masses, ChabrierPMF_TC(masses, taper=True), label="TC", color='g', linestyle='-.')
ax.semilogx(masses, ChabrierPMF_CA(masses, taper=True), label="CA", color='y', linestyle='-.')
ax.semilogx(masses, ChabrierPMF_2CTC(masses, taper=True), label="2CTC", color='b', linestyle='--')

pl.legend(loc='best')

# fig2 = pl.figure(2)
# ax2 = fig2.gca()
# ax2.cla()
# ax2.semilogx(masses, imf.Chabrier2005()(masses), label='IMF', color='k')
# ax2.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-4/3.), label='CA_Num', color='y')
# ax2.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-1), label='IS_Num')
# ax2.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-5/4.), label='TC_Num')
# 
# pl.legend(loc='best')
# 
# fig3 = pl.figure(3)
# ax3 = fig3.gca()
# ax3.cla()
# ax3.semilogx(masses, imf.Chabrier2005()(masses), label='IMF', color='k')
# ax3.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-1.), label='CA_Num', color='y')
# ax3.semilogx(masses, imf.Chabrier2005()(masses)*masses**(0), label='IS_Num')
# ax3.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-3/4.), label='TC_Num')
# 
# pl.legend(loc='best')


fig3 = pl.figure(3)
fig3.clf()
ax3 = fig3.gca()
ax3.set_title("Steady State McKee/Offner + Kroupa PMF")
ax3.semilogx(masses, kroupa(masses), label="IMF", color='k')
ax3.semilogx(masses, KroupaPMF_IS(masses), label="IS", color='r', linestyle=':')
ax3.semilogx(masses, KroupaPMF_TC(masses), label="TC", color='g', linestyle='-.')
ax3.semilogx(masses, KroupaPMF_CA(masses), label="CA", color='y', linestyle='-.')
ax3.semilogx(masses, KroupaPMF_2CTC(masses), label="2CTC", color='b', linestyle='--')

pl.legend(loc='best')


fig4 = pl.figure(4)
fig4.clf()
ax4 = fig4.gca()
ax4.set_title("Tapered McKee/Offner + Kroupa PMF")
ax4.semilogx(masses, kroupa(masses), label="IMF", color='k')
ax4.semilogx(masses, KroupaPMF_IS(masses, taper=True), label="IS", color='r', linestyle=':')
ax4.semilogx(masses, KroupaPMF_TC(masses, taper=True), label="TC", color='g', linestyle='-.')
ax4.semilogx(masses, KroupaPMF_CA(masses, taper=True), label="CA", color='y', linestyle='-.')
ax4.semilogx(masses, KroupaPMF_2CTC(masses, taper=True), label="2CTC", color='b', linestyle='--')

pl.legend(loc='best')

