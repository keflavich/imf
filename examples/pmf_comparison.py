import imf.imf, imf.pmf, imp
imp.reload(imf.imf)
imp.reload(imf.pmf)
imp.reload(imf.imf)
imp.reload(imf.pmf)
from imf.pmf import ChabrierPMF_IS, ChabrierPMF_TC, ChabrierPMF_CA
import pylab as pl
import numpy as np

masses = np.logspace(-2, np.log10(3.0), 100)

pl.clf()
ax = pl.gca()
ax.semilogx(masses, imf.Chabrier2005()(masses), label="IMF", color='k')
ax.semilogx(masses, ChabrierPMF_IS(masses), label="IS", color='r')
ax.semilogx(masses, ChabrierPMF_TC(masses), label="TC", color='g')
ax.semilogx(masses, ChabrierPMF_CA(masses), label="CA", color='y')

pl.legend(loc='best')

fig2 = pl.figure(2)
ax2 = fig2.gca()
ax2.cla()
ax2.semilogx(masses, imf.Chabrier2005()(masses), label='IMF', color='k')
ax2.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-4/3.), label='CA_Num', color='y')
ax2.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-1), label='IS_Num')
ax2.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-5/4.), label='TC_Num')

pl.legend(loc='best')

fig3 = pl.figure(3)
ax3 = fig3.gca()
ax3.cla()
ax3.semilogx(masses, imf.Chabrier2005()(masses), label='IMF', color='k')
ax3.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-1.), label='CA_Num', color='y')
ax3.semilogx(masses, imf.Chabrier2005()(masses)*masses**(0), label='IS_Num')
ax3.semilogx(masses, imf.Chabrier2005()(masses)*masses**(-3/4.), label='TC_Num')

pl.legend(loc='best')


