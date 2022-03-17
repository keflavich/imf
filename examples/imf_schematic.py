"""
I'm  not entirely sure what this figure is supposed to show; it just plots a few variants of the lognormal IMF
"""
from imf import imf
import numpy as np

if __name__ == "__main__":
    import pylab as pl
    pl.matplotlib.rc_file("/Users/adam/.matplotlib/pubfiguresrc")
    pl.rc('font', family='cmr10')

    x = np.logspace(-2,2)
    pl.clf()

    chabrier = imf.chabrier(x)
    chabrier_n = chabrier/chabrier.sum()
    chabrier3x = imf.ChabrierLogNormal(lognormal_center=0.66)(x)
    chabrier3x_n = chabrier3x/chabrier3x.sum()
    seed = imf.ChabrierLogNormal(lognormal_width=0.3)(x)
    seed_n = seed/seed.sum()
    wider = imf.ChabrierLogNormal(lognormal_width=1)(x)
    wider_n = wider/wider.sum()

    pl.loglog(x, chabrier_n, linewidth=3, alpha=0.8, label='Chabrier')
    #pl.loglog(x, imf.kroupa(x), linewidth=3, alpha=0.8, label='Kroupa')
    pl.loglog(x, chabrier3x_n, linestyle='dashed', linewidth=3, alpha=0.8, label='$3\\times$ Chabrier')
    pl.loglog(x, wider_n, linestyle='dotted', linewidth=3, alpha=0.8, label='Wider MF')
    pl.loglog(x, seed_n, linestyle='-.', linewidth=3, alpha=0.8, label='Seed MF')
    pl.xlim(1e-1, 1e2)
    pl.ylim(1e-6, 1.15)
    pl.xlabel("Stellar Mass $M_*$ ($M_{\odot}$)")
    pl.ylabel("$P(M_*)$")
    pl.legend(loc='best', fontsize=22)

    import matplotlib.ticker

    ax = pl.gca()
    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())

    pl.savefig("imf_schematic.png")


    pl.draw()
    pl.show()
