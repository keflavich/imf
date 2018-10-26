import imf
from imf import coolplot,kroupa,make_cluster
from astropy.table import Table
import numpy as np


if __name__ == "__main__":
    import pylab as pl
    pl.matplotlib.style.use('classic')
    pl.rc('font',size=30)
    pl.close(1)

    for massfunc in (imf.kroupa, imf.chabrier2005, imf.salpeter):
        name = massfunc.__class__.__name__
        pl.figure(1, figsize=(10,8))
        pl.clf()
        cluster,yax,colors = coolplot(1000, massfunc=massfunc)
        pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85)
        pl.gca().set_xscale('log')

        masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

        pl.plot(masses,np.log10(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        pl.xlabel("Stellar Mass")
        pl.ylabel("log(dN(M)/dM)")
        pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
        pl.savefig("{0}_imf_figure_log.png".format(name),bbox_inches='tight', dpi=150)
        pl.savefig("{0}_imf_figure_log.pdf".format(name),bbox_inches='tight')

        pl.figure(2, figsize=(20,16))
        pl.clf()
        cluster,yax,colors = coolplot(1000, massfunc=massfunc, log=False)
        pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85)
        pl.gca().set_xscale('log')

        masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

        pl.plot(masses,(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        pl.xlabel("Stellar Mass")
        pl.ylabel("dN(M)/dM")
        pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
        pl.savefig("{0}_imf_figure_linear.png".format(name),bbox_inches='tight')

        #figure(2)
        #clf()
        #def cloud_massfunc(mass,m0=1e3,alpha=1.1):
        #    return (mass/m0)**-alpha

        #clouds = make_cluster(1e8, massfunc=cloud_massfunc)

    massfunc = imf.Kroupa(p3=1.75)
    name='KroupaTopHeavy'
    pl.figure(1, figsize=(10,8))
    pl.clf()
    cluster,yax,colors = coolplot(1000, massfunc=massfunc)
    pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85)
    pl.gca().set_xscale('log')

    masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

    pl.plot(masses,np.log10(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
    pl.xlabel("Stellar Mass")
    pl.ylabel("log(dN(M)/dM)")
    pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
    pl.savefig("{0}_imf_figure_log.png".format(name),bbox_inches='tight', dpi=150)
    pl.savefig("{0}_imf_figure_log.pdf".format(name),bbox_inches='tight')

    # # FAILURE: table is inadequate
    # hr diagram
    # pl.figure(2).clf()

    # #tbl = Table.read('/Users/adam/repos/imf/imf/data/pecaut2013_table_with_lyclum.txt', format='ascii.fixed_width')
    # #tbl = Table.read('/Users/adam/repos/imf/imf/data/EEM_dwarf_UBVIJHK_colors_Teff_noheader.txt', format='ascii.commented_header')
    # tbl = Table.read('/Users/adam/repos/imf/imf/data/EEM_dwarf_UBVIJHK_colors_Teff_noheader.txt', format='ascii.commented_header')
    # ok = tbl['Msun'] < 500
    # colors = imf.color_of_cluster(tbl['Msun'][ok])
    # pl.gca().set_xscale('log')
    # pl.gca().set_yscale('log')
    # pl.scatter(tbl['Teff'][ok], tbl['Msun'][ok], c=colors,
    #            s=tbl['logL'][ok]*85)
