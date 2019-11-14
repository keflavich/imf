"""
Script to make an IMF diagram that shows dN(M)/dM vs M, then populates the area
under the curve with an appropriate number of stars colored by their
"true"(ish) color and sized by their mass.
"""
import imf
from imf import coolplot,kroupa,make_cluster
from astropy.table import Table
import numpy as np


if __name__ == "__main__":
    import pylab as pl
    pl.matplotlib.style.use('classic')
    pl.rc('font',size=30)
    pl.close(1)

    # make three figures of dN/dM vs M, one for each mass function,
    # then do it again in log-scale
    for massfunc in (imf.kroupa, imf.chabrier2005, imf.salpeter):

        # this is not a recommended way to get object names, don't do it in general.
        # (not all classes are guaranteed to have names; I know they do in this
        # case because I made and initialized the classes)
        name = massfunc.__class__.__name__

        pl.figure(1, figsize=(10,8))
        pl.clf()
        cluster,yax,colors = coolplot(1000, massfunc=massfunc)
        pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
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
        pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        pl.gca().set_xscale('log')

        masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

        pl.plot(masses,(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        pl.xlabel("Stellar Mass")
        pl.ylabel("dN(M)/dM")
        pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
        pl.savefig("{0}_imf_figure_linear.png".format(name),bbox_inches='tight')

    # make one more plot, now showing a top-heavy (shallow-tail) IMF
    massfunc = imf.Kroupa(p3=1.75)
    name='KroupaTopHeavy'
    pl.figure(1, figsize=(10,8))
    pl.clf()
    cluster,yax,colors = coolplot(1000, massfunc=massfunc)
    pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
               linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
    pl.gca().set_xscale('log')

    masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

    pl.plot(masses,np.log10(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
    pl.xlabel("Stellar Mass")
    pl.ylabel("log(dN(M)/dM)")
    pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
    pl.savefig("{0}_imf_figure_log.png".format(name),bbox_inches='tight', dpi=150)
    pl.savefig("{0}_imf_figure_log.pdf".format(name),bbox_inches='tight')

    # make two more plots, now showing a bottom- and a top-heavy  IMF
    for massfunc, name in [(imf.Salpeter(alpha=1.5), 'Alpha1p5'),
                           (imf.Salpeter(alpha=3), 'Alpha3p0')]:
        pl.figure(1, figsize=(10,8))
        pl.clf()
        cluster,yax,colors = coolplot(1000, massfunc=massfunc)
        pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        pl.gca().set_xscale('log')

        masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

        pl.plot(masses,np.log10(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        pl.xlabel("Stellar Mass")
        pl.ylabel("log(dN(M)/dM)")
        pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
        pl.savefig("{0}_imf_figure_log.png".format(name),bbox_inches='tight', dpi=150)
        pl.savefig("{0}_imf_figure_log.pdf".format(name),bbox_inches='tight')
