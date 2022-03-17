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
        cluster = np.array(cluster)
        yax = np.array(yax)
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
        cluster = np.array(cluster)
        yax = np.array(yax)
        pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        pl.gca().set_xscale('log')

        masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

        pl.plot(masses,(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        pl.xlabel("Stellar Mass")
        pl.ylabel("dN(M)/dM")
        pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
        pl.savefig("{0}_imf_figure_loglinear.png".format(name),bbox_inches='tight')

        pl.rc('font',size=20)
        pl.figure(3, figsize=(20,16))
        ax1 = pl.subplot(1,3,1)
        ax1.plot(masses,(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        ax1.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        ax2 = pl.subplot(1,3,2)
        ax2.plot(masses,(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        ax2.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        ax3 = pl.subplot(1,3,3)
        ax3.plot(masses,(massfunc(masses)),'r--',linewidth=2,alpha=0.5)
        ax3.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85,
                   linewidths=0.5, edgecolors=(0,0,0,0.25), alpha=0.95)
        ax2.set_xlabel("Stellar Mass", fontsize=30)
        ax1.set_ylabel("dN(M)/dM", fontsize=30)
        ax1.axis([min(cluster)/1.1,1,min(yax)-0.2,max(yax)+0.5])
        ax2.axis([1,5,min(yax)-0.2,max(yax[cluster>1])+0.5])
        ax3.axis([5,max(cluster)*1.1,min(yax)-0.2,max(yax[cluster>5])+0.5])
        pl.tight_layout()
        pl.savefig("{0}_imf_figure_linearlinear.png".format(name),
                   bbox_inches='tight')

        pl.rc('font',size=30)

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
                           (imf.Salpeter(alpha=2), 'Alpha2p0'),
                           (imf.Salpeter(alpha=1), 'Alpha1p0'),
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
