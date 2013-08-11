from imf import coolplot,kroupa,make_cluster
import numpy as np


if __name__ == "__main__":
    import pylab as pl
    pl.rc('font',size=30)
    pl.figure(1)
    pl.clf()
    cluster,yax,colors = coolplot(1000, massfunc=kroupa)
    pl.scatter(cluster, yax, c=colors, s=np.log10(cluster+3)*85)
    pl.gca().set_xscale('log')

    masses = np.logspace(np.log10(cluster.min()), np.log10(cluster.max()),10000)

    pl.plot(masses,np.log10(kroupa(masses)),'r--',linewidth=2,alpha=0.5)
    pl.xlabel("Stellar Mass")
    pl.ylabel("log(dN(M)/dM)")
    pl.gca().axis([min(cluster)/1.1,max(cluster)*1.1,min(yax)-0.2,max(yax)+0.5])
    pl.savefig("plots/imf.png",bbox_inches='tight')

    #figure(2)
    #clf()
    #def cloud_massfunc(mass,m0=1e3,alpha=1.1):
    #    return (mass/m0)**-alpha

    #clouds = make_cluster(1e8, massfunc=cloud_massfunc)
